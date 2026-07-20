#!/usr/bin/env python3
"""用前向因果干预分析 5.7529% 候选的接口失配机制。"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parents[2]
LAB04_ROOT = ROOT / "lab" / "04_tensorshield"
if str(LAB04_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB04_ROOT))
import candidate as lab04  # noqa: E402


def load_lab08():
    path = ROOT / "lab" / "08_leakage" / "run.py"
    spec = importlib.util.spec_from_file_location("tsdp_lab08_run", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lab08 = load_lab08()
prefix = lab04.prefix

EXPERIMENT = "09_mechanism"
ANALYSIS_PROTOCOL = "forward_causal_interface_v1"
SEEDS = tuple(range(43, 53))
STRENGTHS = (0.0, 0.25, 0.50, 0.75, 1.0)
GROUP_ORDER = (
    "layer1_0_conv1",
    "layer1_1_conv1",
    "layer2_0_conv1",
    "layer2_1_conv1",
    "layer3_0_conv1",
    "bn_gamma",
    "head",
)
BN_GROUP_ORDER = ("stem", "block_bn1", "block_bn2", "downsample")
GROUP_STATES = {
    "layer1_0_conv1": ("layer1.0.conv1.weight",),
    "layer1_1_conv1": ("layer1.1.conv1.weight",),
    "layer2_0_conv1": ("layer2.0.conv1.weight",),
    "layer2_1_conv1": ("layer2.1.conv1.weight",),
    "layer3_0_conv1": ("layer3.0.conv1.weight",),
    "head": ("last_linear.weight", "last_linear.bias"),
}
BLOCKS = tuple(f"layer{stage}.{block}" for stage in range(1, 5) for block in range(2))
LAMBDA_FIELDS = (
    "seed",
    "case",
    "utilization_strength",
    "state_sha256",
    "soft_ce",
    "posterior_kl",
    "fidelity",
    "prediction_entropy",
    "feature_rms",
    "feature_l2",
    "logit_rms",
    "norm_matched_soft_ce",
    "norm_matched_posterior_kl",
    "norm_matched_fidelity",
    "norm_matched_prediction_entropy",
    "norm_matched_logit_rms",
)
LATTICE_FIELDS = (
    "seed",
    "subset",
    "restored_group_count",
    "restored_groups",
    "restored_param_count",
    "soft_ce",
    "posterior_kl",
    "fidelity",
    "prediction_entropy",
    "feature_rms",
    "feature_l2",
    "logit_rms",
)
ATTRIBUTION_FIELDS = (
    "seed",
    "group",
    "param_count",
    "reveal_alone_kl_recovery",
    "hide_alone_kl_damage",
    "shapley_kl_recovery",
    "head_context_interaction",
)
SEAM_FIELDS = (
    "seed",
    "block",
    "stage",
    "block_index",
    "variant",
    "public_state_names",
    "public_param_count",
    "candidate_conv1",
    "soft_ce",
    "posterior_kl",
    "fidelity",
    "prediction_entropy",
    "feature_rms",
    "feature_l2",
    "logit_rms",
    "kl_minus_victim",
)
BN_FIELDS = (
    "seed",
    "subset",
    "public_group_count",
    "public_groups",
    "public_state_count",
    "public_param_count",
    "soft_ce",
    "posterior_kl",
    "fidelity",
    "prediction_entropy",
    "feature_rms",
    "feature_l2",
    "logit_rms",
    "kl_minus_victim",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对来源、七组状态、端点和残差块，不运行前向分析。",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_tsv(
    path: Path,
    rows: list[dict[str, object]],
    fields: tuple[str, ...],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def strength_case(strength: float) -> str:
    return f"lambda_{int(round(strength * 100)):03d}"


def load_batches(query, *, device: torch.device, num_workers: int, seed: int):
    _, validation_loader = lab08.build_query_loaders(
        query,
        device=device,
        num_workers=num_workers,
        seed=seed,
    )
    batches = []
    for images, posteriors, labels in validation_loader:
        batches.append((images.cpu(), posteriors.cpu(), labels.cpu()))
    if sum(batch[0].size(0) for batch in batches) != 100:
        raise RuntimeError(f"seed {seed} 的 query-validation 不是 100 条。")
    return batches


def head_features(model: torch.nn.Module, images: torch.Tensor):
    feature_map = model.features(images)
    features = model.avgpool(feature_map).flatten(1)
    return features, model.last_linear(features)


@torch.inference_mode()
def collect_features(
    model: torch.nn.Module,
    batches,
    device: torch.device,
) -> list[torch.Tensor]:
    model.eval()
    return [
        head_features(model, images.to(device, non_blocking=True))[0].cpu()
        for images, _, _ in batches
    ]


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    batches,
    device: torch.device,
    *,
    norm_reference: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    model.eval()
    sample_count = 0
    class_count = None
    feature_count = None
    sums = {
        "soft_ce": 0.0,
        "posterior_kl": 0.0,
        "fidelity": 0.0,
        "prediction_entropy": 0.0,
        "feature_square": 0.0,
        "feature_l2": 0.0,
        "logit_square": 0.0,
    }
    norm_sums = {
        "soft_ce": 0.0,
        "posterior_kl": 0.0,
        "fidelity": 0.0,
        "prediction_entropy": 0.0,
        "logit_square": 0.0,
    }
    for batch_index, (images, posteriors, labels) in enumerate(batches):
        images = images.to(device, non_blocking=True)
        posteriors = posteriors.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        features, logits = head_features(model, images)
        log_probabilities = functional.log_softmax(logits, dim=1)
        probabilities = log_probabilities.exp()
        batch_size = images.size(0)
        sample_count += batch_size
        class_count = logits.size(1)
        feature_count = features.size(1)
        sums["soft_ce"] += float(
            -(posteriors * log_probabilities).sum().item()
        )
        sums["posterior_kl"] += float(
            functional.kl_div(
                log_probabilities,
                posteriors,
                reduction="sum",
            ).item()
        )
        sums["fidelity"] += int((logits.argmax(dim=1) == labels).sum().item())
        sums["prediction_entropy"] += float(
            -(probabilities * log_probabilities).sum().item()
        )
        sums["feature_square"] += float(features.square().sum().item())
        sums["feature_l2"] += float(features.norm(dim=1).sum().item())
        sums["logit_square"] += float(logits.square().sum().item())

        if norm_reference is not None:
            reference = norm_reference[batch_index].to(device, non_blocking=True)
            scale = reference.norm(dim=1, keepdim=True) / features.norm(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-12)
            norm_logits = model.last_linear(features * scale)
            norm_log_probabilities = functional.log_softmax(norm_logits, dim=1)
            norm_probabilities = norm_log_probabilities.exp()
            norm_sums["soft_ce"] += float(
                -(posteriors * norm_log_probabilities).sum().item()
            )
            norm_sums["posterior_kl"] += float(
                functional.kl_div(
                    norm_log_probabilities,
                    posteriors,
                    reduction="sum",
                ).item()
            )
            norm_sums["fidelity"] += int(
                (norm_logits.argmax(dim=1) == labels).sum().item()
            )
            norm_sums["prediction_entropy"] += float(
                -(norm_probabilities * norm_log_probabilities).sum().item()
            )
            norm_sums["logit_square"] += float(norm_logits.square().sum().item())

    if sample_count != 100 or class_count is None or feature_count is None:
        raise RuntimeError("前向分析没有覆盖完整 query-validation。")
    result = {
        "soft_ce": sums["soft_ce"] / sample_count,
        "posterior_kl": sums["posterior_kl"] / sample_count,
        "fidelity": sums["fidelity"] / sample_count,
        "prediction_entropy": sums["prediction_entropy"] / sample_count,
        "feature_rms": math.sqrt(
            sums["feature_square"] / (sample_count * feature_count)
        ),
        "feature_l2": sums["feature_l2"] / sample_count,
        "logit_rms": math.sqrt(
            sums["logit_square"] / (sample_count * class_count)
        ),
    }
    if norm_reference is not None:
        result.update(
            {
                "norm_matched_soft_ce": norm_sums["soft_ce"] / sample_count,
                "norm_matched_posterior_kl":
                    norm_sums["posterior_kl"] / sample_count,
                "norm_matched_fidelity": norm_sums["fidelity"] / sample_count,
                "norm_matched_prediction_entropy":
                    norm_sums["prediction_entropy"] / sample_count,
                "norm_matched_logit_rms": math.sqrt(
                    norm_sums["logit_square"] / (sample_count * class_count)
                ),
            }
        )
    return result


def build_groups(victim: torch.nn.Module, endpoints) -> dict[str, tuple[str, ...]]:
    groups = dict(GROUP_STATES)
    groups["bn_gamma"] = tuple(endpoints["bn_gamma"])
    selected = {
        item["state_name"]
        for item in endpoints["selected_units"]
    }
    grouped = {name for states in groups.values() for name in states}
    if grouped != selected or sum(len(states) for states in groups.values()) != 27:
        raise RuntimeError("Lab09 七组没有精确覆盖 5.7529% 的 27 个 state。")
    state = victim.state_dict()
    if any(name not in state for name in grouped):
        raise RuntimeError("Lab09 七组包含未知 victim state。")
    return groups


def build_bn_groups(
    bn_gamma: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    groups = {
        "stem": tuple(name for name in bn_gamma if name == "bn1.weight"),
        "block_bn1": tuple(
            name
            for name in bn_gamma
            if name != "bn1.weight"
            and ".downsample." not in name
            and name.endswith(".bn1.weight")
        ),
        "block_bn2": tuple(
            name
            for name in bn_gamma
            if ".downsample." not in name and name.endswith(".bn2.weight")
        ),
        "downsample": tuple(
            name for name in bn_gamma if ".downsample.1.weight" in name
        ),
    }
    expected_counts = {
        "stem": 1,
        "block_bn1": 8,
        "block_bn2": 8,
        "downsample": 3,
    }
    if (
        {group: len(names) for group, names in groups.items()} != expected_counts
        or {name for names in groups.values() for name in names} != set(bn_gamma)
    ):
        raise RuntimeError("Lab09 的四类 BN gamma 没有精确覆盖 20 个 state。")
    return groups


def mixed_state(
    base_state: dict[str, torch.Tensor],
    replacement_state: dict[str, torch.Tensor],
    names,
) -> dict[str, torch.Tensor]:
    state = dict(base_state)
    for name in names:
        state[name] = replacement_state[name]
    return state


def state_dict_equal(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[name], right[name])
        for name in left
    )


def build_bn_rows(
    *,
    seed: int,
    victim: torch.nn.Module,
    public: torch.nn.Module,
    groups: dict[str, tuple[str, ...]],
    batches,
    device: torch.device,
) -> list[dict[str, object]]:
    victim_state = {
        name: value.detach().cpu()
        for name, value in victim.state_dict().items()
    }
    public_state = {
        name: value.detach().cpu()
        for name, value in public.state_dict().items()
    }
    model = copy.deepcopy(victim).to(device)
    victim_metrics = evaluate_model(model, batches, device)
    rows = []
    for subset in range(1 << len(BN_GROUP_ORDER)):
        public_groups = [
            group
            for index, group in enumerate(BN_GROUP_ORDER)
            if subset & (1 << index)
        ]
        public_names = [
            name
            for group in public_groups
            for name in groups[group]
        ]
        model.load_state_dict(
            mixed_state(victim_state, public_state, public_names),
            strict=True,
        )
        metrics = evaluate_model(model, batches, device)
        rows.append(
            {
                "seed": seed,
                "subset": subset,
                "public_group_count": len(public_groups),
                "public_groups": ",".join(public_groups),
                "public_state_count": len(public_names),
                "public_param_count": sum(
                    victim_state[name].numel()
                    for name in public_names
                ),
                **metrics,
                "kl_minus_victim":
                    metrics["posterior_kl"] - victim_metrics["posterior_kl"],
            }
        )
    del model
    return rows


def build_lattice_rows(
    *,
    seed: int,
    hybrid: torch.nn.Module,
    victim: torch.nn.Module,
    groups: dict[str, tuple[str, ...]],
    batches,
    device: torch.device,
) -> list[dict[str, object]]:
    base_state = {name: value.detach().cpu() for name, value in hybrid.state_dict().items()}
    victim_state = {name: value.detach().cpu() for name, value in victim.state_dict().items()}
    model = copy.deepcopy(hybrid).to(device)
    rows = []
    for subset in range(1 << len(GROUP_ORDER)):
        restored_groups = [
            group
            for index, group in enumerate(GROUP_ORDER)
            if subset & (1 << index)
        ]
        restored_names = [
            name
            for group in restored_groups
            for name in groups[group]
        ]
        state = mixed_state(base_state, victim_state, restored_names)
        model.load_state_dict(state, strict=True)
        metrics = evaluate_model(model, batches, device)
        rows.append(
            {
                "seed": seed,
                "subset": subset,
                "restored_group_count": len(restored_groups),
                "restored_groups": ",".join(restored_groups),
                "restored_param_count": sum(
                    victim_state[name].numel()
                    for name in restored_names
                    if torch.is_floating_point(victim_state[name])
                    or victim_state[name].dtype == torch.bool
                    or not name.endswith("num_batches_tracked")
                ),
                **metrics,
            }
        )
    full_names = [name for group in GROUP_ORDER for name in groups[group]]
    full_state = mixed_state(base_state, victim_state, full_names)
    if not state_dict_equal(full_state, victim_state):
        raise RuntimeError("Lab09 七组全部恢复后不等于 victim。")
    del model
    return rows


def build_attribution_rows(
    lattice_rows: list[dict[str, object]],
    groups: dict[str, tuple[str, ...]],
    victim: torch.nn.Module,
) -> list[dict[str, object]]:
    by_seed = {}
    for row in lattice_rows:
        by_seed.setdefault(int(row["seed"]), {})[int(row["subset"])] = row
    all_subset = (1 << len(GROUP_ORDER)) - 1
    head_index = GROUP_ORDER.index("head")
    head_bit = 1 << head_index
    victim_state = victim.state_dict()
    rows = []
    for seed in SEEDS:
        subset_rows = by_seed[seed]
        if set(subset_rows) != set(range(all_subset + 1)):
            raise RuntimeError(f"seed {seed} 的 128 个 oracle-reveal 组合不完整。")
        shapley_total = 0.0
        for group_index, group in enumerate(GROUP_ORDER):
            bit = 1 << group_index
            shapley = 0.0
            for subset in range(all_subset + 1):
                if subset & bit:
                    continue
                size = subset.bit_count()
                weight = (
                    math.factorial(size)
                    * math.factorial(len(GROUP_ORDER) - size - 1)
                    / math.factorial(len(GROUP_ORDER))
                )
                marginal = (
                    float(subset_rows[subset]["posterior_kl"])
                    - float(subset_rows[subset | bit]["posterior_kl"])
                )
                shapley += weight * marginal
            reveal_alone = (
                float(subset_rows[0]["posterior_kl"])
                - float(subset_rows[bit]["posterior_kl"])
            )
            hide_alone = (
                float(subset_rows[all_subset ^ bit]["posterior_kl"])
                - float(subset_rows[all_subset]["posterior_kl"])
            )
            if group == "head":
                head_interaction = 0.0
            else:
                without_head = reveal_alone
                with_head = (
                    float(subset_rows[head_bit]["posterior_kl"])
                    - float(subset_rows[head_bit | bit]["posterior_kl"])
                )
                head_interaction = with_head - without_head
            shapley_total += shapley
            rows.append(
                {
                    "seed": seed,
                    "group": group,
                    "param_count": sum(
                        victim_state[name].numel()
                        for name in groups[group]
                    ),
                    "reveal_alone_kl_recovery": reveal_alone,
                    "hide_alone_kl_damage": hide_alone,
                    "shapley_kl_recovery": shapley,
                    "head_context_interaction": head_interaction,
                }
            )
        expected_total = (
            float(subset_rows[0]["posterior_kl"])
            - float(subset_rows[all_subset]["posterior_kl"])
        )
        if not math.isclose(shapley_total, expected_total, abs_tol=1e-8):
            raise RuntimeError(f"seed {seed} 的 Shapley 分解不闭合。")
    return rows


def seam_variants(block: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        ("conv1_weight", (f"{block}.conv1.weight",)),
        ("bn1_gamma", (f"{block}.bn1.weight",)),
        (
            "conv1_weight_bn1_gamma",
            (f"{block}.conv1.weight", f"{block}.bn1.weight"),
        ),
        ("conv2_weight", (f"{block}.conv2.weight",)),
        ("bn2_gamma", (f"{block}.bn2.weight",)),
        (
            "conv2_weight_bn2_gamma",
            (f"{block}.conv2.weight", f"{block}.bn2.weight"),
        ),
    )


def build_seam_rows(
    *,
    seed: int,
    victim: torch.nn.Module,
    public: torch.nn.Module,
    candidate_weights: set[str],
    batches,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    victim_state = {name: value.detach().cpu() for name, value in victim.state_dict().items()}
    public_state = {name: value.detach().cpu() for name, value in public.state_dict().items()}
    model = copy.deepcopy(victim).to(device)
    victim_metrics = evaluate_model(model, batches, device)
    rows = []
    for block in BLOCKS:
        stage, block_index = block.removeprefix("layer").split(".")
        for variant, names in seam_variants(block):
            if any(
                name not in public_state
                or public_state[name].shape != victim_state[name].shape
                for name in names
            ):
                raise RuntimeError(f"{block}/{variant} 的 public/victim state 不兼容。")
            model.load_state_dict(
                mixed_state(victim_state, public_state, names),
                strict=True,
            )
            metrics = evaluate_model(model, batches, device)
            rows.append(
                {
                    "seed": seed,
                    "block": block,
                    "stage": int(stage),
                    "block_index": int(block_index),
                    "variant": variant,
                    "public_state_names": ",".join(names),
                    "public_param_count": sum(victim_state[name].numel() for name in names),
                    "candidate_conv1": f"{block}.conv1.weight" in candidate_weights,
                    **metrics,
                    "kl_minus_victim":
                        metrics["posterior_kl"] - victim_metrics["posterior_kl"],
                }
            )
    del model
    return rows, victim_metrics


def aggregate_results(
    lambda_rows: list[dict[str, object]],
    attribution_rows: list[dict[str, object]],
    seam_rows: list[dict[str, object]],
    bn_rows: list[dict[str, object]],
) -> dict[str, object]:
    lambda_groups = {}
    for strength in STRENGTHS:
        case = strength_case(strength)
        rows = [row for row in lambda_rows if row["case"] == case]
        lambda_groups[case] = {
            "utilization_strength": strength,
            **{
                metric: summarize([float(row[metric]) for row in rows])
                for metric in (
                    "soft_ce",
                    "posterior_kl",
                    "fidelity",
                    "prediction_entropy",
                    "feature_rms",
                    "feature_l2",
                    "logit_rms",
                    "norm_matched_soft_ce",
                    "norm_matched_posterior_kl",
                    "norm_matched_fidelity",
                    "norm_matched_prediction_entropy",
                    "norm_matched_logit_rms",
                )
            },
        }

    attribution_groups = {}
    for group in GROUP_ORDER:
        rows = [row for row in attribution_rows if row["group"] == group]
        attribution_groups[group] = {
            "param_count": int(rows[0]["param_count"]),
            **{
                metric: summarize([float(row[metric]) for row in rows])
                for metric in (
                    "reveal_alone_kl_recovery",
                    "hide_alone_kl_damage",
                    "shapley_kl_recovery",
                    "head_context_interaction",
                )
            },
        }

    seam_groups = {}
    paired = {}
    for block in BLOCKS:
        seam_groups[block] = {}
        for variant, _ in seam_variants(block):
            rows = [
                row
                for row in seam_rows
                if row["block"] == block and row["variant"] == variant
            ]
            seam_groups[block][variant] = {
                "public_param_count": int(rows[0]["public_param_count"]),
                "candidate_conv1": bool(rows[0]["candidate_conv1"]),
                **{
                    metric: summarize([float(row[metric]) for row in rows])
                    for metric in (
                        "soft_ce",
                        "posterior_kl",
                        "fidelity",
                        "feature_rms",
                        "logit_rms",
                        "kl_minus_victim",
                    )
                },
            }
        paired[block] = {}
        for comparison, left_variant, right_variant in (
            ("weight", "conv1_weight", "conv2_weight"),
            ("gamma", "bn1_gamma", "bn2_gamma"),
            (
                "combined",
                "conv1_weight_bn1_gamma",
                "conv2_weight_bn2_gamma",
            ),
        ):
            differences = []
            values_by_seed = {}
            for seed in SEEDS:
                left = next(
                    row
                    for row in seam_rows
                    if row["seed"] == seed
                    and row["block"] == block
                    and row["variant"] == left_variant
                )
                right = next(
                    row
                    for row in seam_rows
                    if row["seed"] == seed
                    and row["block"] == block
                    and row["variant"] == right_variant
                )
                difference = (
                    float(left["posterior_kl"])
                    - float(right["posterior_kl"])
                )
                differences.append(difference)
                values_by_seed[str(seed)] = difference
            paired[block][comparison] = {
                "definition": f"{left_variant}_minus_{right_variant}",
                **summarize(differences),
                "values_by_seed": values_by_seed,
                "conv1_or_bn1_larger_count":
                    sum(value > 0.0 for value in differences),
            }
    bn_by_seed: dict[int, dict[int, dict[str, object]]] = {}
    for row in bn_rows:
        bn_by_seed.setdefault(int(row["seed"]), {})[int(row["subset"])] = row
    bn_attribution = {}
    bn_all_subset = (1 << len(BN_GROUP_ORDER)) - 1
    for group_index, group in enumerate(BN_GROUP_ORDER):
        bit = 1 << group_index
        rows = []
        for seed in SEEDS:
            subset_rows = bn_by_seed[seed]
            shapley = 0.0
            for subset in range(bn_all_subset + 1):
                if subset & bit:
                    continue
                size = subset.bit_count()
                weight = (
                    math.factorial(size)
                    * math.factorial(len(BN_GROUP_ORDER) - size - 1)
                    / math.factorial(len(BN_GROUP_ORDER))
                )
                shapley += weight * (
                    float(subset_rows[subset | bit]["posterior_kl"])
                    - float(subset_rows[subset]["posterior_kl"])
                )
            rows.append(
                {
                    "shapley_kl_damage": shapley,
                    "alone_kl_damage": (
                        float(subset_rows[bit]["posterior_kl"])
                        - float(subset_rows[0]["posterior_kl"])
                    ),
                    "conditional_kl_damage": (
                        float(subset_rows[bn_all_subset]["posterior_kl"])
                        - float(
                            subset_rows[bn_all_subset ^ bit]["posterior_kl"]
                        )
                    ),
                }
            )
        bn_attribution[group] = {
            "state_count": int(
                bn_by_seed[SEEDS[0]][bit]["public_state_count"]
            ),
            **{
                metric: summarize([float(row[metric]) for row in rows])
                for metric in (
                    "alone_kl_damage",
                    "conditional_kl_damage",
                    "shapley_kl_damage",
                )
            },
        }
    for seed in SEEDS:
        shapley_total = sum(
            sum(
                math.factorial(subset.bit_count())
                * math.factorial(
                    len(BN_GROUP_ORDER) - subset.bit_count() - 1
                )
                / math.factorial(len(BN_GROUP_ORDER))
                * (
                    float(
                        bn_by_seed[seed][subset | (1 << group_index)][
                            "posterior_kl"
                        ]
                    )
                    - float(bn_by_seed[seed][subset]["posterior_kl"])
                )
                for subset in range(bn_all_subset + 1)
                if not subset & (1 << group_index)
            )
            for group_index in range(len(BN_GROUP_ORDER))
        )
        expected = (
            float(bn_by_seed[seed][bn_all_subset]["posterior_kl"])
            - float(bn_by_seed[seed][0]["posterior_kl"])
        )
        if not math.isclose(shapley_total, expected, abs_tol=1e-8):
            raise RuntimeError(f"seed {seed} 的 BN Shapley 分解不闭合。")

    return {
        "seed_count": len(SEEDS),
        "sample_standard_deviation_ddof": 1,
        "lambda": lambda_groups,
        "attribution": attribution_groups,
        "seam": seam_groups,
        "seam_paired": paired,
        "bn_attribution": bn_attribution,
        "bn_full_public_gamma": {
            "posterior_kl": summarize(
                [
                    float(bn_by_seed[seed][bn_all_subset]["posterior_kl"])
                    for seed in SEEDS
                ]
            ),
            "sum_of_four_alone_kl_damage": summarize(
                [
                    sum(
                        float(
                            bn_by_seed[seed][1 << group_index][
                                "posterior_kl"
                            ]
                        )
                        - float(bn_by_seed[seed][0]["posterior_kl"])
                        for group_index in range(len(BN_GROUP_ORDER))
                    )
                    for seed in SEEDS
                ]
            ),
        },
    }


def add_lab07_alignment(
    aggregate: dict[str, object],
    lab07_payload: dict[str, object],
) -> None:
    case_by_group = {
        "layer1_0_conv1": "expose_rank_04",
        "layer1_1_conv1": "expose_rank_01",
        "layer2_0_conv1": "expose_rank_02",
        "layer2_1_conv1": "expose_rank_07",
        "layer3_0_conv1": "expose_rank_09",
    }
    paired = lab07_payload.get("aggregate", {}).get(
        "paired_leave_one_out_minus_base",
        {},
    )
    rows = []
    for group, case in case_by_group.items():
        if case not in paired:
            raise ValueError(f"Lab07 dependency 缺少 {case}。")
        rows.append(
            {
                "group": group,
                "lab09_hide_alone_kl_damage": aggregate["attribution"][group][
                    "hide_alone_kl_damage"
                ]["mean"],
                "lab07_accuracy_rebound": paired[case]["metrics"][
                    "surrogate_acc"
                ]["mean"],
                "lab07_fidelity_rebound": paired[case]["metrics"][
                    "fidelity"
                ]["mean"],
                "lab07_kl_rebound": -paired[case]["metrics"][
                    "posterior_kl"
                ]["mean"],
            }
        )
    mechanism = [row["lab09_hide_alone_kl_damage"] for row in rows]
    aggregate["lab07_alignment"] = {
        "status": "five_post_hoc_points_not_selector_or_significance_test",
        "rows": rows,
        "pearson": {
            metric: statistics.correlation(
                mechanism,
                [row[metric] for row in rows],
            )
            for metric in (
                "lab07_accuracy_rebound",
                "lab07_fidelity_rebound",
                "lab07_kl_rebound",
            )
        },
    }


def plot_results(path: Path, aggregate: dict[str, object]) -> None:
    figure, axes = prefix.plt.subplots(2, 3, figsize=(19.0, 9.2))
    x = [strength * 100.0 for strength in STRENGTHS]
    lambda_groups = aggregate["lambda"]

    standard = [
        lambda_groups[strength_case(strength)]["posterior_kl"]["mean"]
        for strength in STRENGTHS
    ]
    standard_std = [
        lambda_groups[strength_case(strength)]["posterior_kl"]["sample_std"]
        for strength in STRENGTHS
    ]
    matched = [
        lambda_groups[strength_case(strength)]["norm_matched_posterior_kl"]["mean"]
        for strength in STRENGTHS
    ]
    matched_std = [
        lambda_groups[strength_case(strength)]["norm_matched_posterior_kl"]["sample_std"]
        for strength in STRENGTHS
    ]
    axes[0, 0].errorbar(x, standard, yerr=standard_std, marker="o", label="Original")
    axes[0, 0].errorbar(
        x,
        matched,
        yerr=matched_std,
        marker="s",
        label="Head-input norm matched",
    )
    axes[0, 0].set_xlabel("Leaked-state utilization (%)")
    axes[0, 0].set_ylabel("Posterior KL")
    axes[0, 0].set_title("Classifier-interface counterfactual")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=0.25)

    feature = [
        lambda_groups[strength_case(strength)]["feature_rms"]["mean"]
        for strength in STRENGTHS
    ]
    logit = [
        lambda_groups[strength_case(strength)]["logit_rms"]["mean"]
        for strength in STRENGTHS
    ]
    feature_ratio = [value / feature[0] for value in feature]
    logit_ratio = [value / logit[0] for value in logit]
    axes[0, 1].plot(x, feature_ratio, marker="o", label="Feature RMS / public")
    axes[0, 1].plot(x, logit_ratio, marker="s", label="Logit RMS / public")
    axes[0, 1].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[0, 1].set_xlabel("Leaked-state utilization (%)")
    axes[0, 1].set_ylabel("Ratio to 0% endpoint")
    axes[0, 1].set_title("Magnitude growth at the protected head")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(alpha=0.25)

    attribution = aggregate["attribution"]
    values = [attribution[group]["shapley_kl_recovery"]["mean"] for group in GROUP_ORDER]
    errors = [
        attribution[group]["shapley_kl_recovery"]["sample_std"]
        for group in GROUP_ORDER
    ]
    labels = [group.replace("_conv1", "") for group in GROUP_ORDER]
    axes[0, 2].bar(range(len(labels)), values, yerr=errors, color="#0072B2")
    axes[0, 2].axhline(0.0, color="black", linewidth=1.0)
    axes[0, 2].set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axes[0, 2].set_ylabel("Shapley KL recovery")
    axes[0, 2].set_title("Exact seven-group oracle attribution")
    axes[0, 2].grid(axis="y", alpha=0.25)

    bn_attribution = aggregate["bn_attribution"]
    bn_values = [
        bn_attribution[group]["shapley_kl_damage"]["mean"]
        for group in BN_GROUP_ORDER
    ]
    bn_errors = [
        bn_attribution[group]["shapley_kl_damage"]["sample_std"]
        for group in BN_GROUP_ORDER
    ]
    axes[1, 0].bar(
        range(len(BN_GROUP_ORDER)),
        bn_values,
        yerr=bn_errors,
        color="#CC79A7",
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=1.0)
    axes[1, 0].set_xticks(
        range(len(BN_GROUP_ORDER)),
        BN_GROUP_ORDER,
        rotation=25,
        ha="right",
    )
    axes[1, 0].set_ylabel("Shapley KL damage")
    axes[1, 0].set_title("Four-group BN gamma closure")
    axes[1, 0].grid(axis="y", alpha=0.25)

    seam = aggregate["seam"]
    block_x = list(range(len(BLOCKS)))
    for variant, label, marker in (
        ("conv1_weight", "conv1 weight", "o"),
        ("conv2_weight", "conv2 weight", "s"),
    ):
        values = [seam[block][variant]["posterior_kl"]["mean"] for block in BLOCKS]
        axes[1, 1].plot(
            block_x,
            values,
            marker=marker,
            label=label,
        )
    axes[1, 1].set_xticks(block_x, [block.removeprefix("layer") for block in BLOCKS])
    axes[1, 1].set_xlabel("BasicBlock")
    axes[1, 1].set_ylabel("Posterior KL")
    axes[1, 1].set_title("Public convolution seam")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(alpha=0.25)

    for variant, label, marker in (
        ("bn1_gamma", "BN1 gamma", "o"),
        ("bn2_gamma", "BN2 gamma", "s"),
    ):
        values = [seam[block][variant]["posterior_kl"]["mean"] for block in BLOCKS]
        axes[1, 2].plot(
            block_x,
            values,
            marker=marker,
            label=label,
        )
    axes[1, 2].set_xticks(
        block_x,
        [block.removeprefix("layer") for block in BLOCKS],
    )
    axes[1, 2].set_xlabel("BasicBlock")
    axes[1, 2].set_ylabel("Posterior KL")
    axes[1, 2].set_title("Public BN gamma seam")
    axes[1, 2].legend(frameon=False)
    axes[1, 2].grid(alpha=0.25)

    figure.suptitle("Lab09: why the protected interfaces matter", fontsize=15)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    prefix.plt.close(figure)


def main() -> int:
    args = parse_args()
    device = prefix.resolve_device(args.device)
    victim_checkpoint = (
        ROOT / "weights" / "MS" / "victim" / prefix.MODEL / prefix.DATASET / "best.pth"
    )
    official_weight = ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    dataset_root = ROOT / "dataset" / "public"
    protocol_root = ROOT / "dataset" / "MS"
    source_candidate = ROOT / "results" / "lab" / "04_tensorshield" / "candidate.json"
    source_lab07 = ROOT / "results" / "lab" / "07_structure" / "dependency.json"
    source_lab08 = ROOT / "results" / "lab" / "08_leakage" / "metrics.json"

    prefix.configure_reproducibility(42, deterministic=True)
    victim, victim_metadata = prefix.build_victim(
        prefix.MODEL,
        prefix.NUM_CLASSES,
        victim_checkpoint,
    )
    queries = {
        seed: prefix.prepare_soft_query(
            dataset=prefix.DATASET,
            model=prefix.MODEL,
            budget=prefix.BUDGET,
            seed=seed,
            dataset_root=dataset_root,
            protocol_root=protocol_root,
        )
        for seed in SEEDS
    }

    prefix.configure_reproducibility(SEEDS[0], deterministic=True)
    endpoint_check = lab08.build_endpoint_models(victim, official_weight, SEEDS[0])
    groups = build_groups(victim, endpoint_check)
    bn_groups = build_bn_groups(tuple(endpoint_check["bn_gamma"]))
    group_metadata = {
        group: {
            "states": list(groups[group]),
            "state_count": len(groups[group]),
            "param_count": sum(
                victim.state_dict()[name].numel()
                for name in groups[group]
            ),
        }
        for group in GROUP_ORDER
    }
    candidate_weights = set(endpoint_check["selected_weights"])
    if tuple(endpoint_check["selected_weights"]) != (
        "layer1.1.conv1.weight",
        "layer2.0.conv1.weight",
        "last_linear.weight",
        "layer1.0.conv1.weight",
        "layer2.1.conv1.weight",
        "layer3.0.conv1.weight",
    ):
        raise RuntimeError("Lab09 的候选 eligible weights 已经漂移。")
    if len(BLOCKS) != 8:
        raise RuntimeError("ResNet18 应包含八个 BasicBlock。")
    del endpoint_check

    print(
        "[PROTOCOL] "
        f"seeds={SEEDS[0]}-{SEEDS[-1]} validation=100 "
        f"groups={len(GROUP_ORDER)} lattice={1 << len(GROUP_ORDER)} "
        f"blocks={len(BLOCKS)} device={device}"
    )
    for group in GROUP_ORDER:
        print(
            f"[GROUP/{group}] states={group_metadata[group]['state_count']} "
            f"params={group_metadata[group]['param_count']}"
        )
    if args.dry_run:
        print("[INFO] dry-run 完成，未运行前向分析或写结果。")
        return 0

    batches_by_seed = {
        seed: load_batches(
            queries[seed],
            device=device,
            num_workers=args.num_workers,
            seed=seed,
        )
        for seed in SEEDS
    }
    victim = victim.to(device)
    lambda_rows = []
    lattice_rows = []
    seam_rows = []
    bn_rows = []
    victim_controls = {}
    for seed in SEEDS:
        batches = batches_by_seed[seed]
        prefix.configure_reproducibility(seed, deterministic=True)
        public_model, _ = lab08.build_strength_model(
            victim.cpu(),
            official_weight,
            seed,
            0.0,
        )
        public_model = public_model.to(device)
        public_features = collect_features(public_model, batches, device)
        reference_head = {
            name: value.detach().cpu().clone()
            for name, value in public_model.last_linear.state_dict().items()
        }

        for strength in STRENGTHS:
            prefix.configure_reproducibility(seed, deterministic=True)
            model, metadata = lab08.build_strength_model(
                victim.cpu(),
                official_weight,
                seed,
                strength,
            )
            if any(
                not torch.equal(reference_head[name], value.detach().cpu())
                for name, value in model.last_linear.state_dict().items()
            ):
                raise RuntimeError(f"seed {seed} 的五个强度没有共享同一受保护分类头。")
            model = model.to(device)
            metrics = evaluate_model(
                model,
                batches,
                device,
                norm_reference=public_features,
            )
            lambda_rows.append(
                {
                    "seed": seed,
                    "case": strength_case(strength),
                    "utilization_strength": strength,
                    "state_sha256": metadata["state_sha256"],
                    **metrics,
                }
            )
            del model

        prefix.configure_reproducibility(seed, deterministic=True)
        endpoints = lab08.build_endpoint_models(victim.cpu(), official_weight, seed)
        seed_groups = build_groups(victim, endpoints)
        lattice_rows.extend(
            build_lattice_rows(
                seed=seed,
                hybrid=endpoints["hybrid"],
                victim=victim,
                groups=seed_groups,
                batches=batches,
                device=device,
            )
        )
        current_seams, victim_metrics = build_seam_rows(
            seed=seed,
            victim=victim,
            public=endpoints["public"],
            candidate_weights=candidate_weights,
            batches=batches,
            device=device,
        )
        seam_rows.extend(current_seams)
        bn_rows.extend(
            build_bn_rows(
                seed=seed,
                victim=victim,
                public=endpoints["public"],
                groups=bn_groups,
                batches=batches,
                device=device,
            )
        )
        victim_controls[str(seed)] = victim_metrics
        del public_model, endpoints
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[SEED {seed}] lambda=5 lattice=128 seams=48 bn=16 "
            f"hybrid_kl={lambda_rows[-1]['posterior_kl']:.6f}",
            flush=True,
        )
    victim = victim.cpu()

    lattice_by_key = {
        (int(row["seed"]), int(row["subset"])): row
        for row in lattice_rows
    }
    bn_by_key = {
        (int(row["seed"]), int(row["subset"])): row
        for row in bn_rows
    }
    all_groups = (1 << len(GROUP_ORDER)) - 1
    gamma_hidden = all_groups ^ (1 << GROUP_ORDER.index("bn_gamma"))
    all_bn_public = (1 << len(BN_GROUP_ORDER)) - 1
    for seed in SEEDS:
        if not math.isclose(
            float(lattice_by_key[(seed, gamma_hidden)]["posterior_kl"]),
            float(bn_by_key[(seed, all_bn_public)]["posterior_kl"]),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise RuntimeError(
                f"seed {seed} 的全部 public BN gamma 端点不一致。"
            )

    attribution_rows = build_attribution_rows(lattice_rows, groups, victim)
    aggregate = aggregate_results(
        lambda_rows,
        attribution_rows,
        seam_rows,
        bn_rows,
    )
    lab07_payload = json.loads(source_lab07.read_text(encoding="utf-8"))
    add_lab07_alignment(aggregate, lab07_payload)
    out_dir = ROOT / "results" / "lab" / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "lambda": f"results/lab/{EXPERIMENT}/lambda.tsv",
        "lattice": f"results/lab/{EXPERIMENT}/lattice.tsv",
        "attribution": f"results/lab/{EXPERIMENT}/attribution.tsv",
        "seam": f"results/lab/{EXPERIMENT}/seam.tsv",
        "bn": f"results/lab/{EXPERIMENT}/bn.tsv",
        "plot": f"results/lab/{EXPERIMENT}/metrics.png",
    }
    write_tsv(ROOT / outputs["lambda"], lambda_rows, LAMBDA_FIELDS)
    write_tsv(ROOT / outputs["lattice"], lattice_rows, LATTICE_FIELDS)
    write_tsv(ROOT / outputs["attribution"], attribution_rows, ATTRIBUTION_FIELDS)
    write_tsv(ROOT / outputs["seam"], seam_rows, SEAM_FIELDS)
    write_tsv(ROOT / outputs["bn"], bn_rows, BN_FIELDS)
    plot_results(ROOT / outputs["plot"], aggregate)
    payload = {
        "schema_version": 1,
        "experiment": "09_attack_dependency_mechanism",
        "analysis_protocol": ANALYSIS_PROTOCOL,
        "scientific_status": "post_hoc_mechanism_analysis_not_selector",
        "dataset": prefix.DATASET,
        "victim_model": prefix.MODEL,
        "query_budget": prefix.BUDGET,
        "analysis_split": "query_validation",
        "analysis_count_per_seed": 100,
        "evaluation_seeds": list(SEEDS),
        "uses_eval_ms": False,
        "trains_surrogate": False,
        "primary_metric": "posterior_kl_to_victim",
        "randomization": {
            "surrogate_initialization": "formal_victim_then_public_v1",
            "per_seed_canonical_replay": True,
        },
        "query_partitions": {
            str(seed): queries[seed].partition.to_metadata()
            for seed in SEEDS
        },
        "source": {
            "lab04_candidate": str(source_candidate.relative_to(ROOT)),
            "lab04_candidate_sha256": prefix.sha256_file(source_candidate),
            "lab07_dependency": str(source_lab07.relative_to(ROOT)),
            "lab07_dependency_sha256": prefix.sha256_file(source_lab07),
            "lab08_metrics": str(source_lab08.relative_to(ROOT)),
            "lab08_metrics_sha256": prefix.sha256_file(source_lab08),
            "victim_checkpoint": str(victim_checkpoint.relative_to(ROOT)),
            "victim_checkpoint_sha256": prefix.sha256_file(victim_checkpoint),
            "victim_checkpoint_epoch": victim_metadata.get("epoch"),
            "official_weight": str(official_weight.relative_to(ROOT)),
            "official_weight_sha256": prefix.sha256_file(official_weight),
            "posterior_path": str(queries[SEEDS[0]].target_path.relative_to(ROOT)),
            "posterior_sha256": queries[SEEDS[0]].target_sha256,
        },
        "system_protection": {
            "source_case": lab04.CANDIDATE_DROP06_CASE,
            "protected_state_count": 27,
            "protected_param_count": 645_924,
            "protected_param_ratio": 645_924 / 11_227_812,
            "groups": group_metadata,
        },
        "lambda_analysis": {
            "strengths": list(STRENGTHS),
            "norm_counterfactual":
                "per_sample_head_input_l2_matched_to_public_same_image",
            "result_count": len(lambda_rows),
        },
        "lattice_analysis": {
            "group_order": list(GROUP_ORDER),
            "subset_count_per_seed": 1 << len(GROUP_ORDER),
            "result_count": len(lattice_rows),
            "oracle_only": True,
        },
        "seam_analysis": {
            "blocks": list(BLOCKS),
            "variants": [variant for variant, _ in seam_variants(BLOCKS[0])],
            "result_count": len(seam_rows),
            "base_model": "victim",
            "replacement_source": "same_seed_canonical_public",
        },
        "bn_analysis": {
            "group_order": list(BN_GROUP_ORDER),
            "groups": {
                group: list(bn_groups[group])
                for group in BN_GROUP_ORDER
            },
            "subset_count_per_seed": 1 << len(BN_GROUP_ORDER),
            "result_count": len(bn_rows),
            "base_model": "victim",
            "replacement_source": "same_seed_canonical_public",
        },
        "victim_controls": victim_controls,
        "results": {
            "lambda": lambda_rows,
            "lattice": lattice_rows,
            "attribution": attribution_rows,
            "seam": seam_rows,
            "bn": bn_rows,
        },
        "aggregate": aggregate,
        "outputs": outputs,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out_dir / "metrics.json", payload)
    print(f"[OK] 写入 {out_dir.relative_to(ROOT)}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
