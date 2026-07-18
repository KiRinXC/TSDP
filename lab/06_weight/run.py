#!/usr/bin/env python3
"""按当前 MS 协议验证 TensorShield Top-10 至 Top-17 的 weight 语义闭包。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
for import_root in (
    ROOT,
    ROOT / "exp" / "MS" / "train_surrogate",
    ROOT / "exp" / "MS" / "train_victim",
):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common.trainer import configure_reproducibility  # noqa: E402
from exp.MS.train_surrogate.core.artifacts import sha256_file  # noqa: E402
from exp.MS.train_surrogate.core.config import (  # noqa: E402
    ATTACK_PROTOCOL_VERSION,
    resolve_device,
)
from exp.MS.train_surrogate.core.data import build_victim  # noqa: E402
from exp.MS.train_surrogate.defense import (  # noqa: E402
    build_resnet18_tensor_units,
    initialize_surrogate,
    load_protection_mask,
    protection_mask_sha256,
    save_protection_mask,
)
from exp.MS.train_surrogate.selector import (  # noqa: E402
    AUTHOR_RESNET18_C100_ELIGIBLE_RANK,
    PUBLISHED_RESNET18_C100_WEIGHTS,
)
from lab.protocol import (  # noqa: E402
    evaluate_once,
    prepare_eval,
    prepare_soft_query,
    protocol_metadata,
    train_validation_best,
)
from models import imagenet as imagenet_models  # noqa: E402


EXPERIMENT = "06_weight"
MODEL = "resnet18"
DATASET = "c100"
NUM_CLASSES = 100
BUDGET = 500
TOP_K_VALUES = tuple(range(10, 18))
SEED = 42
EXPECTED_EXTRA_COUNTS = {
    "bn_gamma": (20, 4_800),
    "downsample_conv": (3, 172_032),
    "bn_gamma_downsample": (23, 176_832),
    "stem_conv": (1, 9_408),
    "all_extras": (24, 186_240),
}
EXPECTED_DOWNSAMPLE_CONV = (
    "layer2.0.downsample.0.weight",
    "layer3.0.downsample.0.weight",
    "layer4.0.downsample.0.weight",
)
VARIANTS = {
    "top_k": {"label": "Top-k", "color": "#555555", "marker": "o"},
    "bn_gamma": {"label": "+ BN gamma", "color": "#009E73", "marker": "s"},
    "downsample_conv": {
        "label": "+ Downsample Conv",
        "color": "#D55E00",
        "marker": "^",
    },
    "bn_gamma_downsample": {
        "label": "+ BN gamma + Downsample",
        "color": "#6A3D9A",
        "marker": "X",
    },
    "stem_conv": {"label": "+ Stem Conv", "color": "#CC79A7", "marker": "D"},
    "all_extras": {"label": "+ All extras", "color": "#0072B2", "marker": "P"},
}
HISTORY_FIELDS = (
    "case",
    "top_k",
    "variant",
    "epoch",
    "learning_rate",
    "query_count",
    "query_loss_sum",
    "query_loss",
    "query_match_count",
    "query_match",
    "validation_count",
    "validation_loss",
    "validation_kl",
    "validation_match_count",
    "validation_match",
    "is_best",
)
DATA_FIELDS = (
    "case",
    "top_k",
    "variant",
    "origin",
    "extra_state_names",
    "protected_unit_count",
    "protected_param_count",
    "protected_param_ratio",
    "protection_mask_sha256",
    "surrogate_acc",
    "fidelity",
    "posterior_kl",
    "accuracy_change_from_top_k",
    "fidelity_change_from_top_k",
    "posterior_kl_change_from_top_k",
)


@dataclass(frozen=True)
class CaseSpec:
    top_k: int
    variant: str
    selected_state_names: tuple[str, ...]
    extra_state_names: tuple[str, ...]

    @property
    def name(self) -> str:
        if self.variant == "top_k":
            return f"top_{self.top_k:02d}"
        return f"top_{self.top_k:02d}_{self.variant}"

    @property
    def trained_here(self) -> bool:
        return self.variant != "top_k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对 Lab04/Lab05 输入、48 个 mask 和保护统计，不写结果。",
    )
    return parser.parse_args()


def canonical_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def unique_names(names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(names))


def derive_extra_states(victim: nn.Module) -> dict[str, tuple[str, ...]]:
    state = victim.state_dict()
    bn_gamma = tuple(
        f"{module_name}.weight"
        for module_name, module in victim.named_modules()
        if module_name and isinstance(module, nn.BatchNorm2d)
    )
    downsample_conv = tuple(
        f"{module_name}.weight"
        for module_name, module in victim.named_modules()
        if isinstance(module, nn.Conv2d) and module_name.endswith(".downsample.0")
    )
    if downsample_conv != EXPECTED_DOWNSAMPLE_CONV:
        raise ValueError(f"downsample Conv 定义已变化：{downsample_conv}")
    groups = {
        "bn_gamma": bn_gamma,
        "downsample_conv": downsample_conv,
        "bn_gamma_downsample": unique_names((*bn_gamma, *downsample_conv)),
        "stem_conv": ("conv1.weight",),
        "all_extras": unique_names(
            (*bn_gamma, *downsample_conv, "conv1.weight")
        ),
    }
    for group_name, names in groups.items():
        missing = set(names) - set(state)
        if missing:
            raise ValueError(f"{group_name} 包含未知 state：{sorted(missing)}")
        actual = (len(names), sum(state[name].numel() for name in names))
        if actual != EXPECTED_EXTRA_COUNTS[group_name]:
            raise ValueError(
                f"{group_name} 统计为 {actual}，"
                f"期望 {EXPECTED_EXTRA_COUNTS[group_name]}。"
            )
    return groups


def build_cases(
    ranking: tuple[str, ...],
    extra_groups: dict[str, tuple[str, ...]],
) -> tuple[CaseSpec, ...]:
    if len(ranking) != 17 or len(ranking) != len(set(ranking)):
        raise ValueError(f"eligible rank 应有 17 个唯一 weight，实际为 {len(ranking)}。")
    if set(ranking[:10]) != set(PUBLISHED_RESNET18_C100_WEIGHTS):
        raise ValueError("eligible Top-10 与 TensorShield Figure 12(d) 集合不一致。")
    if any(set(ranking) & set(names) for names in extra_groups.values()):
        raise ValueError("eligible rank 与额外 weight 语义必须互不重叠。")
    cases = []
    for top_k in TOP_K_VALUES:
        base_names = (*ranking[:top_k], "last_linear.bias")
        for variant in VARIANTS:
            extras = () if variant == "top_k" else extra_groups[variant]
            cases.append(
                CaseSpec(
                    top_k=top_k,
                    variant=variant,
                    selected_state_names=unique_names((*base_names, *extras)),
                    extra_state_names=tuple(extras),
                )
            )
    return tuple(cases)


def initialize_case(
    case: CaseSpec,
    victim: nn.Module,
    official_weight: Path,
) -> tuple[nn.Module, object, dict[str, torch.Tensor], list[dict[str, object]]]:
    units = build_resnet18_tensor_units(victim)
    if len(units) != 122:
        raise RuntimeError(f"ResNet18 unit 数量应为 122，实际为 {len(units)}。")
    unit_by_name = {unit.state_name: unit for unit in units}
    missing = set(case.selected_state_names) - set(unit_by_name)
    if missing:
        raise ValueError(f"{case.name} 包含未知 state：{sorted(missing)}")
    selected_units = [unit_by_name[name] for name in case.selected_state_names]
    surrogate, plan, _, masks = initialize_surrogate(
        factory=imagenet_models.resnet18,
        factory_name=MODEL,
        weight_path=official_weight,
        victim_model=victim,
        num_classes=NUM_CLASSES,
        defense="custom",
        protected_units=",".join(str(unit.index) for unit in selected_units),
        protected_layers=None,
        protected_scalars=None,
        initialization_seed=SEED,
    )
    if not plan.classifier_protected or plan.head_mode != "replace":
        raise RuntimeError(f"{case.name} 必须完整保护分类头，实际为 {plan.head_mode}。")
    state = victim.state_dict()
    expected_params = sum(state[name].numel() for name in case.selected_state_names)
    if (
        plan.protected_unit_count != len(case.selected_state_names)
        or plan.protected_param_count != expected_params
    ):
        raise RuntimeError(f"{case.name} 的保护统计与定义不一致。")
    selected_set = set(case.selected_state_names)
    for name, mask in masks.items():
        if bool(mask.all()) != (name in selected_set) or (
            name not in selected_set and bool(mask.any())
        ):
            raise RuntimeError(f"{case.name} 的 {name} mask 不是完整 unit 选择。")
    ranking = tuple(AUTHOR_RESNET18_C100_ELIGIBLE_RANK)
    extra_set = set(case.extra_state_names)
    metadata = []
    for unit in selected_units:
        if unit.state_name in ranking:
            role = "eligible_prefix"
            eligible_rank = ranking.index(unit.state_name) + 1
        elif unit.state_name == "last_linear.bias":
            role = "fixed_head_bias"
            eligible_rank = None
        elif unit.state_name in extra_set:
            role = "semantic_extra"
            eligible_rank = None
        else:
            raise RuntimeError(f"无法解释 {case.name} 的 state：{unit.state_name}")
        metadata.append(
            {
                "index": unit.index,
                "state_name": unit.state_name,
                "state_kind": unit.state_kind,
                "numel": unit.numel,
                "role": role,
                "eligible_rank": eligible_rank,
            }
        )
    return surrogate, plan, masks, metadata


def load_lab04(path: Path) -> tuple[dict[str, object], dict[int, dict[str, object]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 3,
        "experiment": "04_tensorshield",
        "protocol": "MS",
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "dataset": DATASET,
        "victim_model": MODEL,
        "query_budget": BUDGET,
        "query_train_size": 400,
        "query_validation_size": 100,
        "label_mode": "soft",
        "query_transform": "test",
        "seed": SEED,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"Lab04 {field}={payload.get(field)!r}，期望 {value!r}。")
    ranking = tuple(payload.get("source", {}).get("eligible_rank", ()))
    if ranking != tuple(AUTHOR_RESNET18_C100_ELIGIBLE_RANK):
        raise ValueError("Lab04 eligible rank 与当前作者固定列表不一致。")
    by_k = {int(row["top_k"]): row for row in payload.get("results", [])}
    if set(by_k) != set(range(1, 18)):
        raise ValueError("Lab04 未完整包含 Top-1 至 Top-17。")
    for top_k, row in by_k.items():
        primary = row.get("primary", {})
        if (
            primary.get("checkpoint") != "best.pth"
            or primary.get("selection_metric")
            != "validation_soft_cross_entropy"
            or row.get("result", {}).get("eval_passes") != 1
        ):
            raise ValueError(f"Lab04 Top-{top_k} 不是当前 validation-best 结果。")
        protection = row.get("protection", {})
        mask_path = ROOT / str(protection.get("mask_path", ""))
        if not mask_path.is_file():
            raise FileNotFoundError(f"Lab04 Top-{top_k} 缺少 mask：{mask_path}")
        digest = protection_mask_sha256(load_protection_mask(mask_path))
        if digest != protection.get("protection_mask_sha256"):
            raise ValueError(f"Lab04 Top-{top_k} mask 与 metrics.json 不一致。")
    return payload, by_k


def load_lab05_weight(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 3,
        "experiment": "05_state",
        "protocol": "MS",
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "dataset": DATASET,
        "victim_model": MODEL,
        "query_budget": BUDGET,
        "query_train_size": 400,
        "query_validation_size": 100,
        "label_mode": "soft",
        "query_transform": "test",
        "seed": SEED,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"Lab05 {field}={payload.get(field)!r}，期望 {value!r}。")
    matches = [
        row
        for row in payload.get("results", [])
        if row.get("protection_group") == "weight"
    ]
    if len(matches) != 1:
        raise ValueError("Lab05 必须恰好包含一个 weight 参考结果。")
    result = matches[0]
    if (
        result.get("protection", {}).get("protected_param_count") != 11_222_912
        or result.get("primary", {}).get("checkpoint") != "best.pth"
        or result.get("result", {}).get("eval_passes") != 1
    ):
        raise ValueError("Lab05 weight 参考不符合当前协议或参数语义。")
    return payload, result


def reused_prefix_result(
    case: CaseSpec,
    source: dict[str, object],
    source_sha256: str,
) -> dict[str, object]:
    return {
        "case": case.name,
        "top_k": case.top_k,
        "variant": case.variant,
        "variant_label": VARIANTS[case.variant]["label"],
        "origin": "reused_lab04_prefix",
        "selected_state_names": list(case.selected_state_names),
        "extra_state_names": [],
        "source": {
            "experiment": "04_tensorshield",
            "case": source["case"],
            "metrics_sha256": source_sha256,
        },
        "protection": source["protection"],
        "primary": source["primary"],
        "selection": source["selection"],
        "result": source["result"],
    }


def add_effects(results: list[dict[str, object]]) -> None:
    baselines = {
        int(result["top_k"]): result["result"]
        for result in results
        if result["variant"] == "top_k"
    }
    for result in results:
        baseline = baselines[int(result["top_k"])]
        metrics = result["result"]
        result["effect_vs_top_k"] = {
            "accuracy_change": metrics["surrogate_acc"] - baseline["surrogate_acc"],
            "fidelity_change": metrics["fidelity"] - baseline["fidelity"],
            "posterior_kl_change": metrics["posterior_kl"] - baseline["posterior_kl"],
        }


def write_history(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(
            writer_file,
            fieldnames=HISTORY_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_data(path: Path, results: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(
            writer_file,
            fieldnames=DATA_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for result in results:
            protection = result["protection"]
            metrics = result["result"]
            effect = result["effect_vs_top_k"]
            writer.writerow(
                {
                    "case": result["case"],
                    "top_k": result["top_k"],
                    "variant": result["variant"],
                    "origin": result["origin"],
                    "extra_state_names": ",".join(result["extra_state_names"]),
                    "protected_unit_count": protection["protected_unit_count"],
                    "protected_param_count": protection["protected_param_count"],
                    "protected_param_ratio": protection["protected_param_ratio"],
                    "protection_mask_sha256": protection["protection_mask_sha256"],
                    "surrogate_acc": metrics["surrogate_acc"],
                    "fidelity": metrics["fidelity"],
                    "posterior_kl": metrics["posterior_kl"],
                    "accuracy_change_from_top_k": effect["accuracy_change"],
                    "fidelity_change_from_top_k": effect["fidelity_change"],
                    "posterior_kl_change_from_top_k": effect["posterior_kl_change"],
                }
            )


def set_y_limits(axis: plt.Axes, values: list[float], bounded: bool) -> None:
    minimum = min(values)
    maximum = max(values)
    padding = max((maximum - minimum) * 0.09, 0.02 if bounded else 0.05)
    upper = min(1.0, maximum + padding) if bounded else maximum + padding
    axis.set_ylim(max(0.0, minimum - padding), upper)


def plot_metrics(
    path: Path,
    results: list[dict[str, object]],
    references: dict[str, dict[str, object]],
) -> None:
    specifications = (
        ("surrogate_acc", "Surrogate accuracy"),
        ("fidelity", "Fidelity"),
        ("posterior_kl", "Posterior KL"),
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 240,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.7))
    for axis, (metric, title) in zip(axes, specifications):
        plotted_values: list[float] = []
        for variant, style in VARIANTS.items():
            rows = sorted(
                (row for row in results if row["variant"] == variant),
                key=lambda row: int(row["top_k"]),
            )
            x_values = [int(row["top_k"]) for row in rows]
            y_values = [float(row["result"][metric]) for row in rows]
            plotted_values.extend(y_values)
            axis.plot(
                x_values,
                y_values,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=5.2,
            )
        soft = float(references["full_protection"]["result"][metric])
        hard = float(references["hard_blackbox"]["result"][metric])
        plotted_values.extend((soft, hard))
        axis.axhline(
            soft,
            label="Soft black-box",
            color="#888888",
            linestyle=":",
            linewidth=1.5,
        )
        axis.axhline(
            hard,
            label="Hard-label black-box",
            color="#CC79A7",
            linestyle=(0, (3, 2)),
            linewidth=1.2,
        )
        set_y_limits(axis, plotted_values, bounded=metric != "posterior_kl")
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.set_title(title)
        axis.set_xlabel("TensorShield eligible Top-k")
        axis.set_ylabel(title)
        axis.set_xticks(TOP_K_VALUES)
        axis.set_xlim(9.7, 17.3)
        axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=8,
        frameon=False,
    )
    figure.suptitle(
        "TensorShield Top-k weight-semantic closure on ResNet18 + CIFAR-100",
        y=1.13,
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def clean_outputs(out_dir: Path) -> None:
    for filename in ("metrics.json", "history.tsv", "data.tsv", "metrics.png"):
        (out_dir / filename).unlink(missing_ok=True)
    for path in out_dir.glob("top_*_mask.pt"):
        path.unlink()


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    device = resolve_device(args.device)
    dataset_root = ROOT / "dataset" / "public"
    protocol_root = ROOT / "dataset" / "MS"
    victim_checkpoint = (
        ROOT / "weights" / "MS" / "victim" / MODEL / DATASET / "best.pth"
    )
    official_weight = ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    lab04_path = ROOT / "results" / "lab" / "04_tensorshield" / "metrics.json"
    lab05_path = ROOT / "results" / "lab" / "05_state" / "metrics.json"
    out_dir = ROOT / "results" / "lab" / EXPERIMENT

    lab04_payload, lab04_by_k = load_lab04(lab04_path)
    lab05_payload, lab05_weight = load_lab05_weight(lab05_path)
    lab04_sha256 = sha256_file(lab04_path)
    lab05_sha256 = sha256_file(lab05_path)
    ranking = tuple(AUTHOR_RESNET18_C100_ELIGIBLE_RANK)
    configure_reproducibility(SEED, deterministic=True)
    query = prepare_soft_query(
        dataset=DATASET,
        model=MODEL,
        budget=BUDGET,
        seed=SEED,
        dataset_root=dataset_root,
        protocol_root=protocol_root,
    )
    victim, victim_metadata = build_victim(MODEL, NUM_CLASSES, victim_checkpoint)
    victim_sha256 = sha256_file(victim_checkpoint)
    expected_victim_sha256 = query.manifest.get("victim", {}).get("checkpoint_sha256")
    if expected_victim_sha256 and expected_victim_sha256 != victim_sha256:
        raise ValueError("victim best.pth 与 soft posterior 的来源 checkpoint 不一致。")
    for label, payload in (("Lab04", lab04_payload), ("Lab05", lab05_payload)):
        if payload.get("victim_checkpoint_sha256") != victim_sha256:
            raise ValueError(f"{label} 与当前 victim best.pth 不一致。")
        if payload.get("official_weight_sha256") != sha256_file(official_weight):
            raise ValueError(f"{label} 与当前 ImageNet 预训练权重不一致。")
        if payload.get("posterior_sha256") != query.target_sha256:
            raise ValueError(f"{label} 与当前 soft posterior 不一致。")

    extra_groups = derive_extra_states(victim)
    cases = build_cases(ranking, extra_groups)
    references = dict(lab04_payload["references"])
    expected_by_case: dict[str, dict[str, object]] = {}
    for case in cases:
        configure_reproducibility(SEED, deterministic=True)
        surrogate, plan, _, selected_units = initialize_case(
            case,
            victim,
            official_weight,
        )
        protection = {
            "implementation_defense": "custom",
            **plan.to_metadata(),
            "selected_units": selected_units,
            "mask_path": str(
                (out_dir / f"{case.name}_mask.pt").relative_to(ROOT)
            ),
        }
        expected_by_case[case.name] = {"protection": protection}
        if case.variant == "top_k":
            source = lab04_by_k[case.top_k]["protection"]
            if (
                plan.protection_mask_sha256 != source["protection_mask_sha256"]
                or plan.protected_param_count != source["protected_param_count"]
            ):
                raise ValueError(f"{case.name} 与 Lab04 原始 Top-k mask 不一致。")
        print(
            f"[MASK/{case.name}] variant={case.variant} "
            f"units={plan.protected_unit_count}/122 "
            f"params={plan.protected_param_count}/{plan.total_param_count} "
            f"ratio={plan.protected_param_ratio:.6f} "
            f"sha256={plan.protection_mask_sha256}"
        )
        del surrogate

    endpoint = expected_by_case["top_17_all_extras"]["protection"]
    if (
        endpoint["protected_param_count"]
        != lab05_weight["protection"]["protected_param_count"] + 100
    ):
        raise ValueError("Top-17 + all_extras 未闭合为全部 weight 加分类头 bias。")

    protocol_config = {
        "experiment": EXPERIMENT,
        **protocol_metadata(query),
        "dataset": DATASET,
        "victim_model": MODEL,
        "seed": SEED,
        "top_k_values": list(TOP_K_VALUES),
        "variants": list(VARIANTS),
        "eligible_rank": list(ranking),
        "extra_groups": {name: list(values) for name, values in extra_groups.items()},
        "victim_checkpoint_sha256": victim_sha256,
        "official_weight_sha256": sha256_file(official_weight),
        "posterior_sha256": query.target_sha256,
        "lab04_metrics_sha256": lab04_sha256,
        "lab05_metrics_sha256": lab05_sha256,
    }
    protocol_sha256 = canonical_sha256(protocol_config)
    print(f"[INFO] protocol SHA256：{protocol_sha256}")
    if args.dry_run:
        print("[INFO] dry-run 完成，未写入 Lab06 结果。")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    clean_outputs(out_dir)
    results_by_case: dict[str, dict[str, object]] = {}
    for case in cases:
        if case.variant == "top_k":
            results_by_case[case.name] = reused_prefix_result(
                case,
                lab04_by_k[case.top_k],
                lab04_sha256,
            )

    history_rows: list[dict[str, object]] = []
    evaluation = None
    for case in cases:
        if not case.trained_here:
            continue
        configure_reproducibility(SEED, deterministic=True)
        surrogate, plan, masks, selected_units = initialize_case(
            case,
            victim,
            official_weight,
        )
        expected = expected_by_case[case.name]["protection"]
        if plan.protection_mask_sha256 != expected["protection_mask_sha256"]:
            raise RuntimeError(f"{case.name} 训练前 mask 与预检定义漂移。")
        mask_path = out_dir / f"{case.name}_mask.pt"
        save_protection_mask(mask_path, masks)
        surrogate = surrogate.to(device)
        selection, history = train_validation_best(
            surrogate,
            query,
            device=device,
            num_workers=args.num_workers,
            seed=SEED,
        )
        history_rows.extend(
            {
                "case": case.name,
                "top_k": case.top_k,
                "variant": case.variant,
                **row,
            }
            for row in history
        )
        if evaluation is None:
            evaluation = prepare_eval(
                victim,
                dataset=DATASET,
                dataset_root=dataset_root,
                protocol_root=protocol_root,
                device=device,
                num_workers=args.num_workers,
                seed=SEED,
            )
        result_metrics = evaluate_once(surrogate, evaluation, device)
        results_by_case[case.name] = {
            "case": case.name,
            "top_k": case.top_k,
            "variant": case.variant,
            "variant_label": VARIANTS[case.variant]["label"],
            "origin": "trained_lab06",
            "selected_state_names": list(case.selected_state_names),
            "extra_state_names": list(case.extra_state_names),
            "protection": {
                "implementation_defense": "custom",
                **plan.to_metadata(),
                "selected_units": selected_units,
                "mask_path": str(mask_path.relative_to(ROOT)),
            },
            "primary": {
                "checkpoint": "best.pth",
                "epoch": selection["epoch"],
                "selection_metric": selection["metric"],
            },
            "selection": selection,
            "result": result_metrics,
        }
        print(
            f"[RESULT/{case.name}] epoch={selection['epoch']} "
            f"accuracy={result_metrics['surrogate_acc']:.6f} "
            f"fidelity={result_metrics['fidelity']:.6f} "
            f"posterior_kl={result_metrics['posterior_kl']:.6f}"
        )
        del surrogate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = [results_by_case[case.name] for case in cases]
    add_effects(results)
    metrics_path = out_dir / "metrics.json"
    history_path = out_dir / "history.tsv"
    data_path = out_dir / "data.tsv"
    plot_path = out_dir / "metrics.png"
    payload = {
        "schema_version": 3,
        "experiment": EXPERIMENT,
        "protocol": "MS",
        **protocol_metadata(query),
        "protocol_sha256": protocol_sha256,
        "complete": True,
        "dataset": DATASET,
        "victim_model": MODEL,
        "seed": SEED,
        "randomization": {
            "reset_before_each_surrogate_initialization": True,
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": SEED,
            "query_sampler_seed": SEED,
            "purpose": "controlled_weight_semantic_closure",
        },
        "study": {
            "top_k_values": list(TOP_K_VALUES),
            "variants": {
                name: {
                    "label": style["label"],
                    "extra_state_names": (
                        [] if name == "top_k" else list(extra_groups[name])
                    ),
                }
                for name, style in VARIANTS.items()
            },
            "comparison_scope": "same_top_k_with_disjoint_weight_semantic_extras",
            "trained_case_count": 40,
            "reused_case_count": 8,
        },
        "source": {
            "method": "TensorShield",
            "rank_provenance": "author_confirmed_final_rank",
            "eligible_rank": list(ranking),
            "lab04_metrics": str(lab04_path.relative_to(ROOT)),
            "lab04_metrics_sha256": lab04_sha256,
            "lab05_metrics": str(lab05_path.relative_to(ROOT)),
            "lab05_metrics_sha256": lab05_sha256,
        },
        "victim_checkpoint": str(victim_checkpoint.relative_to(ROOT)),
        "victim_checkpoint_sha256": victim_sha256,
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "official_weight": str(official_weight.relative_to(ROOT)),
        "official_weight_sha256": sha256_file(official_weight),
        "posterior_path": str(query.target_path.relative_to(ROOT)),
        "posterior_sha256": query.target_sha256,
        "training": protocol_metadata(query),
        "primary": {
            "checkpoint": "best.pth",
            "selection_metric": "minimum_validation_soft_cross_entropy",
            "tie_break": "earliest_epoch",
        },
        "results": results,
        "references": references,
        "cross_check": {
            "lab05_weight": {
                "protected_param_count": lab05_weight["protection"][
                    "protected_param_count"
                ],
                "result": lab05_weight["result"],
            },
            "top17_all_extras_relation": "lab05_weight_plus_last_linear_bias",
        },
        "outputs": {
            "data": str(data_path.relative_to(ROOT)),
            "history": str(history_path.relative_to(ROOT)),
            "plot": str(plot_path.relative_to(ROOT)),
            "mask_pattern": str((out_dir / "<case>_mask.pt").relative_to(ROOT)),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_history(history_path, history_rows)
    write_data(data_path, results)
    plot_metrics(plot_path, results, references)
    print(f"[INFO] 结果：{metrics_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
