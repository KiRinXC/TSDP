#!/usr/bin/env python3
"""验证五个前中层 conv1.weight 对当前 MS 攻击的条件依赖。"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
LAB04_ROOT = ROOT / "lab" / "04_tensorshield"
sys.path.insert(0, str(LAB04_ROOT))
import candidate as lab04  # noqa: E402

sys.path.remove(str(LAB04_ROOT))

prefix = lab04.prefix
EXPERIMENT = "07_structure"
BASE_CASE = "five_conv1_bn_gamma_head"
BLACKBOX_CASE = lab04.BLACKBOX_CASE
EVALUATION_SEEDS = lab04.EVALUATION_SEEDS
DEPENDENCY_RANKS = (1, 2, 4, 7, 9)
METRICS = ("surrogate_acc", "fidelity", "posterior_kl")
HISTORY_FIELDS = lab04.HISTORY_FIELDS
DATA_FIELDS = (
    "seed",
    "case",
    "label",
    "exposed_rank",
    "exposed_unit",
    "exposed_state",
    "best_epoch",
    "protected_unit_count",
    "protected_param_count",
    "protected_param_ratio",
    "protection_mask_sha256",
    "surrogate_acc",
    "fidelity",
    "posterior_kl",
    "accuracy_minus_base",
    "fidelity_minus_base",
    "posterior_kl_minus_base",
    "all_metrics_attack_rebound_vs_base",
    "matched_blackbox_accuracy",
    "matched_blackbox_fidelity",
    "matched_blackbox_posterior_kl",
    "all_metrics_at_or_beyond_blackbox",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    label: str
    exposed_rank: int | None


BASE_SPEC = CaseSpec(BASE_CASE, "Protect all five conv1", None)
LEAVE_ONE_OUT_SPECS = tuple(
    CaseSpec(
        f"expose_rank_{rank:02d}",
        f"Expose rank-{rank}",
        rank,
    )
    for rank in DEPENDENCY_RANKS
)
STRATEGY_SPECS = (BASE_SPEC, *LEAVE_ONE_OUT_SPECS)
STRATEGY_CASES = tuple(spec.name for spec in STRATEGY_SPECS)
TRAINED_CASES = tuple(spec.name for spec in LEAVE_ONE_OUT_SPECS)
ALL_CASES = (*STRATEGY_CASES, BLACKBOX_CASE)
SPEC_BY_NAME = {spec.name: spec for spec in STRATEGY_SPECS}
CASE_COLORS = {
    BASE_CASE: "#555555",
    "expose_rank_01": "#0072B2",
    "expose_rank_02": "#56B4E9",
    "expose_rank_04": "#009E73",
    "expose_rank_07": "#E69F00",
    "expose_rank_09": "#D55E00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="复用与当前来源、seed 和 mask 完全一致的逐 case 进度。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对来源、十种子划分、六个策略 mask 与成本，不训练或写结果。",
    )
    return parser.parse_args()


def write_tsv(
    path: Path,
    rows: list[dict[str, object]],
    fields: tuple[str, ...],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def save_progress(
    metrics_path: Path,
    history_path: Path,
    *,
    results: list[dict[str, object]],
    history: list[dict[str, object]],
    source_sha256: str,
    victim_sha256: str,
    expected_hashes: dict[str, str],
) -> None:
    payload = {
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "source_sha256": source_sha256,
        "victim_sha256": victim_sha256,
        "expected_hashes": expected_hashes,
        "results": results,
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_tsv(history_path, history, HISTORY_FIELDS)


def load_progress(
    metrics_path: Path,
    history_path: Path,
    *,
    source_sha256: str,
    victim_sha256: str,
    expected_hashes: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not metrics_path.is_file() or not history_path.is_file():
        return [], []
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        tuple(payload.get("evaluation_seeds", ())) != EVALUATION_SEEDS
        or payload.get("source_sha256") != source_sha256
        or payload.get("victim_sha256") != victim_sha256
        or payload.get("expected_hashes") != expected_hashes
    ):
        raise ValueError("现有 dependency 进度与当前来源、victim 或 mask 不一致。")
    results = list(payload.get("results", ()))
    keys = {
        (int(row["seed"]), str(row["case"]))
        for row in results
    }
    expected_universe = {
        (seed, case)
        for seed in EVALUATION_SEEDS
        for case in TRAINED_CASES
    }
    if len(keys) != len(results) or not keys <= expected_universe:
        raise ValueError("现有 dependency 进度包含重复或未知 case。")
    with history_path.open("r", encoding="utf-8", newline="") as input_file:
        history = list(csv.DictReader(input_file, delimiter="\t"))
    history_by_key: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in history:
        key = (int(row["seed"]), str(row["case"]))
        if key in keys:
            history_by_key.setdefault(key, []).append(row)
    valid_keys = {
        key
        for key, rows in history_by_key.items()
        if len(rows) == prefix.EPOCHS
        and sorted(int(row["epoch"]) for row in rows)
        == list(range(1, prefix.EPOCHS + 1))
    }
    if valid_keys != keys:
        raise ValueError("现有 dependency 进度的训练历史不完整。")
    retained_history = [
        row
        for key in sorted(history_by_key)
        for row in history_by_key[key]
    ]
    return results, retained_history


def build_selected_states(
    victim: torch.nn.Module,
    spec: CaseSpec,
) -> tuple[tuple[str, ...], str | None]:
    base_states, base_weights, bn_gamma = lab04.build_candidate_drop06(victim)
    ranking = tuple(prefix.EXPECTED_ELIGIBLE_RANK)
    expected_weights = tuple(ranking[rank - 1] for rank in (1, 2, 3, 4, 7, 9))
    if base_weights != expected_weights:
        raise ValueError("Lab04 的 5.7529% 基础集合已经变化。")
    conv1_weights = tuple(
        ranking[rank - 1] for rank in DEPENDENCY_RANKS
    )
    if any(not name.endswith(".conv1.weight") for name in conv1_weights):
        raise ValueError("五个依赖候选不再全部是 conv1.weight。")
    if len(bn_gamma) != 20:
        raise ValueError("基础集合不再完整保护 20 个 BN gamma。")
    exposed_state = (
        None
        if spec.exposed_rank is None
        else ranking[spec.exposed_rank - 1]
    )
    if exposed_state is not None and exposed_state not in conv1_weights:
        raise ValueError(f"{spec.name} 暴露的不是固定 conv1 候选。")
    selected = tuple(
        state_name
        for state_name in base_states
        if state_name != exposed_state
    )
    return selected, exposed_state


def initialize_case(
    *,
    victim: torch.nn.Module,
    official_weight: Path,
    seed: int,
    spec: CaseSpec,
):
    selected, exposed_state = build_selected_states(victim, spec)
    units = prefix.build_resnet18_tensor_units(victim)
    unit_by_name = {unit.state_name: unit for unit in units}
    selected_units = [unit_by_name[name] for name in selected]
    unit_spec = ",".join(str(unit.index) for unit in selected_units)
    surrogate, plan, _, masks = prefix.initialize_surrogate(
        factory=prefix.imagenet_models.resnet18,
        factory_name=prefix.MODEL,
        weight_path=official_weight,
        victim_model=victim,
        num_classes=prefix.NUM_CLASSES,
        defense="custom",
        protected_units=unit_spec,
        protected_layers=None,
        protected_scalars=None,
        initialization_seed=seed,
    )
    expected_units = len(selected)
    expected_params = sum(
        victim.state_dict()[name].numel() for name in selected
    )
    expected = (expected_units, expected_params, True, "replace")
    actual = (
        plan.protected_unit_count,
        plan.protected_param_count,
        plan.classifier_protected,
        plan.head_mode,
    )
    if actual != expected:
        raise RuntimeError(f"{spec.name} 保护统计为 {actual}，期望 {expected}。")
    selected_set = set(selected)
    for name, mask in masks.items():
        if bool(mask.all()) != (name in selected_set) or (
            name not in selected_set and bool(mask.any())
        ):
            raise RuntimeError(f"{spec.name} 的 {name} mask 不是完整 unit 选择。")
    if exposed_state is not None and bool(masks[exposed_state].any()):
        raise RuntimeError(f"{spec.name} 没有完整暴露 {exposed_state}。")
    metadata = [
        {
            "index": unit.index,
            "state_name": unit.state_name,
            "state_kind": unit.state_kind,
            "numel": unit.numel,
        }
        for unit in selected_units
    ]
    return surrogate, plan, masks, metadata, exposed_state


def load_source(
    path: Path,
    *,
    victim_sha256: str,
    queries,
) -> tuple[dict[str, object], dict[tuple[int, str], dict[str, object]]]:
    payload = lab04.load_json(path, "Lab04 candidate")
    if (
        tuple(payload.get("evaluation_seeds", ())) != EVALUATION_SEEDS
        or payload.get("victim_checkpoint_sha256") != victim_sha256
        or payload.get("posterior_sha256")
        != queries[EVALUATION_SEEDS[0]].target_sha256
    ):
        raise ValueError("Lab04 candidate 与当前 victim、posterior 或 seeds 不一致。")
    source_cases = {
        lab04.CANDIDATE_DROP06_CASE,
        BLACKBOX_CASE,
    }
    result_by_key = {
        (int(row["seed"]), str(row["case"])): row
        for row in payload.get("results", ())
        if row.get("case") in source_cases
    }
    expected = {
        (seed, case)
        for seed in EVALUATION_SEEDS
        for case in source_cases
    }
    if set(result_by_key) != expected:
        raise ValueError("Lab04 candidate 缺少基础集合或 matched 黑盒结果。")
    for seed in EVALUATION_SEEDS:
        source_partition = payload["query_partitions"][str(seed)]
        current_partition = queries[seed].partition.to_metadata()
        for field in (
            "train_source_indices_sha256",
            "validation_source_indices_sha256",
        ):
            if source_partition[field] != current_partition[field]:
                raise ValueError(f"seed {seed} 的 query 划分与 Lab04 不一致。")
    return payload, result_by_key


def reuse_result(
    source: dict[str, object],
    *,
    local_case: str,
    origin: str,
    mask_path: Path,
) -> dict[str, object]:
    row = copy.deepcopy(source)
    row["case"] = local_case
    row["origin"] = origin
    row["source_case"] = source["case"]
    row["protection"]["mask_path"] = str(mask_path.relative_to(ROOT))
    return row


def trained_result(
    *,
    seed: int,
    spec: CaseSpec,
    plan,
    mask_path: Path,
    selected_units: list[dict[str, object]],
    exposed_state: str,
    selection: dict[str, object],
    result_metrics: dict[str, object],
) -> dict[str, object]:
    ranking = tuple(prefix.EXPECTED_ELIGIBLE_RANK)
    units = {
        unit.state_name: unit
        for unit in prefix.build_resnet18_tensor_units(
            prefix.imagenet_models.resnet18(num_classes=prefix.NUM_CLASSES)
        )
    }
    exposed_unit = units[exposed_state]
    return {
        "seed": seed,
        "case": spec.name,
        "origin": "trained_lab07_leave_one_out",
        "query_partition_seed": seed,
        "randomization": {
            "reset_before_surrogate_initialization": True,
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": seed,
            "query_sampler_seed": seed,
        },
        "ablation": {
            "base_case": BASE_CASE,
            "exposed_rank": spec.exposed_rank,
            "exposed_unit": exposed_unit.index,
            "exposed_state": exposed_state,
            "eligible_rank_check": ranking[spec.exposed_rank - 1],
        },
        "protection": {
            "implementation_defense": "custom",
            **plan.to_metadata(),
            "mask_path": str(mask_path.relative_to(ROOT)),
            "selected_units": selected_units,
        },
        "primary": {
            "checkpoint": "best.pth",
            "epoch": selection["epoch"],
            "selection_metric": selection["metric"],
        },
        "selection": selection,
        "result": result_metrics,
    }


def metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def paired_effect(
    result_by_key: dict[tuple[int, str], dict[str, object]],
    left_case: str,
    right_case: str,
) -> dict[str, object]:
    differences = {
        metric: [
            float(result_by_key[(seed, left_case)]["result"][metric])
            - float(result_by_key[(seed, right_case)]["result"][metric])
            for seed in EVALUATION_SEEDS
        ]
        for metric in METRICS
    }
    rebound = {
        "surrogate_acc": sum(value > 0.0 for value in differences["surrogate_acc"]),
        "fidelity": sum(value > 0.0 for value in differences["fidelity"]),
        "posterior_kl": sum(value < 0.0 for value in differences["posterior_kl"]),
    }
    rebound["all_three"] = sum(
        differences["surrogate_acc"][index] > 0.0
        and differences["fidelity"][index] > 0.0
        and differences["posterior_kl"][index] < 0.0
        for index in range(len(EVALUATION_SEEDS))
    )
    return {
        "left_case": left_case,
        "right_case": right_case,
        "definition": "left_minus_right",
        "attack_rebound_direction": {
            "surrogate_acc": "positive",
            "fidelity": "positive",
            "posterior_kl": "negative",
        },
        "metrics": {
            metric: {
                **metric_summary(values),
                "values_by_seed": {
                    str(seed): value
                    for seed, value in zip(EVALUATION_SEEDS, values)
                },
            }
            for metric, values in differences.items()
        },
        "attack_rebound_counts": rebound,
    }


def build_aggregate(
    result_by_key: dict[tuple[int, str], dict[str, object]],
) -> dict[str, object]:
    groups = {
        case: {
            metric: metric_summary(
                [
                    float(result_by_key[(seed, case)]["result"][metric])
                    for seed in EVALUATION_SEEDS
                ]
            )
            for metric in METRICS
        }
        for case in ALL_CASES
    }
    paired = {
        spec.name: paired_effect(result_by_key, spec.name, BASE_CASE)
        for spec in LEAVE_ONE_OUT_SPECS
    }
    blackbox_counts = {}
    for case in STRATEGY_CASES:
        counts = {
            "surrogate_acc": 0,
            "fidelity": 0,
            "posterior_kl": 0,
            "all_three": 0,
        }
        for seed in EVALUATION_SEEDS:
            current = result_by_key[(seed, case)]["result"]
            blackbox = result_by_key[(seed, BLACKBOX_CASE)]["result"]
            conditions = {
                "surrogate_acc": current["surrogate_acc"]
                <= blackbox["surrogate_acc"],
                "fidelity": current["fidelity"] <= blackbox["fidelity"],
                "posterior_kl": current["posterior_kl"]
                >= blackbox["posterior_kl"],
            }
            conditions["all_three"] = all(conditions.values())
            for metric, condition in conditions.items():
                counts[metric] += int(condition)
        blackbox_counts[case] = counts
    return {
        "seed_count": len(EVALUATION_SEEDS),
        "sample_standard_deviation_ddof": 1,
        "groups": groups,
        "paired_leave_one_out_minus_base": paired,
        "at_or_beyond_matched_blackbox_counts": blackbox_counts,
    }


def build_data_rows(
    result_by_key: dict[tuple[int, str], dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for seed in EVALUATION_SEEDS:
        base = result_by_key[(seed, BASE_CASE)]["result"]
        blackbox = result_by_key[(seed, BLACKBOX_CASE)]["result"]
        for spec in STRATEGY_SPECS:
            row = result_by_key[(seed, spec.name)]
            result = row["result"]
            accuracy_difference = result["surrogate_acc"] - base["surrogate_acc"]
            fidelity_difference = result["fidelity"] - base["fidelity"]
            kl_difference = result["posterior_kl"] - base["posterior_kl"]
            rebound = (
                accuracy_difference > 0.0
                and fidelity_difference > 0.0
                and kl_difference < 0.0
            )
            at_blackbox = (
                result["surrogate_acc"] <= blackbox["surrogate_acc"]
                and result["fidelity"] <= blackbox["fidelity"]
                and result["posterior_kl"] >= blackbox["posterior_kl"]
            )
            ablation = row.get("ablation", {})
            rows.append(
                {
                    "seed": seed,
                    "case": spec.name,
                    "label": spec.label,
                    "exposed_rank": ablation.get("exposed_rank", ""),
                    "exposed_unit": ablation.get("exposed_unit", ""),
                    "exposed_state": ablation.get("exposed_state", ""),
                    "best_epoch": row["primary"]["epoch"],
                    "protected_unit_count": row["protection"][
                        "protected_unit_count"
                    ],
                    "protected_param_count": row["protection"][
                        "protected_param_count"
                    ],
                    "protected_param_ratio": row["protection"][
                        "protected_param_ratio"
                    ],
                    "protection_mask_sha256": row["protection"][
                        "protection_mask_sha256"
                    ],
                    "surrogate_acc": result["surrogate_acc"],
                    "fidelity": result["fidelity"],
                    "posterior_kl": result["posterior_kl"],
                    "accuracy_minus_base": accuracy_difference,
                    "fidelity_minus_base": fidelity_difference,
                    "posterior_kl_minus_base": kl_difference,
                    "all_metrics_attack_rebound_vs_base": rebound,
                    "matched_blackbox_accuracy": blackbox["surrogate_acc"],
                    "matched_blackbox_fidelity": blackbox["fidelity"],
                    "matched_blackbox_posterior_kl": blackbox["posterior_kl"],
                    "all_metrics_at_or_beyond_blackbox": at_blackbox,
                }
            )
    return rows


def plot_result(
    path: Path,
    result_by_key: dict[tuple[int, str], dict[str, object]],
    aggregate: dict[str, object],
) -> None:
    specifications = (
        ("surrogate_acc", "MS accuracy"),
        ("fidelity", "Fidelity"),
        ("posterior_kl", "Posterior KL"),
    )
    figure, axes = prefix.plt.subplots(1, 3, figsize=(19.2, 5.4))
    x_values = list(range(len(STRATEGY_CASES)))
    x_labels = []
    for case in STRATEGY_CASES:
        first = result_by_key[(EVALUATION_SEEDS[0], case)]
        ratio = first["protection"]["protected_param_ratio"] * 100.0
        if case == BASE_CASE:
            x_labels.append(f"Base\n{ratio:.4f}%")
        else:
            rank = first["ablation"]["exposed_rank"]
            unit = first["ablation"]["exposed_unit"]
            x_labels.append(f"Expose #{rank}\nunit {unit}\n{ratio:.4f}%")
    for axis, (metric, title) in zip(axes, specifications):
        means = [
            float(aggregate["groups"][case][metric]["mean"])
            for case in STRATEGY_CASES
        ]
        errors = [
            float(aggregate["groups"][case][metric]["sample_std"])
            for case in STRATEGY_CASES
        ]
        blackbox_mean = float(
            aggregate["groups"][BLACKBOX_CASE][metric]["mean"]
        )
        blackbox_std = float(
            aggregate["groups"][BLACKBOX_CASE][metric]["sample_std"]
        )
        axis.bar(
            x_values,
            means,
            yerr=errors,
            capsize=5,
            width=0.68,
            color=[CASE_COLORS[case] for case in STRATEGY_CASES],
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        for x_value, case in zip(x_values, STRATEGY_CASES):
            values = [
                float(result_by_key[(seed, case)]["result"][metric])
                for seed in EVALUATION_SEEDS
            ]
            offsets = [
                (index - (len(values) - 1) / 2.0) * 0.025
                for index in range(len(values))
            ]
            axis.scatter(
                [x_value + offset for offset in offsets],
                values,
                s=11,
                color="#222222",
                alpha=0.55,
                zorder=3,
            )
        axis.axhline(
            blackbox_mean,
            color="#CC79A7",
            linestyle="--",
            linewidth=1.5,
            label="Matched soft black-box mean",
        )
        axis.axhspan(
            blackbox_mean - blackbox_std,
            blackbox_mean + blackbox_std,
            color="#CC79A7",
            alpha=0.10,
            label="Black-box ± sample std",
        )
        axis.set_title(f"{title}\n10 seeds: mean ± sample std")
        axis.set_ylabel(title)
        axis.set_xticks(x_values, x_labels)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        prefix.set_y_limits(
            axis,
            [
                *means,
                *(mean - error for mean, error in zip(means, errors)),
                *(mean + error for mean, error in zip(means, errors)),
                blackbox_mean - blackbox_std,
                blackbox_mean + blackbox_std,
            ],
            bounded=metric != "posterior_kl",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Conditional dependency of five protected conv1 weights",
        y=1.10,
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white", dpi=240)
    prefix.plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    device = prefix.resolve_device(args.device)
    out_dir = ROOT / "results" / "lab" / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path = ROOT / "results" / "lab" / "04_tensorshield" / "candidate.json"
    dataset_root = ROOT / "dataset" / "public"
    protocol_root = ROOT / "dataset" / "MS"
    victim_checkpoint = (
        ROOT
        / "weights"
        / "MS"
        / "victim"
        / prefix.MODEL
        / prefix.DATASET
        / "best.pth"
    )
    official_weight = ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    prefix.configure_reproducibility(42, deterministic=True)
    victim, victim_metadata = prefix.build_victim(
        prefix.MODEL,
        prefix.NUM_CLASSES,
        victim_checkpoint,
    )
    victim_sha256 = prefix.sha256_file(victim_checkpoint)
    queries = {
        seed: prefix.prepare_soft_query(
            dataset=prefix.DATASET,
            model=prefix.MODEL,
            budget=prefix.BUDGET,
            seed=seed,
            dataset_root=dataset_root,
            protocol_root=protocol_root,
        )
        for seed in EVALUATION_SEEDS
    }
    source_payload, source_by_key = load_source(
        source_path,
        victim_sha256=victim_sha256,
        queries=queries,
    )
    source_sha256 = prefix.sha256_file(source_path)

    templates = {}
    expected_hashes = {}
    for seed in EVALUATION_SEEDS:
        for spec in STRATEGY_SPECS:
            prefix.configure_reproducibility(seed, deterministic=True)
            initialized = initialize_case(
                victim=victim,
                official_weight=official_weight,
                seed=seed,
                spec=spec,
            )
            surrogate, plan, masks, metadata, exposed_state = initialized
            if spec.name not in templates:
                templates[spec.name] = (
                    plan,
                    masks,
                    metadata,
                    exposed_state,
                )
                expected_hashes[spec.name] = plan.protection_mask_sha256
            elif plan.protection_mask_sha256 != expected_hashes[spec.name]:
                raise RuntimeError(f"{spec.name} mask 随 seed 变化。")
            del surrogate
        prefix.configure_reproducibility(seed, deterministic=True)
        surrogate, plan, masks = lab04.initialize_blackbox(
            victim,
            official_weight,
            seed,
        )
        if BLACKBOX_CASE not in templates:
            templates[BLACKBOX_CASE] = (plan, masks)
            expected_hashes[BLACKBOX_CASE] = plan.protection_mask_sha256
        elif plan.protection_mask_sha256 != expected_hashes[BLACKBOX_CASE]:
            raise RuntimeError("matched soft black-box mask 随 seed 变化。")
        del surrogate

    mask_paths = {
        BASE_CASE: out_dir / "dependency_base_mask.pt",
        **{
            spec.name: out_dir / f"dependency_{spec.name}_mask.pt"
            for spec in LEAVE_ONE_OUT_SPECS
        },
        BLACKBOX_CASE: out_dir / "dependency_full_mask.pt",
    }
    for case, mask_path in mask_paths.items():
        prefix.save_protection_mask(mask_path, templates[case][1])
        plan = templates[case][0]
        print(
            f"[MASK/{case}] units={plan.protected_unit_count}/122 "
            f"params={plan.protected_param_count}/{plan.total_param_count} "
            f"ratio={plan.protected_param_ratio:.6f} "
            f"sha256={plan.protection_mask_sha256}"
        )
    if args.dry_run:
        print(f"[INFO] 来源：{source_path.relative_to(ROOT)}")
        print(f"[INFO] 独立验证 seeds：{','.join(map(str, EVALUATION_SEEDS))}")
        print("[INFO] dry-run 完成，未写入依赖实验指标。")
        return 0

    progress_path = out_dir / "dependency_progress.json"
    progress_history_path = out_dir / "dependency_progress_history.tsv"
    if args.resume:
        trained_results, history_rows = load_progress(
            progress_path,
            progress_history_path,
            source_sha256=source_sha256,
            victim_sha256=victim_sha256,
            expected_hashes=expected_hashes,
        )
        print(
            f"[RESUME] 复用 {len(trained_results)} 个 leave-one-out case、"
            f"{len(history_rows)} 条训练历史。"
        )
    else:
        progress_path.unlink(missing_ok=True)
        progress_history_path.unlink(missing_ok=True)
        trained_results = []
        history_rows = []

    reused_results = []
    for seed in EVALUATION_SEEDS:
        reused_results.append(
            reuse_result(
                source_by_key[(seed, lab04.CANDIDATE_DROP06_CASE)],
                local_case=BASE_CASE,
                origin="reused_lab04_candidate",
                mask_path=mask_paths[BASE_CASE],
            )
        )
        reused_results.append(
            reuse_result(
                source_by_key[(seed, BLACKBOX_CASE)],
                local_case=BLACKBOX_CASE,
                origin="reused_lab04_matched_blackbox",
                mask_path=mask_paths[BLACKBOX_CASE],
            )
        )

    completed = {
        (int(row["seed"]), str(row["case"]))
        for row in trained_results
    }
    evaluation = None
    for seed in EVALUATION_SEEDS:
        query = queries[seed]
        for spec in LEAVE_ONE_OUT_SPECS:
            if (seed, spec.name) in completed:
                print(f"[RESUME] 跳过已完成 {seed}/{spec.name}。")
                continue
            prefix.configure_reproducibility(seed, deterministic=True)
            (
                surrogate,
                plan,
                _masks,
                selected_units,
                exposed_state,
            ) = initialize_case(
                victim=victim,
                official_weight=official_weight,
                seed=seed,
                spec=spec,
            )
            if plan.protection_mask_sha256 != expected_hashes[spec.name]:
                raise RuntimeError(f"{seed}/{spec.name} mask 在训练前漂移。")
            surrogate = surrogate.to(device)
            selection, history = prefix.train_validation_best(
                surrogate,
                query,
                device=device,
                num_workers=args.num_workers,
                seed=seed,
            )
            history_rows.extend(
                {"seed": seed, "case": spec.name, **row}
                for row in history
            )
            if evaluation is None:
                evaluation = prefix.prepare_eval(
                    victim,
                    dataset=prefix.DATASET,
                    dataset_root=dataset_root,
                    protocol_root=protocol_root,
                    device=device,
                    num_workers=args.num_workers,
                    seed=seed,
                )
            result_metrics = prefix.evaluate_once(surrogate, evaluation, device)
            trained_results.append(
                trained_result(
                    seed=seed,
                    spec=spec,
                    plan=plan,
                    mask_path=mask_paths[spec.name],
                    selected_units=selected_units,
                    exposed_state=exposed_state,
                    selection=selection,
                    result_metrics=result_metrics,
                )
            )
            print(
                f"[RESULT/seed={seed}/{spec.name}] epoch={selection['epoch']} "
                f"accuracy={result_metrics['surrogate_acc']:.6f} "
                f"fidelity={result_metrics['fidelity']:.6f} "
                f"posterior_kl={result_metrics['posterior_kl']:.6f}",
                flush=True,
            )
            del surrogate
            save_progress(
                progress_path,
                progress_history_path,
                results=trained_results,
                history=history_rows,
                source_sha256=source_sha256,
                victim_sha256=victim_sha256,
                expected_hashes=expected_hashes,
            )

    results = [*reused_results, *trained_results]
    result_by_key = {
        (int(row["seed"]), str(row["case"])): row for row in results
    }
    expected_keys = {
        (seed, case)
        for seed in EVALUATION_SEEDS
        for case in ALL_CASES
    }
    if set(result_by_key) != expected_keys or len(results) != len(expected_keys):
        raise RuntimeError("基础、五组 leave-one-out 与黑盒结果不完整。")
    results = [
        result_by_key[(seed, case)]
        for seed in EVALUATION_SEEDS
        for case in ALL_CASES
    ]
    history_rows.sort(
        key=lambda row: (
            int(row["seed"]),
            TRAINED_CASES.index(str(row["case"])),
            int(row["epoch"]),
        )
    )
    if len(history_rows) != len(EVALUATION_SEEDS) * len(TRAINED_CASES) * prefix.EPOCHS:
        raise RuntimeError("leave-one-out 训练历史不完整。")

    aggregate = build_aggregate(result_by_key)
    data_rows = build_data_rows(result_by_key)
    metrics_path = out_dir / "dependency.json"
    data_path = out_dir / "dependency.tsv"
    history_path = out_dir / "dependency_history.tsv"
    plot_path = out_dir / "dependency.png"
    write_tsv(data_path, data_rows, DATA_FIELDS)
    write_tsv(history_path, history_rows, HISTORY_FIELDS)
    plot_result(plot_path, result_by_key, aggregate)

    first_query = queries[EVALUATION_SEEDS[0]]
    payload = {
        "schema_version": 3,
        "experiment": "07_structure_conv1_dependency",
        "protocol": "MS",
        **prefix.protocol_metadata(first_query),
        "dataset": prefix.DATASET,
        "victim_model": prefix.MODEL,
        "seed": EVALUATION_SEEDS[0],
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "randomization": {
            "reset_before_each_surrogate_initialization": True,
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": EVALUATION_SEEDS[0],
            "query_sampler_seed": EVALUATION_SEEDS[0],
            "per_result_seeded": True,
        },
        "query_partitions": {
            str(seed): queries[seed].partition.to_metadata()
            for seed in EVALUATION_SEEDS
        },
        "scientific_status": "post_hoc_conditional_dependency_validation",
        "base_case": {
            "case": BASE_CASE,
            "source_case": lab04.CANDIDATE_DROP06_CASE,
            "protected_eligible_ranks": [1, 2, 3, 4, 7, 9],
            "tested_conv1_ranks": list(DEPENDENCY_RANKS),
            "add_all_bn_gamma": True,
            "fixed_head_weight_and_bias": True,
        },
        "leave_one_out_cases": {
            spec.name: {
                "label": spec.label,
                "exposed_rank": spec.exposed_rank,
                "exposed_state": templates[spec.name][3],
                "protected_unit_count": templates[spec.name][0].protected_unit_count,
                "protected_param_count": templates[spec.name][0].protected_param_count,
                "protected_param_ratio": templates[spec.name][0].protected_param_ratio,
                "mask": str(mask_paths[spec.name].relative_to(ROOT)),
                "protection_mask_sha256": expected_hashes[spec.name],
            }
            for spec in LEAVE_ONE_OUT_SPECS
        },
        "dependency_definition": {
            "comparison": "leave_one_out_minus_same_seed_base",
            "attack_rebound": {
                "surrogate_acc": "strictly_higher",
                "fidelity": "strictly_higher",
                "posterior_kl": "strictly_lower",
            },
            "member_level_dependency_requires": (
                "stable_paired_attack_rebound_across_evaluation_seeds"
            ),
        },
        "source": {
            "lab04_candidate": str(source_path.relative_to(ROOT)),
            "lab04_candidate_sha256": source_sha256,
        },
        "victim_checkpoint": str(victim_checkpoint.relative_to(ROOT)),
        "victim_checkpoint_sha256": victim_sha256,
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "official_weight": str(official_weight.relative_to(ROOT)),
        "official_weight_sha256": prefix.sha256_file(official_weight),
        "posterior_path": str(first_query.target_path.relative_to(ROOT)),
        "posterior_sha256": first_query.target_sha256,
        "primary": {
            "checkpoint": "best.pth",
            "selection_metric": "minimum_validation_soft_cross_entropy",
            "tie_break": "earliest_epoch",
            "reported_statistic": "mean_and_sample_standard_deviation",
        },
        "results": results,
        "aggregate": aggregate,
        "outputs": {
            "data": str(data_path.relative_to(ROOT)),
            "history": str(history_path.relative_to(ROOT)),
            "plot": str(plot_path.relative_to(ROOT)),
            "masks": {
                case: str(path.relative_to(ROOT))
                for case, path in mask_paths.items()
            },
        },
        "execution": {
            "reused_complete_case_count": len(reused_results),
            "trained_complete_case_count": len(trained_results),
            "resume_requested": args.resume,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    progress_path.unlink(missing_ok=True)
    progress_history_path.unlink(missing_ok=True)
    for spec in LEAVE_ONE_OUT_SPECS:
        effect = aggregate["paired_leave_one_out_minus_base"][spec.name]
        counts = effect["attack_rebound_counts"]
        print(
            f"[SUMMARY/{spec.name}] "
            f"accuracy_delta={effect['metrics']['surrogate_acc']['mean']:+.6f} "
            f"fidelity_delta={effect['metrics']['fidelity']['mean']:+.6f} "
            f"kl_delta={effect['metrics']['posterior_kl']['mean']:+.6f} "
            f"all_three_rebound={counts['all_three']}/{len(EVALUATION_SEEDS)}"
        )
    print(f"[INFO] 结果：{metrics_path.relative_to(ROOT)}")
    print(f"[INFO] 图：{plot_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
