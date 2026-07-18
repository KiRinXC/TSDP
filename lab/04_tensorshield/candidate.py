#!/usr/bin/env python3
"""以十个独立 seed 对比 Top-10、BN gamma 闭包与删点候选。"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

import run as prefix


TOP10_CASE = "tensorshield_top10"
TOP10_BN_CASE = "tensorshield_top10_bn_gamma"
CANDIDATE_CASE = "candidate_drop_05_08_10_bn_gamma"
BLACKBOX_CASE = "soft_full_protection"
STRATEGY_CASES = (TOP10_CASE, TOP10_BN_CASE, CANDIDATE_CASE)
ALL_CASES = (*STRATEGY_CASES, BLACKBOX_CASE)
SELECTION_SEED = 42
EVALUATION_SEEDS = tuple(range(43, 53))
KEPT_ELIGIBLE_RANKS = (1, 2, 3, 4, 6, 7, 9)
DROPPED_TOP10_RANKS = (5, 8, 10)
EXPECTED_STATS = {
    TOP10_CASE: (11, 1_009_764),
    TOP10_BN_CASE: (31, 1_014_564),
    CANDIDATE_CASE: (28, 793_380),
}
CASE_LABELS = {
    TOP10_CASE: "TensorShield Top-10",
    TOP10_BN_CASE: "Top-10 + BN gamma",
    CANDIDATE_CASE: "Drop 5/8/10 + BN gamma",
    BLACKBOX_CASE: "Matched soft black-box",
}
CASE_COLORS = {
    TOP10_CASE: "#555555",
    TOP10_BN_CASE: "#009E73",
    CANDIDATE_CASE: "#AA3377",
}
METRICS = ("surrogate_acc", "fidelity", "posterior_kl")
HISTORY_FIELDS = (
    "seed",
    "case",
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
    "seed",
    "case",
    "label",
    "best_epoch",
    "protected_unit_count",
    "protected_param_count",
    "protected_param_ratio",
    "protection_mask_sha256",
    "surrogate_acc",
    "fidelity",
    "posterior_kl",
    "matched_blackbox_accuracy",
    "matched_blackbox_fidelity",
    "matched_blackbox_posterior_kl",
    "accuracy_minus_blackbox",
    "fidelity_minus_blackbox",
    "posterior_kl_minus_blackbox",
    "all_metrics_at_or_beyond_blackbox",
)


@dataclass(frozen=True)
class StrategySpec:
    name: str
    eligible_ranks: tuple[int, ...]
    add_bn_gamma: bool


STRATEGIES = (
    StrategySpec(TOP10_CASE, tuple(range(1, 11)), False),
    StrategySpec(TOP10_BN_CASE, tuple(range(1, 11)), True),
    StrategySpec(CANDIDATE_CASE, KEPT_ELIGIBLE_RANKS, True),
)
STRATEGY_BY_NAME = {strategy.name: strategy for strategy in STRATEGIES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="复用 candidate.json 中与当前协议、seed、mask 完全一致的已完成 case。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对十种子划分、四个 mask 与保护统计，不训练或写结果。",
    )
    return parser.parse_args()


def load_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("attack_protocol") != prefix.ATTACK_PROTOCOL_VERSION:
        raise ValueError(f"{label} 不是当前 validation-best 协议。")
    return payload


def derive_bn_gamma(victim: nn.Module) -> tuple[str, ...]:
    names = tuple(
        f"{module_name}.weight"
        for module_name, module in victim.named_modules()
        if module_name and isinstance(module, nn.BatchNorm2d)
    )
    state = victim.state_dict()
    if len(names) != 20 or sum(state[name].numel() for name in names) != 4_800:
        raise RuntimeError("ResNet18 BN gamma 应为 20 个 unit、4,800 个参数。")
    return names


def build_strategy_states(
    victim: nn.Module,
    strategy: StrategySpec,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    ranking, _ = prefix.build_cases(victim)
    expected_ranking = tuple(prefix.EXPECTED_ELIGIBLE_RANK)
    if ranking != expected_ranking:
        raise ValueError("候选规则依赖的作者 eligible rank 已变化。")
    if any(rank < 1 or rank > 10 for rank in strategy.eligible_ranks):
        raise ValueError(f"{strategy.name} 只能从作者 Top-10 选择。")
    eligible_weights = tuple(ranking[rank - 1] for rank in strategy.eligible_ranks)
    bn_gamma = derive_bn_gamma(victim) if strategy.add_bn_gamma else ()
    selected = tuple(
        dict.fromkeys((*eligible_weights, "last_linear.bias", *bn_gamma))
    )
    expected_units, expected_params = EXPECTED_STATS[strategy.name]
    protected_params = sum(victim.state_dict()[name].numel() for name in selected)
    if (len(selected), protected_params) != (expected_units, expected_params):
        raise RuntimeError(
            f"{strategy.name} 统计为 {(len(selected), protected_params)}，"
            f"期望 {(expected_units, expected_params)}。"
        )
    return selected, eligible_weights, bn_gamma


def build_candidate(
    victim: nn.Module,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """保留给单元测试与其他 Lab 使用的候选构造入口。"""
    return build_strategy_states(victim, STRATEGY_BY_NAME[CANDIDATE_CASE])


def initialize_strategy(
    strategy: StrategySpec,
    victim: nn.Module,
    official_weight: Path,
    seed: int,
):
    selected, eligible_weights, bn_gamma = build_strategy_states(victim, strategy)
    units = prefix.build_resnet18_tensor_units(victim)
    unit_by_name = {unit.state_name: unit for unit in units}
    missing = set(selected) - set(unit_by_name)
    if missing:
        raise ValueError(f"{strategy.name} 包含未知 state：{sorted(missing)}")
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
    expected_units, expected_params = EXPECTED_STATS[strategy.name]
    actual = (
        plan.protected_unit_count,
        plan.protected_param_count,
        plan.classifier_protected,
        plan.head_mode,
    )
    expected = (expected_units, expected_params, True, "replace")
    if actual != expected:
        raise RuntimeError(f"{strategy.name} 保护统计为 {actual}，期望 {expected}。")
    selected_set = set(selected)
    for name, mask in masks.items():
        if bool(mask.all()) != (name in selected_set) or (
            name not in selected_set and bool(mask.any())
        ):
            raise RuntimeError(f"{strategy.name} 的 {name} mask 不是完整 unit 选择。")

    eligible_set = set(eligible_weights)
    gamma_set = set(bn_gamma)
    metadata = []
    for unit in selected_units:
        if unit.state_name in eligible_set:
            role = "eligible_weight"
            eligible_rank = prefix.EXPECTED_ELIGIBLE_RANK.index(unit.state_name) + 1
        elif unit.state_name == "last_linear.bias":
            role = "fixed_head_bias"
            eligible_rank = None
        elif unit.state_name in gamma_set:
            role = "bn_gamma"
            eligible_rank = None
        else:
            raise RuntimeError(f"无法解释 {strategy.name} 的 state：{unit.state_name}")
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
    return (
        surrogate,
        plan,
        masks,
        metadata,
        eligible_weights,
        bn_gamma,
    )


def initialize_blackbox(
    victim: nn.Module,
    official_weight: Path,
    seed: int,
):
    surrogate, plan, _, masks = prefix.initialize_surrogate(
        factory=prefix.imagenet_models.resnet18,
        factory_name=prefix.MODEL,
        weight_path=official_weight,
        victim_model=victim,
        num_classes=prefix.NUM_CLASSES,
        defense="full_protection",
        protected_units=None,
        protected_layers=None,
        protected_scalars=None,
        initialization_seed=seed,
    )
    expected = (
        plan.tensor_unit_count,
        plan.total_param_count,
        True,
        "replace",
    )
    actual = (
        plan.protected_unit_count,
        plan.protected_param_count,
        plan.classifier_protected,
        plan.head_mode,
    )
    if actual != expected or not all(bool(mask.all()) for mask in masks.values()):
        raise RuntimeError(f"soft full-protection 对照统计异常：{actual}。")
    return surrogate, plan, masks


def load_references(
    lab04_path: Path,
    ablation_path: Path,
    lab06_path: Path,
) -> dict[str, object]:
    lab04 = load_json(lab04_path, "Lab04 metrics.json")
    ablation = load_json(ablation_path, "Lab04 ablation.json")
    lab06 = load_json(lab06_path, "Lab06 metrics.json")
    interaction = ablation.get("source", {}).get("interaction_2x2", {})
    if tuple(interaction.get("base_dropped_ranks", ())) != DROPPED_TOP10_RANKS:
        raise ValueError("Lab04 2×2 的共同基准不再是删除 rank-5/8/10。")
    if tuple(interaction.get("factor_ranks", ())) != (6, 7):
        raise ValueError("Lab04 2×2 不再同时检验 rank-6/rank-7。")
    by_case = {str(row["case"]): row for row in ablation.get("results", [])}
    for case in (
        "drop_05_08_10",
        "drop_05_06_08_10",
        "drop_05_07_08_10",
        "drop_05_06_07_08_10",
    ):
        if case not in by_case:
            raise ValueError(f"Lab04 2×2 缺少 {case}。")
    base = by_case["drop_05_08_10"]["result"]
    for case in (
        "drop_05_06_08_10",
        "drop_05_07_08_10",
        "drop_05_06_07_08_10",
    ):
        current = by_case[case]["result"]
        if not (
            current["surrogate_acc"] > base["surrogate_acc"]
            and current["fidelity"] > base["fidelity"]
            and current["posterior_kl"] < base["posterior_kl"]
        ):
            raise ValueError(f"当前 2×2 不再支持同时保留 rank-6/7：{case}。")

    prefix_by_k = {int(row["top_k"]): row for row in lab04.get("results", [])}
    gamma_matches = [
        row
        for row in lab06.get("results", [])
        if row.get("case") == "top_10_bn_gamma"
    ]
    if 10 not in prefix_by_k or len(gamma_matches) != 1:
        raise ValueError("Lab04/Lab06 缺少 Top-10 或 Top-10 + BN gamma 参考。")
    return {
        "no_protection": lab04["references"]["no_protection"],
        "formal_soft_blackbox_seed42": lab04["references"]["full_protection"],
        "hard_blackbox_seed42": lab04["references"]["hard_blackbox"],
        "tensorshield_top10_seed42": prefix_by_k[10],
        "top10_bn_gamma_seed42": gamma_matches[0],
        "interaction_2x2_seed42": {
            case: by_case[case] for case in interaction["cells"]
        },
    }


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


def metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def difference_summary(
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
    favorable = {
        "surrogate_acc": sum(value <= 0.0 for value in differences["surrogate_acc"]),
        "fidelity": sum(value <= 0.0 for value in differences["fidelity"]),
        "posterior_kl": sum(value >= 0.0 for value in differences["posterior_kl"]),
    }
    favorable["all_three"] = sum(
        differences["surrogate_acc"][index] <= 0.0
        and differences["fidelity"][index] <= 0.0
        and differences["posterior_kl"][index] >= 0.0
        for index in range(len(EVALUATION_SEEDS))
    )
    return {
        "left_case": left_case,
        "right_case": right_case,
        "definition": "left_minus_right",
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
        "left_at_or_better_than_right_counts": favorable,
    }


def build_aggregate(
    result_by_key: dict[tuple[int, str], dict[str, object]],
) -> dict[str, object]:
    groups = {}
    for case in ALL_CASES:
        groups[case] = {
            metric: metric_summary(
                [
                    float(result_by_key[(seed, case)]["result"][metric])
                    for seed in EVALUATION_SEEDS
                ]
            )
            for metric in METRICS
        }
        groups[case]["best_epoch"] = metric_summary(
            [
                float(result_by_key[(seed, case)]["primary"]["epoch"])
                for seed in EVALUATION_SEEDS
            ]
        )
    blackbox_counts = {}
    for case in STRATEGY_CASES:
        comparison = difference_summary(result_by_key, case, BLACKBOX_CASE)
        blackbox_counts[case] = comparison[
            "left_at_or_better_than_right_counts"
        ]
    return {
        "seed_count": len(EVALUATION_SEEDS),
        "sample_standard_deviation_ddof": 1,
        "groups": groups,
        "paired_effects": {
            "bn_gamma": difference_summary(
                result_by_key,
                TOP10_BN_CASE,
                TOP10_CASE,
            ),
            "drop_05_08_10_given_bn_gamma": difference_summary(
                result_by_key,
                CANDIDATE_CASE,
                TOP10_BN_CASE,
            ),
            "candidate_minus_blackbox": difference_summary(
                result_by_key,
                CANDIDATE_CASE,
                BLACKBOX_CASE,
            ),
        },
        "at_or_beyond_matched_blackbox_counts": blackbox_counts,
    }


def build_data_rows(
    result_by_key: dict[tuple[int, str], dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for seed in EVALUATION_SEEDS:
        blackbox = result_by_key[(seed, BLACKBOX_CASE)]["result"]
        for case in STRATEGY_CASES:
            result = result_by_key[(seed, case)]
            metrics = result["result"]
            conditions = (
                metrics["surrogate_acc"] <= blackbox["surrogate_acc"],
                metrics["fidelity"] <= blackbox["fidelity"],
                metrics["posterior_kl"] >= blackbox["posterior_kl"],
            )
            rows.append(
                {
                    "seed": seed,
                    "case": case,
                    "label": CASE_LABELS[case],
                    "best_epoch": result["primary"]["epoch"],
                    "protected_unit_count": result["protection"][
                        "protected_unit_count"
                    ],
                    "protected_param_count": result["protection"][
                        "protected_param_count"
                    ],
                    "protected_param_ratio": result["protection"][
                        "protected_param_ratio"
                    ],
                    "protection_mask_sha256": result["protection"][
                        "protection_mask_sha256"
                    ],
                    "surrogate_acc": metrics["surrogate_acc"],
                    "fidelity": metrics["fidelity"],
                    "posterior_kl": metrics["posterior_kl"],
                    "matched_blackbox_accuracy": blackbox["surrogate_acc"],
                    "matched_blackbox_fidelity": blackbox["fidelity"],
                    "matched_blackbox_posterior_kl": blackbox["posterior_kl"],
                    "accuracy_minus_blackbox": (
                        metrics["surrogate_acc"] - blackbox["surrogate_acc"]
                    ),
                    "fidelity_minus_blackbox": (
                        metrics["fidelity"] - blackbox["fidelity"]
                    ),
                    "posterior_kl_minus_blackbox": (
                        metrics["posterior_kl"] - blackbox["posterior_kl"]
                    ),
                    "all_metrics_at_or_beyond_blackbox": all(conditions),
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
    figure, axes = prefix.plt.subplots(1, 3, figsize=(15.6, 5.1))
    x_values = list(range(len(STRATEGY_CASES)))
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
            width=0.66,
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
                (index - (len(values) - 1) / 2.0) * 0.035
                for index in range(len(values))
            ]
            axis.scatter(
                [x_value + offset for offset in offsets],
                values,
                s=12,
                color="#222222",
                alpha=0.58,
                zorder=3,
            )
        axis.axhline(
            blackbox_mean,
            color="#0072B2",
            linestyle="--",
            linewidth=1.5,
            label="Matched soft black-box mean",
        )
        axis.axhspan(
            blackbox_mean - blackbox_std,
            blackbox_mean + blackbox_std,
            color="#0072B2",
            alpha=0.10,
            label="Black-box ± sample std",
        )
        axis.set_title(
            f"{title}\n10 seeds: mean ± sample std",
        )
        axis.set_ylabel(title)
        axis.set_xticks(
            x_values,
            (
                "Top-10\n8.9934%",
                "+ BN gamma\n9.0362%",
                "Drop 5/8/10\n+ BN gamma\n7.0662%",
            ),
        )
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
        "TensorShield, BN gamma closure, and post-hoc candidate",
        y=1.10,
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white", dpi=240)
    prefix.plt.close(figure)


def result_row(
    *,
    seed: int,
    case: str,
    plan,
    mask_path: Path,
    selection: dict[str, object],
    result_metrics: dict[str, object],
    selected_units: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    protection = {
        "implementation_defense": (
            "full_protection" if case == BLACKBOX_CASE else "custom"
        ),
        **plan.to_metadata(),
        "mask_path": str(mask_path.relative_to(prefix.ROOT)),
    }
    if selected_units is not None:
        protection["selected_units"] = selected_units
    return {
        "seed": seed,
        "case": case,
        "query_partition_seed": seed,
        "randomization": {
            "reset_before_surrogate_initialization": True,
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": seed,
            "query_sampler_seed": seed,
        },
        "protection": protection,
        "primary": {
            "checkpoint": "best.pth",
            "epoch": selection["epoch"],
            "selection_metric": selection["metric"],
        },
        "selection": selection,
        "result": result_metrics,
    }


def load_resume_rows(
    metrics_path: Path,
    history_path: Path,
    *,
    queries,
    victim_sha256: str,
    expected_hashes: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not metrics_path.is_file() or not history_path.is_file():
        return [], []
    payload = load_json(metrics_path, "candidate resume")
    if (
        tuple(payload.get("evaluation_seeds", ())) != EVALUATION_SEEDS
        or payload.get("victim_checkpoint_sha256") != victim_sha256
        or payload.get("posterior_sha256")
        != queries[EVALUATION_SEEDS[0]].target_sha256
    ):
        raise ValueError("现有 candidate 结果与当前 seed、victim 或 posterior 不一致。")
    stored_partitions = payload.get("query_partitions", {})
    for seed in EVALUATION_SEEDS:
        current = queries[seed].partition.to_metadata()
        stored = stored_partitions.get(str(seed), {})
        if (
            stored.get("train_source_indices_sha256")
            != current["train_source_indices_sha256"]
            or stored.get("validation_source_indices_sha256")
            != current["validation_source_indices_sha256"]
        ):
            raise ValueError(f"现有 candidate seed {seed} 的 query 划分不一致。")

    retained_results = []
    retained_keys = set()
    for row in payload.get("results", ()):
        seed = int(row.get("seed", -1))
        case = str(row.get("case", ""))
        if seed not in EVALUATION_SEEDS or case not in ALL_CASES:
            continue
        protection = row.get("protection", {})
        if protection.get("protection_mask_sha256") != expected_hashes[case]:
            raise ValueError(f"现有 candidate {seed}/{case} 的 mask 不一致。")
        if (
            row.get("query_partition_seed") != seed
            or row.get("randomization", {}).get("surrogate_initialization_seed")
            != seed
            or row.get("randomization", {}).get("query_sampler_seed") != seed
            or row.get("result", {}).get("eval_passes") != 1
        ):
            raise ValueError(f"现有 candidate {seed}/{case} 的协议元数据不一致。")
        key = (seed, case)
        if key in retained_keys:
            raise ValueError(f"现有 candidate 重复结果：{key}。")
        retained_keys.add(key)
        retained_results.append(row)

    with history_path.open("r", encoding="utf-8", newline="") as input_file:
        history = list(csv.DictReader(input_file, delimiter="\t"))
    grouped_history: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in history:
        key = (int(row["seed"]), row["case"])
        if key in retained_keys:
            grouped_history.setdefault(key, []).append(row)
    valid_keys = {
        key
        for key, rows in grouped_history.items()
        if len(rows) == prefix.EPOCHS
        and sorted(int(row["epoch"]) for row in rows)
        == list(range(1, prefix.EPOCHS + 1))
    }
    if valid_keys != retained_keys:
        missing = sorted(retained_keys - valid_keys)
        raise ValueError(f"现有 candidate 训练历史不完整：{missing}。")
    retained_history = [
        row
        for key in sorted(grouped_history)
        for row in grouped_history[key]
    ]
    return retained_results, retained_history


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    device = prefix.resolve_device(args.device)
    out_dir = prefix.ROOT / "results" / "lab" / prefix.EXPERIMENT
    lab04_path = out_dir / "metrics.json"
    ablation_path = out_dir / "ablation.json"
    lab06_path = prefix.ROOT / "results" / "lab" / "06_weight" / "metrics.json"
    references = load_references(lab04_path, ablation_path, lab06_path)

    dataset_root = prefix.ROOT / "dataset" / "public"
    protocol_root = prefix.ROOT / "dataset" / "MS"
    victim_checkpoint = (
        prefix.ROOT
        / "weights"
        / "MS"
        / "victim"
        / prefix.MODEL
        / prefix.DATASET
        / "best.pth"
    )
    official_weight = prefix.ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    prefix.configure_reproducibility(SELECTION_SEED, deterministic=True)
    victim, victim_metadata = prefix.build_victim(
        prefix.MODEL, prefix.NUM_CLASSES, victim_checkpoint
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
    posterior_hashes = {query.target_sha256 for query in queries.values()}
    if len(posterior_hashes) != 1:
        raise RuntimeError("十个 seed 未读取同一 soft posterior。")
    for seed, query in queries.items():
        expected_victim_sha256 = query.manifest.get("victim", {}).get(
            "checkpoint_sha256"
        )
        if expected_victim_sha256 and expected_victim_sha256 != victim_sha256:
            raise ValueError(f"seed {seed} 的 posterior 与 victim best.pth 不一致。")
    partition_hashes = {
        str(query.partition.to_metadata()["train_source_indices_sha256"])
        for query in queries.values()
    }
    if len(partition_hashes) != len(EVALUATION_SEEDS):
        raise RuntimeError("十个 seed 没有产生十组唯一 query-train 划分。")

    templates = {}
    expected_hashes: dict[str, str] = {}
    for seed in EVALUATION_SEEDS:
        for strategy in STRATEGIES:
            prefix.configure_reproducibility(seed, deterministic=True)
            initialized = initialize_strategy(
                strategy,
                victim,
                official_weight,
                seed,
            )
            surrogate, plan, masks, metadata, weights, gamma = initialized
            if strategy.name not in templates:
                templates[strategy.name] = (
                    plan,
                    masks,
                    metadata,
                    weights,
                    gamma,
                )
                expected_hashes[strategy.name] = plan.protection_mask_sha256
            elif plan.protection_mask_sha256 != expected_hashes[strategy.name]:
                raise RuntimeError(f"{strategy.name} mask 随 seed 变化。")
            del surrogate
        prefix.configure_reproducibility(seed, deterministic=True)
        surrogate, full_plan, full_masks = initialize_blackbox(
            victim,
            official_weight,
            seed,
        )
        if BLACKBOX_CASE not in templates:
            templates[BLACKBOX_CASE] = (full_plan, full_masks)
            expected_hashes[BLACKBOX_CASE] = full_plan.protection_mask_sha256
        elif full_plan.protection_mask_sha256 != expected_hashes[BLACKBOX_CASE]:
            raise RuntimeError("full-protection mask 随 seed 变化。")
        del surrogate
        partition_metadata = queries[seed].partition.to_metadata()
        print(
            f"[PLAN/seed={seed}] train_hash="
            f"{partition_metadata['train_source_indices_sha256']} "
            f"validation_hash={partition_metadata['validation_source_indices_sha256']}"
        )

    for case in STRATEGY_CASES:
        plan = templates[case][0]
        print(
            f"[MASK/{case}] units={plan.protected_unit_count}/122 "
            f"params={plan.protected_param_count}/{plan.total_param_count} "
            f"ratio={plan.protected_param_ratio:.6f} head={plan.head_mode} "
            f"sha256={plan.protection_mask_sha256}"
        )
    full_plan = templates[BLACKBOX_CASE][0]
    print(
        f"[MASK/{BLACKBOX_CASE}] units={full_plan.protected_unit_count}/122 "
        f"params={full_plan.protected_param_count}/{full_plan.total_param_count} "
        f"sha256={full_plan.protection_mask_sha256}"
    )
    if args.dry_run:
        print(f"[INFO] 独立验证 seeds：{','.join(map(str, EVALUATION_SEEDS))}")
        print("[INFO] dry-run 完成，未写入候选产物。")
        return 0

    metrics_path = out_dir / "candidate.json"
    history_path = out_dir / "candidate_history.tsv"
    retained_results: list[dict[str, object]] = []
    retained_history: list[dict[str, object]] = []
    if args.resume:
        retained_results, retained_history = load_resume_rows(
            metrics_path,
            history_path,
            queries=queries,
            victim_sha256=victim_sha256,
            expected_hashes=expected_hashes,
        )
        print(
            f"[RESUME] 复用 {len(retained_results)} 个完整 case、"
            f"{len(retained_history)} 条训练历史。"
        )

    mask_paths = {
        TOP10_CASE: out_dir / "candidate_top10_mask.pt",
        TOP10_BN_CASE: out_dir / "candidate_top10_bn_gamma_mask.pt",
        CANDIDATE_CASE: out_dir / "candidate_mask.pt",
        BLACKBOX_CASE: out_dir / "candidate_full_mask.pt",
    }
    for case, mask_path in mask_paths.items():
        prefix.save_protection_mask(mask_path, templates[case][1])

    results = list(retained_results)
    history_rows = list(retained_history)
    completed = {
        (int(row["seed"]), str(row["case"]))
        for row in retained_results
    }
    evaluation = None
    for seed in EVALUATION_SEEDS:
        query = queries[seed]
        for case in ALL_CASES:
            if (seed, case) in completed:
                print(f"[RESUME] 跳过已完成 {seed}/{case}。")
                continue
            prefix.configure_reproducibility(seed, deterministic=True)
            if case == BLACKBOX_CASE:
                surrogate, current_plan, masks = initialize_blackbox(
                    victim,
                    official_weight,
                    seed,
                )
                selected_units = None
            else:
                strategy = STRATEGY_BY_NAME[case]
                (
                    surrogate,
                    current_plan,
                    masks,
                    selected_units,
                    _,
                    _,
                ) = initialize_strategy(
                    strategy,
                    victim,
                    official_weight,
                    seed,
                )
            if (
                current_plan.protection_mask_sha256
                != expected_hashes[case]
            ):
                raise RuntimeError(f"seed {seed} 的 {case} mask 在训练前漂移。")
            surrogate = surrogate.to(device)
            selection, history = prefix.train_validation_best(
                surrogate,
                query,
                device=device,
                num_workers=args.num_workers,
                seed=seed,
            )
            history_rows.extend(
                {"seed": seed, "case": case, **row} for row in history
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
            results.append(
                result_row(
                    seed=seed,
                    case=case,
                    plan=current_plan,
                    mask_path=mask_paths[case],
                    selection=selection,
                    result_metrics=result_metrics,
                    selected_units=selected_units,
                )
            )
            print(
                f"[RESULT/seed={seed}/{case}] epoch={selection['epoch']} "
                f"accuracy={result_metrics['surrogate_acc']:.6f} "
                f"fidelity={result_metrics['fidelity']:.6f} "
                f"posterior_kl={result_metrics['posterior_kl']:.6f}",
                flush=True,
            )
            del surrogate

    expected_keys = {
        (seed, case) for seed in EVALUATION_SEEDS for case in ALL_CASES
    }
    result_by_key = {
        (int(row["seed"]), str(row["case"])): row for row in results
    }
    if set(result_by_key) != expected_keys or len(results) != len(expected_keys):
        raise RuntimeError("三策略与配对黑盒的十种子结果不完整。")
    results = [
        result_by_key[(seed, case)]
        for seed in EVALUATION_SEEDS
        for case in ALL_CASES
    ]
    history_rows.sort(
        key=lambda row: (int(row["seed"]), ALL_CASES.index(str(row["case"])), int(row["epoch"]))
    )
    aggregate = build_aggregate(result_by_key)
    data_rows = build_data_rows(result_by_key)
    data_path = out_dir / "candidate.tsv"
    plot_path = out_dir / "candidate.png"
    write_tsv(history_path, history_rows, HISTORY_FIELDS)
    write_tsv(data_path, data_rows, DATA_FIELDS)
    plot_result(plot_path, result_by_key, aggregate)

    first_query = queries[EVALUATION_SEEDS[0]]
    payload = {
        "schema_version": 3,
        "experiment": "04_tensorshield_candidate_multiseed",
        "protocol": "MS",
        **prefix.protocol_metadata(first_query),
        "dataset": prefix.DATASET,
        "victim_model": prefix.MODEL,
        "seed": EVALUATION_SEEDS[0],
        "selection_seed": SELECTION_SEED,
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "randomization": {
            "reset_before_each_surrogate_initialization": True,
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": EVALUATION_SEEDS[0],
            "query_sampler_seed": EVALUATION_SEEDS[0],
            "per_result_seeded": True,
        },
        "multi_seed_protocol": {
            "selection_seed_excluded": True,
            "selection_seed": SELECTION_SEED,
            "evaluation_seeds": list(EVALUATION_SEEDS),
            "strategy_cases_per_seed": list(STRATEGY_CASES),
            "reference_cases_per_seed": [BLACKBOX_CASE],
            "matched_query_partition": True,
            "matched_surrogate_initialization_seed": True,
            "matched_query_sampler_seed": True,
            "candidate_selection_updated_from_current_seed42_ablation": True,
        },
        "query_partitions": {
            str(seed): queries[seed].partition.to_metadata()
            for seed in EVALUATION_SEEDS
        },
        "controlled_comparisons": {
            "bn_gamma_effect": {
                "left": TOP10_BN_CASE,
                "right": TOP10_CASE,
                "only_difference": "protect_all_20_bn_gamma",
            },
            "drop_effect_given_bn_gamma": {
                "left": CANDIDATE_CASE,
                "right": TOP10_BN_CASE,
                "only_difference": "expose_eligible_ranks_5_8_10",
            },
        },
        "selection_rule": {
            "base": "author_eligible_top10",
            "keep_eligible_ranks": list(KEPT_ELIGIBLE_RANKS),
            "drop_top10_ranks": list(DROPPED_TOP10_RANKS),
            "retain_dependency_ranks": [6, 7],
            "add_all_bn_gamma": True,
            "uses_ms_feedback_for_method_selection": True,
            "status": "post_hoc_scientific_validation_only",
        },
        "source": {
            "lab04_metrics": str(lab04_path.relative_to(prefix.ROOT)),
            "lab04_metrics_sha256": prefix.sha256_file(lab04_path),
            "lab04_ablation": str(ablation_path.relative_to(prefix.ROOT)),
            "lab04_ablation_sha256": prefix.sha256_file(ablation_path),
            "lab06_metrics": str(lab06_path.relative_to(prefix.ROOT)),
            "lab06_metrics_sha256": prefix.sha256_file(lab06_path),
        },
        "victim_checkpoint": str(victim_checkpoint.relative_to(prefix.ROOT)),
        "victim_checkpoint_sha256": victim_sha256,
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "official_weight": str(official_weight.relative_to(prefix.ROOT)),
        "official_weight_sha256": prefix.sha256_file(official_weight),
        "posterior_path": str(first_query.target_path.relative_to(prefix.ROOT)),
        "posterior_sha256": first_query.target_sha256,
        "strategies": {
            case: {
                "label": CASE_LABELS[case],
                "eligible_ranks": list(STRATEGY_BY_NAME[case].eligible_ranks),
                "add_all_bn_gamma": STRATEGY_BY_NAME[case].add_bn_gamma,
                "protected_unit_count": templates[case][0].protected_unit_count,
                "protected_param_count": templates[case][0].protected_param_count,
                "protected_param_ratio": templates[case][0].protected_param_ratio,
                "mask": str(mask_paths[case].relative_to(prefix.ROOT)),
                "protection_mask_sha256": expected_hashes[case],
            }
            for case in STRATEGY_CASES
        },
        "matched_soft_blackbox": {
            "mask": str(mask_paths[BLACKBOX_CASE].relative_to(prefix.ROOT)),
            "protection_mask_sha256": expected_hashes[BLACKBOX_CASE],
        },
        "primary": {
            "checkpoint": "best.pth",
            "selection_metric": "minimum_validation_soft_cross_entropy",
            "tie_break": "earliest_epoch",
            "reported_statistic": "mean_and_sample_standard_deviation",
        },
        "results": results,
        "aggregate": aggregate,
        "references": references,
        "execution": {
            "resume_requested": args.resume,
            "reused_complete_case_count": len(retained_results),
            "trained_complete_case_count": len(results) - len(retained_results),
        },
        "outputs": {
            "data": str(data_path.relative_to(prefix.ROOT)),
            "history": str(history_path.relative_to(prefix.ROOT)),
            "plot": str(plot_path.relative_to(prefix.ROOT)),
            "strategy_masks": {
                case: str(mask_paths[case].relative_to(prefix.ROOT))
                for case in STRATEGY_CASES
            },
            "blackbox_mask": str(
                mask_paths[BLACKBOX_CASE].relative_to(prefix.ROOT)
            ),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for case in STRATEGY_CASES:
        stats = aggregate["groups"][case]
        counts = aggregate["at_or_beyond_matched_blackbox_counts"][case]
        print(
            f"[SUMMARY/{case}] "
            f"accuracy={stats['surrogate_acc']['mean']:.6f}±"
            f"{stats['surrogate_acc']['sample_std']:.6f} "
            f"fidelity={stats['fidelity']['mean']:.6f}±"
            f"{stats['fidelity']['sample_std']:.6f} "
            f"posterior_kl={stats['posterior_kl']['mean']:.6f}±"
            f"{stats['posterior_kl']['sample_std']:.6f} "
            f"all_three_vs_bb={counts['all_three']}/{len(EVALUATION_SEEDS)}"
        )
    print(f"[INFO] 结果：{metrics_path.relative_to(prefix.ROOT)}")
    print(f"[INFO] 三柱图：{plot_path.relative_to(prefix.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
