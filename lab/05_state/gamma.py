#!/usr/bin/env python3
"""在固定候选保护集合上执行四类 BN gamma 的十种子 MS 消融。"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
LAB04_ROOT = ROOT / "lab" / "04_tensorshield"
if str(LAB04_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB04_ROOT))

import candidate as lab04  # noqa: E402


prefix = lab04.prefix
EXPERIMENT = "05_state_gamma_ablation"
OUT_DIR = ROOT / "results" / "lab" / "05_state"
SOURCE_PATH = ROOT / "results" / "lab" / "04_tensorshield" / "candidate.json"
SOURCE_HISTORY_PATH = (
    ROOT / "results" / "lab" / "04_tensorshield" / "candidate_history.tsv"
)
RESULT_PATH = OUT_DIR / "gamma.json"
DATA_PATH = OUT_DIR / "gamma.tsv"
HISTORY_PATH = OUT_DIR / "gamma_history.tsv"
PLOT_PATH = OUT_DIR / "gamma.png"
EVALUATION_SEEDS = tuple(range(43, 53))
SOURCE_ALL_CASE = lab04.CANDIDATE_DROP06_CASE
SOURCE_BLACKBOX_CASE = lab04.BLACKBOX_CASE

BASE_STATES = (
    "layer1.1.conv1.weight",
    "layer2.0.conv1.weight",
    "last_linear.weight",
    "layer1.0.conv1.weight",
    "layer2.1.conv1.weight",
    "layer3.0.conv1.weight",
    "last_linear.bias",
)
GAMMA_GROUP_ORDER = ("stem", "block_bn1", "block_bn2", "downsample")
GAMMA_GROUP_LABELS = {
    "stem": "Stem",
    "block_bn1": "Block BN1",
    "block_bn2": "Block BN2",
    "downsample": "Downsample BN",
}
GAMMA_GROUP_EXPECTED = {
    "stem": (1, 64),
    "block_bn1": (8, 1_920),
    "block_bn2": (8, 1_920),
    "downsample": (3, 896),
}

NO_GAMMA = "no_gamma"
ALL_GAMMA = "all_gamma"
DROP_STEM = "drop_stem"
DROP_BLOCK_BN1 = "drop_block_bn1"
DROP_BLOCK_BN2 = "drop_block_bn2"
DROP_DOWNSAMPLE = "drop_downsample"
CASES = (
    NO_GAMMA,
    ALL_GAMMA,
    DROP_STEM,
    DROP_BLOCK_BN1,
    DROP_BLOCK_BN2,
    DROP_DOWNSAMPLE,
)
TRAINED_CASES = tuple(case for case in CASES if case != ALL_GAMMA)
CASE_LABELS = {
    NO_GAMMA: "No gamma",
    ALL_GAMMA: "All 20 gamma",
    DROP_STEM: "All - Stem",
    DROP_BLOCK_BN1: "All - Block BN1",
    DROP_BLOCK_BN2: "All - Block BN2",
    DROP_DOWNSAMPLE: "All - Downsample",
}
CASE_COLORS = {
    NO_GAMMA: "#777777",
    ALL_GAMMA: "#0072B2",
    DROP_STEM: "#56B4E9",
    DROP_BLOCK_BN1: "#E69F00",
    DROP_BLOCK_BN2: "#D55E00",
    DROP_DOWNSAMPLE: "#009E73",
}
CASE_DROPPED_GROUP = {
    NO_GAMMA: tuple(GAMMA_GROUP_ORDER),
    ALL_GAMMA: (),
    DROP_STEM: ("stem",),
    DROP_BLOCK_BN1: ("block_bn1",),
    DROP_BLOCK_BN2: ("block_bn2",),
    DROP_DOWNSAMPLE: ("downsample",),
}
METRICS = ("surrogate_acc", "fidelity", "posterior_kl")
HISTORY_FIELDS = lab04.HISTORY_FIELDS
DATA_FIELDS = (
    "seed",
    "case",
    "label",
    "best_epoch",
    "protected_unit_count",
    "protected_param_count",
    "protected_param_ratio",
    "protected_gamma_count",
    "protected_gamma_param_count",
    "dropped_gamma_group",
    "protection_mask_sha256",
    "surrogate_acc",
    "fidelity",
    "posterior_kl",
    "matched_soft_blackbox_accuracy",
    "matched_soft_blackbox_fidelity",
    "matched_soft_blackbox_posterior_kl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="复用 gamma.json 中与当前协议、seed 和 mask 一致的已完成训练 case。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对四类 gamma、六个 mask、来源和十种子划分，不训练或写结果。",
    )
    return parser.parse_args()


def derive_gamma_groups(victim: nn.Module) -> dict[str, tuple[str, ...]]:
    all_gamma = lab04.derive_bn_gamma(victim)
    groups = {
        "stem": tuple(name for name in all_gamma if name == "bn1.weight"),
        "block_bn1": tuple(
            name
            for name in all_gamma
            if name.startswith("layer")
            and ".downsample." not in name
            and name.endswith(".bn1.weight")
        ),
        "block_bn2": tuple(
            name
            for name in all_gamma
            if name.startswith("layer")
            and ".downsample." not in name
            and name.endswith(".bn2.weight")
        ),
        "downsample": tuple(
            name for name in all_gamma if name.endswith(".downsample.1.weight")
        ),
    }
    state = victim.state_dict()
    for group, expected in GAMMA_GROUP_EXPECTED.items():
        actual = (len(groups[group]), sum(state[name].numel() for name in groups[group]))
        if actual != expected:
            raise RuntimeError(f"{group} gamma 统计为 {actual}，期望 {expected}。")
    flattened = tuple(name for group in GAMMA_GROUP_ORDER for name in groups[group])
    if len(flattened) != len(set(flattened)) or set(flattened) != set(all_gamma):
        raise RuntimeError("四类 gamma 没有互斥且完整地覆盖全部 20 个 BN gamma。")
    return groups


def selected_states(
    case: str,
    gamma_groups: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    dropped = set(CASE_DROPPED_GROUP[case])
    selected_gamma = tuple(
        name
        for group in GAMMA_GROUP_ORDER
        if group not in dropped
        for name in gamma_groups[group]
    )
    return tuple(dict.fromkeys((*BASE_STATES, *selected_gamma)))


def initialize_case(
    case: str,
    victim: nn.Module,
    official_weight: Path,
    gamma_groups: dict[str, tuple[str, ...]],
    seed: int,
):
    selected = selected_states(case, gamma_groups)
    units = prefix.build_resnet18_tensor_units(victim)
    unit_by_name = {unit.state_name: unit for unit in units}
    missing = set(selected) - set(unit_by_name)
    if missing:
        raise ValueError(f"{case} 包含未知 state：{sorted(missing)}")
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
    if not plan.classifier_protected or plan.head_mode != "replace":
        raise RuntimeError(f"{case} 没有固定保护完整分类头。")
    selected_set = set(selected)
    for name, mask in masks.items():
        expected = name in selected_set
        if bool(mask.all()) != expected or (not expected and bool(mask.any())):
            raise RuntimeError(f"{case} 的 {name} mask 不是完整 tensor 选择。")

    gamma_by_name = {
        name: group for group, names in gamma_groups.items() for name in names
    }
    metadata = []
    for unit in selected_units:
        if unit.state_name in BASE_STATES:
            role = "fixed_head" if unit.state_name.startswith("last_linear.") else "fixed_conv1"
            gamma_group = None
        else:
            role = "bn_gamma"
            gamma_group = gamma_by_name[unit.state_name]
        metadata.append(
            {
                "index": unit.index,
                "state_name": unit.state_name,
                "state_kind": unit.state_kind,
                "numel": unit.numel,
                "role": role,
                "gamma_group": gamma_group,
            }
        )
    return surrogate, plan, masks, metadata


def load_source(
    victim: nn.Module,
    victim_sha256: str,
    official_weight_sha256: str,
    posterior_sha256: str,
    expected_mask_sha256: str,
    queries: dict[int, object],
) -> tuple[
    dict[int, dict[str, object]],
    dict[int, dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    if not SOURCE_PATH.is_file() or not SOURCE_HISTORY_PATH.is_file():
        raise FileNotFoundError("缺少 Lab04 candidate 十种子来源。")
    payload = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 3,
        "attack_protocol": prefix.ATTACK_PROTOCOL_VERSION,
        "dataset": prefix.DATASET,
        "victim_model": prefix.MODEL,
        "query_budget": prefix.BUDGET,
        "label_mode": "soft",
        "victim_checkpoint_sha256": victim_sha256,
        "official_weight_sha256": official_weight_sha256,
        "posterior_sha256": posterior_sha256,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"Lab04 candidate 的 {field} 与当前实验不一致。")
    if tuple(payload.get("evaluation_seeds", ())) != EVALUATION_SEEDS:
        raise ValueError("Lab04 candidate 的十个 seed 与当前实验不一致。")

    all_rows = {
        int(row["seed"]): row
        for row in payload["results"]
        if row["case"] == SOURCE_ALL_CASE
    }
    blackbox_rows = {
        int(row["seed"]): row
        for row in payload["results"]
        if row["case"] == SOURCE_BLACKBOX_CASE
    }
    if tuple(sorted(all_rows)) != EVALUATION_SEEDS or tuple(sorted(blackbox_rows)) != EVALUATION_SEEDS:
        raise ValueError("Lab04 candidate 缺少 all-gamma 或 matched soft 黑盒十种子结果。")

    expected_states = set(selected_states(ALL_GAMMA, derive_gamma_groups(victim)))
    for seed in EVALUATION_SEEDS:
        row = all_rows[seed]
        actual_states = {
            unit["state_name"] for unit in row["protection"]["selected_units"]
        }
        if actual_states != expected_states:
            raise ValueError(f"Lab04 seed {seed} 的 all-gamma mask 与当前实验不一致。")
        if row["protection"]["protection_mask_sha256"] != expected_mask_sha256:
            raise ValueError(f"Lab04 seed {seed} 的 all-gamma mask 哈希不一致。")
        for candidate_row in (row, blackbox_rows[seed]):
            if candidate_row["query_partition_seed"] != seed:
                raise ValueError(f"Lab04 seed {seed} 的 query 划分元数据错误。")
            if candidate_row["result"].get("eval_passes") != 1:
                raise ValueError(f"Lab04 seed {seed} 不是单次 eval_ms 结果。")
        source_partition = payload["query_partitions"][str(seed)]
        current_partition = queries[seed].partition.to_metadata()
        for field in (
            "train_source_indices_sha256",
            "validation_source_indices_sha256",
        ):
            if source_partition[field] != current_partition[field]:
                raise ValueError(f"Lab04 seed {seed} 的 {field} 与当前划分不一致。")

    with SOURCE_HISTORY_PATH.open(encoding="utf-8", newline="") as source_file:
        source_history = [
            row
            for row in csv.DictReader(source_file, delimiter="\t")
            if row["case"] == SOURCE_ALL_CASE
        ]
    if len(source_history) != len(EVALUATION_SEEDS) * prefix.EPOCHS:
        raise ValueError("Lab04 all-gamma history 不完整。")
    transformed_history = [{**row, "case": ALL_GAMMA} for row in source_history]
    source = {
        "path": str(SOURCE_PATH.relative_to(ROOT)),
        "sha256": prefix.sha256_file(SOURCE_PATH),
        "history_path": str(SOURCE_HISTORY_PATH.relative_to(ROOT)),
        "history_sha256": prefix.sha256_file(SOURCE_HISTORY_PATH),
        "all_gamma_source_case": SOURCE_ALL_CASE,
        "matched_soft_blackbox_source_case": SOURCE_BLACKBOX_CASE,
    }
    return all_rows, blackbox_rows, transformed_history, source


def clone_reused_result(
    source_row: dict[str, object],
    mask_path: Path,
) -> dict[str, object]:
    row = json.loads(json.dumps(source_row))
    row["case"] = ALL_GAMMA
    row["label"] = CASE_LABELS[ALL_GAMMA]
    row["source_reuse"] = {
        "path": str(SOURCE_PATH.relative_to(ROOT)),
        "case": SOURCE_ALL_CASE,
    }
    row["protection"]["mask_path"] = str(mask_path.relative_to(ROOT))
    return row


def result_row(
    *,
    seed: int,
    case: str,
    plan,
    mask_path: Path,
    metadata: list[dict[str, object]],
    gamma_groups: dict[str, tuple[str, ...]],
    selection: dict[str, object],
    result: dict[str, object],
) -> dict[str, object]:
    protected_gamma = [row for row in metadata if row["role"] == "bn_gamma"]
    return {
        "seed": seed,
        "case": case,
        "label": CASE_LABELS[case],
        "query_partition_seed": seed,
        "randomization": {
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": seed,
            "query_sampler_seed": seed,
            "reset_before_surrogate_initialization": True,
        },
        "gamma": {
            "protected_groups": [
                group for group in GAMMA_GROUP_ORDER if group not in CASE_DROPPED_GROUP[case]
            ],
            "dropped_groups": list(CASE_DROPPED_GROUP[case]),
            "protected_state_count": len(protected_gamma),
            "protected_param_count": sum(int(row["numel"]) for row in protected_gamma),
            "group_states": {
                group: list(gamma_groups[group]) for group in GAMMA_GROUP_ORDER
            },
        },
        "protection": {
            "implementation_defense": "custom",
            "defense": "custom",
            "tensor_unit_count": plan.tensor_unit_count,
            "protected_unit_count": plan.protected_unit_count,
            "protection_mask_sha256": plan.protection_mask_sha256,
            "classifier_protected": plan.classifier_protected,
            "head_mode": plan.head_mode,
            "total_param_count": plan.total_param_count,
            "protected_param_count": plan.protected_param_count,
            "protected_param_ratio": plan.protected_param_ratio,
            "mask_path": str(mask_path.relative_to(ROOT)),
            "selected_units": metadata,
        },
        "primary": {
            "checkpoint": "best.pth",
            "epoch": selection["epoch"],
            "selection_metric": selection["metric"],
        },
        "selection": selection,
        "result": result,
    }


def normalize_reused_result(
    row: dict[str, object],
    gamma_groups: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    selected_units = row["protection"]["selected_units"]
    gamma_names = {name for names in gamma_groups.values() for name in names}
    row["gamma"] = {
        "protected_groups": list(GAMMA_GROUP_ORDER),
        "dropped_groups": [],
        "protected_state_count": 20,
        "protected_param_count": 4_800,
        "group_states": {group: list(gamma_groups[group]) for group in GAMMA_GROUP_ORDER},
    }
    for unit in selected_units:
        if unit["state_name"] in gamma_names:
            unit["gamma_group"] = next(
                group for group in GAMMA_GROUP_ORDER if unit["state_name"] in gamma_groups[group]
            )
        else:
            unit["gamma_group"] = None
    return row


def metric_summary(values: list[float]) -> dict[str, object]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "values_by_seed": {
            str(seed): value for seed, value in zip(EVALUATION_SEEDS, values)
        },
    }


def paired_effect(
    result_by_key: dict[tuple[int, str], dict[str, object]],
    left_case: str,
    right_case: str,
) -> dict[str, object]:
    values = {
        metric: [
            float(result_by_key[(seed, left_case)]["result"][metric])
            - float(result_by_key[(seed, right_case)]["result"][metric])
            for seed in EVALUATION_SEEDS
        ]
        for metric in METRICS
    }
    protection_harmed = {
        "surrogate_acc": sum(value > 0.0 for value in values["surrogate_acc"]),
        "fidelity": sum(value > 0.0 for value in values["fidelity"]),
        "posterior_kl": sum(value < 0.0 for value in values["posterior_kl"]),
    }
    protection_harmed["all_three"] = sum(
        values["surrogate_acc"][index] > 0.0
        and values["fidelity"][index] > 0.0
        and values["posterior_kl"][index] < 0.0
        for index in range(len(EVALUATION_SEEDS))
    )
    protection_improved = {
        "surrogate_acc": sum(value < 0.0 for value in values["surrogate_acc"]),
        "fidelity": sum(value < 0.0 for value in values["fidelity"]),
        "posterior_kl": sum(value > 0.0 for value in values["posterior_kl"]),
    }
    protection_improved["all_three"] = sum(
        values["surrogate_acc"][index] < 0.0
        and values["fidelity"][index] < 0.0
        and values["posterior_kl"][index] > 0.0
        for index in range(len(EVALUATION_SEEDS))
    )
    return {
        "left_case": left_case,
        "right_case": right_case,
        "definition": "left_minus_right",
        "metrics": {metric: metric_summary(metric_values) for metric, metric_values in values.items()},
        "left_harms_protection_counts": protection_harmed,
        "left_improves_protection_counts": protection_improved,
    }


def build_aggregate(
    result_by_key: dict[tuple[int, str], dict[str, object]],
    blackbox_by_seed: dict[int, dict[str, object]],
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
        for case in CASES
    }
    groups["matched_soft_blackbox"] = {
        metric: metric_summary(
            [float(blackbox_by_seed[seed]["result"][metric]) for seed in EVALUATION_SEEDS]
        )
        for metric in METRICS
    }
    blackbox_counts = {}
    for case in CASES:
        counts = {
            "surrogate_acc": 0,
            "fidelity": 0,
            "posterior_kl": 0,
            "all_three": 0,
        }
        for seed in EVALUATION_SEEDS:
            result = result_by_key[(seed, case)]["result"]
            blackbox = blackbox_by_seed[seed]["result"]
            conditions = {
                "surrogate_acc": result["surrogate_acc"] <= blackbox["surrogate_acc"],
                "fidelity": result["fidelity"] <= blackbox["fidelity"],
                "posterior_kl": result["posterior_kl"] >= blackbox["posterior_kl"],
            }
            conditions["all_three"] = all(conditions.values())
            for metric, condition in conditions.items():
                counts[metric] += int(condition)
        blackbox_counts[case] = counts
    return {
        "seed_count": len(EVALUATION_SEEDS),
        "sample_standard_deviation_ddof": 1,
        "groups": groups,
        "paired_effects": {
            "all_minus_no_gamma": paired_effect(result_by_key, ALL_GAMMA, NO_GAMMA),
            **{
                f"{case}_minus_all_gamma": paired_effect(result_by_key, case, ALL_GAMMA)
                for case in CASES
                if case.startswith("drop_")
            },
        },
        "at_or_beyond_matched_soft_blackbox_counts": blackbox_counts,
    }


def build_data_rows(
    result_by_key: dict[tuple[int, str], dict[str, object]],
    blackbox_by_seed: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for seed in EVALUATION_SEEDS:
        blackbox = blackbox_by_seed[seed]["result"]
        for case in CASES:
            row = result_by_key[(seed, case)]
            protection = row["protection"]
            gamma = row["gamma"]
            result = row["result"]
            rows.append(
                {
                    "seed": seed,
                    "case": case,
                    "label": CASE_LABELS[case],
                    "best_epoch": row["primary"]["epoch"],
                    "protected_unit_count": protection["protected_unit_count"],
                    "protected_param_count": protection["protected_param_count"],
                    "protected_param_ratio": protection["protected_param_ratio"],
                    "protected_gamma_count": gamma["protected_state_count"],
                    "protected_gamma_param_count": gamma["protected_param_count"],
                    "dropped_gamma_group": ",".join(gamma["dropped_groups"]),
                    "protection_mask_sha256": protection["protection_mask_sha256"],
                    "surrogate_acc": result["surrogate_acc"],
                    "fidelity": result["fidelity"],
                    "posterior_kl": result["posterior_kl"],
                    "matched_soft_blackbox_accuracy": blackbox["surrogate_acc"],
                    "matched_soft_blackbox_fidelity": blackbox["fidelity"],
                    "matched_soft_blackbox_posterior_kl": blackbox["posterior_kl"],
                }
            )
    return rows


def write_tsv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_result(
    aggregate: dict[str, object],
    result_by_key: dict[tuple[int, str], dict[str, object]],
    hard_blackbox: dict[str, object],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 240,
        }
    )
    specifications = (
        ("surrogate_acc", "MS accuracy"),
        ("fidelity", "Fidelity"),
        ("posterior_kl", "Posterior KL"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18.2, 5.5))
    x_values = list(range(len(CASES)))
    labels = [
        CASE_LABELS[case]
        .replace("All 20 gamma", "All 20\ngamma")
        .replace("All - ", "Drop\n")
        for case in CASES
    ]
    for axis, (metric, title) in zip(axes, specifications):
        means = [float(aggregate["groups"][case][metric]["mean"]) for case in CASES]
        errors = [float(aggregate["groups"][case][metric]["sample_std"]) for case in CASES]
        axis.bar(
            x_values,
            means,
            yerr=errors,
            capsize=4,
            width=0.68,
            color=[CASE_COLORS[case] for case in CASES],
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        for x_value, case in zip(x_values, CASES):
            values = [
                float(result_by_key[(seed, case)]["result"][metric])
                for seed in EVALUATION_SEEDS
            ]
            offsets = [(-0.18 + index * 0.04) for index in range(len(values))]
            axis.scatter(
                [x_value + offset for offset in offsets],
                values,
                s=13,
                color="#222222",
                alpha=0.52,
                linewidths=0,
                zorder=3,
            )
        soft = aggregate["groups"]["matched_soft_blackbox"][metric]
        soft_mean = float(soft["mean"])
        soft_std = float(soft["sample_std"])
        axis.axhspan(
            soft_mean - soft_std,
            soft_mean + soft_std,
            color="#999999",
            alpha=0.14,
            zorder=0,
        )
        axis.axhline(
            soft_mean,
            color="#555555",
            linestyle="--",
            linewidth=1.5,
            label="Matched soft BB",
            zorder=1,
        )
        hard_value = float(hard_blackbox["result"][metric])
        axis.axhline(
            hard_value,
            color="#AA3377",
            linestyle=(0, (3, 2)),
            linewidth=1.35,
            label="Hard-label BB (seed 42)",
            zorder=1,
        )
        all_values = means + [soft_mean - soft_std, soft_mean + soft_std, hard_value]
        padding = max((max(all_values) - min(all_values)) * 0.10, 0.008 if metric != "posterior_kl" else 0.04)
        lower = min(all_values) - padding
        upper = max(all_values) + padding
        axis.set_ylim(max(0.0, lower), upper)
        axis.set_xticks(x_values, labels, rotation=0)
        axis.set_title(title)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    figure.suptitle("BN gamma group ablation on the fixed five-conv protection set")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(PLOT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def load_resume_rows(
    *,
    expected_hashes: dict[str, str],
    victim_sha256: str,
    posterior_sha256: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not RESULT_PATH.is_file() or not HISTORY_PATH.is_file():
        return [], []
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    if (
        payload.get("attack_protocol") != prefix.ATTACK_PROTOCOL_VERSION
        or payload.get("victim_checkpoint_sha256") != victim_sha256
        or payload.get("posterior_sha256") != posterior_sha256
        or tuple(payload.get("evaluation_seeds", ())) != EVALUATION_SEEDS
    ):
        raise ValueError("现有 gamma.json 不能在当前协议下 resume。")
    retained = []
    for row in payload.get("results", []):
        case = str(row["case"])
        seed = int(row["seed"])
        if case in TRAINED_CASES and seed in EVALUATION_SEEDS:
            if row["protection"]["protection_mask_sha256"] != expected_hashes[case]:
                raise ValueError(f"现有 {seed}/{case} mask 与当前定义不一致。")
            retained.append(row)
    completed = {(int(row["seed"]), str(row["case"])) for row in retained}
    with HISTORY_PATH.open(encoding="utf-8", newline="") as history_file:
        history = [
            row
            for row in csv.DictReader(history_file, delimiter="\t")
            if (int(row["seed"]), str(row["case"])) in completed
        ]
    grouped = {}
    for row in history:
        grouped.setdefault((int(row["seed"]), str(row["case"])), []).append(row)
    if any(len(grouped.get(key, ())) != prefix.EPOCHS for key in completed):
        raise ValueError("现有 gamma history 不是每个已完成 case 恰好 100 epoch。")
    return retained, history


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    device = prefix.resolve_device(args.device)
    dataset_root = ROOT / "dataset" / "public"
    protocol_root = ROOT / "dataset" / "MS"
    victim_checkpoint = ROOT / "weights" / "MS" / "victim" / prefix.MODEL / prefix.DATASET / "best.pth"
    official_weight = ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    prefix.configure_reproducibility(42, deterministic=True)
    victim, victim_metadata = prefix.build_victim(prefix.MODEL, prefix.NUM_CLASSES, victim_checkpoint)
    victim_sha256 = prefix.sha256_file(victim_checkpoint)
    official_weight_sha256 = prefix.sha256_file(official_weight)
    gamma_groups = derive_gamma_groups(victim)
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
    posterior_sha256 = next(iter(posterior_hashes))
    for seed, query in queries.items():
        expected_victim_sha256 = query.manifest.get("victim", {}).get("checkpoint_sha256")
        if expected_victim_sha256 and expected_victim_sha256 != victim_sha256:
            raise ValueError(f"seed {seed} 的 posterior 与 victim best.pth 不一致。")

    templates = {}
    expected_hashes = {}
    for case in CASES:
        prefix.configure_reproducibility(EVALUATION_SEEDS[0], deterministic=True)
        surrogate, plan, masks, metadata = initialize_case(
            case,
            victim,
            official_weight,
            gamma_groups,
            EVALUATION_SEEDS[0],
        )
        templates[case] = (plan, masks, metadata)
        expected_hashes[case] = plan.protection_mask_sha256
        del surrogate
        print(
            f"[MASK/{case}] units={plan.protected_unit_count}/122 "
            f"params={plan.protected_param_count}/{plan.total_param_count} "
            f"ratio={plan.protected_param_ratio:.6f} "
            f"gamma={sum(row['role'] == 'bn_gamma' for row in metadata)} "
            f"sha256={plan.protection_mask_sha256}",
            flush=True,
        )

    all_source, blackbox_by_seed, source_history, source_metadata = load_source(
        victim,
        victim_sha256,
        official_weight_sha256,
        posterior_sha256,
        expected_hashes[ALL_GAMMA],
        queries,
    )
    hard_blackbox_path = (
        ROOT / "results" / "MS" / prefix.MODEL / prefix.DATASET / "hard_blackbox" / "metrics.json"
    )
    hard_blackbox = prefix.load_formal_bound(
        hard_blackbox_path,
        "hard_blackbox",
        label_mode="hard",
        model=prefix.MODEL,
        dataset=prefix.DATASET,
        budget=prefix.BUDGET,
    )
    print(
        f"[SOURCE] 复用 Lab04 {len(all_source)} 个 all-gamma 与 "
        f"{len(blackbox_by_seed)} 个 matched soft 黑盒结果。",
        flush=True,
    )
    if args.dry_run:
        print("[INFO] dry-run 完成，未写入 Lab05 gamma 产物。")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_paths = {case: OUT_DIR / f"gamma_{case}_mask.pt" for case in CASES}
    for case in CASES:
        prefix.save_protection_mask(mask_paths[case], templates[case][1])

    retained_results = []
    retained_history = []
    if args.resume:
        retained_results, retained_history = load_resume_rows(
            expected_hashes=expected_hashes,
            victim_sha256=victim_sha256,
            posterior_sha256=posterior_sha256,
        )
        print(
            f"[RESUME] 复用 {len(retained_results)} 个训练 case、"
            f"{len(retained_history)} 条 history。",
            flush=True,
        )

    results = list(retained_results)
    history_rows = list(retained_history)
    completed = {(int(row["seed"]), str(row["case"])) for row in results}
    evaluation = None
    for seed in EVALUATION_SEEDS:
        query = queries[seed]
        for case in TRAINED_CASES:
            if (seed, case) in completed:
                print(f"[RESUME] 跳过 {seed}/{case}。", flush=True)
                continue
            prefix.configure_reproducibility(seed, deterministic=True)
            surrogate, plan, _, metadata = initialize_case(
                case,
                victim,
                official_weight,
                gamma_groups,
                seed,
            )
            if plan.protection_mask_sha256 != expected_hashes[case]:
                raise RuntimeError(f"seed {seed} 的 {case} mask 漂移。")
            surrogate = surrogate.to(device)
            selection, history = prefix.train_validation_best(
                surrogate,
                query,
                device=device,
                num_workers=args.num_workers,
                seed=seed,
            )
            history_rows.extend({"seed": seed, "case": case, **row} for row in history)
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
            result = prefix.evaluate_once(surrogate, evaluation, device)
            results.append(
                result_row(
                    seed=seed,
                    case=case,
                    plan=plan,
                    mask_path=mask_paths[case],
                    metadata=metadata,
                    gamma_groups=gamma_groups,
                    selection=selection,
                    result=result,
                )
            )
            print(
                f"[RESULT/seed={seed}/{case}] epoch={selection['epoch']} "
                f"accuracy={result['surrogate_acc']:.6f} "
                f"fidelity={result['fidelity']:.6f} "
                f"posterior_kl={result['posterior_kl']:.6f}",
                flush=True,
            )
            del surrogate
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for seed in EVALUATION_SEEDS:
        reused = clone_reused_result(all_source[seed], mask_paths[ALL_GAMMA])
        results.append(normalize_reused_result(reused, gamma_groups))
    history_rows.extend(source_history)

    expected_keys = {(seed, case) for seed in EVALUATION_SEEDS for case in CASES}
    result_by_key = {(int(row["seed"]), str(row["case"])): row for row in results}
    if set(result_by_key) != expected_keys or len(results) != len(expected_keys):
        raise RuntimeError("六种 gamma case 的十种子结果不完整或重复。")
    results = [result_by_key[(seed, case)] for seed in EVALUATION_SEEDS for case in CASES]
    history_rows.sort(
        key=lambda row: (int(row["seed"]), CASES.index(str(row["case"])), int(row["epoch"]))
    )
    if len(history_rows) != len(EVALUATION_SEEDS) * len(CASES) * prefix.EPOCHS:
        raise RuntimeError("gamma history 行数错误。")
    aggregate = build_aggregate(result_by_key, blackbox_by_seed)
    data_rows = build_data_rows(result_by_key, blackbox_by_seed)
    write_tsv(HISTORY_PATH, history_rows, HISTORY_FIELDS)
    write_tsv(DATA_PATH, data_rows, DATA_FIELDS)
    plot_result(aggregate, result_by_key, hard_blackbox)

    payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "protocol": "MS_gamma_group_ablation",
        "attack_protocol": prefix.ATTACK_PROTOCOL_VERSION,
        "dataset": prefix.DATASET,
        "victim_model": prefix.MODEL,
        "query_budget": prefix.BUDGET,
        "label_mode": "soft",
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "randomization": {
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seeds": list(EVALUATION_SEEDS),
            "query_sampler_seeds": list(EVALUATION_SEEDS),
            "reset_before_each_surrogate_initialization": True,
        },
        "query_partitions": {
            str(seed): queries[seed].partition.to_metadata() for seed in EVALUATION_SEEDS
        },
        "base_protection_states": list(BASE_STATES),
        "gamma_group_order": list(GAMMA_GROUP_ORDER),
        "gamma_groups": {group: list(gamma_groups[group]) for group in GAMMA_GROUP_ORDER},
        "cases": {
            case: {
                "label": CASE_LABELS[case],
                "dropped_gamma_groups": list(CASE_DROPPED_GROUP[case]),
                "protection_mask_sha256": expected_hashes[case],
                "mask_path": str(mask_paths[case].relative_to(ROOT)),
            }
            for case in CASES
        },
        "training": {
            "max_epochs": prefix.EPOCHS,
            "batch_size": prefix.BATCH_SIZE,
            "eval_batch_size": prefix.EVAL_BATCH_SIZE,
            "optimizer": "SGD",
            "learning_rate": prefix.LEARNING_RATE,
            "momentum": prefix.MOMENTUM,
            "weight_decay": prefix.WEIGHT_DECAY,
            "lr_scheduler": "StepLR",
            "lr_step": prefix.LR_STEP,
            "lr_gamma": prefix.LR_GAMMA,
            "checkpoint": "best.pth",
            "checkpoint_selection": "minimum query-validation soft cross-entropy",
            "checkpoint_tie_break": "earliest epoch",
            "eval_ms_passes_per_case": 1,
        },
        "victim_checkpoint": str(victim_checkpoint.relative_to(ROOT)),
        "victim_checkpoint_sha256": victim_sha256,
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "official_weight": str(official_weight.relative_to(ROOT)),
        "official_weight_sha256": official_weight_sha256,
        "posterior_path": str(queries[EVALUATION_SEEDS[0]].target_path.relative_to(ROOT)),
        "posterior_sha256": posterior_sha256,
        "source_reuse": source_metadata,
        "hard_blackbox": {
            "path": str(hard_blackbox_path.relative_to(ROOT)),
            "sha256": prefix.sha256_file(hard_blackbox_path),
            "seed": 42,
            "result": hard_blackbox["result"],
        },
        "matched_soft_blackbox": {
            "source": source_metadata,
            "results_by_seed": {str(seed): blackbox_by_seed[seed]["result"] for seed in EVALUATION_SEEDS},
        },
        "results": results,
        "aggregate": aggregate,
        "outputs": {
            "data": str(DATA_PATH.relative_to(ROOT)),
            "history": str(HISTORY_PATH.relative_to(ROOT)),
            "plot": str(PLOT_PATH.relative_to(ROOT)),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[DONE] {RESULT_PATH.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
