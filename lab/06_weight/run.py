#!/usr/bin/env python3
"""验证 TensorShield Top-10 至 Top-17 的遗漏 weight 语义闭包。"""

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
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = ROOT / "exp" / "MS" / "train_surrogate"
TRAIN_VICTIM_ROOT = ROOT / "exp" / "MS" / "train_victim"
for import_root in (ROOT, TRAIN_ROOT, TRAIN_VICTIM_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common.trainer import (  # noqa: E402
    build_generator,
    configure_reproducibility,
    seed_worker,
)
from exp.MS.train_surrogate.core.artifacts import sha256_file  # noqa: E402
from exp.MS.train_surrogate.core.config import (  # noqa: E402
    ATTACK_PROTOCOL_VERSION,
    resolve_device,
)
from exp.MS.train_surrogate.core.data import (  # noqa: E402
    build_eval_dataset,
    build_query_dataset,
    build_victim,
    load_query_targets,
)
from exp.MS.train_surrogate.core.engine import (  # noqa: E402
    collect_eval_reference,
    evaluate_surrogate,
    train_one_epoch,
)
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
from models import imagenet as imagenet_models  # noqa: E402


EXPERIMENT = "06_weight"
MODEL = "resnet18"
DATASET = "c100"
NUM_CLASSES = 100
BUDGET = 500
TOP_K_VALUES = tuple(range(10, 18))
EPOCHS = 100
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.5
WEIGHT_DECAY = 5e-4
LR_STEP = 60
LR_GAMMA = 0.1
SEED = 42
LAB04_METRICS_SHA256 = "c05c7d158ff243878200ad40414c153630355d51a5de010e4ac12b570585c9a7"
LAB05_METRICS_SHA256 = "b96e6fcbf6bf7157ef80fe440a2b2789441c5cefdfe47b793f5505a6ce16b1c4"
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
    "top_k": {
        "label": "Top-k",
        "color": "#555555",
        "marker": "o",
    },
    "bn_gamma": {
        "label": "+ BN gamma",
        "color": "#009E73",
        "marker": "s",
    },
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
    "stem_conv": {
        "label": "+ Stem Conv",
        "color": "#CC79A7",
        "marker": "D",
    },
    "all_extras": {
        "label": "+ All extras",
        "color": "#0072B2",
        "marker": "P",
    },
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
        help="只验证输入哈希、48 个 mask 与保护统计，不训练或写结果。",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="严格核对并复用已有完整组合，只训练缺失组合。",
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
    stem_conv = ("conv1.weight",)
    if downsample_conv != EXPECTED_DOWNSAMPLE_CONV:
        raise ValueError(f"downsample Conv 定义已变化：{downsample_conv}")
    groups = {
        "bn_gamma": bn_gamma,
        "downsample_conv": downsample_conv,
        "bn_gamma_downsample": unique_names((*bn_gamma, *downsample_conv)),
        "stem_conv": stem_conv,
        "all_extras": unique_names((*bn_gamma, *downsample_conv, *stem_conv)),
    }
    for group_name, names in groups.items():
        missing = set(names) - set(state)
        if missing:
            raise ValueError(f"{group_name} 包含未知 state：{sorted(missing)}")
        count = sum(state[name].numel() for name in names)
        expected = EXPECTED_EXTRA_COUNTS[group_name]
        if (len(names), count) != expected:
            raise ValueError(
                f"{group_name} 统计为 {(len(names), count)}，期望 {expected}。"
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
            selected = unique_names((*base_names, *extras))
            cases.append(CaseSpec(top_k, variant, selected, tuple(extras)))
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
    unit_spec = ",".join(str(unit.index) for unit in selected_units)
    surrogate, plan, _, masks = initialize_surrogate(
        factory=imagenet_models.resnet18,
        factory_name=MODEL,
        weight_path=official_weight,
        victim_model=victim,
        num_classes=NUM_CLASSES,
        defense="custom",
        protected_units=unit_spec,
        protected_layers=None,
        protected_scalars=None,
    )
    if not plan.classifier_protected or plan.head_mode != "replace":
        raise RuntimeError(f"{case.name} 必须完整保护分类头，实际为 {plan.head_mode}。")
    if plan.protected_unit_count != len(case.selected_state_names):
        raise RuntimeError(f"{case.name} 保护 unit 数量与定义不一致。")
    state = victim.state_dict()
    expected_params = sum(state[name].numel() for name in case.selected_state_names)
    if plan.protected_param_count != expected_params:
        raise RuntimeError(
            f"{case.name} 保护参数为 {plan.protected_param_count}，期望 {expected_params}。"
        )
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
            rank = ranking.index(unit.state_name) + 1
        elif unit.state_name == "last_linear.bias":
            role = "fixed_head_bias"
            rank = None
        elif unit.state_name in extra_set:
            role = "semantic_extra"
            rank = None
        else:
            raise RuntimeError(f"无法解释 {case.name} 的 state：{unit.state_name}")
        metadata.append(
            {
                "index": unit.index,
                "state_name": unit.state_name,
                "state_kind": unit.state_kind,
                "numel": unit.numel,
                "role": role,
                "eligible_rank": rank,
            }
        )
    return surrogate, plan, masks, metadata


def load_lab04(path: Path) -> tuple[dict[str, object], dict[int, dict[str, object]]]:
    if sha256_file(path) != LAB04_METRICS_SHA256:
        raise ValueError("Lab04 metrics.json 已变化，必须先重新固化 Lab06 协议。")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 2,
        "experiment": "04_tensorshield",
        "protocol": "MS",
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "dataset": DATASET,
        "victim_model": MODEL,
        "query_budget": BUDGET,
        "label_mode": "soft",
        "query_transform": "test",
        "seed": SEED,
        "primary": {"evaluation": "end", "epoch": EPOCHS},
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"Lab04 {field}={payload.get(field)!r}，期望 {value!r}。")
    if payload.get("randomization", {}).get(
        "reset_before_each_surrogate_initialization"
    ) is not True:
        raise ValueError("Lab04 没有固定每个 Top-k 的初始化 RNG。")
    ranking = tuple(payload.get("source", {}).get("eligible_rank", ()))
    if ranking != tuple(AUTHOR_RESNET18_C100_ELIGIBLE_RANK):
        raise ValueError("Lab04 eligible rank 与当前作者固定列表不一致。")
    by_k = {int(row["top_k"]): row for row in payload.get("results", [])}
    if set(by_k) != set(range(1, 18)):
        raise ValueError("Lab04 未完整包含 Top-1 至 Top-17。")
    for top_k, row in by_k.items():
        if row.get("primary") != {"evaluation": "end", "epoch": EPOCHS}:
            raise ValueError(f"Lab04 Top-{top_k} 不是固定 end 主结果。")
        protection = row.get("protection", {})
        mask_path = ROOT / str(protection.get("mask_path", ""))
        if not mask_path.is_file():
            raise FileNotFoundError(f"Lab04 Top-{top_k} 缺少 mask：{mask_path}")
        digest = protection_mask_sha256(load_protection_mask(mask_path))
        if digest != protection.get("protection_mask_sha256"):
            raise ValueError(f"Lab04 Top-{top_k} mask 与 metrics.json 不一致。")
    return payload, by_k


def load_lab05_weight(path: Path) -> dict[str, object]:
    if sha256_file(path) != LAB05_METRICS_SHA256:
        raise ValueError("Lab05 metrics.json 已变化，必须先重新固化 Lab06 协议。")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 2,
        "experiment": "05_state",
        "protocol": "MS",
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "dataset": DATASET,
        "victim_model": MODEL,
        "query_budget": BUDGET,
        "label_mode": "soft",
        "query_transform": "test",
        "seed": SEED,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"Lab05 {field}={payload.get(field)!r}，期望 {value!r}。")
    matches = [
        row for row in payload.get("results", []) if row.get("protection_group") == "weight"
    ]
    if len(matches) != 1:
        raise ValueError("Lab05 必须恰好包含一个 weight 参考结果。")
    result = matches[0]
    protection = result.get("protection", {})
    if protection.get("protected_param_count") != 11_222_912:
        raise ValueError("Lab05 weight 保护参数量已变化。")
    if result.get("primary") != {"evaluation": "end", "epoch": EPOCHS}:
        raise ValueError("Lab05 weight 不是固定 end 主结果。")
    return result


def load_bound(path: Path, artifact_id: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "artifact_id": artifact_id,
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "dataset": DATASET,
        "victim_model": MODEL,
        "query_budget": BUDGET,
        "label_mode": "soft",
        "query_transform": "test",
        "lr_step": LR_STEP,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"参考结果 {path} 的 {field} 不符合当前协议。")
    if payload.get("primary", {}).get("checkpoint") != "end.pth":
        raise ValueError(f"参考结果 {path} 未使用 end.pth。")
    return {
        "artifact_id": artifact_id,
        "run_id": payload["run_id"],
        "protection": payload["protection"],
        "end": payload["end"],
    }


def reused_prefix_result(case: CaseSpec, source: dict[str, object]) -> dict[str, object]:
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
            "metrics_sha256": LAB04_METRICS_SHA256,
        },
        "protection": source["protection"],
        "primary": source["primary"],
        "end": source["end"],
    }


def add_effects(results: list[dict[str, object]]) -> None:
    baselines = {
        int(result["top_k"]): result["end"]
        for result in results
        if result["variant"] == "top_k"
    }
    for result in results:
        baseline = baselines[int(result["top_k"])]
        end = result["end"]
        result["effect_vs_top_k"] = {
            "accuracy_change": end["surrogate_acc"] - baseline["surrogate_acc"],
            "fidelity_change": end["fidelity"] - baseline["fidelity"],
            "posterior_kl_change": end["posterior_kl"] - baseline["posterior_kl"],
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


def read_history(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少可复用历史：{path}")
    with path.open("r", newline="", encoding="utf-8") as reader_file:
        return list(csv.DictReader(reader_file, delimiter="\t"))


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
            end = result["end"]
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
                    "surrogate_acc": end["surrogate_acc"],
                    "fidelity": end["fidelity"],
                    "posterior_kl": end["posterior_kl"],
                    "accuracy_change_from_top_k": effect["accuracy_change"],
                    "fidelity_change_from_top_k": effect["fidelity_change"],
                    "posterior_kl_change_from_top_k": effect["posterior_kl_change"],
                }
            )


def set_y_limits(axis: plt.Axes, values: list[float], bounded: bool) -> None:
    minimum = min(values)
    maximum = max(values)
    padding = max((maximum - minimum) * 0.09, 0.02 if bounded else 0.05)
    upper = maximum + padding
    if bounded:
        upper = min(1.0, upper)
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
        plotted_values = []
        for variant, style in VARIANTS.items():
            rows = sorted(
                (row for row in results if row["variant"] == variant),
                key=lambda row: int(row["top_k"]),
            )
            x_values = [int(row["top_k"]) for row in rows]
            y_values = [float(row["end"][metric]) for row in rows]
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
        full_value = float(references["full_protection"]["end"][metric])
        plotted_values.append(full_value)
        axis.axhline(
            full_value,
            label="Soft black-box (full protection)",
            color="#888888",
            linestyle=":",
            linewidth=1.5,
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
        ncol=7,
        frameon=False,
    )
    figure.suptitle(
        "TensorShield Top-k weight-semantic closure on ResNet18 + CIFAR-100",
        y=1.13,
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def validate_reusable_result(
    case: CaseSpec,
    result: dict[str, object],
    expected_protection: dict[str, object],
    history: list[dict[str, str]],
    mask_path: Path,
) -> None:
    if result.get("top_k") != case.top_k or result.get("variant") != case.variant:
        raise ValueError(f"{case.name} 已有 case 定义与当前协议不一致。")
    if result.get("selected_state_names") != list(case.selected_state_names):
        raise ValueError(f"{case.name} 已有保护 state 集合与当前协议不一致。")
    protection = result.get("protection", {})
    for field in (
        "protected_unit_count",
        "protected_param_count",
        "protection_mask_sha256",
        "classifier_protected",
        "head_mode",
    ):
        if protection.get(field) != expected_protection[field]:
            raise ValueError(f"{case.name} 已有 {field} 与当前 mask 不一致。")
    if result.get("primary") != {"evaluation": "end", "epoch": EPOCHS}:
        raise ValueError(f"{case.name} 已有结果不是固定第 100 轮 end。")
    if not mask_path.is_file():
        raise FileNotFoundError(f"{case.name} 缺少可复用 mask：{mask_path}")
    digest = protection_mask_sha256(load_protection_mask(mask_path))
    if digest != expected_protection["protection_mask_sha256"]:
        raise ValueError(f"{case.name} 已有 mask 内容与当前定义不一致。")
    rows = [row for row in history if row.get("case") == case.name]
    if [int(row["epoch"]) for row in rows] != list(range(1, EPOCHS + 1)):
        raise ValueError(f"{case.name} 已有历史不是完整的 1-{EPOCHS} 轮。")
    if any(
        int(row["top_k"]) != case.top_k or row["variant"] != case.variant
        for row in rows
    ):
        raise ValueError(f"{case.name} 已有历史的组合标签不一致。")
    end = result.get("end", {})
    if end.get("eval_count") != 10_000 or end.get("victim_correct") != 6_182:
        raise ValueError(f"{case.name} 已有 end 不是当前完整 eval_ms。")


def validate_reuse_payload(
    payload: dict[str, object],
    training_protocol: dict[str, object],
    ranking: tuple[str, ...],
    victim_sha256: str,
    official_weight_sha256: str,
    posterior_sha256: str,
) -> None:
    """核对不随新增受控曲线变化的协议字段，具体 case 另逐项核对。"""
    expected = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "protocol": "MS",
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "dataset": DATASET,
        "victim_model": MODEL,
        "query_budget": BUDGET,
        "label_mode": "soft",
        "query_transform": "test",
        "seed": SEED,
        "victim_checkpoint_sha256": victim_sha256,
        "official_weight_sha256": official_weight_sha256,
        "posterior_sha256": posterior_sha256,
        "training": training_protocol,
        "primary": {"evaluation": "end", "epoch": EPOCHS},
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"已有 Lab06 {field} 与当前协议不一致，拒绝复用。")
    study = payload.get("study", {})
    if study.get("top_k_values") != list(TOP_K_VALUES):
        raise ValueError("已有 Lab06 Top-k 范围与当前协议不一致，拒绝复用。")
    source = payload.get("source", {})
    source_expected = {
        "method": "TensorShield",
        "rank_provenance": "author_confirmed_final_rank",
        "eligible_rank": list(ranking),
        "lab04_metrics_sha256": LAB04_METRICS_SHA256,
        "lab05_metrics_sha256": LAB05_METRICS_SHA256,
    }
    for field, value in source_expected.items():
        if source.get(field) != value:
            raise ValueError(f"已有 Lab06 source.{field} 与当前协议不一致，拒绝复用。")


def clean_outputs(out_dir: Path) -> None:
    for filename in (
        "metrics.json",
        "history.tsv",
        "data.tsv",
        "metrics.png",
    ):
        (out_dir / filename).unlink(missing_ok=True)
    for path in out_dir.glob("top_*_mask.pt"):
        path.unlink()


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    if args.dry_run and args.reuse_existing:
        raise ValueError("--dry-run 与 --reuse-existing 不能同时使用。")
    device = resolve_device(args.device)
    dataset_root = ROOT / "dataset" / "public"
    protocol_root = ROOT / "dataset" / "MS"
    victim_checkpoint = ROOT / "weights" / "MS" / "victim" / MODEL / DATASET / "best.pth"
    official_weight = ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    lab04_path = ROOT / "results" / "lab" / "04_tensorshield" / "metrics.json"
    lab05_path = ROOT / "results" / "lab" / "05_state" / "metrics.json"
    out_dir = ROOT / "results" / "lab" / EXPERIMENT

    lab04_payload, lab04_by_k = load_lab04(lab04_path)
    lab05_weight = load_lab05_weight(lab05_path)
    ranking = tuple(AUTHOR_RESNET18_C100_ELIGIBLE_RANK)
    configure_reproducibility(SEED, deterministic=True)
    query_indices, query_posteriors, query_labels, posterior_path, query_manifest = (
        load_query_targets(protocol_root, DATASET, MODEL, BUDGET, "soft")
    )
    victim, victim_metadata = build_victim(MODEL, NUM_CLASSES, victim_checkpoint)
    victim_sha256 = sha256_file(victim_checkpoint)
    expected_victim_sha256 = query_manifest.get("victim", {}).get("checkpoint_sha256")
    if expected_victim_sha256 and expected_victim_sha256 != victim_sha256:
        raise ValueError("victim best.pth 与生成 soft posterior 时使用的 checkpoint 不一致。")
    if lab04_payload.get("victim_checkpoint_sha256") != victim_sha256:
        raise ValueError("Lab04 与当前 victim best.pth 不一致。")
    if lab04_payload.get("official_weight_sha256") != sha256_file(official_weight):
        raise ValueError("Lab04 与当前 ImageNet 预训练权重不一致。")
    if lab04_payload.get("posterior_sha256") != sha256_file(posterior_path):
        raise ValueError("Lab04 与当前 soft posterior 不一致。")

    extra_groups = derive_extra_states(victim)
    cases = build_cases(ranking, extra_groups)
    references_root = ROOT / "results" / "MS" / MODEL / DATASET
    references = {
        "no_protection": load_bound(
            references_root / "no_protection" / "metrics.json", "no_protection"
        ),
        "full_protection": load_bound(
            references_root / "full_protection" / "metrics.json", "full_protection"
        ),
    }

    expected_by_case: dict[str, dict[str, object]] = {}
    for case in cases:
        configure_reproducibility(SEED, deterministic=True)
        surrogate, plan, _, selected_units = initialize_case(case, victim, official_weight)
        expected_by_case[case.name] = {
            "protection": {
                "implementation_defense": "custom",
                **plan.to_metadata(),
                "selected_units": selected_units,
                "mask_path": str(
                    (
                        ROOT
                        / "results"
                        / "lab"
                        / EXPERIMENT
                        / f"{case.name}_mask.pt"
                    ).relative_to(ROOT)
                ),
            }
        }
        if case.variant == "top_k":
            source_protection = lab04_by_k[case.top_k]["protection"]
            if (
                plan.protection_mask_sha256
                != source_protection["protection_mask_sha256"]
                or plan.protected_param_count
                != source_protection["protected_param_count"]
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

    all_extras_top17 = expected_by_case["top_17_all_extras"]["protection"]
    expected_endpoint = lab05_weight["protection"]["protected_param_count"] + 100
    if all_extras_top17["protected_param_count"] != expected_endpoint:
        raise ValueError("Top-17 + all_extras 没有闭合为全部 weight 加分类头 bias。")

    training_protocol = {
        "mode": "finetune",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "optimizer": "SGD",
        "learning_rate": LEARNING_RATE,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "lr_scheduler": "StepLR",
        "lr_step": LR_STEP,
        "lr_gamma": LR_GAMMA,
        "evaluation_schedule": "end_only",
    }
    protocol_config = {
        "experiment": EXPERIMENT,
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "dataset": DATASET,
        "victim_model": MODEL,
        "query_budget": BUDGET,
        "label_mode": "soft",
        "query_transform": "test",
        "seed": SEED,
        "top_k_values": list(TOP_K_VALUES),
        "variants": list(VARIANTS),
        "eligible_rank": list(ranking),
        "extra_groups": {name: list(values) for name, values in extra_groups.items()},
        "victim_checkpoint_sha256": victim_sha256,
        "official_weight_sha256": sha256_file(official_weight),
        "posterior_sha256": sha256_file(posterior_path),
        "lab04_metrics_sha256": LAB04_METRICS_SHA256,
        "lab05_metrics_sha256": LAB05_METRICS_SHA256,
        "training": training_protocol,
    }
    protocol_sha256 = canonical_sha256(protocol_config)
    print(f"[INFO] protocol SHA256：{protocol_sha256}")
    print(
        "[INFO] Top-17 + all_extras 参数："
        f"{all_extras_top17['protected_param_count']}/"
        f"{all_extras_top17['total_param_count']}"
    )
    if args.dry_run:
        print("[INFO] dry-run 完成，未写入 Lab06 结果。")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    existing_results: dict[str, dict[str, object]] = {}
    existing_history: list[dict[str, str]] = []
    metrics_path = out_dir / "metrics.json"
    history_path = out_dir / "history.tsv"
    data_path = out_dir / "data.tsv"
    plot_path = out_dir / "metrics.png"
    if args.reuse_existing:
        if not metrics_path.is_file():
            raise FileNotFoundError(f"--reuse-existing 要求已有结果：{metrics_path}")
        existing_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        validate_reuse_payload(
            existing_payload,
            training_protocol,
            ranking,
            victim_sha256,
            sha256_file(official_weight),
            sha256_file(posterior_path),
        )
        existing_history = read_history(history_path)
        known_cases = {case.name for case in cases}
        for result in existing_payload.get("results", []):
            case_name = result.get("case")
            if case_name not in known_cases:
                raise ValueError(f"已有 Lab06 包含未知 case：{case_name}")
            if case_name in existing_results:
                raise ValueError(f"已有 Lab06 case 重复：{case_name}")
            existing_results[str(case_name)] = result
    else:
        clean_outputs(out_dir)

    results_by_case: dict[str, dict[str, object]] = {}
    history_rows: list[dict[str, object]] = []
    for case in cases:
        if case.variant == "top_k":
            results_by_case[case.name] = reused_prefix_result(
                case, lab04_by_k[case.top_k]
            )
            continue
        if case.name not in existing_results:
            continue
        expected = expected_by_case[case.name]["protection"]
        mask_path = out_dir / f"{case.name}_mask.pt"
        validate_reusable_result(
            case,
            existing_results[case.name],
            expected,
            existing_history,
            mask_path,
        )
        results_by_case[case.name] = existing_results[case.name]
        history_rows.extend(
            row for row in existing_history if row.get("case") == case.name
        )
        print(f"[REUSE/{case.name}] 已核对 100 轮历史、mask 和 end 指标。")

    query_dataset = build_query_dataset(
        DATASET,
        dataset_root,
        query_indices,
        query_posteriors,
        query_labels,
        input_transform="test",
    )
    eval_dataset = build_eval_dataset(DATASET, dataset_root, protocol_root, None)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=build_generator(SEED, offset=1),
    )
    victim = victim.to(device)
    eval_reference = collect_eval_reference(victim, eval_loader, device)
    victim = victim.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    case_order = {case.name: index for index, case in enumerate(cases)}

    def persist(complete: bool) -> None:
        ordered_results = [
            results_by_case[case.name]
            for case in cases
            if case.name in results_by_case
        ]
        add_effects(ordered_results)
        ordered_history = sorted(
            history_rows,
            key=lambda row: (case_order[str(row["case"])], int(row["epoch"])),
        )
        missing_cases = [
            case.name for case in cases if case.name not in results_by_case
        ]
        payload = {
            "schema_version": 1,
            "experiment": EXPERIMENT,
            "protocol": "MS",
            "attack_protocol": ATTACK_PROTOCOL_VERSION,
            "protocol_sha256": protocol_sha256,
            "complete": complete,
            "dataset": DATASET,
            "victim_model": MODEL,
            "query_budget": BUDGET,
            "label_mode": "soft",
            "query_transform": "test",
            "seed": SEED,
            "randomization": {
                "reset_before_each_surrogate_initialization": True,
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
                "lab04_metrics_sha256": LAB04_METRICS_SHA256,
                "lab05_metrics": str(lab05_path.relative_to(ROOT)),
                "lab05_metrics_sha256": LAB05_METRICS_SHA256,
            },
            "victim_checkpoint": str(victim_checkpoint.relative_to(ROOT)),
            "victim_checkpoint_sha256": victim_sha256,
            "victim_checkpoint_epoch": victim_metadata.get("epoch"),
            "official_weight": str(official_weight.relative_to(ROOT)),
            "official_weight_sha256": sha256_file(official_weight),
            "posterior_path": str(posterior_path.relative_to(ROOT)),
            "posterior_sha256": sha256_file(posterior_path),
            "training": training_protocol,
            "primary": {"evaluation": "end", "epoch": EPOCHS},
            "results": ordered_results,
            "references": references,
            "cross_check": {
                "lab05_weight": {
                    "protected_param_count": lab05_weight["protection"][
                        "protected_param_count"
                    ],
                    "end": lab05_weight["end"],
                },
                "top17_all_extras_relation": "lab05_weight_plus_last_linear_bias",
            },
            "missing_cases": missing_cases,
            "outputs": {
                "data": str(data_path.relative_to(ROOT)),
                "history": str(history_path.relative_to(ROOT)),
                "plot": str(plot_path.relative_to(ROOT)),
                "mask_pattern": str(
                    (out_dir / "<case>_mask.pt").relative_to(ROOT)
                ),
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        metrics_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_history(history_path, ordered_history)
        write_data(data_path, ordered_results)
        if complete:
            plot_metrics(plot_path, ordered_results, references)
        else:
            plot_path.unlink(missing_ok=True)

    for case in cases:
        if not case.trained_here or case.name in results_by_case:
            continue
        configure_reproducibility(SEED, deterministic=True)
        surrogate, plan, masks, selected_units = initialize_case(
            case, victim, official_weight
        )
        expected = expected_by_case[case.name]["protection"]
        if plan.protection_mask_sha256 != expected["protection_mask_sha256"]:
            raise RuntimeError(f"{case.name} 训练前 mask 与 dry-run 定义漂移。")
        mask_path = out_dir / f"{case.name}_mask.pt"
        save_protection_mask(mask_path, masks)
        surrogate = surrogate.to(device)
        query_loader = DataLoader(
            query_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker,
            generator=build_generator(SEED),
        )
        optimizer = torch.optim.SGD(
            surrogate.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=LR_STEP,
            gamma=LR_GAMMA,
        )
        for epoch in range(1, EPOCHS + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            train_metrics = train_one_epoch(
                surrogate,
                query_loader,
                optimizer,
                device,
                "soft",
                epoch,
                EPOCHS,
                None,
            )
            scheduler.step()
            history_rows.append(
                {
                    "case": case.name,
                    "top_k": case.top_k,
                    "variant": case.variant,
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    **train_metrics,
                }
            )
        end_metrics = evaluate_surrogate(
            surrogate,
            eval_loader,
            eval_reference,
            device,
        )
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
            "primary": {"evaluation": "end", "epoch": EPOCHS},
            "end": end_metrics,
        }
        print(
            f"[END/{case.name}] accuracy={end_metrics['surrogate_acc']:.6f} "
            f"fidelity={end_metrics['fidelity']:.6f} "
            f"posterior_kl={end_metrics['posterior_kl']:.6f}"
        )
        persist(complete=False)
        del surrogate, optimizer, scheduler, query_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(results_by_case) != len(cases):
        missing = [case.name for case in cases if case.name not in results_by_case]
        raise RuntimeError(f"Lab06 仍缺少组合：{missing}")
    persist(complete=True)
    print(f"[INFO] 结果：{metrics_path.relative_to(ROOT)}")
    print(f"[INFO] 三联图：{plot_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
