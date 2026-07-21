#!/usr/bin/env python3
"""计算 Test01 的 40/16 个 Conv weight/BN affine 候选指标。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Subset


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
TRAIN_ROOT = ROOT / "exp" / "MS" / "train_surrogate"
TRAIN_VICTIM_ROOT = ROOT / "exp" / "MS" / "train_victim"
for import_root in (ROOT, TRAIN_ROOT, TRAIN_VICTIM_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common.trainer import (  # noqa: E402
    build_generator,
    build_public_split_dataset,
    build_transforms,
    configure_reproducibility,
    seed_worker,
)
from exp.MS.train_surrogate.core.artifacts import sha256_file  # noqa: E402
from exp.MS.train_surrogate.core.config import resolve_device  # noqa: E402
from exp.MS.train_surrogate.core.data import (  # noqa: E402
    build_victim,
    read_query_indices,
)
from models import imagenet as imagenet_models  # noqa: E402


EXPERIMENT = "01_cross"
MODEL = "resnet18"
DATASET = "c100"
NUM_CLASSES = 100
QUERY_COUNT = 500
TRAIN_COUNT = 50_000
SEED = 42
BATCH_SIZE = 64
EXPECTED_CONV_COUNT = 20
EXPECTED_BN_COUNT = 20
MAIN_CONV_NAMES = tuple(
    f"layer{stage}.{block}.conv{conv}"
    for stage in range(1, 5)
    for block in range(2)
    for conv in range(1, 3)
)
RANK_TARGETS = ("cross", "z_vv", "z_vp", "z_pv", "z_pp", "natural")
METRIC_SPECS = (
    {
        "field": "cross_abs_mean",
        "file": "cross",
        "title": "Mean absolute cross residual",
        "label": "mean(|I|) over C×H×W and 500 query images",
        "signed": False,
    },
    {
        "field": "natural_abs_mean",
        "file": "natural",
        "title": "Mean absolute natural residual",
        "label": "mean(|z_uu - z_pp|) over C×H×W and 500 query images",
        "signed": False,
    },
    {
        "field": "cross_rank_mean",
        "file": "cross_rank",
        "title": "Cross-residual entropy effective rank",
        "label": "mean(r(I)) over 500 query images",
        "signed": False,
    },
    {
        "field": "rank_gap_vv_vp_mean",
        "file": "rank_gap_vv_vp",
        "title": "Effective-rank gap: z_vv versus z_vp",
        "label": "mean(r(z_vv) - r(z_vp)) over 500 query images",
        "signed": True,
    },
    {
        "field": "rank_gap_pv_pp_mean",
        "file": "rank_gap_pv_pp",
        "title": "Effective-rank gap: z_pv versus z_pp",
        "label": "mean(r(z_pv) - r(z_pp)) over 500 query images",
        "signed": True,
    },
    {
        "field": "rank_interaction_mean",
        "file": "rank_interaction",
        "title": "Effective-rank interaction",
        "label": "mean((r(z_vv)-r(z_vp))-(r(z_pv)-r(z_pp)))",
        "signed": True,
    },
    {
        "field": "natural_rank_mean",
        "file": "natural_rank",
        "title": "Natural-residual entropy effective rank",
        "label": "mean(r(z_uu - z_pp)) over 500 query images",
        "signed": False,
    },
    {
        "field": "rank_gap_uu_pp_mean",
        "file": "rank_gap_uu_pp",
        "title": "Effective-rank gap: z_uu versus z_pp",
        "label": "mean(r(z_uu) - r(z_pp)) over 500 query images",
        "signed": True,
    },
    {
        "field": "product_score",
        "file": "product",
        "title": "Cross × natural residual score",
        "label": "mean(|I|) × mean(|z_uu - z_pp|)",
        "signed": False,
    },
)
METRIC_FIELDS = tuple(str(spec["field"]) for spec in METRIC_SPECS)
COMMON_FIELDS = (
    "operator_type",
    "module",
    "weight_state",
    "bias_state",
    "weight_shape",
    "input_shape",
    "output_shape",
    "parameter_count",
    "image_count",
    "rank_capacity",
) + METRIC_FIELDS
ALL_FIELDS = ("index",) + COMMON_FIELDS
MAIN_FIELDS = ("index", "all_index") + COMMON_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只处理第一个 batch 并核对公式，不写结果。",
    )
    return parser.parse_args()


def hash_integer_sequence(values: list[int]) -> str:
    encoded = json.dumps(values, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def shape_text(tensor: torch.Tensor) -> str:
    return "×".join(str(value) for value in tensor.shape[1:])


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


def select_modules(
    model: torch.nn.Module,
) -> tuple[
    dict[str, torch.nn.Module],
    dict[str, torch.nn.Conv2d],
    dict[str, torch.nn.BatchNorm2d],
]:
    candidates: dict[str, torch.nn.Module] = {}
    conv: dict[str, torch.nn.Conv2d] = {}
    bn: dict[str, torch.nn.BatchNorm2d] = {}
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, torch.nn.Conv2d):
            candidates[name] = module
            conv[name] = module
        elif isinstance(module, torch.nn.BatchNorm2d):
            candidates[name] = module
            bn[name] = module
    if len(conv) != EXPECTED_CONV_COUNT:
        raise ValueError(f"ResNet18 Conv2d 应为 20 个，实际为 {len(conv)}。")
    if len(bn) != EXPECTED_BN_COUNT:
        raise ValueError(f"ResNet18 BatchNorm2d 应为 20 个，实际为 {len(bn)}。")
    if len(candidates) != EXPECTED_CONV_COUNT + EXPECTED_BN_COUNT:
        raise ValueError("ResNet18 候选集合不是 40 项。")
    if any(module.bias is not None for module in conv.values()):
        raise ValueError("当前 ResNet18 的 Conv2d 意外包含 bias。")
    expected_extra_conv = {
        "conv1",
        "layer2.0.downsample.0",
        "layer3.0.downsample.0",
        "layer4.0.downsample.0",
    }
    if set(conv) - set(MAIN_CONV_NAMES) != expected_extra_conv:
        raise ValueError("20 个 Conv 没有形成 16 个主分支加 stem/downsample。")
    return candidates, conv, bn


def check_module_pairs(
    public_candidates: dict[str, torch.nn.Module],
    victim_candidates: dict[str, torch.nn.Module],
) -> None:
    if list(public_candidates) != list(victim_candidates):
        raise ValueError("public/victim 候选名称或顺序不一致。")
    for name, public_module in public_candidates.items():
        victim_module = victim_candidates[name]
        if type(public_module) is not type(victim_module):
            raise ValueError(f"{name} 的 public/victim 类型不一致。")
        if public_module.weight.shape != victim_module.weight.shape:
            raise ValueError(f"{name} 的 public/victim weight 形状不一致。")
        if isinstance(public_module, torch.nn.Conv2d):
            for field in ("stride", "padding", "dilation", "groups"):
                if getattr(public_module, field) != getattr(victim_module, field):
                    raise ValueError(f"{name} 的卷积几何字段 {field} 不一致。")
        elif isinstance(public_module, torch.nn.BatchNorm2d):
            if (
                public_module.bias.shape != victim_module.bias.shape
                or public_module.running_mean.shape
                != victim_module.running_mean.shape
                or public_module.running_var.shape
                != victim_module.running_var.shape
                or public_module.eps != victim_module.eps
            ):
                raise ValueError(f"{name} 的 BN 几何或状态形状不一致。")


def register_capture(
    modules: dict[str, torch.nn.Module],
    captured_inputs: dict[str, torch.Tensor],
    captured_outputs: dict[str, torch.Tensor],
) -> list[torch.utils.hooks.RemovableHandle]:
    handles = []
    for name, module in modules.items():
        def capture(_module, inputs, output, current=name):
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise ValueError(f"{current} 的算子输入不可识别。")
            captured_inputs[current] = inputs[0].detach().clone()
            captured_outputs[current] = output.detach().clone()

        handles.append(module.register_forward_hook(capture))
    return handles


def apply_conv(
    input_tensor: torch.Tensor,
    geometry: torch.nn.Conv2d,
    weight: torch.Tensor,
) -> torch.Tensor:
    return functional.conv2d(
        input_tensor,
        weight,
        bias=None,
        stride=geometry.stride,
        padding=geometry.padding,
        dilation=geometry.dilation,
        groups=geometry.groups,
    )


def normalize_bn_input(
    input_tensor: torch.Tensor,
    module: torch.nn.BatchNorm2d,
) -> torch.Tensor:
    if module.running_mean is None or module.running_var is None:
        raise ValueError("BN affine 计算要求 track_running_stats=True。")
    mean = module.running_mean.reshape(1, -1, 1, 1)
    inverse_std = torch.rsqrt(
        module.running_var.reshape(1, -1, 1, 1) + module.eps
    )
    return (input_tensor - mean) * inverse_std


def apply_gamma(
    normalized_input: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    return normalized_input * gamma.reshape(1, -1, 1, 1)


def apply_beta(
    gamma_output: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    return gamma_output + beta.reshape(1, -1, 1, 1)


def effective_rank_per_image(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if tensor.ndim != 4:
        raise ValueError("有效秩只接受 N×C×H×W 张量。")
    batch, channels, height, width = tensor.shape
    matrices = (
        tensor.permute(0, 2, 3, 1)
        .reshape(batch, height * width, channels)
        .to(dtype=torch.float64)
    )
    singular_values = torch.linalg.svdvals(matrices)
    energy = singular_values.square()
    total = energy.sum(dim=1, keepdim=True)
    nonzero = total.squeeze(1) > 0
    probabilities = torch.zeros_like(energy)
    probabilities[nonzero] = energy[nonzero] / total[nonzero]
    entropy_terms = torch.zeros_like(probabilities)
    positive = probabilities > 0
    entropy_terms[positive] = (
        probabilities[positive] * probabilities[positive].log()
    )
    ranks = torch.zeros(batch, dtype=torch.float64, device=tensor.device)
    ranks[nonzero] = torch.exp(-entropy_terms[nonzero].sum(dim=1))
    if not torch.isfinite(ranks).all():
        raise ValueError("逐图片有效秩出现非有限值。")
    return ranks, min(height * width, channels)


def initialize_statistics(
    candidates: dict[str, torch.nn.Module],
) -> dict[str, dict[str, object]]:
    statistics = {}
    for name, module in candidates.items():
        operator_type = (
            "conv_weight"
            if isinstance(module, torch.nn.Conv2d)
            else "bn_affine"
        )
        if isinstance(module, torch.nn.Conv2d):
            state_shape = "×".join(str(value) for value in module.weight.shape)
            parameter_count = module.weight.numel()
        else:
            state_shape = ";".join(
                (
                    "gamma=" + "×".join(str(value) for value in module.weight.shape),
                    "beta=" + "×".join(str(value) for value in module.bias.shape),
                )
            )
            parameter_count = module.weight.numel() + module.bias.numel()
        statistics[name] = {
            "operator_type": operator_type,
            "weight_state": f"{name}.weight",
            "bias_state": (
                "" if isinstance(module, torch.nn.Conv2d) else f"{name}.bias"
            ),
            "weight_shape": state_shape,
            "parameter_count": parameter_count,
            "image_count": 0,
            "cross_abs_sum": 0.0,
            "natural_abs_sum": 0.0,
            "rank_sums": {target: 0.0 for target in RANK_TARGETS},
            "rank_capacity": None,
            "input_shape": None,
            "output_shape": None,
        }
    return statistics


def update_statistics(
    state: dict[str, object],
    *,
    cross: torch.Tensor,
    z_vv: torch.Tensor,
    z_vp: torch.Tensor,
    z_pv: torch.Tensor,
    z_pp: torch.Tensor,
    operator_input: torch.Tensor,
) -> None:
    tensors = {
        "cross": cross,
        "z_vv": z_vv,
        "z_vp": z_vp,
        "z_pv": z_pv,
        "z_pp": z_pp,
        "natural": z_vv - z_pp,
    }
    if any(tensor.shape != cross.shape for tensor in tensors.values()):
        raise ValueError("同一候选的六个秩张量形状不一致。")
    batch = cross.size(0)
    state["image_count"] = int(state["image_count"]) + batch
    state["cross_abs_sum"] = float(state["cross_abs_sum"]) + float(
        cross.abs().mean(dim=(1, 2, 3)).double().sum().item()
    )
    state["natural_abs_sum"] = float(state["natural_abs_sum"]) + float(
        tensors["natural"].abs().mean(dim=(1, 2, 3)).double().sum().item()
    )
    stacked = torch.cat([tensors[target] for target in RANK_TARGETS], dim=0)
    stacked_ranks, capacity = effective_rank_per_image(stacked)
    if state["rank_capacity"] not in (None, capacity):
        raise ValueError("同一候选的有效秩上限在 batch 间变化。")
    state["rank_capacity"] = capacity
    rank_chunks = stacked_ranks.split(batch)
    if len(rank_chunks) != len(RANK_TARGETS):
        raise RuntimeError("批量有效秩没有按六个目标正确拆分。")
    for target, ranks in zip(RANK_TARGETS, rank_chunks):
        state["rank_sums"][target] = float(
            state["rank_sums"][target]
        ) + float(ranks.sum().item())
    input_shape = shape_text(operator_input)
    output_shape = shape_text(cross)
    if state["input_shape"] not in (None, input_shape):
        raise ValueError("同一候选的输入形状在 batch 间变化。")
    if state["output_shape"] not in (None, output_shape):
        raise ValueError("同一候选的输出形状在 batch 间变化。")
    state["input_shape"] = input_shape
    state["output_shape"] = output_shape


def build_rows(
    statistics: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for index, (module, state) in enumerate(statistics.items(), start=1):
        count = int(state["image_count"])
        if count != QUERY_COUNT:
            raise ValueError(f"{module} 只统计了 {count}/{QUERY_COUNT} 张图片。")
        capacity = state["rank_capacity"]
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"{module} 缺少有效秩上限。")
        rank_means = {
            target: float(state["rank_sums"][target]) / count
            for target in RANK_TARGETS
        }
        if any(
            not math.isfinite(value)
            or value < -1e-12
            or value > capacity + 1e-9
            for value in rank_means.values()
        ):
            raise ValueError(f"{module} 的基础有效秩超出合法范围。")
        cross_abs = float(state["cross_abs_sum"]) / count
        natural_abs = float(state["natural_abs_sum"]) / count
        row = {
            "index": index,
            "operator_type": state["operator_type"],
            "module": module,
            "weight_state": state["weight_state"],
            "bias_state": state["bias_state"],
            "weight_shape": state["weight_shape"],
            "input_shape": state["input_shape"],
            "output_shape": state["output_shape"],
            "parameter_count": state["parameter_count"],
            "image_count": count,
            "rank_capacity": capacity,
            "cross_abs_mean": cross_abs,
            "natural_abs_mean": natural_abs,
            "cross_rank_mean": rank_means["cross"],
            "rank_gap_vv_vp_mean": (
                rank_means["z_vv"] - rank_means["z_vp"]
            ),
            "rank_gap_pv_pp_mean": (
                rank_means["z_pv"] - rank_means["z_pp"]
            ),
            "rank_interaction_mean": (
                (rank_means["z_vv"] - rank_means["z_vp"])
                - (rank_means["z_pv"] - rank_means["z_pp"])
            ),
            "natural_rank_mean": rank_means["natural"],
            "rank_gap_uu_pp_mean": (
                rank_means["z_vv"] - rank_means["z_pp"]
            ),
            "product_score": cross_abs * natural_abs,
        }
        if any(not math.isfinite(float(row[field])) for field in METRIC_FIELDS):
            raise ValueError(f"{module} 的派生指标包含非有限值。")
        rows.append(row)
    if len(rows) != 40:
        raise ValueError("all 结果不是 40 项。")
    stem = next(row for row in rows if row["module"] == "conv1")
    if (
        float(stem["cross_abs_mean"]) != 0.0
        or float(stem["cross_rank_mean"]) != 0.0
        or float(stem["product_score"]) != 0.0
    ):
        raise ValueError("stem conv1 的紧凑交叉指标不是严格零。")
    return rows


def extract_main_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_module = {str(row["module"]): row for row in rows}
    if not set(MAIN_CONV_NAMES).issubset(by_module):
        raise ValueError("all 结果缺少 16 个主分支卷积。")
    extracted = []
    for index, module in enumerate(MAIN_CONV_NAMES, start=1):
        source = by_module[module]
        extracted.append(
            {
                "index": index,
                "all_index": source["index"],
                **{field: source[field] for field in COMMON_FIELDS},
            }
        )
    return extracted


def metric_order(
    rows: list[dict[str, object]],
    field: str,
) -> list[dict[str, object]]:
    ordered = sorted(
        rows,
        key=lambda row: (-abs(float(row[field])), str(row["module"])),
    )
    values = [abs(float(row[field])) for row in ordered]
    if any(right > left + 1e-12 for left, right in zip(values, values[1:])):
        raise ValueError(f"{field} 没有按绝对值降序排列。")
    return ordered


def plot_metric(
    path: Path,
    rows: list[dict[str, object]],
    spec: dict[str, object],
    *,
    scope: str,
) -> None:
    field = str(spec["field"])
    ordered = metric_order(rows, field)
    labels = [str(row["module"]) for row in ordered]
    values = [float(row[field]) for row in ordered]
    positions = list(range(len(ordered)))
    if scope == "all":
        colors = [
            "#0072B2" if row["operator_type"] == "conv_weight" else "#E69F00"
            for row in ordered
        ]
        handles = [
            plt.Rectangle((0, 0), 1, 1, color="#0072B2"),
            plt.Rectangle((0, 0), 1, 1, color="#E69F00"),
        ]
        legend_labels = ("Conv weight", "BN affine")
        figure_size = (12.6, 14.0)
        scope_title = "all 40 Conv weight and BN affine candidates"
    else:
        colors = [
            "#0072B2" if str(row["module"]).endswith("conv1") else "#D55E00"
            for row in ordered
        ]
        handles = [
            plt.Rectangle((0, 0), 1, 1, color="#0072B2"),
            plt.Rectangle((0, 0), 1, 1, color="#D55E00"),
        ]
        legend_labels = ("conv1", "conv2")
        figure_size = (11.2, 7.6)
        scope_title = "sixteen BasicBlock convolution weight candidates"
    figure, axis = plt.subplots(figsize=figure_size)
    axis.barh(positions, values, color=colors)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel(str(spec["label"]))
    axis.set_title(f"{spec['title']}\n{scope_title}")
    axis.grid(axis="x", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    max_abs = max(abs(value) for value in values)
    scale = max(max_abs, 1e-12)
    signed = bool(spec["signed"])
    if signed:
        axis.axvline(0.0, color="#333333", linewidth=0.8)
        axis.set_xlim(-scale * 1.28, scale * 1.28)
    else:
        axis.set_xlim(0.0, scale * 1.18)
    for position, value in zip(positions, values):
        if value < 0:
            x = value - scale * 0.012
            horizontal_alignment = "right"
        else:
            x = value + scale * 0.012
            horizontal_alignment = "left"
        text_value = f"{value:+.4f}" if signed else f"{value:.4f}"
        axis.text(
            x,
            position,
            text_value,
            va="center",
            ha=horizontal_alignment,
            fontsize=7,
        )
    axis.legend(handles, legend_labels, frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=230, bbox_inches="tight")
    plt.close(figure)


def measure(
    *,
    loader: DataLoader,
    public: torch.nn.Module,
    victim: torch.nn.Module,
    public_candidates: dict[str, torch.nn.Module],
    victim_candidates: dict[str, torch.nn.Module],
    device: torch.device,
    dry_run: bool,
) -> tuple[list[dict[str, object]] | None, dict[str, float], int]:
    public_inputs: dict[str, torch.Tensor] = {}
    public_outputs: dict[str, torch.Tensor] = {}
    victim_inputs: dict[str, torch.Tensor] = {}
    victim_outputs: dict[str, torch.Tensor] = {}
    handles = [
        *register_capture(public_candidates, public_inputs, public_outputs),
        *register_capture(victim_candidates, victim_inputs, victim_outputs),
    ]
    statistics = initialize_statistics(public_candidates)
    correctness = {
        "captured_candidate_count": float(len(public_candidates)),
        "captured_conv_count": float(
            sum(isinstance(module, torch.nn.Conv2d) for module in public_candidates.values())
        ),
        "captured_bn_affine_count": float(
            sum(
                isinstance(module, torch.nn.BatchNorm2d)
                for module in public_candidates.values()
            )
        ),
        "max_conv_public_hook_error": 0.0,
        "max_conv_victim_hook_error": 0.0,
        "max_conv_compact_identity_error": 0.0,
        "max_bn_affine_public_hook_error": 0.0,
        "max_bn_affine_victim_hook_error": 0.0,
        "max_bn_affine_compact_identity_error": 0.0,
        "max_stem_compact_cross_abs": 0.0,
    }
    processed = 0
    try:
        with torch.no_grad():
            for batch_index, (images, _labels) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                public_inputs.clear()
                public_outputs.clear()
                victim_inputs.clear()
                victim_outputs.clear()
                public(images)
                victim(images)
                expected_names = set(public_candidates)
                if any(
                    set(captured) != expected_names
                    for captured in (
                        public_inputs,
                        public_outputs,
                        victim_inputs,
                        victim_outputs,
                    )
                ):
                    raise ValueError("forward hook 没有完整捕获 40 个候选。")
                for name, public_module in public_candidates.items():
                    victim_module = victim_candidates[name]
                    if isinstance(public_module, torch.nn.Conv2d):
                        h_public = public_inputs[name]
                        h_victim = victim_inputs[name]
                        z_pp = apply_conv(
                            h_public,
                            public_module,
                            public_module.weight,
                        )
                        z_pv = apply_conv(
                            h_public,
                            victim_module,
                            victim_module.weight,
                        )
                        z_vp = apply_conv(
                            h_victim,
                            public_module,
                            public_module.weight,
                        )
                        z_vv = apply_conv(
                            h_victim,
                            victim_module,
                            victim_module.weight,
                        )
                        expanded = z_vv - z_vp - z_pv + z_pp
                        cross = apply_conv(
                            h_victim - h_public,
                            public_module,
                            victim_module.weight - public_module.weight,
                        )
                        if batch_index == 0:
                            correctness["max_conv_public_hook_error"] = max(
                                correctness["max_conv_public_hook_error"],
                                float((z_pp - public_outputs[name]).abs().max().item()),
                            )
                            correctness["max_conv_victim_hook_error"] = max(
                                correctness["max_conv_victim_hook_error"],
                                float((z_vv - victim_outputs[name]).abs().max().item()),
                            )
                            correctness["max_conv_compact_identity_error"] = max(
                                correctness["max_conv_compact_identity_error"],
                                float((expanded - cross).abs().max().item()),
                            )
                        if name == "conv1":
                            correctness["max_stem_compact_cross_abs"] = max(
                                correctness["max_stem_compact_cross_abs"],
                                float(cross.abs().max().item()),
                            )
                        operator_input = h_victim
                    elif isinstance(public_module, torch.nn.BatchNorm2d):
                        h_public = normalize_bn_input(
                            public_inputs[name],
                            public_module,
                        )
                        h_victim = normalize_bn_input(
                            victim_inputs[name],
                            victim_module,
                        )
                        z_pp = apply_beta(
                            apply_gamma(h_public, public_module.weight),
                            public_module.bias,
                        )
                        z_pv = apply_beta(
                            apply_gamma(h_public, victim_module.weight),
                            victim_module.bias,
                        )
                        z_vp = apply_beta(
                            apply_gamma(h_victim, public_module.weight),
                            public_module.bias,
                        )
                        z_vv = apply_beta(
                            apply_gamma(h_victim, victim_module.weight),
                            victim_module.bias,
                        )
                        expanded = z_vv - z_vp - z_pv + z_pp
                        cross = apply_gamma(
                            h_victim - h_public,
                            victim_module.weight - public_module.weight,
                        )
                        if batch_index == 0:
                            correctness["max_bn_affine_public_hook_error"] = max(
                                correctness["max_bn_affine_public_hook_error"],
                                float(
                                    (z_pp - public_outputs[name])
                                    .abs()
                                    .max()
                                    .item()
                                ),
                            )
                            correctness["max_bn_affine_victim_hook_error"] = max(
                                correctness["max_bn_affine_victim_hook_error"],
                                float(
                                    (z_vv - victim_outputs[name])
                                    .abs()
                                    .max()
                                    .item()
                                ),
                            )
                            correctness["max_bn_affine_compact_identity_error"] = max(
                                correctness["max_bn_affine_compact_identity_error"],
                                float((expanded - cross).abs().max().item()),
                            )
                        operator_input = h_victim
                    else:
                        raise TypeError(f"不支持的候选类型：{name}。")
                    update_statistics(
                        statistics[name],
                        cross=cross,
                        z_vv=z_vv,
                        z_vp=z_vp,
                        z_pv=z_pv,
                        z_pp=z_pp,
                        operator_input=operator_input,
                    )
                processed += images.size(0)
                if batch_index == 0 or batch_index + 1 == len(loader):
                    print(
                        f"[QUERY {batch_index + 1:03d}/{len(loader):03d}] "
                        f"processed={processed}/{QUERY_COUNT}",
                        flush=True,
                    )
                if dry_run:
                    break
    finally:
        for handle in handles:
            handle.remove()
    tolerances = {
        "max_conv_public_hook_error": 1e-6,
        "max_conv_victim_hook_error": 1e-6,
        "max_conv_compact_identity_error": 2e-5,
        "max_bn_affine_public_hook_error": 2e-6,
        "max_bn_affine_victim_hook_error": 2e-6,
        "max_bn_affine_compact_identity_error": 2e-6,
        "max_stem_compact_cross_abs": 0.0,
    }
    for field, tolerance in tolerances.items():
        if float(correctness[field]) > tolerance:
            raise RuntimeError(f"正确性检查失败：{field}={correctness[field]}。")
    if dry_run:
        return None, correctness, processed
    if processed != QUERY_COUNT:
        raise RuntimeError(f"只处理了 {processed}/{QUERY_COUNT} 张 query。")
    return build_rows(statistics), correctness, processed


def save_results(
    *,
    rows: list[dict[str, object]],
    correctness: dict[str, float],
    processed: int,
    output_root: Path,
    data_metadata: dict[str, object],
    public_checkpoint: Path,
    victim_checkpoint: Path,
    victim_metadata: dict[str, object],
    device: torch.device,
    num_workers: int,
) -> None:
    main_rows = extract_main_rows(rows)
    all_path = output_root / "all.tsv"
    main_path = output_root / "main.tsv"
    write_tsv(all_path, rows, ALL_FIELDS)
    write_tsv(main_path, main_rows, MAIN_FIELDS)
    outputs = {
        "all": str(all_path.relative_to(ROOT)),
        "main": str(main_path.relative_to(ROOT)),
    }
    for scope, scope_rows in (("all", rows), ("main", main_rows)):
        for spec in METRIC_SPECS:
            filename = f"{scope}_{spec['file']}.png"
            path = output_root / filename
            plot_metric(path, scope_rows, spec, scope=scope)
            outputs[f"{scope}_{spec['file']}"] = str(path.relative_to(ROOT))
    payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "protocol": "cross_natural_rank_bn_affine_500_query_v2",
        "scientific_status": "data_only_no_ms_feedback",
        "dataset": DATASET,
        "model": MODEL,
        "seed": SEED,
        "data": data_metadata,
        "models": {
            "public": {
                "checkpoint": str(public_checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(public_checkpoint),
                "num_classes": 1000,
            },
            "victim": {
                "checkpoint": str(victim_checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(victim_checkpoint),
                "checkpoint_epoch": victim_metadata.get("epoch"),
                "num_classes": NUM_CLASSES,
            },
        },
        "selection": {
            "all_candidate_count": len(rows),
            "main_candidate_count": len(main_rows),
            "all_conv_weight_count": sum(
                row["operator_type"] == "conv_weight" for row in rows
            ),
            "all_bn_affine_count": sum(
                row["operator_type"] == "bn_affine" for row in rows
            ),
            "bn_affine_states_per_candidate": ["weight", "bias"],
            "bn_running_state_role": "normalize_each_natural_model_input",
            "bn_cross_formula": (
                "I=(gamma_v-gamma_p)*(h_hat_v-h_hat_p);beta_cancels"
            ),
            "bn_natural_formula": (
                "N=(gamma_v*h_hat_v+beta_v)-(gamma_p*h_hat_p+beta_p)"
            ),
            "main_modules": list(MAIN_CONV_NAMES),
            "ranking": "descending_absolute_metric_then_module",
            "display": "signed_true_metric_value",
            "ranking_uses_ms_feedback": False,
        },
        "effective_rank": {
            "matrix": "per_image_H_times_W_by_C_channels_as_columns",
            "signed_tensor": True,
            "centered": False,
            "spectral_probability": "sigma_squared_normalized",
            "formula": "exp(-sum_i(p_i*log(p_i)))",
            "zero_energy_rank": 0.0,
            "aggregation": "per_image_then_mean_over_500_query",
        },
        "metrics": {
            "cross_abs_mean": "mean_image(mean_chw(abs(I)))",
            "natural_abs_mean": "mean_image(mean_chw(abs(z_vv-z_pp)))",
            "cross_rank_mean": "mean_image(r(I))",
            "rank_gap_vv_vp_mean": "mean_image(r(z_vv)-r(z_vp))",
            "rank_gap_pv_pp_mean": "mean_image(r(z_pv)-r(z_pp))",
            "rank_interaction_mean": (
                "mean_image((r(z_vv)-r(z_vp))-(r(z_pv)-r(z_pp)))"
            ),
            "natural_rank_mean": "mean_image(r(z_vv-z_pp))",
            "rank_gap_uu_pp_mean": "mean_image(r(z_vv)-r(z_pp))",
            "product_score": "cross_abs_mean*natural_abs_mean",
        },
        "execution": {
            "device": str(device),
            "batch_size": BATCH_SIZE,
            "num_workers": num_workers,
            "processed_image_count": processed,
            "model_mode": "eval",
            "gradient_enabled": False,
        },
        "correctness": correctness,
        "results": {"all": rows, "main": main_rows},
        "outputs": outputs,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_root / "metrics.json", payload)


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    device = resolve_device(args.device)
    configure_reproducibility(SEED, deterministic=True)
    output_root = ROOT / "results" / "test" / "MS" / EXPERIMENT
    output_root.mkdir(parents=True, exist_ok=True)
    victim_checkpoint = (
        ROOT / "weights" / "MS" / "victim" / MODEL / DATASET / "best.pth"
    )
    public_checkpoint = ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    victim, victim_metadata = build_victim(
        MODEL,
        NUM_CLASSES,
        victim_checkpoint,
    )
    public = imagenet_models.resnet18(num_classes=1000)
    imagenet_models.load_official_imagenet_weights(
        MODEL,
        public,
        str(public_checkpoint),
        strict=True,
    )
    public_candidates, _public_conv, _public_bn = select_modules(public)
    victim_candidates, _victim_conv, _victim_bn = select_modules(victim)
    check_module_pairs(public_candidates, victim_candidates)
    _, test_transform = build_transforms(DATASET)
    public_dataset = build_public_split_dataset(
        DATASET,
        ROOT / "dataset" / "public",
        "train",
        test_transform,
    )
    if len(public_dataset) != TRAIN_COUNT:
        raise ValueError(
            f"CIFAR-100 official_train 应为 {TRAIN_COUNT} 张，"
            f"实际为 {len(public_dataset)}。"
        )
    query_indices = read_query_indices(
        ROOT / "dataset" / "MS",
        DATASET,
    )[:QUERY_COUNT]
    if len(query_indices) != QUERY_COUNT or len(set(query_indices)) != QUERY_COUNT:
        raise ValueError("没有得到固定且不重复的 500 张 query。")
    data_metadata = {
        "split": "query_pool_ms",
        "count": QUERY_COUNT,
        "selection": "canonical_query_rank_prefix",
        "source_indices": query_indices,
        "source_indices_sha256": hash_integer_sequence(query_indices),
        "input_transform": "test",
        "gradient_enabled": False,
    }
    loader = DataLoader(
        Subset(public_dataset, query_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=build_generator(SEED),
    )
    public = public.to(device).eval()
    victim = victim.to(device).eval()
    rows, correctness, processed = measure(
        loader=loader,
        public=public,
        victim=victim,
        public_candidates=public_candidates,
        victim_candidates=victim_candidates,
        device=device,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(
            f"[INFO] dry-run 通过：batch={processed} correctness={correctness}"
        )
        return 0
    assert rows is not None
    save_results(
        rows=rows,
        correctness=correctness,
        processed=processed,
        output_root=output_root,
        data_metadata=data_metadata,
        public_checkpoint=public_checkpoint,
        victim_checkpoint=victim_checkpoint,
        victim_metadata=victim_metadata,
        device=device,
        num_workers=args.num_workers,
    )
    print(f"[OK] 写入 {output_root.relative_to(ROOT)} 的精简结果。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
