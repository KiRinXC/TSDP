#!/usr/bin/env python3
"""计算 ResNet18 全部 Conv weight 与 BN gamma 的交叉残差绝对均值。"""

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
WEIGHT_FIELDS = (
    "rank",
    "type_rank",
    "operator_type",
    "module",
    "weight_state",
    "weight_shape",
    "input_shape",
    "output_shape",
    "parameter_count",
    "image_count",
    "cross_abs_mean",
)
TENSOR_FIELDS = (
    "rank",
    "overall_rank",
    "module",
    "weight_state",
    "input_shape",
    "output_shape",
    "image_count",
    "cross_abs_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--scope",
        choices=("both", "query", "full"),
        default="both",
        help="both 依次计算 500-query 与 50,000-image；也可单独选择其一。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="每个选定范围只处理第一个 batch 并核对恒等式，不写结果。",
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
    dict[str, torch.nn.Conv2d],
    dict[str, torch.nn.BatchNorm2d],
]:
    conv = {
        name: module
        for name, module in model.named_modules()
        if name and isinstance(module, torch.nn.Conv2d)
    }
    bn = {
        name: module
        for name, module in model.named_modules()
        if name and isinstance(module, torch.nn.BatchNorm2d)
    }
    if len(conv) != EXPECTED_CONV_COUNT:
        raise ValueError(
            f"ResNet18 Conv2d 应为 {EXPECTED_CONV_COUNT} 个，实际为 {len(conv)}。"
        )
    if len(bn) != EXPECTED_BN_COUNT:
        raise ValueError(
            f"ResNet18 BatchNorm2d 应为 {EXPECTED_BN_COUNT} 个，实际为 {len(bn)}。"
        )
    if any(module.bias is not None for module in conv.values()):
        raise ValueError("当前 ResNet18 的 Conv2d 意外包含 bias。")
    expected_extra_conv = {
        "conv1",
        "layer2.0.downsample.0",
        "layer3.0.downsample.0",
        "layer4.0.downsample.0",
    }
    if set(conv) - set(MAIN_CONV_NAMES) != expected_extra_conv:
        raise ValueError("20 个 Conv2d 没有形成 16 个主分支加 stem/downsample。")
    expected_extra_bn = {
        "bn1",
        "layer2.0.downsample.1",
        "layer3.0.downsample.1",
        "layer4.0.downsample.1",
    }
    main_bn = {
        f"layer{stage}.{block}.bn{bn_index}"
        for stage in range(1, 5)
        for block in range(2)
        for bn_index in range(1, 3)
    }
    if set(bn) != expected_extra_bn | main_bn:
        raise ValueError("20 个 BatchNorm2d 的模块集合不符合当前 ResNet18。")
    return conv, bn


def check_module_pairs(
    public_conv: dict[str, torch.nn.Conv2d],
    victim_conv: dict[str, torch.nn.Conv2d],
    public_bn: dict[str, torch.nn.BatchNorm2d],
    victim_bn: dict[str, torch.nn.BatchNorm2d],
) -> None:
    if set(public_conv) != set(victim_conv):
        raise ValueError("public/victim Conv2d 名称集合不同。")
    if set(public_bn) != set(victim_bn):
        raise ValueError("public/victim BatchNorm2d 名称集合不同。")
    for name, left in public_conv.items():
        right = victim_conv[name]
        fields = ("stride", "padding", "dilation", "groups")
        if left.weight.shape != right.weight.shape or any(
            getattr(left, field) != getattr(right, field)
            for field in fields
        ):
            raise ValueError(f"{name} 的 public/victim 卷积几何不一致。")
    for name, left in public_bn.items():
        right = victim_bn[name]
        if (
            left.weight.shape != right.weight.shape
            or left.bias.shape != right.bias.shape
            or left.running_mean.shape != right.running_mean.shape
            or left.running_var.shape != right.running_var.shape
            or left.eps != right.eps
        ):
            raise ValueError(f"{name} 的 public/victim BN 几何不一致。")


def register_capture(
    modules: dict[str, torch.nn.Module],
    captured_inputs: dict[str, torch.Tensor],
    captured_outputs: dict[str, torch.Tensor],
):
    handles = []
    for name, module in modules.items():
        def capture(_module, inputs, output, current=name):
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise ValueError(f"{current} 的算子输入不可识别。")
            # BN 后可能接 inplace ReLU，必须立即 clone。
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
        raise ValueError("BN gamma 排名要求 track_running_stats=True。")
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


def initialize_statistics(
    public_conv: dict[str, torch.nn.Conv2d],
    public_bn: dict[str, torch.nn.BatchNorm2d],
) -> dict[str, dict[str, object]]:
    statistics: dict[str, dict[str, object]] = {}
    for name, module in public_conv.items():
        statistics[name] = {
            "operator_type": "conv_weight",
            "weight_state": f"{name}.weight",
            "weight_shape": "×".join(str(value) for value in module.weight.shape),
            "parameter_count": module.weight.numel(),
            "image_count": 0,
            "cross_abs_mean_sum": 0.0,
            "input_shape": None,
            "output_shape": None,
        }
    for name, module in public_bn.items():
        statistics[name] = {
            "operator_type": "bn_gamma",
            "weight_state": f"{name}.weight",
            "weight_shape": "×".join(str(value) for value in module.weight.shape),
            "parameter_count": module.weight.numel(),
            "image_count": 0,
            "cross_abs_mean_sum": 0.0,
            "input_shape": None,
            "output_shape": None,
        }
    if len(statistics) != EXPECTED_CONV_COUNT + EXPECTED_BN_COUNT:
        raise RuntimeError("Conv 与 BN gamma 模块名称发生冲突。")
    return statistics


def update_statistics(
    state: dict[str, object],
    interaction: torch.Tensor,
    operator_input: torch.Tensor,
) -> None:
    batch = interaction.size(0)
    # 每张图片先对 C、H、W 的绝对交叉残差取平均，再跨图片平均。
    per_image = interaction.abs().mean(dim=(1, 2, 3))
    state["image_count"] = int(state["image_count"]) + batch
    state["cross_abs_mean_sum"] = float(
        state["cross_abs_mean_sum"]
    ) + float(per_image.double().sum().item())
    input_shape = shape_text(operator_input)
    output_shape = shape_text(interaction)
    if state["input_shape"] not in (None, input_shape):
        raise ValueError("同一 weight 算子的输入形状在 batch 间变化。")
    if state["output_shape"] not in (None, output_shape):
        raise ValueError("同一 weight 算子的输出形状在 batch 间变化。")
    state["input_shape"] = input_shape
    state["output_shape"] = output_shape


def build_rows(
    statistics: dict[str, dict[str, object]],
    expected_count: int,
) -> list[dict[str, object]]:
    rows = []
    for module, state in statistics.items():
        count = int(state["image_count"])
        if count != expected_count:
            raise ValueError(f"{module} 只统计了 {count}/{expected_count} 张图片。")
        score = float(state["cross_abs_mean_sum"]) / count
        if not math.isfinite(score):
            raise ValueError(f"{module} 的交叉残差绝对均值不是有限值。")
        rows.append(
            {
                "rank": 0,
                "type_rank": 0,
                "operator_type": state["operator_type"],
                "module": module,
                "weight_state": state["weight_state"],
                "weight_shape": state["weight_shape"],
                "input_shape": state["input_shape"],
                "output_shape": state["output_shape"],
                "parameter_count": state["parameter_count"],
                "image_count": count,
                "cross_abs_mean": score,
            }
        )
    rows.sort(key=lambda row: (-float(row["cross_abs_mean"]), str(row["module"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    for operator_type in ("conv_weight", "bn_gamma"):
        typed = [row for row in rows if row["operator_type"] == operator_type]
        if len(typed) != 20:
            raise ValueError(f"{operator_type} 排名不是 20 个候选。")
        for type_rank, row in enumerate(typed, start=1):
            row["type_rank"] = type_rank
    return rows


def extract_main_conv(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = [
        row
        for row in rows
        if row["operator_type"] == "conv_weight"
        and str(row["module"]) in MAIN_CONV_NAMES
    ]
    if len(selected) != 16 or {str(row["module"]) for row in selected} != set(
        MAIN_CONV_NAMES
    ):
        raise ValueError("没有从 40 项排名中恰好提取 16 个主分支卷积。")
    selected.sort(
        key=lambda row: (-float(row["cross_abs_mean"]), str(row["module"]))
    )
    return [
        {
            "rank": rank,
            "overall_rank": int(row["rank"]),
            "module": row["module"],
            "weight_state": row["weight_state"],
            "input_shape": row["input_shape"],
            "output_shape": row["output_shape"],
            "image_count": row["image_count"],
            "cross_abs_mean": row["cross_abs_mean"],
        }
        for rank, row in enumerate(selected, start=1)
    ]


def plot_weights(
    path: Path,
    rows: list[dict[str, object]],
    *,
    image_count: int,
) -> None:
    ordered = list(reversed(rows))
    labels = [
        f"{str(row['module'])}  [{str(row['operator_type'])}]"
        for row in ordered
    ]
    scores = [float(row["cross_abs_mean"]) for row in ordered]
    colors = [
        "#0072B2" if row["operator_type"] == "conv_weight" else "#E69F00"
        for row in ordered
    ]
    figure, axis = plt.subplots(figsize=(12.4, 14.0))
    axis.barh(range(len(ordered)), scores, color=colors)
    axis.set_yticks(range(len(ordered)), labels)
    axis.set_xlabel("Mean absolute cross residual over C×H×W")
    scope_text = (
        "500 fixed query images"
        if image_count == QUERY_COUNT
        else "all 50,000 CIFAR-100 training images"
    )
    axis.set_title(
        "ResNet18 Conv weights and BN gamma cross residuals\n"
        f"{scope_text}; mean over C×H×W"
    )
    axis.grid(axis="x", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for index, score in enumerate(scores):
        axis.text(score, index, f" {score:.4f}", va="center", fontsize=7)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#0072B2"),
        plt.Rectangle((0, 0), 1, 1, color="#E69F00"),
    ]
    axis.legend(
        handles,
        ("Conv weight", "BN gamma"),
        frameon=False,
        loc="lower right",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=230, bbox_inches="tight")
    plt.close(figure)


def plot_main_conv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    ordered = list(reversed(rows))
    labels = [str(row["module"]) for row in ordered]
    scores = [float(row["cross_abs_mean"]) for row in ordered]
    colors = [
        "#0072B2" if str(row["module"]).endswith("conv1") else "#D55E00"
        for row in ordered
    ]
    figure, axis = plt.subplots(figsize=(10.6, 7.2))
    axis.barh(range(len(ordered)), scores, color=colors)
    axis.set_yticks(range(len(ordered)), labels)
    axis.set_xlabel("Mean absolute cross residual over C×H×W")
    axis.set_title(
        "Sixteen ResNet18 BasicBlock convolution weights\n"
        "extracted from the 40-candidate ranking on 500 query images"
    )
    axis.grid(axis="x", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for index, score in enumerate(scores):
        axis.text(score, index, f" {score:.4f}", va="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=230, bbox_inches="tight")
    plt.close(figure)


def measure(
    *,
    scope: str,
    loader: DataLoader,
    expected_count: int,
    public: torch.nn.Module,
    victim: torch.nn.Module,
    public_conv: dict[str, torch.nn.Conv2d],
    victim_conv: dict[str, torch.nn.Conv2d],
    public_bn: dict[str, torch.nn.BatchNorm2d],
    victim_bn: dict[str, torch.nn.BatchNorm2d],
    device: torch.device,
    dry_run: bool,
) -> tuple[
    list[dict[str, object]] | None,
    dict[str, float],
    int,
]:
    public_modules = {**public_conv, **public_bn}
    victim_modules = {**victim_conv, **victim_bn}
    public_inputs: dict[str, torch.Tensor] = {}
    public_outputs: dict[str, torch.Tensor] = {}
    victim_inputs: dict[str, torch.Tensor] = {}
    victim_outputs: dict[str, torch.Tensor] = {}
    handles = [
        *register_capture(public_modules, public_inputs, public_outputs),
        *register_capture(victim_modules, victim_inputs, victim_outputs),
    ]
    statistics = initialize_statistics(public_conv, public_bn)
    correctness = {
        "captured_conv_count": float(len(public_conv)),
        "captured_bn_gamma_count": float(len(public_bn)),
        "max_conv_public_hook_error": 0.0,
        "max_conv_victim_hook_error": 0.0,
        "max_conv_compact_identity_error": 0.0,
        "max_bn_public_hook_error": 0.0,
        "max_bn_victim_hook_error": 0.0,
        "max_bn_gamma_compact_identity_error": 0.0,
        "max_stem_conv_cross_abs": 0.0,
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
                expected_names = set(statistics)
                captured_sets = (
                    set(public_inputs),
                    set(public_outputs),
                    set(victim_inputs),
                    set(victim_outputs),
                )
                if any(names != expected_names for names in captured_sets):
                    raise ValueError("forward hook 没有完整捕获 20 Conv 和 20 BN。")

                for name, public_module in public_conv.items():
                    victim_module = victim_conv[name]
                    h_public = public_inputs[name]
                    h_victim = victim_inputs[name]
                    if h_public.shape != h_victim.shape:
                        raise ValueError(f"{name} 的 public/victim 输入形状不同。")
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
                    interaction = z_vv - z_vp - z_pv + z_pp
                    if batch_index == 0:
                        correctness["max_conv_public_hook_error"] = max(
                            correctness["max_conv_public_hook_error"],
                            float((z_pp - public_outputs[name]).abs().max().item()),
                        )
                        correctness["max_conv_victim_hook_error"] = max(
                            correctness["max_conv_victim_hook_error"],
                            float((z_vv - victim_outputs[name]).abs().max().item()),
                        )
                        compact = apply_conv(
                            h_victim - h_public,
                            public_module,
                            victim_module.weight - public_module.weight,
                        )
                        correctness["max_conv_compact_identity_error"] = max(
                            correctness["max_conv_compact_identity_error"],
                            float((interaction - compact).abs().max().item()),
                        )
                    if name == "conv1":
                        correctness["max_stem_conv_cross_abs"] = max(
                            correctness["max_stem_conv_cross_abs"],
                            float(interaction.abs().max().item()),
                        )
                    update_statistics(statistics[name], interaction, h_victim)

                for name, public_module in public_bn.items():
                    victim_module = victim_bn[name]
                    h_public = normalize_bn_input(
                        public_inputs[name],
                        public_module,
                    )
                    h_victim = normalize_bn_input(
                        victim_inputs[name],
                        victim_module,
                    )
                    if h_public.shape != h_victim.shape:
                        raise ValueError(
                            f"{name} 的 public/victim 标准化输入形状不同。"
                        )
                    z_pp = apply_gamma(h_public, public_module.weight)
                    z_pv = apply_gamma(h_public, victim_module.weight)
                    z_vp = apply_gamma(h_victim, public_module.weight)
                    z_vv = apply_gamma(h_victim, victim_module.weight)
                    interaction = z_vv - z_vp - z_pv + z_pp
                    if batch_index == 0:
                        public_natural = apply_beta(z_pp, public_module.bias)
                        victim_natural = apply_beta(z_vv, victim_module.bias)
                        correctness["max_bn_public_hook_error"] = max(
                            correctness["max_bn_public_hook_error"],
                            float(
                                (public_natural - public_outputs[name])
                                .abs()
                                .max()
                                .item()
                            ),
                        )
                        correctness["max_bn_victim_hook_error"] = max(
                            correctness["max_bn_victim_hook_error"],
                            float(
                                (victim_natural - victim_outputs[name])
                                .abs()
                                .max()
                                .item()
                            ),
                        )
                        compact = apply_gamma(
                            h_victim - h_public,
                            victim_module.weight - public_module.weight,
                        )
                        correctness["max_bn_gamma_compact_identity_error"] = max(
                            correctness["max_bn_gamma_compact_identity_error"],
                            float((interaction - compact).abs().max().item()),
                        )
                    update_statistics(statistics[name], interaction, h_victim)

                processed += images.size(0)
                if (
                    batch_index == 0
                    or (batch_index + 1) % 10 == 0
                    or batch_index + 1 == len(loader)
                ):
                    print(
                        f"[{scope.upper()} {batch_index + 1:03d}/"
                        f"{len(loader):03d}] processed={processed}/"
                        f"{expected_count}",
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
        "max_bn_public_hook_error": 2e-6,
        "max_bn_victim_hook_error": 2e-6,
        "max_bn_gamma_compact_identity_error": 2e-6,
        "max_stem_conv_cross_abs": 2e-6,
    }
    if any(
        float(correctness[field]) > tolerance
        for field, tolerance in tolerances.items()
    ):
        raise RuntimeError(
            f"{scope} 的交叉残差恒等式未通过：{correctness}"
        )
    if dry_run:
        return None, correctness, processed
    if processed != expected_count:
        raise RuntimeError(f"{scope} 只处理了 {processed}/{expected_count} 张图片。")
    return build_rows(statistics, expected_count), correctness, processed


def output_names(scope: str) -> dict[str, str]:
    if scope == "query":
        return {
            "metrics": "weights.json",
            "weights": "weights.tsv",
            "conv": "weights_conv.tsv",
            "bn": "weights_bn.tsv",
            "plot": "weights.png",
        }
    return {
        "metrics": "weights_full.json",
        "weights": "weights_full.tsv",
        "conv": "weights_full_conv.tsv",
        "bn": "weights_full_bn.tsv",
        "plot": "weights_full.png",
    }


def save_scope(
    *,
    scope: str,
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
    public_conv: dict[str, torch.nn.Conv2d],
    public_bn: dict[str, torch.nn.BatchNorm2d],
) -> None:
    names = output_names(scope)
    paths = {
        key: output_root / name
        for key, name in names.items()
    }
    conv_rows = [row for row in rows if row["operator_type"] == "conv_weight"]
    bn_rows = [row for row in rows if row["operator_type"] == "bn_gamma"]
    write_tsv(paths["weights"], rows, WEIGHT_FIELDS)
    write_tsv(paths["conv"], conv_rows, WEIGHT_FIELDS)
    write_tsv(paths["bn"], bn_rows, WEIGHT_FIELDS)
    plot_weights(paths["plot"], rows, image_count=processed)

    extra_outputs: dict[str, str] = {}
    if scope == "query":
        main_rows = extract_main_conv(rows)
        tensor_path = output_root / "tensors.tsv"
        tensor_plot_path = output_root / "tensors.png"
        write_tsv(tensor_path, main_rows, TENSOR_FIELDS)
        plot_main_conv(tensor_plot_path, main_rows)
        extra_outputs = {
            "main_conv_tensors": str(tensor_path.relative_to(ROOT)),
            "main_conv_plot": str(tensor_plot_path.relative_to(ROOT)),
        }

    payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "protocol": f"cross_weight_conv_bn_gamma_chw_mean_{scope}_v1",
        "scientific_status": "data_only_weight_selector_no_ms_feedback",
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
            "candidate_count": len(rows),
            "conv_weight_count": len(conv_rows),
            "bn_gamma_count": len(bn_rows),
            "included_conv": list(public_conv),
            "included_bn_gamma": list(public_bn),
            "excluded": [
                "convolution filters as independent candidates",
                "BatchNorm bias",
                "BatchNorm running_mean and running_var as candidates",
                "BatchNorm num_batches_tracked",
                "last_linear",
                "pooling, ReLU and residual addition",
            ],
            "conv_formula": "I=Conv(h_v-h_p,W_v-W_p)",
            "bn_gamma_input": (
                "h_hat=(h-running_mean)/sqrt(running_var+eps)"
            ),
            "bn_gamma_formula": (
                "I_gamma=(gamma_v-gamma_p)*(h_hat_v-h_hat_p)"
            ),
            "bn_running_state_role": (
                "construct_each_model_normalized_input_not_ranked_candidate"
            ),
            "bn_beta_role": "additive_after_gamma_excluded_and_cancels",
            "normalization": "mean_absolute_value_over_c_h_w_then_mean_images",
            "score": "mean_image(mean_chw(abs(I)))",
            "ranking_uses_ms_feedback": False,
        },
        "execution": {
            "device": str(device),
            "batch_size": BATCH_SIZE,
            "num_workers": num_workers,
            "processed_image_count": processed,
            "model_mode": "eval",
        },
        "correctness": correctness,
        "results": rows,
        "outputs": {
            "weights": str(paths["weights"].relative_to(ROOT)),
            "conv_weights": str(paths["conv"].relative_to(ROOT)),
            "bn_gamma": str(paths["bn"].relative_to(ROOT)),
            "rank_plot": str(paths["plot"].relative_to(ROOT)),
            **extra_outputs,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(paths["metrics"], payload)


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
    public_checkpoint = (
        ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    )
    dataset_root = ROOT / "dataset" / "public"

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
    public_conv, public_bn = select_modules(public)
    victim_conv, victim_bn = select_modules(victim)
    check_module_pairs(public_conv, victim_conv, public_bn, victim_bn)

    _, test_transform = build_transforms(DATASET)
    public_dataset = build_public_split_dataset(
        DATASET,
        dataset_root,
        "train",
        test_transform,
    )
    if len(public_dataset) != TRAIN_COUNT:
        raise ValueError(
            f"CIFAR-100 official_train 应为 {TRAIN_COUNT} 张，"
            f"实际为 {len(public_dataset)}。"
        )

    scopes = ("query", "full") if args.scope == "both" else (args.scope,)
    query_indices: list[int] | None = None
    if "query" in scopes:
        query_indices = read_query_indices(
            ROOT / "dataset" / "MS",
            DATASET,
        )[:QUERY_COUNT]
        if (
            len(query_indices) != QUERY_COUNT
            or len(set(query_indices)) != QUERY_COUNT
        ):
            raise ValueError("没有得到固定且不重复的 500 张 query。")

    public = public.to(device).eval()
    victim = victim.to(device).eval()
    for scope in scopes:
        if scope == "query":
            assert query_indices is not None
            dataset = Subset(public_dataset, query_indices)
            expected_count = QUERY_COUNT
            data_metadata = {
                "split": "query_pool_ms",
                "count": QUERY_COUNT,
                "selection": "canonical_query_rank_prefix",
                "source_indices": query_indices,
                "source_indices_sha256": hash_integer_sequence(query_indices),
                "input_transform": "test",
                "gradient_enabled": False,
            }
        else:
            dataset = public_dataset
            expected_count = TRAIN_COUNT
            data_metadata = {
                "split": "official_train",
                "count": TRAIN_COUNT,
                "selection": "all_source_indices_in_ascending_order",
                "source_index_start": 0,
                "source_index_end": TRAIN_COUNT - 1,
                "input_transform": "test",
                "gradient_enabled": False,
            }
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker,
            generator=build_generator(SEED),
        )
        rows, correctness, processed = measure(
            scope=scope,
            loader=loader,
            expected_count=expected_count,
            public=public,
            victim=victim,
            public_conv=public_conv,
            victim_conv=victim_conv,
            public_bn=public_bn,
            victim_bn=victim_bn,
            device=device,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(
                f"[INFO] {scope} dry-run 通过："
                f"conv=20 bn_gamma=20 batch={processed} "
                f"correctness={correctness}"
            )
            continue
        assert rows is not None
        save_scope(
            scope=scope,
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
            public_conv=public_conv,
            public_bn=public_bn,
        )
        print(f"[{scope.upper()} WEIGHT RANK]")
        for row in rows:
            print(
                f"{int(row['rank']):02d} "
                f"{str(row['operator_type']):<11} "
                f"{str(row['module']):<28} "
                f"mean_abs={float(row['cross_abs_mean']):.6f}"
            )

    if not args.dry_run:
        print(f"[OK] 写入 {output_root.relative_to(ROOT)}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
