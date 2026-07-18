#!/usr/bin/env python3
"""统计公开模型与受害者模型四路交叉前向的逐 filter weight 残差。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SURROGATE_ROOT = REPO_ROOT / "exp" / "MS" / "train_surrogate"
TEMP_ROOT = Path(__file__).resolve().parent
for search_root in (REPO_ROOT, TRAIN_SURROGATE_ROOT, TEMP_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import support as rec  # noqa: E402


MODEL = "resnet18"
DATASET = "c100"
NUM_CLASSES = 100
SEED = 42
QUERY_COUNT = 400
DEFAULT_BATCH_SIZE = 64
EXPECTED_CONV_COUNT = 20
EXPECTED_FILTER_COUNT = 4_800
ENTRY_CONVS = tuple(
    f"layer{stage}.0.conv{conv}"
    for stage in range(1, 5)
    for conv in (1, 2)
)
COMPONENTS = ("weight", "input", "total", "interaction")
DIRECT_EFFECTS = (
    "weight_effect_public_input",
    "weight_effect_victim_input",
    "input_effect_public_weight",
    "input_effect_victim_weight",
)

OUTPUT_ROOT = REPO_ROOT / "temp" / "output"
METRICS_PATH = OUTPUT_ROOT / "residual.json"
FILTER_PATH = OUTPUT_ROOT / "residual_filters.tsv"
UNIT_PATH = OUTPUT_ROOT / "residual_units.tsv"
ENTRY_PATH = OUTPUT_ROOT / "residual_entry.tsv"
PLOT_PATH = OUTPUT_ROOT / "residual.png"
OUTPUT_PATHS = (
    METRICS_PATH,
    FILTER_PATH,
    UNIT_PATH,
    ENTRY_PATH,
    PLOT_PATH,
)


@dataclass(frozen=True)
class ConvSpec:
    """一个与 TensorShield weight unit 对齐的 Conv2d。"""

    index: int
    module_name: str
    state_name: str
    role: str
    stage: int
    block: int
    conv_in_block: int
    paired_bn: str
    in_channels: int
    out_channels: int
    kernel_height: int
    kernel_width: int
    stride_height: int
    stride_width: int
    groups: int
    weight_param_count: int
    filter_param_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对模型、Conv/filter 数量和 query 输入，不运行统计或写产物。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有 temp/output/residual* 临时产物。",
    )
    return parser.parse_args()


def pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"无法解析二维参数：{value}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def conv_role(module_name: str) -> tuple[str, int, int, int, str]:
    if module_name == "conv1":
        return "stem", 0, -1, 1, "bn1"
    shortcut = re.fullmatch(r"layer([1-4])\.(\d+)\.downsample\.0", module_name)
    if shortcut:
        stage = int(shortcut.group(1))
        block = int(shortcut.group(2))
        return (
            "downsample",
            stage,
            block,
            0,
            f"layer{stage}.{block}.downsample.1",
        )
    main = re.fullmatch(r"layer([1-4])\.(\d+)\.conv([12])", module_name)
    if not main:
        raise ValueError(f"无法识别 ResNet18 Conv：{module_name}")
    stage = int(main.group(1))
    block = int(main.group(2))
    conv_index = int(main.group(3))
    return (
        f"main_conv{conv_index}",
        stage,
        block,
        conv_index,
        f"layer{stage}.{block}.bn{conv_index}",
    )


def build_conv_specs(
    public: nn.Module,
    victim: nn.Module,
) -> tuple[list[ConvSpec], dict[str, nn.Conv2d], dict[str, nn.Conv2d]]:
    public_convs = {
        name: module
        for name, module in public.named_modules()
        if isinstance(module, nn.Conv2d)
    }
    victim_convs = {
        name: module
        for name, module in victim.named_modules()
        if isinstance(module, nn.Conv2d)
    }
    if set(public_convs) != set(victim_convs):
        raise RuntimeError("公开模型与受害者模型的 Conv 集合不一致。")
    specs = []
    for index, (module_name, victim_conv) in enumerate(victim_convs.items()):
        public_conv = public_convs[module_name]
        if (
            public_conv.weight.shape != victim_conv.weight.shape
            or public_conv.stride != victim_conv.stride
            or public_conv.padding != victim_conv.padding
            or public_conv.dilation != victim_conv.dilation
            or public_conv.groups != victim_conv.groups
        ):
            raise RuntimeError(f"{module_name} 的公开/受害者 Conv 结构不一致。")
        if public_conv.bias is not None or victim_conv.bias is not None:
            raise RuntimeError(
                f"{module_name} 含 bias，当前 weight-only 残差口径不适用。"
            )
        role, stage, block, conv_index, paired_bn = conv_role(module_name)
        kernel_height, kernel_width = pair(victim_conv.kernel_size)
        stride_height, stride_width = pair(victim_conv.stride)
        specs.append(
            ConvSpec(
                index=index,
                module_name=module_name,
                state_name=f"{module_name}.weight",
                role=role,
                stage=stage,
                block=block,
                conv_in_block=conv_index,
                paired_bn=paired_bn,
                in_channels=victim_conv.in_channels,
                out_channels=victim_conv.out_channels,
                kernel_height=kernel_height,
                kernel_width=kernel_width,
                stride_height=stride_height,
                stride_width=stride_width,
                groups=victim_conv.groups,
                weight_param_count=int(victim_conv.weight.numel()),
                filter_param_count=int(victim_conv.weight[0].numel()),
            )
        )
    if len(specs) != EXPECTED_CONV_COUNT:
        raise RuntimeError(
            f"Conv 数量为 {len(specs)}，期望 {EXPECTED_CONV_COUNT}。"
        )
    filter_count = sum(spec.out_channels for spec in specs)
    if filter_count != EXPECTED_FILTER_COUNT:
        raise RuntimeError(
            f"filter 数量为 {filter_count}，期望 {EXPECTED_FILTER_COUNT}。"
        )
    if not set(ENTRY_CONVS).issubset({spec.module_name for spec in specs}):
        raise RuntimeError("四个阶段入口块的 8 个卷积不完整。")
    return specs, public_convs, victim_convs


class ConvCapture:
    """捕获一次真实前向中各 Conv 的输入和 pre-BN 输出。"""

    def __init__(self, modules: dict[str, nn.Conv2d]):
        self.inputs: dict[str, torch.Tensor] = {}
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles = []
        for name, module in modules.items():
            self.handles.append(
                module.register_forward_pre_hook(self._make_pre_hook(name))
            )
            self.handles.append(
                module.register_forward_hook(self._make_forward_hook(name))
            )

    def _make_pre_hook(self, name: str):
        def hook(_module, inputs):
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise RuntimeError(f"{name} 的 Conv 输入无法识别。")
            self.inputs[name] = inputs[0].detach()

        return hook

    def _make_forward_hook(self, name: str):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                raise RuntimeError(f"{name} 的 Conv 输出无法识别。")
            self.outputs[name] = output.detach()

        return hook

    def clear(self) -> None:
        self.inputs.clear()
        self.outputs.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def make_accumulator(channels: int) -> dict[str, object]:
    accumulator: dict[str, object] = {
        "element_count_per_filter": 0,
        "sample_count": 0,
        "output_height": 0,
        "output_width": 0,
    }
    for component in COMPONENTS:
        accumulator[f"{component}_signed_sum"] = torch.zeros(
            channels, dtype=torch.float64
        )
        accumulator[f"{component}_abs_sum"] = torch.zeros(
            channels, dtype=torch.float64
        )
        accumulator[f"{component}_square_sum"] = torch.zeros(
            channels, dtype=torch.float64
        )
    for effect in DIRECT_EFFECTS:
        accumulator[f"{effect}_abs_sum"] = torch.zeros(
            channels, dtype=torch.float64
        )
    return accumulator


def accumulate_tensor(
    accumulator: dict[str, object],
    prefix: str,
    value: torch.Tensor,
) -> None:
    dimensions = (0, 2, 3)
    signed = value.sum(dim=dimensions).double().cpu()
    absolute = value.abs().sum(dim=dimensions).double().cpu()
    square = value.square().sum(dim=dimensions).double().cpu()
    accumulator[f"{prefix}_signed_sum"] += signed
    accumulator[f"{prefix}_abs_sum"] += absolute
    accumulator[f"{prefix}_square_sum"] += square


def accumulate_abs_effect(
    accumulator: dict[str, object],
    prefix: str,
    value: torch.Tensor,
) -> None:
    accumulator[f"{prefix}_abs_sum"] += (
        value.abs().sum(dim=(0, 2, 3)).double().cpu()
    )


def crossed_conv(
    conv: nn.Conv2d,
    inputs: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return functional.conv2d(
        inputs,
        weight,
        bias=None,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
    )


@torch.no_grad()
def collect_residuals(
    public: nn.Module,
    victim: nn.Module,
    public_convs: dict[str, nn.Conv2d],
    victim_convs: dict[str, nn.Conv2d],
    specs: list[ConvSpec],
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, dict[str, object]], dict[str, float | int]]:
    accumulators = {
        spec.module_name: make_accumulator(spec.out_channels)
        for spec in specs
    }
    public_capture = ConvCapture(public_convs)
    victim_capture = ConvCapture(victim_convs)
    sample_count = 0
    completeness_max_abs_error = 0.0
    public_forward_max_abs_error = 0.0
    victim_forward_max_abs_error = 0.0
    try:
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            public_capture.clear()
            victim_capture.clear()
            public(images)
            victim(images)
            expected_names = set(public_convs)
            if (
                set(public_capture.inputs) != expected_names
                or set(public_capture.outputs) != expected_names
                or set(victim_capture.inputs) != expected_names
                or set(victim_capture.outputs) != expected_names
            ):
                raise RuntimeError("真实前向没有完整捕获全部 Conv 输入输出。")

            for spec in specs:
                name = spec.module_name
                public_conv = public_convs[name]
                victim_conv = victim_convs[name]
                public_input = public_capture.inputs[name]
                victim_input = victim_capture.inputs[name]
                z_pp = public_capture.outputs[name]
                z_vv = victim_capture.outputs[name]
                z_pv = crossed_conv(
                    victim_conv,
                    public_input,
                    victim_conv.weight,
                )
                z_vp = crossed_conv(
                    public_conv,
                    victim_input,
                    public_conv.weight,
                )
                recomputed_pp = crossed_conv(
                    public_conv,
                    public_input,
                    public_conv.weight,
                )
                recomputed_vv = crossed_conv(
                    victim_conv,
                    victim_input,
                    victim_conv.weight,
                )
                public_forward_max_abs_error = max(
                    public_forward_max_abs_error,
                    float((recomputed_pp - z_pp).abs().max().item()),
                )
                victim_forward_max_abs_error = max(
                    victim_forward_max_abs_error,
                    float((recomputed_vv - z_vv).abs().max().item()),
                )

                weight_public_input = z_pv - z_pp
                weight_victim_input = z_vv - z_vp
                input_public_weight = z_vp - z_pp
                input_victim_weight = z_vv - z_pv
                weight_residual = 0.5 * (
                    weight_public_input + weight_victim_input
                )
                input_residual = 0.5 * (
                    input_public_weight + input_victim_weight
                )
                total_residual = z_vv - z_pp
                interaction = z_vv - z_vp - z_pv + z_pp
                completeness_max_abs_error = max(
                    completeness_max_abs_error,
                    float(
                        (
                            weight_residual
                            + input_residual
                            - total_residual
                        )
                        .abs()
                        .max()
                        .item()
                    ),
                )

                accumulator = accumulators[name]
                batch_size, channels, height, width = total_residual.shape
                if channels != spec.out_channels:
                    raise RuntimeError(f"{name} 的输出通道数量变化。")
                if int(accumulator["output_height"]) == 0:
                    accumulator["output_height"] = height
                    accumulator["output_width"] = width
                elif (
                    int(accumulator["output_height"]) != height
                    or int(accumulator["output_width"]) != width
                ):
                    raise RuntimeError(f"{name} 的输出空间大小在批次间变化。")
                accumulator["sample_count"] += batch_size
                accumulator["element_count_per_filter"] += (
                    batch_size * height * width
                )
                accumulate_tensor(accumulator, "weight", weight_residual)
                accumulate_tensor(accumulator, "input", input_residual)
                accumulate_tensor(accumulator, "total", total_residual)
                accumulate_tensor(accumulator, "interaction", interaction)
                accumulate_abs_effect(
                    accumulator,
                    "weight_effect_public_input",
                    weight_public_input,
                )
                accumulate_abs_effect(
                    accumulator,
                    "weight_effect_victim_input",
                    weight_victim_input,
                )
                accumulate_abs_effect(
                    accumulator,
                    "input_effect_public_weight",
                    input_public_weight,
                )
                accumulate_abs_effect(
                    accumulator,
                    "input_effect_victim_weight",
                    input_victim_weight,
                )
            sample_count += images.size(0)
    finally:
        public_capture.close()
        victim_capture.close()
    if sample_count != QUERY_COUNT:
        raise RuntimeError(
            f"残差统计样本数为 {sample_count}，期望 {QUERY_COUNT}。"
        )
    for name, accumulator in accumulators.items():
        if int(accumulator["sample_count"]) != sample_count:
            raise RuntimeError(f"{name} 的样本累计数量不一致。")
    diagnostics: dict[str, float | int] = {
        "sample_count": sample_count,
        "completeness_max_abs_error": completeness_max_abs_error,
        "public_forward_max_abs_error": public_forward_max_abs_error,
        "victim_forward_max_abs_error": victim_forward_max_abs_error,
    }
    return accumulators, diagnostics


def build_filter_rows(
    specs: list[ConvSpec],
    accumulators: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for spec in specs:
        accumulator = accumulators[spec.module_name]
        count = int(accumulator["element_count_per_filter"])
        if count <= 0:
            raise RuntimeError(f"{spec.module_name} 没有有效累计元素。")
        for channel in range(spec.out_channels):
            row: dict[str, object] = {
                **asdict(spec),
                "filter_index": channel,
                "sample_count": int(accumulator["sample_count"]),
                "output_height": int(accumulator["output_height"]),
                "output_width": int(accumulator["output_width"]),
                "element_count": count,
            }
            for component in COMPONENTS:
                signed_sum = accumulator[f"{component}_signed_sum"]
                absolute_sum = accumulator[f"{component}_abs_sum"]
                square_sum = accumulator[f"{component}_square_sum"]
                row[f"{component}_residual_signed_mean"] = float(
                    signed_sum[channel].item() / count
                )
                row[f"{component}_residual_abs_mean"] = float(
                    absolute_sum[channel].item() / count
                )
                row[f"{component}_residual_rms"] = math.sqrt(
                    float(square_sum[channel].item() / count)
                )
            for effect in DIRECT_EFFECTS:
                effect_sum = accumulator[f"{effect}_abs_sum"]
                row[f"{effect}_abs_mean"] = float(
                    effect_sum[channel].item() / count
                )
            rows.append(row)
    if len(rows) != EXPECTED_FILTER_COUNT:
        raise RuntimeError(
            f"逐 filter 结果为 {len(rows)} 行，期望 {EXPECTED_FILTER_COUNT}。"
        )
    return rows


def build_unit_rows(
    specs: list[ConvSpec],
    filter_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in filter_rows:
        grouped.setdefault(str(row["module_name"]), []).append(row)
    rows = []
    for spec in specs:
        filters = grouped[spec.module_name]
        if len(filters) != spec.out_channels:
            raise RuntimeError(f"{spec.module_name} 的 filter 聚合数量不一致。")
        row: dict[str, object] = {
            **asdict(spec),
            "filter_count": len(filters),
            "sample_count": filters[0]["sample_count"],
            "output_height": filters[0]["output_height"],
            "output_width": filters[0]["output_width"],
        }
        for component in COMPONENTS:
            absolute_values = [
                float(item[f"{component}_residual_abs_mean"])
                for item in filters
            ]
            square_means = [
                float(item[f"{component}_residual_rms"]) ** 2
                for item in filters
            ]
            row[f"{component}_residual_filter_sum"] = sum(absolute_values)
            row[f"{component}_residual_filter_mean"] = (
                sum(absolute_values) / len(absolute_values)
            )
            row[f"{component}_residual_rms"] = math.sqrt(
                sum(square_means) / len(square_means)
            )
        for effect in DIRECT_EFFECTS:
            values = [
                float(item[f"{effect}_abs_mean"]) for item in filters
            ]
            row[f"{effect}_filter_sum"] = sum(values)
            row[f"{effect}_filter_mean"] = sum(values) / len(values)
        rows.append(row)
    if len(rows) != EXPECTED_CONV_COUNT:
        raise RuntimeError(
            f"unit 结果为 {len(rows)} 行，期望 {EXPECTED_CONV_COUNT}。"
        )
    return rows


def write_tsv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_entry_residuals(
    path: Path,
    entry_rows: list[dict[str, object]],
) -> None:
    labels = [
        str(row["module_name"]).replace("layer", "L").replace(".conv", " C")
        for row in entry_rows
    ]
    specifications = (
        ("weight_residual_filter_sum", "Weight residual"),
        ("input_residual_filter_sum", "Input residual"),
        ("total_residual_filter_sum", "Total output residual"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.2))
    colors = ("#228833", "#4477AA", "#CC6677")
    for axis, (field, title), color in zip(axes, specifications, colors):
        values = [float(row[field]) for row in entry_rows]
        bars = axis.bar(labels, values, color=color)
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=35)
        axis.set_ylabel("Sum of per-filter mean absolute residual")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    figure.suptitle("Four-way crossed-forward residuals on 400 query-train images")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers 不能小于 0。")
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0。")
    device = rec.resolve_device(args.device)
    victim_path = (
        REPO_ROOT / "weights" / "MS" / "victim" / MODEL / DATASET / "best.pth"
    )
    public_path = (
        REPO_ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    )
    split_path = REPO_ROOT / "dataset" / "MS" / DATASET / "splits.tsv"
    for path in (victim_path, public_path, split_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    rec.configure_reproducibility(SEED, deterministic=True)
    victim, victim_metadata = rec.build_victim(
        MODEL,
        NUM_CLASSES,
        victim_path,
    )
    public = rec.build_public_model()
    specs, public_convs, victim_convs = build_conv_specs(public, victim)
    query_indices, query_partition = rec.discovery_indices()
    if len(query_indices) != QUERY_COUNT:
        raise RuntimeError(
            f"query-train 数量为 {len(query_indices)}，期望 {QUERY_COUNT}。"
        )
    _, test_transform = rec.build_transforms(DATASET)
    public_dataset = rec.build_public_split_dataset(
        DATASET,
        REPO_ROOT / "dataset" / "public",
        "train",
        test_transform,
    )
    if min(query_indices) < 0 or max(query_indices) >= len(public_dataset):
        raise ValueError("query_pool_ms source index 越界。")

    print(
        f"[RESIDUAL] device={device} query={len(query_indices)} "
        f"conv={len(specs)} filters={sum(s.out_channels for s in specs)} "
        f"query_hash={rec.digest_indices(query_indices)}",
        flush=True,
    )
    print(
        "[RESIDUAL] entry convs: " + ",".join(ENTRY_CONVS),
        flush=True,
    )
    if args.dry_run:
        print("[RESIDUAL] dry-run 通过：未运行前向统计，也未写入 temp/output。")
        return 0

    existing = [path for path in OUTPUT_PATHS if path.exists()]
    if existing and not args.overwrite:
        paths = ", ".join(str(path.relative_to(REPO_ROOT)) for path in existing)
        raise FileExistsError(f"临时产物已存在：{paths}；请使用 --overwrite。")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in existing:
            path.unlink()

    loader = DataLoader(
        Subset(public_dataset, query_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=rec.seed_worker,
        generator=rec.build_generator(SEED, offset=410),
    )
    public = public.to(device).eval()
    victim = victim.to(device).eval()
    accumulators, diagnostics = collect_residuals(
        public,
        victim,
        public_convs,
        victim_convs,
        specs,
        loader,
        device,
    )
    filter_rows = build_filter_rows(specs, accumulators)
    unit_rows = build_unit_rows(specs, filter_rows)
    unit_by_name = {str(row["module_name"]): row for row in unit_rows}
    entry_rows = [unit_by_name[name] for name in ENTRY_CONVS]
    if len(entry_rows) != 8:
        raise RuntimeError("四个阶段入口块的 8 个卷积摘要不完整。")

    write_tsv(FILTER_PATH, filter_rows, list(filter_rows[0]))
    write_tsv(UNIT_PATH, unit_rows, list(unit_rows[0]))
    write_tsv(ENTRY_PATH, entry_rows, list(entry_rows[0]))
    plot_entry_residuals(PLOT_PATH, entry_rows)

    top_filters = sorted(
        filter_rows,
        key=lambda row: (
            -float(row["weight_residual_abs_mean"]),
            int(row["index"]),
            int(row["filter_index"]),
        ),
    )[:20]
    top_units = sorted(
        unit_rows,
        key=lambda row: (
            -float(row["weight_residual_filter_sum"]),
            int(row["index"]),
        ),
    )
    payload = {
        "schema_version": 1,
        "experiment": "temp_four_way_filter_weight_residual",
        "scope": "temporary_forward_only_xai_diagnostic",
        "model": MODEL,
        "dataset": DATASET,
        "seed": SEED,
        "definition": {
            "unit": "one Conv2d weight tensor",
            "filter": "one Conv2d output-channel weight slice",
            "measurement_boundary": "Conv2d pre-BN output",
            "four_outputs": {
                "PP": "public weight on public-model layer input",
                "PV": "victim weight on public-model layer input",
                "VP": "public weight on victim-model layer input",
                "VV": "victim weight on victim-model layer input",
            },
            "weight_residual": "0.5 * ((PV - PP) + (VV - VP))",
            "input_residual": "0.5 * ((VP - PP) + (VV - PV))",
            "total_residual": "VV - PP",
            "filter_scalar": (
                "mean absolute residual over query images and spatial positions"
            ),
            "unit_scalar": "sum of filter scalar values",
            "bn_included": False,
        },
        "protocol": {
            "input_split": "query_pool_ms/query_train",
            "input_source_split": "official_train",
            "input_count": QUERY_COUNT,
            "query_partition": query_partition,
            "input_transform": "test",
            "query_labels_consumed": False,
            "query_posteriors_consumed": False,
            "surrogate_or_ms_metrics_consumed": False,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "model_mode": "eval",
            "gradient_enabled": False,
        },
        "randomization": {
            "seed": SEED,
            "shuffle": False,
            "dataloader_generator_seed": SEED,
            "dataloader_generator_offset": 410,
        },
        "graph": {
            "conv_count": len(specs),
            "filter_count": len(filter_rows),
            "conv_specs": [asdict(spec) for spec in specs],
            "entry_convs": list(ENTRY_CONVS),
        },
        "inputs": {
            "query_source_indices": query_indices,
            "query_source_indices_sha256": rec.digest_indices(query_indices),
            "splits": str(split_path.relative_to(REPO_ROOT)),
            "splits_sha256": rec.sha256_file(split_path),
            "victim_checkpoint": str(victim_path.relative_to(REPO_ROOT)),
            "victim_checkpoint_sha256": rec.sha256_file(victim_path),
            "victim_checkpoint_epoch": victim_metadata.get("epoch"),
            "public_checkpoint": str(public_path.relative_to(REPO_ROOT)),
            "public_checkpoint_sha256": rec.sha256_file(public_path),
        },
        "diagnostics": diagnostics,
        "entry_results": entry_rows,
        "unit_ranking_by_weight_residual_sum": top_units,
        "top20_filters_by_weight_residual": top_filters,
        "outputs": {
            "filters": str(FILTER_PATH.relative_to(REPO_ROOT)),
            "units": str(UNIT_PATH.relative_to(REPO_ROOT)),
            "entry": str(ENTRY_PATH.relative_to(REPO_ROOT)),
            "plot": str(PLOT_PATH.relative_to(REPO_ROOT)),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    METRICS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[RESIDUAL] complete "
        f"max_error={diagnostics['completeness_max_abs_error']:.8g}",
        flush=True,
    )
    for row in entry_rows:
        print(
            f"[RESIDUAL] {row['module_name']} "
            f"weight_sum={row['weight_residual_filter_sum']:.6f} "
            f"weight_mean={row['weight_residual_filter_mean']:.6f} "
            f"input_sum={row['input_residual_filter_sum']:.6f} "
            f"total_sum={row['total_residual_filter_sum']:.6f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
