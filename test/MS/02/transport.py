#!/usr/bin/env python3
"""计算 ResNet18 Conv weight 与 BN affine 的任务特定表征传输分数。"""

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


EXPERIMENT = "02"
MODEL = "resnet18"
DATASET = "c100"
NUM_CLASSES = 100
QUERY_COUNT = 500
SEED = 42
BATCH_SIZE = 64
EXPECTED_CONV_COUNT = 20
EXPECTED_BN_COUNT = 20
DENOMINATOR_EPSILON = 1e-12
NUMERICAL_EIGENVALUE_RELATIVE_TOLERANCE = 1e-8
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
    "bias_state",
    "weight_shape",
    "input_shape",
    "output_shape",
    "parameter_count",
    "image_count",
    "observation_count",
    "mean_transport",
    "covariance_transport",
    "wasserstein2",
    "symmetric_second_moment",
    "rt_score",
)
TENSOR_FIELDS = (
    "rank",
    "overall_rank",
    "module",
    "weight_state",
    "input_shape",
    "output_shape",
    "parameter_count",
    "image_count",
    "mean_transport",
    "covariance_transport",
    "wasserstein2",
    "rt_score",
)
COMPARISON_FIELDS = (
    "module",
    "operator_type",
    "rt_rank",
    "rt_score",
    "cross_rank",
    "cross_abs_mean",
    "rank_delta_rt_minus_cross",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只处理第一个 batch 并核对算子输出，不写结果。",
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


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 Test01 对照表：{path}")
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file, delimiter="\t"))


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
    for name, public_module in public_conv.items():
        victim_module = victim_conv[name]
        geometry_fields = ("stride", "padding", "dilation", "groups")
        if public_module.weight.shape != victim_module.weight.shape or any(
            getattr(public_module, field) != getattr(victim_module, field)
            for field in geometry_fields
        ):
            raise ValueError(f"{name} 的 public/victim 卷积几何不一致。")
    for name, public_module in public_bn.items():
        victim_module = victim_bn[name]
        if (
            public_module.weight.shape != victim_module.weight.shape
            or public_module.bias.shape != victim_module.bias.shape
            or public_module.running_mean.shape != victim_module.running_mean.shape
            or public_module.running_var.shape != victim_module.running_var.shape
            or public_module.eps != victim_module.eps
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
        raise ValueError("BN affine 表征传输要求 track_running_stats=True。")
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


def apply_affine(
    normalized_input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    return apply_gamma(normalized_input, gamma) + beta.reshape(1, -1, 1, 1)


def initialize_statistics(
    public_conv: dict[str, torch.nn.Conv2d],
    public_bn: dict[str, torch.nn.BatchNorm2d],
    device: torch.device,
) -> dict[str, dict[str, object]]:
    statistics: dict[str, dict[str, object]] = {}
    for operator_type, modules in (
        ("conv_weight", public_conv),
        ("bn_affine", public_bn),
    ):
        for name, module in modules.items():
            channels = (
                module.out_channels
                if isinstance(module, torch.nn.Conv2d)
                else module.num_features
            )
            statistics[name] = {
                "operator_type": operator_type,
                "weight_state": f"{name}.weight",
                "bias_state": (
                    "" if isinstance(module, torch.nn.Conv2d) else f"{name}.bias"
                ),
                "weight_shape": (
                    "×".join(str(value) for value in module.weight.shape)
                    if isinstance(module, torch.nn.Conv2d)
                    else ";".join(
                        (
                            "gamma="
                            + "×".join(str(value) for value in module.weight.shape),
                            "beta="
                            + "×".join(str(value) for value in module.bias.shape),
                        )
                    )
                ),
                "parameter_count": (
                    module.weight.numel()
                    if isinstance(module, torch.nn.Conv2d)
                    else module.weight.numel() + module.bias.numel()
                ),
                "image_count": 0,
                "observation_count": 0,
                "input_shape": None,
                "output_shape": None,
                "public_sum": torch.zeros(channels, dtype=torch.float64, device=device),
                "victim_sum": torch.zeros(channels, dtype=torch.float64, device=device),
                "public_second": torch.zeros(
                    (channels, channels), dtype=torch.float64, device=device
                ),
                "victim_second": torch.zeros(
                    (channels, channels), dtype=torch.float64, device=device
                ),
            }
    if len(statistics) != EXPECTED_CONV_COUNT + EXPECTED_BN_COUNT:
        raise RuntimeError("Conv 与 BN affine 模块名称发生冲突。")
    return statistics


def activation_matrix(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError("表征传输只接受 N×C×H×W 张量。")
    return (
        tensor.permute(0, 2, 3, 1)
        .reshape(-1, tensor.size(1))
        .to(dtype=torch.float64)
    )


def update_statistics(
    state: dict[str, object],
    public_output: torch.Tensor,
    victim_output: torch.Tensor,
    operator_input: torch.Tensor,
) -> None:
    if public_output.shape != victim_output.shape:
        raise ValueError("public/victim 算子输出形状不一致。")
    public_matrix = activation_matrix(public_output)
    victim_matrix = activation_matrix(victim_output)
    observations = public_matrix.size(0)
    state["public_sum"].add_(public_matrix.sum(dim=0))
    state["victim_sum"].add_(victim_matrix.sum(dim=0))
    state["public_second"].add_(public_matrix.T @ public_matrix)
    state["victim_second"].add_(victim_matrix.T @ victim_matrix)
    state["image_count"] = int(state["image_count"]) + public_output.size(0)
    state["observation_count"] = int(state["observation_count"]) + observations

    input_shape = shape_text(operator_input)
    output_shape = shape_text(public_output)
    if state["input_shape"] not in (None, input_shape):
        raise ValueError("同一算子的输入形状在 batch 间变化。")
    if state["output_shape"] not in (None, output_shape):
        raise ValueError("同一算子的输出形状在 batch 间变化。")
    state["input_shape"] = input_shape
    state["output_shape"] = output_shape


def moments(
    value_sum: torch.Tensor,
    second_sum: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if count <= 0:
        raise ValueError("表征统计样本数必须大于零。")
    mean = value_sum / count
    covariance = second_sum / count - torch.outer(mean, mean)
    covariance = (covariance + covariance.T) * 0.5
    return mean, covariance


def psd_sqrt(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, float, int]:
    matrix = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    min_eigenvalue = float(eigenvalues.min().item())
    eigenvalue_scale = max(
        abs(float(torch.trace(matrix).item())) / matrix.size(0),
        DENOMINATOR_EPSILON,
    )
    if (
        min_eigenvalue
        < -NUMERICAL_EIGENVALUE_RELATIVE_TOLERANCE * eigenvalue_scale
    ):
        raise RuntimeError(
            "半正定矩阵出现超出浮点容差的负特征值："
            f"min={min_eigenvalue} scale={eigenvalue_scale}"
        )
    negative_count = int((eigenvalues < 0).sum().item())
    clipped = eigenvalues.clamp_min(0).sqrt()
    square_root = (eigenvectors * clipped.unsqueeze(0)) @ eigenvectors.T
    return square_root, min_eigenvalue, negative_count


def representation_transport(
    public_mean: torch.Tensor,
    public_covariance: torch.Tensor,
    victim_mean: torch.Tensor,
    victim_covariance: torch.Tensor,
) -> tuple[dict[str, float], dict[str, float]]:
    public_sqrt, public_min_eigenvalue, public_negative_count = psd_sqrt(
        public_covariance
    )
    victim_eigenvalues = torch.linalg.eigvalsh(victim_covariance)
    victim_min_eigenvalue = float(victim_eigenvalues.min().item())
    victim_eigenvalue_scale = max(
        abs(float(torch.trace(victim_covariance).item()))
        / victim_covariance.size(0),
        DENOMINATOR_EPSILON,
    )
    if (
        victim_min_eigenvalue
        < -NUMERICAL_EIGENVALUE_RELATIVE_TOLERANCE
        * victim_eigenvalue_scale
    ):
        raise RuntimeError(
            "victim 协方差出现超出浮点容差的负特征值："
            f"min={victim_min_eigenvalue} scale={victim_eigenvalue_scale}"
        )
    victim_negative_count = int((victim_eigenvalues < 0).sum().item())
    middle = public_sqrt @ victim_covariance @ public_sqrt
    middle_sqrt, middle_min_eigenvalue, middle_negative_count = psd_sqrt(middle)

    mean_transport = float((victim_mean - public_mean).square().sum().item())
    covariance_transport_raw = float(
        (
            torch.trace(public_covariance)
            + torch.trace(victim_covariance)
            - 2.0 * torch.trace(middle_sqrt)
        ).item()
    )
    scale = max(
        float(
            (
                torch.trace(public_covariance)
                + torch.trace(victim_covariance)
            ).abs().item()
        ),
        1.0,
    )
    if covariance_transport_raw < -1e-7 * scale:
        raise RuntimeError(
            f"Wasserstein 协方差项出现过大负值：{covariance_transport_raw}"
        )
    covariance_transport = max(covariance_transport_raw, 0.0)
    wasserstein2 = mean_transport + covariance_transport
    public_energy = float(
        (public_mean.square().sum() + torch.trace(public_covariance)).item()
    )
    victim_energy = float(
        (victim_mean.square().sum() + torch.trace(victim_covariance)).item()
    )
    symmetric_second_moment = 0.5 * (public_energy + victim_energy)
    rt_score = wasserstein2 / (symmetric_second_moment + DENOMINATOR_EPSILON)
    values = {
        "mean_transport": mean_transport,
        "covariance_transport": covariance_transport,
        "wasserstein2": wasserstein2,
        "symmetric_second_moment": symmetric_second_moment,
        "rt_score": rt_score,
    }
    diagnostics = {
        "public_covariance_min_eigenvalue": public_min_eigenvalue,
        "victim_covariance_min_eigenvalue": victim_min_eigenvalue,
        "middle_min_eigenvalue": middle_min_eigenvalue,
        "public_covariance_negative_eigenvalue_count": public_negative_count,
        "victim_covariance_negative_eigenvalue_count": victim_negative_count,
        "middle_negative_eigenvalue_count": middle_negative_count,
    }
    return values, diagnostics


def transport_self_check(device: torch.device) -> dict[str, float]:
    public_mean = torch.tensor([0.25, -0.5, 0.75], dtype=torch.float64, device=device)
    victim_mean = torch.tensor([-0.25, 0.0, 1.0], dtype=torch.float64, device=device)
    covariance = torch.tensor(
        [
            [1.0, 0.2, 0.0],
            [0.2, 0.8, 0.1],
            [0.0, 0.1, 0.5],
        ],
        dtype=torch.float64,
        device=device,
    )
    identity_values, _ = representation_transport(
        public_mean,
        covariance,
        public_mean,
        covariance,
    )
    mean_only_values, _ = representation_transport(
        public_mean,
        covariance,
        victim_mean,
        covariance,
    )
    expected_mean_transport = float(
        (victim_mean - public_mean).square().sum().item()
    )
    identity_error = abs(identity_values["wasserstein2"])
    mean_only_error = abs(
        mean_only_values["wasserstein2"] - expected_mean_transport
    )
    if identity_error > 1e-10 or mean_only_error > 1e-10:
        raise RuntimeError(
            "Wasserstein 自检未通过："
            f"identity_error={identity_error} mean_only_error={mean_only_error}"
        )
    return {
        "wasserstein_identity_error": identity_error,
        "wasserstein_mean_only_error": mean_only_error,
    }


def build_rows(
    statistics: dict[str, dict[str, object]],
    expected_image_count: int,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, float]] = []
    for module, state in statistics.items():
        image_count = int(state["image_count"])
        observation_count = int(state["observation_count"])
        if image_count != expected_image_count:
            raise ValueError(
                f"{module} 只统计了 {image_count}/{expected_image_count} 张图片。"
            )
        public_mean, public_covariance = moments(
            state["public_sum"], state["public_second"], observation_count
        )
        victim_mean, victim_covariance = moments(
            state["victim_sum"], state["victim_second"], observation_count
        )
        values, current_diagnostics = representation_transport(
            public_mean,
            public_covariance,
            victim_mean,
            victim_covariance,
        )
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError(f"{module} 的表征传输结果不是有限值。")
        diagnostics.append(current_diagnostics)
        rows.append(
            {
                "rank": 0,
                "type_rank": 0,
                "operator_type": state["operator_type"],
                "module": module,
                "weight_state": state["weight_state"],
                "bias_state": state["bias_state"],
                "weight_shape": state["weight_shape"],
                "input_shape": state["input_shape"],
                "output_shape": state["output_shape"],
                "parameter_count": state["parameter_count"],
                "image_count": image_count,
                "observation_count": observation_count,
                **values,
            }
        )
    rows.sort(key=lambda row: (-float(row["rt_score"]), str(row["module"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    for operator_type in ("conv_weight", "bn_affine"):
        typed = [row for row in rows if row["operator_type"] == operator_type]
        for type_rank, row in enumerate(typed, start=1):
            row["type_rank"] = type_rank

    diagnostic_summary = {
        "min_public_covariance_eigenvalue": min(
            item["public_covariance_min_eigenvalue"] for item in diagnostics
        ),
        "min_victim_covariance_eigenvalue": min(
            item["victim_covariance_min_eigenvalue"] for item in diagnostics
        ),
        "min_middle_eigenvalue": min(
            item["middle_min_eigenvalue"] for item in diagnostics
        ),
        "max_public_negative_eigenvalue_count": max(
            item["public_covariance_negative_eigenvalue_count"]
            for item in diagnostics
        ),
        "max_victim_negative_eigenvalue_count": max(
            item["victim_covariance_negative_eigenvalue_count"]
            for item in diagnostics
        ),
        "max_middle_negative_eigenvalue_count": max(
            item["middle_negative_eigenvalue_count"] for item in diagnostics
        ),
    }
    return rows, diagnostic_summary


def extract_main_conv(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_module = {str(row["module"]): row for row in rows}
    if any(name not in by_module for name in MAIN_CONV_NAMES):
        raise ValueError("40 项 RT 排名缺少 BasicBlock 主分支卷积。")
    selected = sorted(
        (by_module[name] for name in MAIN_CONV_NAMES),
        key=lambda row: int(row["rank"]),
    )
    return [
        {
            "rank": rank,
            "overall_rank": row["rank"],
            "module": row["module"],
            "weight_state": row["weight_state"],
            "input_shape": row["input_shape"],
            "output_shape": row["output_shape"],
            "parameter_count": row["parameter_count"],
            "image_count": row["image_count"],
            "mean_transport": row["mean_transport"],
            "covariance_transport": row["covariance_transport"],
            "wasserstein2": row["wasserstein2"],
            "rt_score": row["rt_score"],
        }
        for rank, row in enumerate(selected, start=1)
    ]


def correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("相关性计算的两个序列长度不一致或为空。")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0 or right_scale == 0:
        raise ValueError("相关性计算遇到常数序列。")
    return numerator / (left_scale * right_scale)


def kendall_tau(left: list[int], right: list[int]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Kendall 序列长度不一致或样本过少。")
    concordant = 0
    discordant = 0
    for first in range(len(left) - 1):
        for second in range(first + 1, len(left)):
            product = (left[first] - left[second]) * (
                right[first] - right[second]
            )
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    pairs = concordant + discordant
    if pairs == 0:
        raise ValueError("Kendall 序列没有可比较数对。")
    return (concordant - discordant) / pairs


def build_comparison(
    rows: list[dict[str, object]],
    cross_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    cross_rows = read_tsv(cross_path)
    if len(cross_rows) != EXPECTED_CONV_COUNT + EXPECTED_BN_COUNT:
        raise ValueError("Test01 对照表不是 40 项。")
    cross_by_module = {row["module"]: row for row in cross_rows}
    if set(cross_by_module) != {str(row["module"]) for row in rows}:
        raise ValueError("Test01/Test02 的 40 个候选模块集合不一致。")
    cross_order = sorted(
        cross_rows,
        key=lambda row: (
            -abs(float(row["cross_abs_mean"])),
            str(row["module"]),
        ),
    )
    cross_rank_by_module = {
        str(row["module"]): rank
        for rank, row in enumerate(cross_order, start=1)
    }
    comparison_rows = []
    for row in rows:
        cross = cross_by_module[str(row["module"])]
        if cross["operator_type"] != row["operator_type"]:
            raise ValueError(f"{row['module']} 在 Test01/Test02 的算子类型不一致。")
        comparison_rows.append(
            {
                "module": row["module"],
                "operator_type": row["operator_type"],
                "rt_rank": row["rank"],
                "rt_score": row["rt_score"],
                "cross_rank": cross_rank_by_module[str(row["module"])],
                "cross_abs_mean": float(cross["cross_abs_mean"]),
                "rank_delta_rt_minus_cross": int(row["rank"])
                - cross_rank_by_module[str(row["module"])],
            }
        )
    rt_ranks = [int(row["rt_rank"]) for row in comparison_rows]
    cross_ranks = [int(row["cross_rank"]) for row in comparison_rows]
    statistics: dict[str, object] = {
        "spearman_rank_correlation": correlation(
            [float(value) for value in rt_ranks],
            [float(value) for value in cross_ranks],
        ),
        "kendall_rank_correlation": kendall_tau(rt_ranks, cross_ranks),
        "top_overlap": {},
    }
    for top_k in (5, 10, 20):
        rt_top = {
            str(row["module"])
            for row in comparison_rows
            if int(row["rt_rank"]) <= top_k
        }
        cross_top = {
            str(row["module"])
            for row in comparison_rows
            if int(row["cross_rank"]) <= top_k
        }
        statistics["top_overlap"][str(top_k)] = {
            "count": len(rt_top & cross_top),
            "fraction": len(rt_top & cross_top) / top_k,
            "modules": sorted(rt_top & cross_top),
        }
    return comparison_rows, statistics


def plot_weights(path: Path, rows: list[dict[str, object]]) -> None:
    ordered = list(reversed(rows))
    labels = [
        f"{row['module']}  [{row['operator_type']}]"
        for row in ordered
    ]
    scores = [float(row["rt_score"]) for row in ordered]
    colors = [
        "#0b7fab" if row["operator_type"] == "conv_weight" else "#eda600"
        for row in ordered
    ]
    figure, axis = plt.subplots(figsize=(14, 15))
    axis.barh(range(len(ordered)), scores, color=colors)
    axis.set_yticks(range(len(ordered)), labels=labels, fontsize=8)
    axis.set_xlabel("Normalized Gaussian representation transport (RT)")
    axis.set_title(
        "ResNet18 task-specific representation transport\n"
        "same victim input; 500 fixed CIFAR-100 query images"
    )
    axis.grid(axis="x", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for index, score in enumerate(scores):
        axis.text(score, index, f" {score:.4f}", va="center", fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=230, bbox_inches="tight")
    plt.close(figure)


def plot_tensors(path: Path, rows: list[dict[str, object]]) -> None:
    ordered = list(reversed(rows))
    labels = [str(row["module"]) for row in ordered]
    scores = [float(row["rt_score"]) for row in ordered]
    colors = [
        "#0b7fab" if ".conv1" in str(row["module"]) else "#d96b36"
        for row in ordered
    ]
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.barh(range(len(ordered)), scores, color=colors)
    axis.set_yticks(range(len(ordered)), labels=labels, fontsize=9)
    axis.set_xlabel("Normalized Gaussian representation transport (RT)")
    axis.set_title(
        "Sixteen BasicBlock convolution weights\n"
        "extracted from the Test02 40-candidate ranking"
    )
    axis.grid(axis="x", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for index, score in enumerate(scores):
        axis.text(score, index, f" {score:.4f}", va="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=230, bbox_inches="tight")
    plt.close(figure)


def plot_comparison(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    figure, axis = plt.subplots(figsize=(9, 8))
    for operator_type, color, label in (
        ("conv_weight", "#0b7fab", "Conv weight"),
        ("bn_affine", "#eda600", "BN affine"),
    ):
        selected = [row for row in rows if row["operator_type"] == operator_type]
        axis.scatter(
            [int(row["cross_rank"]) for row in selected],
            [int(row["rt_rank"]) for row in selected],
            color=color,
            label=label,
            alpha=0.85,
            s=48,
        )
    axis.plot([1, 40], [1, 40], color="#666666", linestyle="--", linewidth=1)
    for row in rows:
        if int(row["rt_rank"]) <= 8 or int(row["cross_rank"]) <= 8:
            axis.annotate(
                str(row["module"]),
                (int(row["cross_rank"]), int(row["rt_rank"])),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
    axis.set_xlim(0, 41)
    axis.set_ylim(41, 0)
    axis.set_xlabel("Test01 cross-residual rank")
    axis.set_ylabel("Test02 representation-transport rank")
    axis.set_title("Test01 vs Test02 ranking")
    axis.grid(alpha=0.2)
    axis.legend()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=230, bbox_inches="tight")
    plt.close(figure)


def measure(
    *,
    loader: DataLoader,
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
    dict[str, float] | None,
    dict[str, float],
    int,
]:
    victim_modules = {**victim_conv, **victim_bn}
    victim_inputs: dict[str, torch.Tensor] = {}
    victim_outputs: dict[str, torch.Tensor] = {}
    handles = register_capture(victim_modules, victim_inputs, victim_outputs)
    statistics = initialize_statistics(public_conv, public_bn, device)
    correctness = {
        "captured_conv_count": float(len(public_conv)),
        "captured_bn_affine_count": float(len(public_bn)),
        "max_conv_victim_hook_error": 0.0,
        "max_bn_affine_victim_hook_error": 0.0,
    }
    processed = 0
    try:
        with torch.no_grad():
            for batch_index, (images, _labels) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                victim_inputs.clear()
                victim_outputs.clear()
                victim(images)
                expected_names = set(statistics)
                if (
                    set(victim_inputs) != expected_names
                    or set(victim_outputs) != expected_names
                ):
                    raise ValueError("victim forward hook 没有完整捕获 20 Conv 和 20 BN。")

                for name, public_module in public_conv.items():
                    victim_module = victim_conv[name]
                    victim_input = victim_inputs[name]
                    public_output = apply_conv(
                        victim_input, public_module, public_module.weight
                    )
                    victim_output = apply_conv(
                        victim_input, victim_module, victim_module.weight
                    )
                    if batch_index == 0:
                        correctness["max_conv_victim_hook_error"] = max(
                            correctness["max_conv_victim_hook_error"],
                            float(
                                (victim_output - victim_outputs[name])
                                .abs()
                                .max()
                                .item()
                            ),
                        )
                    update_statistics(
                        statistics[name],
                        public_output,
                        victim_output,
                        victim_input,
                    )

                for name, public_module in public_bn.items():
                    victim_module = victim_bn[name]
                    victim_input = victim_inputs[name]
                    normalized_input = normalize_bn_input(
                        victim_input, victim_module
                    )
                    public_output = apply_affine(
                        normalized_input,
                        public_module.weight,
                        public_module.bias,
                    )
                    victim_output = apply_affine(
                        normalized_input,
                        victim_module.weight,
                        victim_module.bias,
                    )
                    if batch_index == 0:
                        correctness["max_bn_affine_victim_hook_error"] = max(
                            correctness["max_bn_affine_victim_hook_error"],
                            float(
                                (victim_output - victim_outputs[name])
                                .abs()
                                .max()
                                .item()
                            ),
                        )
                    update_statistics(
                        statistics[name],
                        public_output,
                        victim_output,
                        normalized_input,
                    )

                processed += images.size(0)
                print(
                    f"[RT {batch_index + 1:02d}/{len(loader):02d}] "
                    f"processed={processed}/{QUERY_COUNT}",
                    flush=True,
                )
                if dry_run:
                    break
    finally:
        for handle in handles:
            handle.remove()

    tolerances = {
        "max_conv_victim_hook_error": 1e-6,
        "max_bn_affine_victim_hook_error": 2e-6,
    }
    if any(
        float(correctness[field]) > tolerance
        for field, tolerance in tolerances.items()
    ):
        raise RuntimeError(f"算子输出正确性检查未通过：{correctness}")
    if dry_run:
        return None, None, correctness, processed
    if processed != QUERY_COUNT:
        raise RuntimeError(f"只处理了 {processed}/{QUERY_COUNT} 张图片。")
    rows, numerical = build_rows(statistics, QUERY_COUNT)
    return rows, numerical, correctness, processed


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    device = resolve_device(args.device)
    configure_reproducibility(SEED, deterministic=True)
    transport_check = transport_self_check(device)

    output_root = ROOT / "results" / "test" / "MS" / EXPERIMENT
    output_root.mkdir(parents=True, exist_ok=True)
    victim_checkpoint = (
        ROOT / "weights" / "MS" / "victim" / MODEL / DATASET / "best.pth"
    )
    public_checkpoint = ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    cross_path = ROOT / "results" / "test" / "MS" / "01_cross" / "all.tsv"

    victim, victim_metadata = build_victim(
        MODEL, NUM_CLASSES, victim_checkpoint
    )
    public = imagenet_models.resnet18(num_classes=1000)
    imagenet_models.load_official_imagenet_weights(
        MODEL, public, str(public_checkpoint), strict=True
    )
    public_conv, public_bn = select_modules(public)
    victim_conv, victim_bn = select_modules(victim)
    check_module_pairs(public_conv, victim_conv, public_bn, victim_bn)

    _, test_transform = build_transforms(DATASET)
    public_dataset = build_public_split_dataset(
        DATASET,
        ROOT / "dataset" / "public",
        "train",
        test_transform,
    )
    query_indices = read_query_indices(ROOT / "dataset" / "MS", DATASET)[
        :QUERY_COUNT
    ]
    if len(query_indices) != QUERY_COUNT or len(set(query_indices)) != QUERY_COUNT:
        raise ValueError("没有得到固定且不重复的 500 张 query。")
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
    rows, numerical, correctness, processed = measure(
        loader=loader,
        public=public,
        victim=victim,
        public_conv=public_conv,
        victim_conv=victim_conv,
        public_bn=public_bn,
        victim_bn=victim_bn,
        device=device,
        dry_run=args.dry_run,
    )
    correctness.update(transport_check)
    if args.dry_run:
        print(
            "[INFO] Test02 dry-run 通过："
            f"conv=20 bn_affine=20 batch={processed} correctness={correctness}"
        )
        return 0

    assert rows is not None
    assert numerical is not None
    conv_rows = [row for row in rows if row["operator_type"] == "conv_weight"]
    bn_rows = [row for row in rows if row["operator_type"] == "bn_affine"]
    main_rows = extract_main_conv(rows)
    comparison_rows, comparison_statistics = build_comparison(rows, cross_path)

    weights_path = output_root / "weights.tsv"
    conv_path = output_root / "weights_conv.tsv"
    bn_path = output_root / "weights_bn.tsv"
    tensors_path = output_root / "tensors.tsv"
    comparison_path = output_root / "comparison.tsv"
    weights_plot_path = output_root / "weights.png"
    tensors_plot_path = output_root / "tensors.png"
    comparison_plot_path = output_root / "comparison.png"
    metrics_path = output_root / "metrics.json"
    write_tsv(weights_path, rows, WEIGHT_FIELDS)
    write_tsv(conv_path, conv_rows, WEIGHT_FIELDS)
    write_tsv(bn_path, bn_rows, WEIGHT_FIELDS)
    write_tsv(tensors_path, main_rows, TENSOR_FIELDS)
    write_tsv(comparison_path, comparison_rows, COMPARISON_FIELDS)
    plot_weights(weights_plot_path, rows)
    plot_tensors(tensors_plot_path, main_rows)
    plot_comparison(comparison_plot_path, comparison_rows)

    head_parameters = sum(
        parameter.numel() for parameter in victim.last_linear.parameters()
    )
    total_parameters = sum(parameter.numel() for parameter in victim.parameters())
    payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "protocol": "same_victim_input_gaussian_representation_transport_bn_affine_v2",
        "scientific_status": "data_only_operator_selector_no_ms_feedback",
        "dataset": DATASET,
        "model": MODEL,
        "seed": SEED,
        "data": {
            "split": "query_pool_ms",
            "count": QUERY_COUNT,
            "selection": "canonical_query_rank_prefix",
            "source_indices": query_indices,
            "source_indices_sha256": hash_integer_sequence(query_indices),
            "input_transform": "test",
            "gradient_enabled": False,
        },
        "models": {
            "public": {
                "checkpoint": str(public_checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(public_checkpoint),
                "num_classes": 1000,
                "classification_head_role": "excluded_semantically_incompatible",
            },
            "victim": {
                "checkpoint": str(victim_checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(victim_checkpoint),
                "checkpoint_epoch": victim_metadata.get("epoch"),
                "num_classes": NUM_CLASSES,
            },
        },
        "classification_head": {
            "selection_role": "mandatory_private_boundary_not_ranked",
            "states": ["last_linear.weight", "last_linear.bias"],
            "parameter_count": head_parameters,
            "model_parameter_count": total_parameters,
            "model_parameter_fraction": head_parameters / total_parameters,
            "reason": "public_imagenet_and_victim_cifar_class_coordinates_do_not_match",
        },
        "selection": {
            "candidate_count": len(rows),
            "conv_weight_count": len(conv_rows),
            "bn_affine_count": len(bn_rows),
            "included_conv": list(public_conv),
            "included_bn_affine": list(public_bn),
            "bn_affine_states_per_candidate": ["weight", "bias"],
            "conv_intervention": "z_p=Conv(h_v,W_p);z_v=Conv(h_v,W_v)",
            "bn_affine_intervention": (
                "h_hat_v=normalize(h_v,running_state_v);"
                "z_p=gamma_p*h_hat_v+beta_p;"
                "z_v=gamma_v*h_hat_v+beta_v"
            ),
            "observation_axis": "reshape_N_C_H_W_to_NHW_by_C",
            "moment_estimator": "population_mean_and_covariance_float64",
            "wasserstein": (
                "W2^2=||mu_v-mu_p||^2+Tr(Sigma_v+Sigma_p-"
                "2*sqrt(sqrt(Sigma_p)*Sigma_v*sqrt(Sigma_p)))"
            ),
            "normalization": (
                "half_sum_of_public_and_victim_second_moment_energy"
            ),
            "score": "rt_score=wasserstein2/(symmetric_second_moment+1e-12)",
            "ranking_uses_ms_feedback": False,
            "excluded": [
                "classification head because public/victim class coordinates differ",
                "BatchNorm running state as a candidate",
                "pooling, ReLU and residual addition",
                "convolution filters as independent candidates",
            ],
        },
        "execution": {
            "device": str(device),
            "batch_size": BATCH_SIZE,
            "num_workers": args.num_workers,
            "processed_image_count": processed,
            "model_mode": "eval",
        },
        "correctness": {
            **correctness,
            **numerical,
            "numerical_negative_eigenvalues_are_clipped_to_zero": True,
            "empirical_covariance_shrinkage": False,
            "eigenvalue_relative_tolerance": (
                NUMERICAL_EIGENVALUE_RELATIVE_TOLERANCE
            ),
        },
        "test01_comparison": {
            "source": str(cross_path.relative_to(ROOT)),
            "source_sha256": sha256_file(cross_path),
            **comparison_statistics,
        },
        "results": rows,
        "outputs": {
            "weights": str(weights_path.relative_to(ROOT)),
            "conv_weights": str(conv_path.relative_to(ROOT)),
            "bn_affine": str(bn_path.relative_to(ROOT)),
            "rank_plot": str(weights_plot_path.relative_to(ROOT)),
            "main_conv_tensors": str(tensors_path.relative_to(ROOT)),
            "main_conv_plot": str(tensors_plot_path.relative_to(ROOT)),
            "test01_comparison": str(comparison_path.relative_to(ROOT)),
            "test01_comparison_plot": str(comparison_plot_path.relative_to(ROOT)),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(metrics_path, payload)

    print("[TEST02 REPRESENTATION TRANSPORT RANK]")
    for row in rows:
        print(
            f"{int(row['rank']):02d} "
            f"{str(row['operator_type']):<11} "
            f"{float(row['rt_score']):.8f} {row['module']}"
        )
    print(
        "[TEST01 COMPARISON] "
        f"spearman={comparison_statistics['spearman_rank_correlation']:.6f} "
        f"kendall={comparison_statistics['kendall_rank_correlation']:.6f} "
        f"top_overlap={comparison_statistics['top_overlap']}"
    )
    print(f"[INFO] Test02 结果已写入：{output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
