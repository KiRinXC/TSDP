#!/usr/bin/env python3
"""Playground 残差实验共享的原始数据读取、校验与绘图工具。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "results" / "playground" / "01_raw"
MAIN_MODULES = tuple(
    f"layer{stage}.{block}.conv{conv}"
    for stage in range(1, 5)
    for block in range(2)
    for conv in range(1, 3)
)
ROUTES = ("z_pp", "z_pv", "z_vp", "z_vv")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_integer_sequence(values: Iterable[int]) -> str:
    encoded = json.dumps(list(values), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_tsv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: tuple[str, ...],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if not reader.fieldnames or any(not field for field in reader.fieldnames):
            raise ValueError(f"{path} 的 TSV 表头无效。")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"{path} 存在超出表头的列。")
    return rows


def load_raw_manifest() -> dict[str, object]:
    manifest = read_json(RAW_ROOT / "manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("experiment") != "01_raw_weight_routes"
        or manifest.get("candidate_count") != 40
    ):
        raise ValueError("PG01 raw manifest 协议或候选数量不正确。")
    return manifest


def load_raw_rows() -> list[dict[str, str]]:
    rows = read_tsv(RAW_ROOT / "data.tsv")
    if len(rows) != 40:
        raise ValueError(f"PG01 raw 应包含 40 个 weight，实际为 {len(rows)}。")
    modules = [row["module"] for row in rows]
    if len(modules) != len(set(modules)):
        raise ValueError("PG01 raw 包含重复 module。")
    if any(
        not row["state_name"].endswith(".weight")
        or row["operator_type"] not in {"conv_weight", "bn_gamma"}
        or row["bias_state"]
        for row in rows
    ):
        raise ValueError("PG01 raw 混入非 weight、bias 或未知算子。")
    return rows


def extract_main_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_module = {str(row["module"]): row for row in rows}
    missing = set(MAIN_MODULES) - set(by_module)
    if missing:
        raise ValueError(f"main 16 个主分支卷积不完整：{sorted(missing)}")
    return [by_module[module] for module in MAIN_MODULES]


def load_activation(
    row: dict[str, str],
    *,
    verify_hash: bool = True,
) -> dict[str, object]:
    path = ROOT / row["activation_path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    if verify_hash and sha256_file(path) != row["activation_sha256"]:
        raise ValueError(f"PG01 activation 哈希不一致：{path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("state_name") != row["state_name"]
        or payload.get("module") != row["module"]
        or payload.get("query_source_indices_sha256")
        != row["query_source_indices_sha256"]
    ):
        raise ValueError(f"PG01 activation 元数据不一致：{path}")
    tensors = payload.get("routes")
    if not isinstance(tensors, dict) or set(tensors) != set(ROUTES):
        raise ValueError(f"PG01 activation 四路输出不完整：{path}")
    shape = None
    for route in ROUTES:
        tensor = tensors[route]
        if not torch.is_tensor(tensor) or tensor.dtype != torch.float32:
            raise ValueError(f"PG01 {path.name}:{route} 不是 float32 tensor。")
        if tensor.size(0) != 500:
            raise ValueError(f"PG01 {path.name}:{route} 不是 500 张 query。")
        if shape is None:
            shape = tuple(tensor.shape)
        elif tuple(tensor.shape) != shape:
            raise ValueError(f"PG01 {path.name} 的四路输出形状不同。")
    cross = payload.get("cross")
    if (
        not torch.is_tensor(cross)
        or cross.dtype != torch.float32
        or tuple(cross.shape) != shape
    ):
        raise ValueError(f"PG01 {path.name} 缺少紧凑公式计算的 float32 交叉残差。")
    return payload


def residual_tensors(
    routes: dict[str, torch.Tensor],
    *,
    exact_cross: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cross = (
        exact_cross
        if exact_cross is not None
        else routes["z_vv"] - routes["z_vp"] - routes["z_pv"] + routes["z_pp"]
    )
    natural = routes["z_vv"] - routes["z_pp"]
    return cross, natural


def effective_rank_per_image(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
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
    terms = torch.zeros_like(probabilities)
    positive = probabilities > 0
    terms[positive] = probabilities[positive] * probabilities[positive].log()
    ranks = torch.zeros(batch, dtype=torch.float64, device=tensor.device)
    ranks[nonzero] = torch.exp(-terms[nonzero].sum(dim=1))
    if not torch.isfinite(ranks).all():
        raise ValueError("逐图片有效秩出现非有限值。")
    return ranks, min(height * width, channels)


def assert_finite_rows(rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    for row in rows:
        for field in fields:
            if not math.isfinite(float(row[field])):
                raise ValueError(f"{row.get('module')}:{field} 不是有限值。")


def add_product_ranks(
    rows: list[dict[str, object]],
    *,
    product_field: str = "product_score",
) -> list[dict[str, object]]:
    ordered = sorted(
        rows,
        key=lambda row: (-float(row[product_field]), str(row["state_name"])),
    )
    return [
        {"product_rank": rank, **row}
        for rank, row in enumerate(ordered, start=1)
    ]


def plot_metric(
    path: Path,
    rows: list[dict[str, object]],
    *,
    field: str,
    title: str,
    xlabel: str,
    scope: str,
    signed: bool = False,
) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (-abs(float(row[field])), str(row["state_name"])),
    )
    labels = [str(row["module"]) for row in ordered]
    values = [float(row[field]) for row in ordered]
    positions = list(range(len(rows)))
    if scope == "all":
        colors = [
            "#0072B2" if row["operator_type"] == "conv_weight" else "#E69F00"
            for row in ordered
        ]
        figure_size = (12.6, 14.0)
        handles = [
            plt.Rectangle((0, 0), 1, 1, color="#0072B2"),
            plt.Rectangle((0, 0), 1, 1, color="#E69F00"),
        ]
        legend = ("Conv weight", "BN gamma")
        scope_text = "all 40 Conv weight and BN gamma candidates"
    elif scope == "main":
        colors = [
            "#0072B2" if str(row["module"]).endswith("conv1") else "#D55E00"
            for row in ordered
        ]
        figure_size = (11.2, 7.6)
        handles = [
            plt.Rectangle((0, 0), 1, 1, color="#0072B2"),
            plt.Rectangle((0, 0), 1, 1, color="#D55E00"),
        ]
        legend = ("conv1", "conv2")
        scope_text = "sixteen BasicBlock main-path Conv weight candidates"
    elif scope == "bn":
        if any(row["operator_type"] != "bn_gamma" for row in ordered):
            raise ValueError("bn scope 混入非 BN gamma 候选。")
        colors = ["#E69F00" for _row in ordered]
        figure_size = (11.2, 8.8)
        handles = [plt.Rectangle((0, 0), 1, 1, color="#E69F00")]
        legend = ("BN gamma",)
        scope_text = "twenty BN gamma candidates"
    else:
        raise ValueError(f"未知绘图 scope：{scope}")
    figure, axis = plt.subplots(figsize=figure_size)
    axis.barh(positions, values, color=colors)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel(xlabel)
    axis.set_title(f"{title}\n{scope_text}")
    axis.grid(axis="x", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    max_abs = max((abs(value) for value in values), default=1.0)
    scale = max(max_abs, 1e-12)
    if signed:
        axis.axvline(0.0, color="#333333", linewidth=0.8)
        axis.set_xlim(-scale * 1.28, scale * 1.28)
    else:
        axis.set_xlim(0.0, scale * 1.18)
    for position, value in zip(positions, values):
        if value < 0:
            x = value - scale * 0.012
            alignment = "right"
        else:
            x = value + scale * 0.012
            alignment = "left"
        axis.text(
            x,
            position,
            f"{value:+.4g}" if signed else f"{value:.4g}",
            va="center",
            ha=alignment,
            fontsize=7,
        )
    axis.legend(handles, legend, frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=230, bbox_inches="tight")
    plt.close(figure)
