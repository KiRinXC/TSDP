#!/usr/bin/env python3
"""query、eval_ms 和 victim 模型输入。"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset

from .config import MODEL_SPECS

from common.trainer import (  # noqa: E402
    MS_EVAL_SOURCES,
    build_public_split_dataset,
    build_transforms,
    read_ms_split_indices,
)
from models import imagenet as imagenet_models  # noqa: E402


class QueryDataset(Dataset):
    """使用 canonical query 顺序绑定公开图像与 victim 查询目标。"""

    def __init__(
        self,
        public_dataset,
        source_indices: list[int],
        posteriors: torch.Tensor | None,
        pseudo_labels: torch.Tensor,
    ):
        if len(source_indices) != pseudo_labels.size(0):
            raise ValueError("query 索引和伪标签数量不一致。")
        if posteriors is not None and len(source_indices) != posteriors.size(0):
            raise ValueError("query 索引和 posterior 数量不一致。")
        self.public_dataset = public_dataset
        self.source_indices = source_indices
        self.posteriors = posteriors
        self.pseudo_labels = pseudo_labels

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, index: int):
        image, _ = self.public_dataset[self.source_indices[index]]
        if self.posteriors is None:
            return image, self.pseudo_labels[index]
        return image, self.posteriors[index], self.pseudo_labels[index]


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 JSON 文件：{path}")
    with path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def load_checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint
    if isinstance(checkpoint, dict) and checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint, {}
    raise ValueError(f"无法识别 checkpoint 格式：{path}")


def read_query_indices(protocol_root: Path, dataset_name: str) -> list[int]:
    split_path = protocol_root / dataset_name / "splits.tsv"
    if not split_path.is_file():
        raise FileNotFoundError(f"找不到 MS 划分：{split_path}")
    with split_path.open("r", newline="", encoding="utf-8") as reader_file:
        reader = csv.DictReader(reader_file, delimiter="\t")
        required = {"split", "source_split", "source_index", "query_rank"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{split_path} 缺少字段：{sorted(missing)}")
        rows = [row for row in reader if row["split"] == "query_pool_ms"]
    if any(row["source_split"] != "official_train" for row in rows):
        raise ValueError("query_pool_ms 必须来自 official_train。")
    rows.sort(key=lambda row: int(row["query_rank"]))
    ranks = [int(row["query_rank"]) for row in rows]
    if ranks != list(range(len(rows))):
        raise ValueError("query_pool_ms 的 query_rank 必须从 0 连续递增。")
    indices = [int(row["source_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("query_pool_ms 包含重复 source_index。")
    return indices


def read_hard_labels(labels_path: Path, query_indices: list[int]) -> torch.Tensor:
    if not labels_path.is_file():
        raise FileNotFoundError(f"找不到 victim hard label：{labels_path}")
    with labels_path.open("r", newline="", encoding="utf-8") as reader_file:
        reader = csv.DictReader(reader_file, delimiter="\t")
        required = {"query_rank", "source_split", "source_index", "pseudo_label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{labels_path} 缺少字段：{sorted(missing)}")
        rows = list(reader)
    rows.sort(key=lambda row: int(row["query_rank"]))
    ranks = [int(row["query_rank"]) for row in rows]
    if ranks != list(range(len(query_indices))):
        raise ValueError("labels.tsv 的 query_rank 与 query_pool_ms 不一致。")
    if any(row["source_split"] != "official_train" for row in rows):
        raise ValueError("labels.tsv 中的 query 必须来自 official_train。")
    label_indices = [int(row["source_index"]) for row in rows]
    if label_indices != query_indices:
        raise ValueError("labels.tsv 的 source_index 顺序与 query_pool_ms 不一致。")
    return torch.tensor([int(row["pseudo_label"]) for row in rows], dtype=torch.long)


def load_query_targets(
    protocol_root: Path,
    dataset_name: str,
    model_name: str,
    budget: int,
    label_mode: str,
) -> tuple[list[int], torch.Tensor | None, torch.Tensor, Path, dict]:
    protocol_manifest = load_json(protocol_root / dataset_name / "manifest.json")
    planned_budgets = protocol_manifest.get("query", {}).get("planned_budgets", [])
    if budget not in planned_budgets:
        raise ValueError(f"预算 {budget} 不在固定预算 {planned_budgets} 中。")

    model_root = protocol_root / dataset_name / model_name
    query_manifest = load_json(model_root / "manifest.json")
    if (
        query_manifest.get("protocol") != "MS"
        or query_manifest.get("dataset") != dataset_name
        or query_manifest.get("model") != model_name
    ):
        raise ValueError(f"query manifest 与 {model_name}+{dataset_name} 不一致。")
    query_indices = read_query_indices(protocol_root, dataset_name)
    if label_mode == "hard":
        labels_path = model_root / query_manifest.get("outputs", {}).get("labels", "labels.tsv")
        pseudo_labels = read_hard_labels(labels_path, query_indices)
        return query_indices[:budget], None, pseudo_labels[:budget], labels_path, query_manifest

    posterior_path = model_root / query_manifest.get("outputs", {}).get("posteriors", "posteriors.pt")
    if not posterior_path.is_file():
        raise FileNotFoundError(f"找不到 victim posterior：{posterior_path}")
    package = torch.load(posterior_path, map_location="cpu", weights_only=False)
    if not isinstance(package, dict):
        raise ValueError(f"无法识别 posterior 文件格式：{posterior_path}")
    if package.get("protocol") != "MS" or package.get("dataset") != dataset_name or package.get("model") != model_name:
        raise ValueError(f"posterior 元数据与 {model_name}+{dataset_name} 不一致。")
    posteriors = package.get("posteriors")
    pseudo_labels = package.get("pseudo_labels")
    if not torch.is_tensor(posteriors) or not torch.is_tensor(pseudo_labels):
        raise ValueError("posteriors.pt 缺少 posteriors 或 pseudo_labels tensor。")

    if posteriors.ndim != 2 or posteriors.size(0) != len(query_indices):
        raise ValueError("posterior 数量与 query_pool_ms 不一致。")
    if pseudo_labels.shape != (len(query_indices),):
        raise ValueError("pseudo_labels 数量与 query_pool_ms 不一致。")
    if not torch.equal(posteriors.argmax(dim=1), pseudo_labels.long()):
        raise ValueError("pseudo_labels 与 posterior argmax 不一致。")
    if not torch.allclose(posteriors.sum(dim=1), torch.ones(len(query_indices)), atol=1e-5, rtol=1e-5):
        raise ValueError("posterior 每行概率和不为 1。")
    return (
        query_indices[:budget],
        posteriors[:budget].float(),
        pseudo_labels[:budget].long(),
        posterior_path,
        query_manifest,
    )


def build_query_dataset(
    dataset_name: str,
    dataset_root: Path,
    source_indices: list[int],
    posteriors: torch.Tensor | None,
    pseudo_labels: torch.Tensor,
) -> QueryDataset:
    train_transform, _ = build_transforms(dataset_name)
    public_dataset = build_public_split_dataset(dataset_name, dataset_root, "train", train_transform)
    invalid = [index for index in source_indices if index < 0 or index >= len(public_dataset)]
    if invalid:
        raise ValueError(f"query_pool_ms 包含越界索引：{invalid[0]}")
    return QueryDataset(public_dataset, source_indices, posteriors, pseudo_labels)


def build_eval_dataset(dataset_name: str, dataset_root: Path, protocol_root: Path, subset: int | None):
    _, eval_transform = build_transforms(dataset_name)
    public_split = "val" if dataset_name == "t200" else "test"
    public_dataset = build_public_split_dataset(dataset_name, dataset_root, public_split, eval_transform)
    indices = read_ms_split_indices(protocol_root, dataset_name, "eval_ms", MS_EVAL_SOURCES[dataset_name])
    if subset is not None:
        if subset <= 0:
            raise ValueError("eval_subset 必须大于 0。")
        indices = indices[:subset]
    return Subset(public_dataset, indices)


def build_victim(model_name: str, num_classes: int, checkpoint_path: Path) -> tuple[nn.Module, dict]:
    factory_name, _ = MODEL_SPECS[model_name]
    factory: Callable[..., nn.Module] = getattr(imagenet_models, factory_name)
    model = factory(num_classes=num_classes)
    state_dict, metadata = load_checkpoint_state(checkpoint_path)
    model.load_state_dict(state_dict, strict=True)
    if metadata.get("arch") not in (None, model_name):
        raise ValueError(f"checkpoint arch={metadata.get('arch')}，期望 {model_name}。")
    return model, metadata
