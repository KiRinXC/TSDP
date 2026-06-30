#!/usr/bin/env python3
"""伪标签数据集构造公共逻辑。"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tv_datasets
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.train_victim.common.trainer import build_transforms, resolve_dataset_name  # noqa: E402
from models import imagenet as imagenet_models  # noqa: E402

try:
    import numpy as np
except ImportError:
    np = None


@dataclass(frozen=True)
class ModelSpec:
    """描述一个伪标签生成模型入口。"""

    name: str
    display_name: str
    factory_name: str


class IndexedSubset(Dataset):
    """按 manifest 中的 source_index 读取样本，并额外返回 rank 和 source_index。"""

    def __init__(self, source_dataset, indices: list[int]):
        self.source_dataset = source_dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, rank: int):
        source_index = self.indices[rank]
        image, _ignored_label = self.source_dataset[source_index]
        return image, rank, source_index


def configure_reproducibility(seed: int | None) -> None:
    """固定推理流程中可控的随机状态。"""
    if seed is None or seed < 0:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if np is not None:
        np.random.seed(seed)


def default_victim_weight_path(spec: ModelSpec, dataset_name: str) -> Path:
    """返回训练好的 victim 权重默认路径。"""
    return REPO_ROOT / "weights" / "victim" / spec.name / dataset_name / "target.pth"


def default_source_split(dataset_name: str) -> str:
    """返回该数据集默认的评估 split。"""
    return "val" if dataset_name == "tiny-imagenet-200" else "test"


def resolve_split_name(dataset_name: str, split: str) -> str:
    """把外部 split 输入统一到当前项目的评估 split 名称。"""
    normalized = split.strip().lower()
    if normalized in {"eval", "validation", "valid", "val", "test"}:
        return default_source_split(dataset_name)
    raise ValueError("当前伪标签入口只支持评估 split，可选 split: eval / val / test")


def default_query_manifest_path(derived_root: Path, dataset_name: str, split_name: str) -> Path:
    """返回无标签查询集 manifest 默认路径。"""
    return derived_root / dataset_name / split_name / "manifest.json"


def read_manifest(path: Path) -> dict:
    """读取并校验无标签查询集 manifest。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到无标签查询集 manifest：{path}")
    with path.open("r", encoding="utf-8") as reader:
        manifest = json.load(reader)

    if "indices" not in manifest or not isinstance(manifest["indices"], list):
        raise ValueError(f"manifest 缺少 indices 列表：{path}")
    if manifest.get("kind") != "unlabeled_query_subset":
        print(f"[WARN] manifest kind 不是 unlabeled_query_subset：{manifest.get('kind')}")
    return manifest


def build_source_dataset(dataset_name: str, dataset_root: Path, split_name: str):
    """按训练协议中的测试 transform 读取源评估数据集。"""
    _train_transform, test_transform = build_transforms(dataset_name)

    if dataset_name == "cifar10":
        return tv_datasets.CIFAR10(
            root=str(dataset_root / "cifar10"),
            train=(split_name != "test"),
            download=False,
            transform=test_transform,
        )

    if dataset_name == "cifar100":
        return tv_datasets.CIFAR100(
            root=str(dataset_root / "cifar100"),
            train=(split_name != "test"),
            download=False,
            transform=test_transform,
        )

    if dataset_name == "stl10":
        return tv_datasets.STL10(
            root=str(dataset_root / "stl10"),
            split=split_name,
            download=False,
            transform=test_transform,
        )

    if dataset_name == "tiny-imagenet-200":
        root = dataset_root / "tiny-imagenet-200" / split_name
        if not root.is_dir() and split_name == "val":
            fallback = dataset_root / "tiny-imagenet-200" / "val2"
            if fallback.is_dir():
                root = fallback
        if not root.is_dir():
            raise FileNotFoundError(f"找不到 Tiny-ImageNet split 目录：{root}")
        return tv_datasets.ImageFolder(root=str(root), transform=test_transform)

    raise ValueError(f"未知的数据集：{dataset_name}")


def infer_num_classes(dataset_name: str, source_dataset) -> int:
    """返回当前数据集类别数。"""
    classes = getattr(source_dataset, "classes", None)
    if classes is not None:
        return len(classes)
    if dataset_name in {"cifar10", "stl10"}:
        return 10
    if dataset_name == "cifar100":
        return 100
    if dataset_name == "tiny-imagenet-200":
        return 200
    raise ValueError(f"无法推断类别数：{dataset_name}")


def get_class_names(source_dataset, num_classes: int) -> list[str]:
    """返回类别名。"""
    classes = getattr(source_dataset, "classes", None)
    if classes is not None:
        return [str(name) for name in classes]
    return [str(index) for index in range(num_classes)]


def build_model(spec: ModelSpec, num_classes: int, weight_path: Path, device: torch.device) -> nn.Module:
    """创建模型并加载训练好的 victim 权重。"""
    if not weight_path.is_file():
        raise FileNotFoundError(f"找不到 victim 权重：{weight_path}")

    factory: Callable[..., nn.Module] = getattr(imagenet_models, spec.factory_name)
    model = factory(num_classes=num_classes)
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def resolve_device(name: str) -> torch.device:
    """把用户输入的设备名转换成 torch.device。"""
    normalized = name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用的 CUDA 设备。")
    return torch.device(normalized)


def default_output_dir(
    pseudo_label_root: Path,
    dataset_name: str,
    model_name: str,
    split_name: str,
) -> Path:
    """返回伪标签数据集默认输出目录。"""
    return pseudo_label_root / dataset_name / model_name / split_name


@torch.inference_mode()
def make_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> list[dict]:
    """对无标签查询集生成伪标签。"""
    records: list[dict] = []
    progress = tqdm(loader, desc="[PSEUDO]", dynamic_ncols=True)
    for inputs, ranks, source_indices in progress:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)

        for rank, source_index, pred, confidence in zip(
            ranks.tolist(),
            source_indices.tolist(),
            predictions.cpu().tolist(),
            confidences.cpu().tolist(),
        ):
            records.append(
                {
                    "rank": int(rank),
                    "source_index": int(source_index),
                    "pseudo_label": int(pred),
                    "pseudo_label_name": class_names[int(pred)],
                    "confidence": float(confidence),
                }
            )
    records.sort(key=lambda item: item["rank"])
    return records


def write_samples(path: Path, records: list[dict]) -> None:
    """写出伪标签样本表。"""
    fieldnames = ["rank", "source_index", "pseudo_label", "pseudo_label_name", "confidence"]
    with path.open("w", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(writer_file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def parse_args(spec: ModelSpec) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=f"使用 {spec.display_name} victim 模型构造伪标签数据集")
    parser.add_argument("dataset", nargs="?", default=None, help="数据集名称")
    parser.add_argument("--dataset", dest="dataset_flag", default=None, help="数据集名称")
    parser.add_argument("--split", default="eval", help="评估 split，默认 eval")
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "dataset" / "public"), help="公开数据集根目录")
    parser.add_argument("--derived-root", default=str(REPO_ROOT / "dataset" / "derived"), help="无标签查询集根目录")
    parser.add_argument(
        "--pseudo-label-root",
        default=str(REPO_ROOT / "dataset" / "pseudo_labels"),
        help="伪标签数据集根目录",
    )
    parser.add_argument("--query-manifest", default=None, help="无标签查询集 manifest 路径")
    parser.add_argument("--victim-weight-path", default=None, help="训练好的 victim 权重路径")
    parser.add_argument("--out-dir", default=None, help="伪标签数据集输出目录")
    parser.add_argument("--device", default="auto", help="运行设备：auto / cpu / cuda / cuda:0")
    parser.add_argument("--batch-size", type=int, default=256, help="推理 batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--force", action="store_true", help="允许覆盖已有伪标签数据集")
    parser.add_argument("--dry-run", action="store_true", help="只检查路径和计划，不加载模型、不写文件")
    return parser.parse_args()


def pseudo_label_main(spec: ModelSpec) -> None:
    """执行伪标签数据集构造流程。"""
    args = parse_args(spec)
    dataset_name = resolve_dataset_name(args.dataset_flag or args.dataset or "cifar10")
    split_name = resolve_split_name(dataset_name, args.split)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    derived_root = Path(args.derived_root).expanduser().resolve()
    pseudo_label_root = Path(args.pseudo_label_root).expanduser().resolve()
    query_manifest_path = (
        Path(args.query_manifest).expanduser().resolve()
        if args.query_manifest
        else default_query_manifest_path(derived_root, dataset_name, split_name)
    )
    victim_weight_path = (
        Path(args.victim_weight_path).expanduser().resolve()
        if args.victim_weight_path
        else default_victim_weight_path(spec, dataset_name)
    )

    manifest = read_manifest(query_manifest_path)
    indices = [int(index) for index in manifest["indices"]]
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else default_output_dir(pseudo_label_root, dataset_name, spec.name, split_name)
    )

    print(f"[INFO] 模型: {spec.display_name}")
    print(f"[INFO] 数据集: {dataset_name}")
    print(f"[INFO] source split: {split_name}")
    print(f"[INFO] 查询集 manifest: {query_manifest_path}")
    print(f"[INFO] 查询样本数: {len(indices)}")
    print(f"[INFO] victim 权重: {victim_weight_path}")
    print(f"[INFO] 伪标签根目录: {pseudo_label_root}")
    print(f"[INFO] 输出目录: {out_dir}")

    if args.dry_run:
        print("[INFO] dry-run 模式结束，没有加载模型或写入文件。")
        return

    manifest_path = out_dir / "manifest.json"
    samples_path = out_dir / "samples.tsv"
    if (manifest_path.exists() or samples_path.exists()) and not args.force:
        raise FileExistsError(f"输出已存在，如需覆盖请加 --force：{out_dir}")

    configure_reproducibility(args.seed)
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    source_dataset = build_source_dataset(dataset_name, dataset_root, split_name)
    num_classes = infer_num_classes(dataset_name, source_dataset)
    class_names = get_class_names(source_dataset, num_classes)

    if any(index < 0 or index >= len(source_dataset) for index in indices):
        raise ValueError("manifest 中存在越界 source_index。")

    query_dataset = IndexedSubset(source_dataset, indices)
    loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    model = build_model(spec, num_classes, victim_weight_path, device)
    records = make_predictions(model, loader, device, class_names)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = {
        "kind": "pseudo_label_dataset",
        "version": 1,
        "dataset": dataset_name,
        "source_split": split_name,
        "source_query_manifest": str(query_manifest_path),
        "source_sample_count": len(indices),
        "model": spec.name,
        "display_name": spec.display_name,
        "victim_weight_path": str(victim_weight_path),
        "num_classes": num_classes,
        "class_names": class_names,
        "sample_file": "samples.tsv",
        "sample_columns": ["rank", "source_index", "pseudo_label", "pseudo_label_name", "confidence"],
        "label_policy": "不写入真实标签；pseudo_label 是 victim 模型对无标签查询样本的预测。",
        "dataset_root": str(dataset_root),
        "derived_root": str(derived_root),
        "pseudo_label_root": str(pseudo_label_root),
        "source_query_summary": {
            "sample_basis_split": manifest.get("sample_basis_split"),
            "sample_basis_size": manifest.get("sample_basis_size"),
            "sample_count": manifest.get("sample_count", len(indices)),
            "actual_ratio": manifest.get("actual_ratio"),
            "train_reference_ratio": manifest.get("train_reference_ratio"),
            "seed": manifest.get("seed"),
        },
        "seed": args.seed,
        "created_on": str(datetime.now()),
    }
    with manifest_path.open("w", encoding="utf-8") as writer:
        json.dump(output_manifest, writer, ensure_ascii=False, indent=2)
        writer.write("\n")
    write_samples(samples_path, records)

    print(f"[INFO] manifest written: {manifest_path}")
    print(f"[INFO] samples written: {samples_path}")
