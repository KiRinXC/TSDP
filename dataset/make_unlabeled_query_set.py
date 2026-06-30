#!/usr/bin/env python3
"""从验证集中构造无标签查询子集，采样规模以训练集大小为基准。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from torchvision import datasets as tv_datasets


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset" / "public"
DEFAULT_DERIVED_ROOT = REPO_ROOT / "dataset" / "derived"
DEFAULT_RATIO = 0.01


DATASET_ALIASES = {
    "cifar10": "cifar10",
    "cifar_10": "cifar10",
    "cifar-10": "cifar10",
    "cifar100": "cifar100",
    "cifar_100": "cifar100",
    "cifar-100": "cifar100",
    "stl10": "stl10",
    "stl-10": "stl10",
    "tiny-imagenet-200": "tiny-imagenet-200",
    "tinyimagenet200": "tiny-imagenet-200",
    "tiny_imagenet200": "tiny-imagenet-200",
    "tiny_imagenet_200": "tiny-imagenet-200",
    "tinyimagenet": "tiny-imagenet-200",
}


def resolve_dataset_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized not in DATASET_ALIASES:
        valid = ", ".join(sorted(DATASET_ALIASES))
        raise ValueError(f"不支持的数据集：{name}。可选值：{valid}")
    return DATASET_ALIASES[normalized]


def resolve_split_name(dataset_name: str, split: str) -> str:
    normalized = split.strip().lower()
    if normalized in {"eval", "validation", "valid", "val", "test"}:
        return "val" if dataset_name == "tiny-imagenet-200" else "test"
    raise ValueError("当前脚本只从验证集构造无标签查询集，可选 split: eval / val / test")


def build_dataset(dataset_name: str, dataset_root: Path, split_name: str):
    if dataset_name == "cifar10":
        return tv_datasets.CIFAR10(
            root=str(dataset_root / "cifar10"),
            train=(split_name != "test"),
            download=False,
            transform=None,
        )

    if dataset_name == "cifar100":
        return tv_datasets.CIFAR100(
            root=str(dataset_root / "cifar100"),
            train=(split_name != "test"),
            download=False,
            transform=None,
        )

    if dataset_name == "stl10":
        return tv_datasets.STL10(
            root=str(dataset_root / "stl10"),
            split=split_name,
            download=False,
            transform=None,
        )

    if dataset_name == "tiny-imagenet-200":
        root = dataset_root / "tiny-imagenet-200" / split_name
        if not root.is_dir():
            raise FileNotFoundError(f"找不到 Tiny-ImageNet split 目录：{root}")
        return tv_datasets.ImageFolder(root=str(root), transform=None)

    raise ValueError(f"未知的数据集：{dataset_name}")


def build_source_dataset(dataset_name: str, dataset_root: Path, split_name: str):
    return build_dataset(dataset_name, dataset_root, split_name)


def build_train_reference_dataset(dataset_name: str, dataset_root: Path):
    if dataset_name == "tiny-imagenet-200":
        return build_dataset(dataset_name, dataset_root, "train")
    return build_dataset(dataset_name, dataset_root, "train")


def default_output_dir(
    derived_root: Path,
    dataset_name: str,
) -> Path:
    return derived_root / dataset_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从验证集中构造无标签查询子集，默认采样训练集规模的 1%。")
    parser.add_argument("--dataset", required=True, help="数据集名称，如 cifar10 / stl10")
    parser.add_argument(
        "--split",
        default="eval",
        help="验证集来源，默认 eval。CIFAR/STL 使用官方 test，Tiny-ImageNet 使用 val。",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="公开数据集根目录")
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT, help="派生数据根目录")
    parser.add_argument("--ratio", type=float, default=DEFAULT_RATIO, help="相对训练集大小的采样比例，默认 0.01")
    parser.add_argument("--max-samples", type=int, default=None, help="直接指定样本数，仍需不超过训练集 1%%")
    parser.add_argument("--seed", type=int, default=42, help="采样随机种子")
    parser.add_argument("--force", action="store_true", help="允许覆盖已有 manifest")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不写入文件")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_name = resolve_dataset_name(args.dataset)
    split_name = resolve_split_name(dataset_name, args.split)
    dataset_root = args.dataset_root.expanduser().resolve()
    derived_root = args.derived_root.expanduser().resolve()

    if args.ratio <= 0:
        raise ValueError("--ratio 必须大于 0")

    train_reference_dataset = build_train_reference_dataset(dataset_name, dataset_root)
    train_reference_size = len(train_reference_dataset)
    source_dataset = build_source_dataset(dataset_name, dataset_root, split_name)
    source_size = len(source_dataset)
    one_percent = train_reference_size * 0.01

    if args.max_samples is not None:
        sample_count = args.max_samples
    else:
        sample_count = int(train_reference_size * args.ratio)

    if sample_count <= 0:
        raise ValueError(f"采样数量为 {sample_count}，请增大 --ratio 或 --max-samples")
    if sample_count > one_percent:
        raise ValueError(
            f"采样数量必须不超过训练集 1%%。当前 {sample_count}/{train_reference_size}，"
            f"1%% 阈值为 {one_percent:.2f}"
        )
    if sample_count > source_size:
        raise ValueError(f"采样数量 {sample_count} 超过 split 大小 {source_size}")

    rng = random.Random(args.seed)
    indices = rng.sample(range(source_size), sample_count)
    actual_ratio = sample_count / source_size
    output_dir = default_output_dir(
        derived_root=derived_root,
        dataset_name=dataset_name,
    )
    manifest_path = output_dir / "manifest.json"
    samples_path = output_dir / "samples.tsv"

    print(f"[INFO] dataset: {dataset_name}")
    print(f"[INFO] source split: {split_name}")
    print(f"[INFO] source size: {source_size}")
    print(f"[INFO] train reference size: {train_reference_size}")
    print(f"[INFO] sample count: {sample_count}")
    print(f"[INFO] actual ratio: {actual_ratio:.6f}")
    print(f"[INFO] train reference ratio: {sample_count / train_reference_size:.6f}")
    print(f"[INFO] seed: {args.seed}")
    print(f"[INFO] output dir: {output_dir}")

    if args.dry_run:
        print("[INFO] dry-run 模式结束，没有写入文件。")
        return 0

    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"manifest 已存在，如需覆盖请加 --force：{manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": "unlabeled_query_subset",
        "version": 1,
        "dataset": dataset_name,
        "source_split": split_name,
        "source_size": source_size,
        "sample_basis_split": "train",
        "sample_basis_size": train_reference_size,
        "sample_count": sample_count,
        "actual_ratio": actual_ratio,
        "train_reference_ratio": sample_count / train_reference_size,
        "max_ratio_policy": "默认样本数为 floor(训练集大小 * 1%)；样本从验证集中抽取。CIFAR/STL 使用官方 test，Tiny-ImageNet 使用 val。",
        "seed": args.seed,
        "public_dataset_root": str(dataset_root),
        "derived_root": str(derived_root),
        "label_policy": "真实标签不写入 manifest；后续查询脚本只能把 source labels 当作应忽略字段。",
        "indices": indices,
        "created_on": str(datetime.now()),
    }
    classes = getattr(source_dataset, "classes", None)
    if classes is not None:
        manifest["num_classes"] = len(classes)

    with manifest_path.open("w", encoding="utf-8") as writer:
        json.dump(manifest, writer, ensure_ascii=False, indent=2)

    with samples_path.open("w", encoding="utf-8") as writer:
        writer.write("rank\tsource_index\n")
        for rank, index in enumerate(indices):
            writer.write(f"{rank}\t{index}\n")

    print(f"[INFO] manifest written: {manifest_path}")
    print(f"[INFO] samples written: {samples_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
