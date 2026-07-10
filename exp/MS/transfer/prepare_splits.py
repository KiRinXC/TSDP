#!/usr/bin/env python3
"""Build reference-compatible overlapping MS splits for public datasets."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_VICTIM_ROOT = REPO_ROOT / "exp" / "MS" / "train_victim"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRAIN_VICTIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_VICTIM_ROOT))

from common.trainer import build_public_split_dataset, resolve_dataset_name  # noqa: E402


DATASET_IDS = ("c10", "c100", "s10", "t200")
SEED = 42
QUERY_RATIO = 0.01
PLANNED_BUDGETS = {
    "c10": [50, 100, 300, 500],
    "c100": [50, 100, 300, 500],
    "s10": [50],
    "t200": [50, 100, 300, 500, 1000],
}
EVAL_SOURCES = {
    "c10": ("test", "official_test"),
    "c100": ("test", "official_test"),
    "s10": ("test", "official_test"),
    "t200": ("val", "official_val"),
}
SPLIT_FIELDS = ["record_id", "split", "source_split", "source_index", "global_index", "query_rank"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构造与参考开源实现一致的随机重叠 MS 数据划分。")
    parser.add_argument("datasets", nargs="*", default=["all"], help="all 或 c10/c100/s10/t200")
    parser.add_argument(
        "--dataset-root",
        default=str(REPO_ROOT / "dataset" / "public"),
        help="公开数据集根目录。",
    )
    parser.add_argument(
        "--protocol-root",
        default=str(REPO_ROOT / "dataset" / "MS"),
        help="MS 协议产物根目录。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 manifest.json 和 splits.tsv。")
    return parser.parse_args()


def resolve_datasets(values: list[str]) -> list[str]:
    normalized = [value.strip().lower() for value in values if value.strip()]
    if not normalized or normalized == ["all"]:
        return list(DATASET_IDS)
    if "all" in normalized:
        raise ValueError("all 不能与具体数据集 id 同时使用。")
    result = [resolve_dataset_name(value) for value in normalized]
    if len(set(result)) != len(result):
        raise ValueError("数据集 id 不能重复。")
    return result


def query_size_for(train_size: int) -> int:
    query_size = int(train_size * QUERY_RATIO)
    if query_size <= 0:
        raise ValueError(f"训练集大小 {train_size} 无法按 {QUERY_RATIO:.0%} 构造 query_pool_ms。")
    return query_size


def write_splits(path: Path, train_size: int, query_indices: list[int], eval_size: int, eval_source: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(writer_file, fieldnames=SPLIT_FIELDS, delimiter="\t")
        writer.writeheader()
        for source_index in range(train_size):
            writer.writerow(
                {
                    "record_id": f"official_train:{source_index}",
                    "split": "victim_train",
                    "source_split": "official_train",
                    "source_index": source_index,
                    "global_index": source_index,
                    "query_rank": "",
                }
            )
        for query_rank, source_index in enumerate(query_indices):
            writer.writerow(
                {
                    "record_id": f"official_train:{source_index}",
                    "split": "query_pool_ms",
                    "source_split": "official_train",
                    "source_index": source_index,
                    "global_index": source_index,
                    "query_rank": query_rank,
                }
            )
        for source_index in range(eval_size):
            writer.writerow(
                {
                    "record_id": f"{eval_source}:{source_index}",
                    "split": "eval_ms",
                    "source_split": eval_source,
                    "source_index": source_index,
                    "global_index": source_index,
                    "query_rank": "",
                }
            )


def write_manifest(
    path: Path,
    dataset_name: str,
    train_size: int,
    eval_size: int,
    query_size: int,
    eval_source: str,
) -> None:
    manifest = {
        "schema_version": 1,
        "protocol": "MS",
        "protocol_variant": "reference_random_overlap",
        "dataset": dataset_name,
        "seed": SEED,
        "split_file": "splits.tsv",
        "source": {
            "train_split": "official_train",
            "train_size": train_size,
            "eval_split": eval_source,
            "eval_size": eval_size,
        },
        "splits": {
            "victim_train": {
                "source_split": "official_train",
                "count": train_size,
                "selection": "official_train_full",
            },
            "query_pool_ms": {
                "source_split": "official_train",
                "count": query_size,
                "selection": "uniform_random_without_replacement",
                "query_rank_field": "query_rank",
            },
            "eval_ms": {
                "source_split": eval_source,
                "count": eval_size,
                "selection": "official_eval_full",
            },
        },
        "query": {
            "split": "query_pool_ms",
            "ratio_of_victim_train": QUERY_RATIO,
            "max_budget": query_size,
            "planned_budgets": PLANNED_BUDGETS[dataset_name],
        },
        "constraints": {
            "victim_train_query_pool_overlap": query_size,
            "query_pool_eval_overlap": 0,
        },
    }
    with path.open("w", encoding="utf-8") as writer:
        json.dump(manifest, writer, ensure_ascii=False, indent=2)
        writer.write("\n")


def prepare_dataset(dataset_name: str, dataset_root: Path, protocol_root: Path, overwrite: bool) -> None:
    out_dir = protocol_root / dataset_name
    manifest_path = out_dir / "manifest.json"
    splits_path = out_dir / "splits.tsv"
    if not overwrite and any(path.exists() for path in (manifest_path, splits_path)):
        raise FileExistsError(f"协议产物已存在：{out_dir}。使用 --overwrite 重新生成。")

    train_dataset = build_public_split_dataset(dataset_name, dataset_root, "train", transform=None)
    eval_loader_split, eval_source = EVAL_SOURCES[dataset_name]
    eval_dataset = build_public_split_dataset(dataset_name, dataset_root, eval_loader_split, transform=None)
    train_size = len(train_dataset)
    query_size = query_size_for(train_size)
    if PLANNED_BUDGETS[dataset_name][-1] > query_size:
        raise ValueError(f"{dataset_name} 的最大预算超过 1% query pool：{query_size}")

    query_indices = random.Random(SEED).sample(range(train_size), query_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_splits(splits_path, train_size, query_indices, len(eval_dataset), eval_source)
    write_manifest(manifest_path, dataset_name, train_size, len(eval_dataset), query_size, eval_source)
    print(
        f"[INFO] {dataset_name}: victim_train={train_size} query_pool_ms={query_size} "
        f"eval_ms={len(eval_dataset)} overlap={query_size}"
    )


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    protocol_root = Path(args.protocol_root).expanduser().resolve()
    for dataset_name in resolve_datasets(args.datasets):
        prepare_dataset(dataset_name, dataset_root, protocol_root, args.overwrite)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
