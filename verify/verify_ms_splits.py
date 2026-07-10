#!/usr/bin/env python3
"""Validate canonical reference-compatible MS split manifests."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_ROOT = REPO_ROOT / "dataset" / "MS"
EXPECTED = {
    "c10": {"train": 50000, "query": 500, "eval": 10000, "eval_source": "official_test", "budgets": [50, 100, 300, 500]},
    "c100": {"train": 50000, "query": 500, "eval": 10000, "eval_source": "official_test", "budgets": [50, 100, 300, 500]},
    "s10": {"train": 5000, "query": 50, "eval": 8000, "eval_source": "official_test", "budgets": [50]},
    "t200": {"train": 100000, "query": 1000, "eval": 10000, "eval_source": "official_val", "budgets": [50, 100, 300, 500, 1000]},
}
REQUIRED_FIELDS = {"record_id", "split", "source_split", "source_index", "global_index", "query_rank"}


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise ValueError(message)


def rows_for(rows: list[dict[str, str]], split_name: str) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == split_name]


def indices(rows: list[dict[str, str]], dataset: str, split_name: str) -> list[int]:
    try:
        result = [int(row["source_index"]) for row in rows]
    except ValueError as exc:
        fail(f"{dataset}: {split_name} 包含非整数 source_index: {exc}")
    if len(result) != len(set(result)):
        fail(f"{dataset}: {split_name} 包含重复 source_index")
    return result


def verify_dataset(protocol_root: Path, dataset: str, expected: dict[str, int | str | list[int]]) -> None:
    root = protocol_root / dataset
    manifest_path = root / "manifest.json"
    splits_path = root / "splits.tsv"
    if not manifest_path.is_file() or not splits_path.is_file():
        fail(f"{dataset}: 缺少 manifest.json 或 splits.tsv")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "MS" or manifest.get("protocol_variant") != "reference_random_overlap":
        fail(f"{dataset}: protocol 或 protocol_variant 不正确")
    if manifest.get("dataset") != dataset or manifest.get("seed") != 42:
        fail(f"{dataset}: dataset 或 seed 不正确")
    query = manifest.get("query", {})
    if query.get("split") != "query_pool_ms" or query.get("ratio_of_victim_train") != 0.01:
        fail(f"{dataset}: query 定义不正确")
    if query.get("max_budget") != expected["query"] or query.get("planned_budgets") != expected["budgets"]:
        fail(f"{dataset}: query 预算不正确")

    with splits_path.open("r", newline="", encoding="utf-8") as reader_file:
        reader = csv.DictReader(reader_file, delimiter="\t")
        missing = REQUIRED_FIELDS - set(reader.fieldnames or [])
        if missing:
            fail(f"{dataset}: splits.tsv 缺少字段 {sorted(missing)}")
        rows = list(reader)

    victim = rows_for(rows, "victim_train")
    pool = rows_for(rows, "query_pool_ms")
    evaluation = rows_for(rows, "eval_ms")
    for split_name, selected, expected_count in (
        ("victim_train", victim, expected["train"]),
        ("query_pool_ms", pool, expected["query"]),
        ("eval_ms", evaluation, expected["eval"]),
    ):
        if len(selected) != expected_count:
            fail(f"{dataset}: {split_name} 应为 {expected_count}，实际为 {len(selected)}")

    victim_indices = indices(victim, dataset, "victim_train")
    query_indices = indices(pool, dataset, "query_pool_ms")
    eval_indices = indices(evaluation, dataset, "eval_ms")
    if victim_indices != list(range(int(expected["train"]))):
        fail(f"{dataset}: victim_train 必须覆盖官方训练集全部索引")
    if eval_indices != list(range(int(expected["eval"]))):
        fail(f"{dataset}: eval_ms 必须覆盖官方评估集全部索引")
    if {row["source_split"] for row in victim} != {"official_train"}:
        fail(f"{dataset}: victim_train source_split 不正确")
    if {row["source_split"] for row in pool} != {"official_train"}:
        fail(f"{dataset}: query_pool_ms source_split 不正确")
    if {row["source_split"] for row in evaluation} != {expected["eval_source"]}:
        fail(f"{dataset}: eval_ms source_split 不正确")
    if any(row["query_rank"] for row in victim + evaluation):
        fail(f"{dataset}: 非 query 行不得设置 query_rank")
    try:
        query_ranks = [int(row["query_rank"]) for row in pool]
    except ValueError as exc:
        fail(f"{dataset}: query_rank 必须是整数: {exc}")
    if query_ranks != list(range(int(expected["query"]))):
        fail(f"{dataset}: query_rank 必须为连续的 [0, max_budget) 顺序")
    if not set(query_indices).issubset(set(victim_indices)):
        fail(f"{dataset}: query_pool_ms 必须是 victim_train 的子集")
    if len(set(query_indices) & set(victim_indices)) != int(expected["query"]):
        fail(f"{dataset}: victim_train 与 query_pool_ms 的重叠量不正确")
    print(
        f"[OK] {dataset}: victim_train={len(victim)} query_pool_ms={len(pool)} "
        f"eval_ms={len(evaluation)} overlap={len(query_indices)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="验证 reference-random-overlap MS splits。")
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT, help="MS 协议根目录。")
    args = parser.parse_args()
    protocol_root = args.protocol_root.resolve()
    expected_top = {"README.md", *EXPECTED}
    actual_top = {path.name for path in protocol_root.iterdir()} if protocol_root.is_dir() else set()
    if actual_top != expected_top:
        fail(f"MS 顶层应为 {sorted(expected_top)}，实际为 {sorted(actual_top)}")
    for dataset, expected in EXPECTED.items():
        verify_dataset(protocol_root, dataset, expected)
    print("MS split verification passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
