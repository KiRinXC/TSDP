#!/usr/bin/env python3
"""核对 ResNet18+CIFAR-100 正式 surrogate 新协议产物。"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.MS.train_surrogate.core.artifacts import INDEX_FIELDS  # noqa: E402
from exp.MS.train_surrogate.core.config import (  # noqa: E402
    ATTACK_PROTOCOL_VERSION,
    HARD_BLACKBOX_ATTACK_PROTOCOL_VERSION,
)


MODEL = "resnet18"
DATASET = "c100"
BUDGET = 500
EVAL_COUNT = 10_000
VICTIM_CORRECT = 6_182
RESULTS_ROOT = ROOT / "results" / "MS" / MODEL / DATASET
WEIGHTS_ROOT = ROOT / "weights" / "MS" / "surrogate" / MODEL / DATASET


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as reader_file:
        reader = csv.DictReader(reader_file, delimiter="\t")
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields or any(None in row for row in rows):
        raise ValueError(f"TSV 表头或列数无效：{path}")
    return fields, rows


def expected_artifacts() -> tuple[set[str], set[str]]:
    baseline = load_json(ROOT / "exp" / "MS" / "train_surrogate" / "baseline.json")
    planned = {
        str(config["id"])
        for group in ("layer_sweep", "large_weight_sweep")
        for config in baseline[group]["configurations"]
    }
    if len(planned) != 32:
        raise ValueError("baseline.json 不是 32 个固定扫描点。")
    ordinary = planned | {
        "no_protection",
        "full_protection",
        "hard_blackbox",
        "head_only",
        "tensorshield",
    }
    return ordinary, ordinary | {"teeslice"}


def validate_partition(payload: dict, path: Path) -> None:
    partition = payload.get("query_partition", {})
    expected = {
        "method": "fixed_seeded_random_partition_of_query_rank",
        "budget": BUDGET,
        "train_size": 400,
        "validation_size": 100,
        "seed": 42,
        "seed_offset": 100,
    }
    for field, value in expected.items():
        if partition.get(field) != value:
            raise ValueError(f"{path} 的 query_partition.{field} 不正确。")
    train_ranks = partition.get("train_ranks", [])
    validation_ranks = partition.get("validation_ranks", [])
    if (
        len(train_ranks) != 400
        or len(validation_ranks) != 100
        or set(train_ranks) & set(validation_ranks)
        or set(train_ranks) | set(validation_ranks) != set(range(BUDGET))
    ):
        raise ValueError(f"{path} 的 query train/validation 划分无效。")


def validate_history(
    artifact_id: str,
    payload: dict,
    protocol: str,
) -> None:
    path = WEIGHTS_ROOT / artifact_id / "train.log.tsv"
    fields, rows = read_tsv(path)
    forbidden = {
        "eval_count",
        "victim_correct",
        "surrogate_correct",
        "agreement_count",
        "victim_acc",
        "surrogate_acc",
        "fidelity",
        "posterior_kl",
    }
    if forbidden & set(fields):
        raise ValueError(f"{path} 泄漏了逐 epoch eval_ms 字段。")
    expected_rows = 1 if artifact_id == "no_protection" else 100
    if len(rows) != expected_rows:
        raise ValueError(f"{path} 行数应为 {expected_rows}，实际为 {len(rows)}。")
    epochs = [int(row["epoch"]) for row in rows]
    expected_epochs = [0] if artifact_id == "no_protection" else list(range(1, 101))
    if epochs != expected_epochs:
        raise ValueError(f"{path} epoch 不完整或顺序错误。")
    for row in rows:
        if int(row["validation_count"]) != 100:
            raise ValueError(f"{path} validation_count 不是 100。")
        expected_query_count = 0 if artifact_id == "no_protection" else 400
        if int(row["query_count"]) != expected_query_count:
            raise ValueError(f"{path} query_count 不是 {expected_query_count}。")
    best_rows = [row for row in rows if row["is_best"] == "True"]
    if not best_rows:
        raise ValueError(f"{path} 没有 best 轨迹。")
    minimum = min(float(row["validation_loss"]) for row in rows)
    earliest = next(
        int(row["epoch"])
        for row in rows
        if float(row["validation_loss"]) == minimum
    )
    primary_epoch = int(payload["primary"]["epoch"])
    if primary_epoch != earliest or best_rows[-1]["epoch"] != str(earliest):
        raise ValueError(f"{path} 没有选择 validation loss 最低的最早 epoch。")

    checkpoint_path = WEIGHTS_ROOT / artifact_id / "best.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if (WEIGHTS_ROOT / artifact_id / "end.pth").exists():
        raise ValueError(f"{artifact_id} 仍残留失效 surrogate end.pth。")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("attack_protocol") != protocol
        or checkpoint.get("epoch") != primary_epoch
        or checkpoint.get("run_id") != payload.get("run_id")
        or not isinstance(checkpoint.get("state_dict"), dict)
    ):
        raise ValueError(f"{checkpoint_path} 元数据与 metrics.json 不一致。")


def validate_ordinary(artifact_id: str) -> dict:
    path = RESULTS_ROOT / artifact_id / "metrics.json"
    payload = load_json(path)
    label_mode = "hard" if artifact_id == "hard_blackbox" else "soft"
    protocol = (
        HARD_BLACKBOX_ATTACK_PROTOCOL_VERSION
        if label_mode == "hard"
        else ATTACK_PROTOCOL_VERSION
    )
    expected = {
        "schema_version": 3,
        "protocol": "MS",
        "attack_protocol": protocol,
        "artifact_id": artifact_id,
        "dataset": DATASET,
        "victim_model": MODEL,
        "query_budget": BUDGET,
        "label_mode": label_mode,
        "query_transform": "test",
        "lr_step": 60,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"{path} 的 {field}={payload.get(field)!r}，期望 {value!r}。")
    validate_partition(payload, path)
    primary = payload.get("primary", {})
    expected_metric = (
        "identity_epoch_0"
        if artifact_id == "no_protection"
        else (
            "validation_hard_cross_entropy"
            if label_mode == "hard"
            else "validation_soft_cross_entropy"
        )
    )
    if (
        primary.get("checkpoint") != "best.pth"
        or primary.get("selection_metric") != expected_metric
    ):
        raise ValueError(f"{path} 的 primary checkpoint 或 selection metric 不正确。")
    epoch = int(primary.get("epoch", -1))
    if (artifact_id == "no_protection" and epoch != 0) or (
        artifact_id != "no_protection" and not 1 <= epoch <= 100
    ):
        raise ValueError(f"{path} 的 primary epoch 不正确。")
    result = payload.get("result", {})
    if (
        result.get("eval_count") != EVAL_COUNT
        or result.get("eval_passes") != 1
        or result.get("victim_correct") != VICTIM_CORRECT
        or result.get("victim_acc") != VICTIM_CORRECT / EVAL_COUNT
    ):
        raise ValueError(f"{path} 不是选模后唯一一次完整 eval_ms 结果。")
    if artifact_id == "no_protection" and (
        result.get("surrogate_correct") != VICTIM_CORRECT
        or result.get("agreement_count") != EVAL_COUNT
        or result.get("fidelity") != 1.0
    ):
        raise ValueError("no_protection 没有逐样本复现 victim。")
    protection = payload.get("protection", {})
    if artifact_id in {"full_protection", "hard_blackbox"} and (
        protection.get("protected_param_ratio") != 1.0
        or protection.get("defense") != "full_protection"
    ):
        raise ValueError(f"{artifact_id} 不是完整参数保护。")
    if not (WEIGHTS_ROOT / artifact_id / "protection_mask.pt").is_file():
        raise FileNotFoundError(WEIGHTS_ROOT / artifact_id / "protection_mask.pt")
    validate_history(artifact_id, payload, protocol)
    return payload


def validate_teeslice() -> None:
    path = RESULTS_ROOT / "teeslice" / "metrics.json"
    payload = load_json(path)
    if (
        payload.get("schema_version") != 3
        or payload.get("attack_protocol") != ATTACK_PROTOCOL_VERSION
        or payload.get("comparison_scope") != "standalone_reproduction"
        or payload.get("artifact_id") != "teeslice"
        or payload.get("label_mode") != "soft"
        or payload.get("primary", {}).get("checkpoint") != "best.pth"
        or payload.get("primary", {}).get("selection_metric")
        != "validation_soft_cross_entropy"
    ):
        raise ValueError("TEESlice surrogate 不是 standalone validation-best 新协议。")
    validate_partition(payload, path)
    result = payload.get("result", {})
    whitebox = payload.get("whitebox", {})
    if (
        result.get("eval_count") != EVAL_COUNT
        or result.get("eval_passes") != 1
        or whitebox.get("eval_count") != EVAL_COUNT
        or whitebox.get("eval_passes") != 1
        or whitebox.get("agreement_count") != EVAL_COUNT
    ):
        raise ValueError("TEESlice 黑盒或白盒评估不完整。")
    validate_history("teeslice", payload, ATTACK_PROTOCOL_VERSION)
    if not (WEIGHTS_ROOT / "teeslice" / "topology.json").is_file():
        raise FileNotFoundError(WEIGHTS_ROOT / "teeslice" / "topology.json")
    if not (RESULTS_ROOT / "teeslice" / "victim.json").is_file():
        raise FileNotFoundError("TEESlice surrogate 覆盖时误删了 victim.json。")


def validate_index(ordinary_artifacts: set[str]) -> None:
    fields, rows = read_tsv(RESULTS_ROOT / "metrics.tsv")
    if fields != INDEX_FIELDS:
        raise ValueError("正式 metrics.tsv 表头不是新协议字段。")
    if {row["artifact_id"] for row in rows} != ordinary_artifacts:
        raise ValueError("正式 metrics.tsv artifact 集合不完整或混入 TEESlice。")
    if any(row["primary_checkpoint"] != "best.pth" for row in rows):
        raise ValueError("正式 metrics.tsv 仍引用非 best checkpoint。")


def main() -> int:
    ordinary_artifacts, all_artifacts = expected_artifacts()
    actual_artifacts = {
        path.parent.name for path in RESULTS_ROOT.glob("*/metrics.json")
    }
    if actual_artifacts != all_artifacts:
        raise ValueError(
            f"正式 artifact 集合不一致：missing={sorted(all_artifacts - actual_artifacts)}, "
            f"extra={sorted(actual_artifacts - all_artifacts)}"
        )
    partitions = []
    for artifact_id in sorted(ordinary_artifacts):
        payload = validate_ordinary(artifact_id)
        partitions.append(payload["query_partition"]["train_ranks_sha256"])
    if len(set(partitions)) != 1:
        raise ValueError("普通正式 artifact 没有共享同一个 query train 划分。")
    validate_teeslice()
    validate_index(ordinary_artifacts)
    print(
        f"[OK] formal MS results: ordinary={len(ordinary_artifacts)} "
        f"standalone=1 protocol={ATTACK_PROTOCOL_VERSION}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
