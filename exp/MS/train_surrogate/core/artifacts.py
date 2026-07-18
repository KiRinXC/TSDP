#!/usr/bin/env python3
"""surrogate checkpoint、日志和结果索引。"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

from .config import REPO_ROOT


INDEX_FIELDS = [
    "artifact_id", "plan_id", "run_id", "attack_protocol", "dataset", "victim_model", "defense",
    "protected_layer_count", "source_ratio", "training_mode", "label_mode", "query_transform",
    "query_budget", "query_train_size", "query_validation_size", "query_split_seed",
    "query_sampler_seed", "lr_step",
    "protected_unit_count", "protection_mask_sha256", "protected_scalar_count", "protected_param_count",
    "total_param_count", "protected_param_ratio", "head_mode", "primary_checkpoint", "primary_epoch",
    "selection_metric", "validation_loss", "validation_match",
    "eval_count", "victim_correct",
    "surrogate_correct", "agreement_count", "victim_acc", "surrogate_acc", "fidelity", "posterior_kl_sum",
    "posterior_kl", "metrics_path",
]
HISTORY_FIELDS = [
    "epoch", "learning_rate", "query_count", "query_loss_sum", "query_loss", "query_match_count", "query_match",
    "validation_count", "validation_loss", "validation_kl", "validation_match_count",
    "validation_match", "is_best",
]
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    model_name: str,
    dataset_name: str,
    run_id: str,
    attack_protocol: str,
    metrics: dict[str, int | float],
) -> None:
    checkpoint = {
        "schema_version": 2,
        "protocol": "MS",
        "attack_protocol": attack_protocol,
        "arch": model_name,
        "dataset": dataset_name,
        "run_id": run_id,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(checkpoint, path)


def make_run_id(config: dict[str, object]) -> str:
    canonical = json.dumps(config, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def make_artifact_id(plan_id: str | None, defense: str, run_id: str) -> str:
    if plan_id is not None:
        artifact_id = plan_id
    elif defense in {"no_protection", "full_protection", "head_only", "tensorshield"}:
        artifact_id = defense
    else:
        artifact_id = run_id
    parts = artifact_id.split("_")
    if len(parts) > 2 or not all(part and part.isalnum() and part == part.lower() for part in parts):
        raise ValueError(f"artifact_id 不符合目录命名约定：{artifact_id}")
    return artifact_id


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as writer:
        json.dump(payload, writer, ensure_ascii=False, indent=2)
        writer.write("\n")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_history_row(path: Path, row: dict[str, object], initialize: bool = False) -> None:
    mode = "w" if initialize else "a"
    with path.open(mode, newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(
            writer_file,
            fieldnames=HISTORY_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        if initialize:
            writer.writeheader()
        else:
            writer.writerow(row)


def update_index(
    path: Path,
    row: dict[str, object],
    *,
    replace_incompatible: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        rows: list[dict[str, str]] = []
        if path.is_file():
            with path.open("r", newline="", encoding="utf-8") as reader_file:
                reader = csv.DictReader(reader_file, delimiter="\t")
                if reader.fieldnames != INDEX_FIELDS:
                    if not replace_incompatible:
                        raise ValueError(f"结果索引字段不兼容：{path}")
                else:
                    rows = [
                        existing
                        for existing in reader
                        if existing["run_id"] != row["run_id"]
                        and existing["artifact_id"] != row["artifact_id"]
                    ]
        rows.append({name: str(row[name]) for name in INDEX_FIELDS})
        rows.sort(key=lambda existing: existing["artifact_id"])
        with temporary_path.open("w", newline="", encoding="utf-8") as writer_file:
            writer = csv.DictWriter(
                writer_file,
                fieldnames=INDEX_FIELDS,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            writer_file.flush()
            os.fsync(writer_file.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
        fcntl.flock(directory_fd, fcntl.LOCK_UN)
        os.close(directory_fd)
