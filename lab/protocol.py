#!/usr/bin/env python3
"""Lab 共用的正式 MS surrogate 训练、选模与边界读取协议。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from exp.MS.train_surrogate.core.artifacts import sha256_file
from exp.MS.train_surrogate.core.config import (
    ATTACK_PROTOCOL_VERSION,
    FORMAL_BATCH_SIZE,
    FORMAL_EPOCHS,
    FORMAL_EVAL_BATCH_SIZE,
    FORMAL_LEARNING_RATE,
    FORMAL_LR_GAMMA,
    FORMAL_LR_STEP,
    FORMAL_MOMENTUM,
    FORMAL_WEIGHT_DECAY,
    HARD_BLACKBOX_ATTACK_PROTOCOL_VERSION,
)
from exp.MS.train_surrogate.core.data import (
    QueryPartition,
    build_eval_dataset,
    build_query_partition_datasets,
    load_query_targets,
    make_query_partition,
)
from exp.MS.train_surrogate.core.engine import (
    EvalReference,
    collect_eval_reference,
    evaluate_surrogate,
    select_validation_best,
)
from exp.MS.train_surrogate.defense import ExposureFreezer
from exp.MS.train_victim.common.trainer import build_generator, seed_worker


@dataclass(frozen=True)
class QueryData:
    """同一个 query budget 派生出的训练与选模数据。"""

    train: Dataset
    validation: Dataset
    partition: QueryPartition
    target_path: Path
    target_sha256: str
    manifest: dict[str, object]


@dataclass(frozen=True)
class EvalData:
    """只在 checkpoint 固定后构造的 eval_ms loader 与 victim 参考。"""

    loader: DataLoader
    reference: EvalReference


def prepare_soft_query(
    *,
    dataset: str,
    model: str,
    budget: int,
    seed: int,
    dataset_root: Path,
    protocol_root: Path,
) -> QueryData:
    indices, posteriors, labels, target_path, manifest = load_query_targets(
        protocol_root,
        dataset,
        model,
        budget,
        "soft",
    )
    partition = make_query_partition(indices, seed=seed)
    train, validation = build_query_partition_datasets(
        dataset,
        dataset_root,
        indices,
        posteriors,
        labels,
        partition,
    )
    return QueryData(
        train=train,
        validation=validation,
        partition=partition,
        target_path=target_path,
        target_sha256=sha256_file(target_path),
        manifest=manifest,
    )


def build_query_loaders(
    query: QueryData,
    *,
    device: torch.device,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    pin_memory = device.type == "cuda"
    return (
        DataLoader(
            query.train,
            batch_size=FORMAL_BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=build_generator(seed),
        ),
        DataLoader(
            query.validation,
            batch_size=FORMAL_EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=build_generator(seed, offset=1),
        ),
    )


def train_validation_best(
    model: torch.nn.Module,
    query: QueryData,
    *,
    device: torch.device,
    num_workers: int,
    seed: int,
    freezer: ExposureFreezer | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    train_loader, validation_loader = build_query_loaders(
        query,
        device=device,
        num_workers=num_workers,
        seed=seed,
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Lab surrogate 没有可训练参数。")
    optimizer = torch.optim.SGD(
        parameters,
        lr=FORMAL_LEARNING_RATE,
        momentum=FORMAL_MOMENTUM,
        weight_decay=FORMAL_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=FORMAL_LR_STEP,
        gamma=FORMAL_LR_GAMMA,
    )
    selection, history = select_validation_best(
        model,
        train_loader,
        validation_loader,
        optimizer,
        scheduler,
        device,
        "soft",
        FORMAL_EPOCHS,
        query.partition.validation_size,
        freezer,
    )
    return dict(selection), [dict(row) for row in history]


def prepare_eval(
    victim: torch.nn.Module,
    *,
    dataset: str,
    dataset_root: Path,
    protocol_root: Path,
    device: torch.device,
    num_workers: int,
    seed: int,
) -> EvalData:
    dataset_object = build_eval_dataset(
        dataset,
        dataset_root,
        protocol_root,
        None,
    )
    loader = DataLoader(
        dataset_object,
        batch_size=FORMAL_EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=build_generator(seed, offset=2),
    )
    victim = victim.to(device)
    reference = collect_eval_reference(victim, loader, device)
    return EvalData(loader=loader, reference=reference)


def evaluate_once(
    model: torch.nn.Module,
    evaluation: EvalData,
    device: torch.device,
) -> dict[str, int | float]:
    return {
        **evaluate_surrogate(
            model,
            evaluation.loader,
            evaluation.reference,
            device,
        ),
        "eval_passes": 1,
    }


def load_formal_bound(
    path: Path,
    artifact_id: str,
    *,
    label_mode: str,
    model: str,
    dataset: str,
    budget: int,
) -> dict[str, object]:
    import json

    if label_mode not in {"soft", "hard"}:
        raise ValueError(f"未知黑盒标签模式：{label_mode}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    attack_protocol = (
        ATTACK_PROTOCOL_VERSION
        if label_mode == "soft"
        else HARD_BLACKBOX_ATTACK_PROTOCOL_VERSION
    )
    expected = {
        "schema_version": 3,
        "artifact_id": artifact_id,
        "attack_protocol": attack_protocol,
        "dataset": dataset,
        "victim_model": model,
        "query_budget": budget,
        "label_mode": label_mode,
        "query_transform": "test",
        "lr_step": FORMAL_LR_STEP,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"正式边界 {path} 的 {field}={payload.get(field)!r}，"
                f"期望 {value!r}。"
            )
    primary = payload.get("primary", {})
    if primary.get("checkpoint") != "best.pth":
        raise ValueError(f"正式边界 {path} 未使用 validation-best checkpoint。")
    partition = payload.get("query_partition", {})
    if (
        partition.get("train_size") != 400
        or partition.get("validation_size") != 100
        or partition.get("seed") != 42
        or partition.get("seed_offset") != 100
    ):
        raise ValueError(f"正式边界 {path} 的 query 划分不正确。")
    result = payload.get("result")
    if not isinstance(result, dict) or result.get("eval_passes") != 1:
        raise ValueError(f"正式边界 {path} 缺少单次 eval_ms 结果。")
    return {
        "artifact_id": artifact_id,
        "run_id": payload["run_id"],
        "label_mode": label_mode,
        "attack_protocol": attack_protocol,
        "primary": primary,
        "protection": payload["protection"],
        "result": result,
        "path": str(path),
        "sha256": sha256_file(path),
    }


def protocol_metadata(query: QueryData) -> dict[str, object]:
    return {
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "query_budget": query.partition.budget,
        "query_train_size": query.partition.train_size,
        "query_validation_size": query.partition.validation_size,
        "query_partition": query.partition.to_metadata(),
        "label_mode": "soft",
        "query_transform": "test",
        "training_mode": "finetune",
        "max_epochs": FORMAL_EPOCHS,
        "batch_size": FORMAL_BATCH_SIZE,
        "eval_batch_size": FORMAL_EVAL_BATCH_SIZE,
        "optimizer": "SGD",
        "learning_rate": FORMAL_LEARNING_RATE,
        "momentum": FORMAL_MOMENTUM,
        "weight_decay": FORMAL_WEIGHT_DECAY,
        "lr_scheduler": "StepLR",
        "lr_step": FORMAL_LR_STEP,
        "lr_gamma": FORMAL_LR_GAMMA,
        "checkpoint": "best.pth",
        "checkpoint_selection": "minimum_validation_soft_cross_entropy",
        "checkpoint_tie_break": "earliest_epoch",
        "eval_ms_passes_per_case": 1,
    }
