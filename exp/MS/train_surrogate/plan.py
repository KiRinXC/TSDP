#!/usr/bin/env python3
"""生成 ResNet18+C100 baseline 的固定保护计划，不启动训练。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[3]
TRAIN_VICTIM_ROOT = ROOT / "exp" / "MS" / "train_victim"
for path in (ROOT, TRAIN_VICTIM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.trainer import configure_reproducibility  # noqa: E402
from core.artifacts import display_path, sha256_file  # noqa: E402
from core.config import ATTACK_PROTOCOL_VERSION  # noqa: E402
from core.data import (
    QUERY_SPLIT_SEED_OFFSET,
    build_victim,
    make_query_partition,
    read_query_indices,
)  # noqa: E402
from defense import (  # noqa: E402
    build_mask_selection,
    build_public_model as build_seeded_public_model,
    protection_mask_sha256,
)
from defense.base import DefenseOptions  # noqa: E402
from defense.mask import build_protection_mask_payload  # noqa: E402
from models import imagenet as imagenet_models  # noqa: E402


MODEL_NAME = "resnet18"
DATASET_NAME = "c100"
NUM_CLASSES = 100
QUERY_BUDGET = 500
SEED = 42
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.5
WEIGHT_DECAY = 5e-4
LR_STEP = 60
LR_GAMMA = 0.1
LAYER_COUNTS = (2, 4, 6, 8, 10, 12, 14, 16)
MAGNITUDE_ANCHORS = (
    ("0.01", 1, 100),
    ("0.1", 1, 10),
    ("0.3", 3, 10),
    ("0.5", 1, 2),
    ("0.7", 7, 10),
    ("0.8", 4, 5),
    ("0.9", 9, 10),
    ("0.95", 19, 20),
)


def build_public_model(weight_path: Path) -> nn.Module:
    return build_seeded_public_model(
        imagenet_models.resnet18,
        MODEL_NAME,
        weight_path,
        NUM_CLASSES,
        initialization_seed=SEED,
    )


def magnitude_eligible_count(model: nn.Module) -> int:
    return sum(
        module.weight.numel()
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear))
    )


def summarize_selection(
    *,
    config_id: str,
    defense: str,
    victim_model: nn.Module,
    public_model: nn.Module,
    protected_layers: str | None = None,
    protected_scalars: int | None = None,
    source_ratio: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    selection = build_mask_selection(
        defense,
        victim_model,
        public_model,
        DefenseOptions(
            architecture=MODEL_NAME,
            protected_units=None,
            protected_layers=protected_layers,
            protected_scalars=protected_scalars,
        ),
    )
    total_params = 0
    protected_params = 0
    for name, parameter in victim_model.named_parameters():
        total_params += parameter.numel()
        protected_params += int(selection.masks[name].sum().item())

    summary: dict[str, object] = {
        "id": config_id,
        "defense": defense,
        "protected_layers": protected_layers,
        "protected_scalars": protected_scalars,
        "source_ratio": source_ratio,
        "protected_unit_count": sum(bool(mask.any()) for mask in selection.masks.values()),
        "protected_param_count": protected_params,
        "total_param_count": total_params,
        "protected_param_ratio": protected_params / total_params,
        "classifier_protected": selection.classifier_protected,
        "head_mode": selection.head_mode,
        "magnitude_eligible_count": selection.magnitude_eligible_count,
        "protection_mask_sha256": protection_mask_sha256(selection.masks),
    }
    return summary, build_protection_mask_payload(selection.masks)


def main() -> int:
    manifest_path = Path(__file__).resolve().with_name("baseline.json")
    mask_path = ROOT / "weights" / "MS" / "surrogate" / MODEL_NAME / DATASET_NAME / "baseline.pt"
    victim_path = ROOT / "weights" / "MS" / "victim" / MODEL_NAME / DATASET_NAME / "best.pth"
    official_path = ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    protocol_manifest_path = ROOT / "dataset" / "MS" / DATASET_NAME / "manifest.json"
    posterior_path = ROOT / "dataset" / "MS" / DATASET_NAME / MODEL_NAME / "posteriors.pt"

    protocol_manifest = json.loads(protocol_manifest_path.read_text(encoding="utf-8"))
    if protocol_manifest.get("query", {}).get("max_budget") != QUERY_BUDGET:
        raise ValueError("C100 manifest 的正式 query budget 不再是 500。")
    query_partition = make_query_partition(
        read_query_indices(ROOT / "dataset" / "MS", DATASET_NAME)[:QUERY_BUDGET],
        seed=SEED,
    )

    configure_reproducibility(SEED, deterministic=True)
    victim_model, _ = build_victim(MODEL_NAME, NUM_CLASSES, victim_path)
    public_model = build_public_model(official_path)

    layer_configs: list[dict[str, object]] = []
    magnitude_configs: list[dict[str, object]] = []
    mask_payloads: dict[str, dict[str, object]] = {}

    for layer_count in LAYER_COUNTS:
        half = layer_count // 2
        ranges = {
            "shallow": f"1-{layer_count}",
            "middle": f"{10 - half}-{9 + half}",
            "deep": f"{19 - layer_count}-18",
        }
        for defense, protected_layers in ranges.items():
            config_id = f"{defense}_{layer_count:02d}"
            summary, masks = summarize_selection(
                config_id=config_id,
                defense=defense,
                victim_model=victim_model,
                public_model=public_model,
                protected_layers=protected_layers,
            )
            summary["protected_layer_count"] = layer_count
            layer_configs.append(summary)
            mask_payloads[config_id] = masks

    eligible_count = magnitude_eligible_count(public_model)
    for ordinal, (ratio_text, numerator, denominator) in enumerate(MAGNITUDE_ANCHORS, start=1):
        protected_scalars = eligible_count * numerator // denominator
        config_id = f"large_{ordinal:02d}"
        summary, masks = summarize_selection(
            config_id=config_id,
            defense="large_weight",
            victim_model=victim_model,
            public_model=public_model,
            protected_scalars=protected_scalars,
            source_ratio=ratio_text,
        )
        magnitude_configs.append(summary)
        mask_payloads[config_id] = masks

    if len(layer_configs) != 24 or len(magnitude_configs) != 8 or len(mask_payloads) != 32:
        raise RuntimeError("baseline 保护计划数量不正确。")

    mask_package = {
        "schema_version": 1,
        "protocol": "MS",
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "config_count": len(mask_payloads),
        "configurations": mask_payloads,
    }
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mask_package, mask_path)

    manifest = {
        "schema_version": 1,
        "protocol": "MS",
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "purpose": "resnet18_c100_baseline_protection_plan",
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "seed": SEED,
        "query_budget": QUERY_BUDGET,
        "query_train_size": query_partition.train_size,
        "query_validation_size": query_partition.validation_size,
        "query_split_seed": SEED,
        "query_split_seed_offset": QUERY_SPLIT_SEED_OFFSET,
        "query_partition": query_partition.to_metadata(),
        "label_mode": "soft",
        "query_transform": "test",
        "training_mode": "finetune",
        "primary_checkpoint": "best.pth",
        "checkpoint_selection": "minimum_validation_soft_cross_entropy",
        "checkpoint_tie_break": "earliest_epoch",
        "eval_ms_passes": 1,
        "training_hyperparameters": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "lr_step": LR_STEP,
            "lr_gamma": LR_GAMMA,
        },
        "protocol_manifest": display_path(protocol_manifest_path),
        "protocol_manifest_sha256": sha256_file(protocol_manifest_path),
        "posterior_path": display_path(posterior_path),
        "posterior_sha256": sha256_file(posterior_path),
        "victim_checkpoint": display_path(victim_path),
        "victim_checkpoint_sha256": sha256_file(victim_path),
        "official_weight": display_path(official_path),
        "official_weight_sha256": sha256_file(official_path),
        "mask_package": display_path(mask_path),
        "mask_config_count": len(mask_payloads),
        "layer_sweep": {
            "layer_counts": list(LAYER_COUNTS),
            "middle_rule": "symmetric_around_layers_9_and_10",
            "configurations": layer_configs,
        },
        "large_weight_sweep": {
            "source": "../Demo/TEESlice-artifact/model-stealing/knockoff/adversary/train_mag.py",
            "selection": "exact_global_topk_by_public_weight_absolute_value",
            "eligible_modules": ["Conv2d.weight", "BatchNorm2d.weight", "Linear.weight"],
            "magnitude_eligible_count": eligible_count,
            "source_ratios": [ratio for ratio, _, _ in MAGNITUDE_ANCHORS],
            "configurations": magnitude_configs,
        },
        "training_run_count": 32,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] baseline 配置: {len(mask_payloads)}")
    print(f"[INFO] 配置清单: {manifest_path}")
    print(f"[INFO] mask 包: {mask_path}")
    print(f"[INFO] large_weight eligible scalars: {eligible_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
