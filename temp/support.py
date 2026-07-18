#!/usr/bin/env python3
"""temp 中交叉残差与因果残差实验共用的最小依赖。"""

from __future__ import annotations

from pathlib import Path
import sys

import torch.nn as nn
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_VICTIM_ROOT = REPO_ROOT / "exp" / "MS" / "train_victim"
for search_root in (REPO_ROOT, TRAIN_VICTIM_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from exp.MS.train_surrogate.core.artifacts import sha256_file
from exp.MS.train_surrogate.core.config import MODEL_SPECS, resolve_device
from exp.MS.train_surrogate.core.data import (
    build_victim,
    hash_integer_sequence,
    make_query_partition,
    read_query_indices,
)
from exp.MS.train_surrogate.defense import build_public_model as build_seeded_public_model
from exp.MS.train_victim.common.trainer import (
    build_generator,
    build_public_split_dataset,
    build_transforms,
    configure_reproducibility,
    seed_worker,
)
from models import imagenet as imagenet_models


MODEL = "resnet18"
DATASET = "c100"
NUM_CLASSES = 100
QUERY_BUDGET = 500
SEED = 42


def digest_indices(indices) -> str:
    """对 source index 顺序做与正式 query 划分一致的摘要。"""

    return hash_integer_sequence([int(index) for index in indices])


def discovery_indices() -> tuple[list[int], dict[str, object]]:
    """只返回 query-train 的 400 条图像，保留 validation 专用于选模。"""

    query_indices = read_query_indices(REPO_ROOT / "dataset" / "MS", DATASET)
    if len(query_indices) != QUERY_BUDGET:
        raise RuntimeError(
            f"query_pool_ms 数量为 {len(query_indices)}，期望 {QUERY_BUDGET}。"
        )
    partition = make_query_partition(query_indices, seed=SEED)
    return list(partition.train_source_indices), partition.to_metadata()


def build_public_model() -> nn.Module:
    """按普通 surrogate 的 canonical RNG 轨迹构造公开初始化。"""

    factory_name, weight_filename = MODEL_SPECS[MODEL]
    factory = getattr(imagenet_models, factory_name)
    return build_seeded_public_model(
        factory,
        factory_name,
        REPO_ROOT / "weights" / "pre_train" / weight_filename,
        NUM_CLASSES,
        initialization_seed=SEED,
    )


def initialize_masked_surrogate(
    victim: nn.Module,
    masks: dict[str, torch.Tensor],
) -> nn.Module:
    """让未保护状态直接暴露，保护状态保留公开初始化。"""

    surrogate = build_public_model()
    victim_state = victim.state_dict()
    surrogate_state = surrogate.state_dict()
    if set(masks) != set(victim_state) or set(surrogate_state) != set(victim_state):
        raise ValueError("victim、surrogate 与保护 mask 的 state 集合不一致。")
    for name, current in surrogate_state.items():
        protected = masks[name]
        exposed = victim_state[name]
        if protected.shape != current.shape or exposed.shape != current.shape:
            raise ValueError(f"{name} 的 victim/public/mask 形状不一致。")
        protected = protected.to(device=current.device, dtype=torch.bool)
        exposed = exposed.to(device=current.device, dtype=current.dtype)
        if protected.all():
            continue
        if protected.any():
            surrogate_state[name] = torch.where(
                protected,
                current,
                exposed,
            )
        else:
            surrogate_state[name] = exposed.clone()
    surrogate.load_state_dict(surrogate_state, strict=True)
    return surrogate
