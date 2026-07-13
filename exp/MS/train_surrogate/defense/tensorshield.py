#!/usr/bin/env python3
"""根据作者确认 rank 构造 TensorShield 的固定保护 mask。"""

from __future__ import annotations

import torch
import torch.nn as nn

from selector import (
    AUTHOR_RESNET18_C100_ELIGIBLE_RANK,
    PUBLISHED_RESNET18_C100_STATES,
    PUBLISHED_RESNET18_C100_WEIGHTS,
)

from .base import DefenseOptions, MaskSelection


def build_tensorshield(
    defense: str,
    victim_model: nn.Module,
    options: DefenseOptions,
) -> MaskSelection:
    if defense != "tensorshield":
        raise ValueError(f"TensorShield builder 收到未知策略：{defense}")
    if any(
        value is not None
        for value in (
            options.protected_units,
            options.protected_layers,
            options.protected_scalars,
        )
    ):
        raise ValueError("tensorshield 不接受 unit、layer 或 scalar 选择参数。")
    if options.architecture != "resnet18":
        raise ValueError("当前只固定了 ResNet18+CIFAR-100 的 TensorShield 作者 rank。")
    classifier = getattr(victim_model, "last_linear", None)
    if not isinstance(classifier, nn.Linear) or classifier.out_features != 100:
        raise ValueError("当前只固定了 ResNet18+CIFAR-100 的 TensorShield 作者 rank。")

    state = victim_model.state_dict()
    selected = set(PUBLISHED_RESNET18_C100_STATES)
    missing = selected - set(state)
    if missing:
        raise ValueError(f"作者固定保护集合包含未知 state：{sorted(missing)}")
    if set(AUTHOR_RESNET18_C100_ELIGIBLE_RANK[:10]) != set(
        PUBLISHED_RESNET18_C100_WEIGHTS
    ):
        raise RuntimeError("作者 eligible Top-10 与 Figure 12(d) 固定集合不一致。")
    masks = {
        name: torch.full_like(value, name in selected, dtype=torch.bool)
        for name, value in state.items()
    }
    return MaskSelection(
        masks=masks,
        classifier_protected=True,
        head_mode="replace",
    )
