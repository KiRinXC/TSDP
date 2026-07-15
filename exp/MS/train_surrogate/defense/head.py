#!/usr/bin/env python3
"""只保护 victim 分类头的固定控制策略。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import DefenseOptions, MaskSelection


HEAD_STATE_NAMES = ("last_linear.weight", "last_linear.bias")


def build_head_only(
    defense: str,
    victim_model: nn.Module,
    options: DefenseOptions,
) -> MaskSelection:
    if defense != "head_only":
        raise ValueError(f"分类头 builder 收到未知策略：{defense}")
    if any(
        value is not None
        for value in (
            options.protected_units,
            options.protected_layers,
            options.protected_scalars,
        )
    ):
        raise ValueError("head_only 不接受 unit、layer 或 scalar 选择参数。")

    classifier = getattr(victim_model, "last_linear", None)
    if not isinstance(classifier, nn.Linear) or classifier.bias is None:
        raise ValueError("head_only 要求模型使用带 bias 的 last_linear 分类头。")
    state = victim_model.state_dict()
    missing = set(HEAD_STATE_NAMES) - set(state)
    if missing:
        raise ValueError(f"分类头包含未知 state：{sorted(missing)}")

    selected = set(HEAD_STATE_NAMES)
    masks = {
        name: torch.full_like(value, name in selected, dtype=torch.bool)
        for name, value in state.items()
    }
    return MaskSelection(
        masks=masks,
        classifier_protected=True,
        head_mode="replace",
    )
