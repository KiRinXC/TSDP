#!/usr/bin/env python3
"""大权重标量保护策略。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import DefenseOptions, MaskSelection
from .mask import state_name


def build_magnitude_masks(
    public_model: nn.Module,
    protected_count: int,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """按公开预训练权重绝对值全局排序，选择受保护标量。"""
    state = public_model.state_dict()
    masks = {name: torch.zeros_like(value, dtype=torch.bool) for name, value in state.items()}
    eligible: list[tuple[str, nn.Module, torch.Tensor]] = []
    flat_values: list[torch.Tensor] = []
    for module_name, module in public_model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            continue
        name = state_name(module_name, "weight")
        values = state[name].detach().abs().reshape(-1).cpu()
        eligible.append((module_name, module, values))
        flat_values.append(values)

    all_values = torch.cat(flat_values)
    total = all_values.numel()
    if not 0 < protected_count <= total:
        raise ValueError(f"protected_scalars 必须在 1-{total} 之间。")
    selected = torch.topk(all_values, protected_count, largest=True, sorted=False).indices
    flat_mask = torch.zeros(total, dtype=torch.bool)
    flat_mask[selected] = True

    offset = 0
    for module_name, module, values in eligible:
        count = values.numel()
        weight_name = state_name(module_name, "weight")
        weight_mask = flat_mask[offset : offset + count].reshape_as(state[weight_name])
        masks[weight_name] = weight_mask
        if isinstance(module, nn.BatchNorm2d):
            for local_name in ("bias", "running_mean", "running_var"):
                name = state_name(module_name, local_name)
                if name in masks:
                    masks[name] = weight_mask.clone()
        offset += count
    return masks, total, protected_count


def build_large_weight(
    defense: str,
    public_model: nn.Module,
    options: DefenseOptions,
) -> MaskSelection:
    if defense != "large_weight":
        raise ValueError(f"大权重 builder 收到未知策略：{defense}")
    if options.protected_units is not None:
        raise ValueError("large_weight 不接受 --protected-units。")
    if options.protected_layers is not None:
        raise ValueError("large_weight 不接受 --protected-layers。")
    if options.protected_scalars is None:
        raise ValueError("large_weight 必须指定 --protected-scalars。")
    masks, eligible_count, protected_count = build_magnitude_masks(public_model, options.protected_scalars)
    head_weight_mask = masks["last_linear.weight"]
    head_bias_mask = masks["last_linear.bias"]
    classifier_protected = bool(head_weight_mask.any() or head_bias_mask.any())
    classifier_fully_protected = bool(head_weight_mask.all() and head_bias_mask.all())
    if classifier_fully_protected:
        head_mode = "replace"
    elif classifier_protected:
        head_mode = "mixed"
    else:
        head_mode = "exposed"
    return MaskSelection(
        masks=masks,
        classifier_protected=classifier_protected,
        head_mode=head_mode,
        magnitude_eligible_count=eligible_count,
        magnitude_protected_count=protected_count,
    )
