#!/usr/bin/env python3
"""保护策略注册与统一分发。"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from .base import DefenseOptions, MaskSelection
from .head import build_head_only
from .magnitude import build_large_weight
from .tensorshield import build_tensorshield
from .unit import build_unit_selection


Builder = Callable[[str, nn.Module, nn.Module, DefenseOptions], MaskSelection]


def _unit_builder(
    defense: str,
    victim_model: nn.Module,
    public_model: nn.Module,
    options: DefenseOptions,
) -> MaskSelection:
    del public_model
    return build_unit_selection(defense, victim_model, options)


def _magnitude_builder(
    defense: str,
    victim_model: nn.Module,
    public_model: nn.Module,
    options: DefenseOptions,
) -> MaskSelection:
    del victim_model
    return build_large_weight(defense, public_model, options)


def _head_builder(
    defense: str,
    victim_model: nn.Module,
    public_model: nn.Module,
    options: DefenseOptions,
) -> MaskSelection:
    del public_model
    return build_head_only(defense, victim_model, options)


def _tensorshield_builder(
    defense: str,
    victim_model: nn.Module,
    public_model: nn.Module,
    options: DefenseOptions,
) -> MaskSelection:
    del public_model
    return build_tensorshield(defense, victim_model, options)


DEFENSE_REGISTRY: dict[str, Builder] = {
    "no_protection": _unit_builder,
    "full_protection": _unit_builder,
    "head_only": _head_builder,
    "shallow": _unit_builder,
    "middle": _unit_builder,
    "deep": _unit_builder,
    "custom": _unit_builder,
    "large_weight": _magnitude_builder,
    "tensorshield": _tensorshield_builder,
}
DEFENSES = tuple(DEFENSE_REGISTRY)


def build_mask_selection(
    defense: str,
    victim_model: nn.Module,
    public_model: nn.Module,
    options: DefenseOptions,
) -> MaskSelection:
    try:
        builder = DEFENSE_REGISTRY[defense]
    except KeyError as exc:
        raise ValueError(f"未知保护策略：{defense}") from exc
    return builder(defense, victim_model, public_model, options)
