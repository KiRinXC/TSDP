#!/usr/bin/env python3
"""MS surrogate 保护策略插件。"""

from .initialize import ExposureFreezer, initialize_surrogate
from .magnitude import build_magnitude_masks
from .mask import (
    build_layer_groups,
    build_unit_masks,
    load_protection_mask,
    parse_unit_selection,
    protection_mask_sha256,
    resolve_unit_selection,
    save_protection_mask,
)
from .registry import DEFENSES, DEFENSE_REGISTRY, build_mask_selection
from .resnet18 import build_resnet18_layer_groups, build_resnet18_tensor_units
from .unit import parse_official_layer_selection, resolve_resnet18_layer_units

__all__ = [
    "DEFENSES",
    "DEFENSE_REGISTRY",
    "ExposureFreezer",
    "build_layer_groups",
    "build_magnitude_masks",
    "build_mask_selection",
    "build_resnet18_layer_groups",
    "build_resnet18_tensor_units",
    "build_unit_masks",
    "initialize_surrogate",
    "load_protection_mask",
    "parse_unit_selection",
    "parse_official_layer_selection",
    "protection_mask_sha256",
    "resolve_unit_selection",
    "resolve_resnet18_layer_units",
    "save_protection_mask",
]
