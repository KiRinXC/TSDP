#!/usr/bin/env python3
"""完整 unit 保护策略。"""

from __future__ import annotations

import torch.nn as nn

from .base import DefenseOptions, MaskSelection
from .mask import build_unit_masks, parse_unit_selection, resolve_unit_selection
from .resnet18 import build_resnet18_layer_groups


def parse_official_layer_selection(specification: str, layer_count: int) -> tuple[int, ...]:
    """解析 1-based 官方层表达式，例如 `1-3,6`。"""
    selected: set[int] = set()
    for raw_token in specification.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError(f"官方层表达式包含空项：{specification}")
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
                raise ValueError(f"无效官方层区间：{token}")
            start, end = (int(part.strip()) for part in parts)
            if start > end:
                raise ValueError(f"官方层区间起点大于终点：{token}")
            values = range(start, end + 1)
        elif token.isdigit():
            values = (int(token),)
        else:
            raise ValueError(f"无效官方层索引：{token}")
        for index in values:
            if not 1 <= index <= layer_count:
                raise ValueError(f"官方层索引越界：{index}，有效范围为 1-{layer_count}")
            if index in selected:
                raise ValueError(f"官方层索引重复：{index}")
            selected.add(index)
    if not selected:
        raise ValueError("至少需要选择一个官方层。")
    return tuple(sorted(selected))


def _validate_layer_direction(defense: str, selected_layers: tuple[int, ...], layer_count: int) -> None:
    if not selected_layers:
        raise ValueError(f"{defense} 至少需要选择一个完整官方层。")
    contiguous = selected_layers == tuple(range(selected_layers[0], selected_layers[-1] + 1))
    if defense == "shallow" and (selected_layers[0] != 1 or not contiguous):
        raise ValueError("shallow 必须是从官方第 1 层开始的连续层范围。")
    if defense == "deep" and (selected_layers[-1] != layer_count or not contiguous):
        raise ValueError(f"deep 必须是到官方第 {layer_count} 层结束的连续层范围。")
    if defense == "middle" and (
        not contiguous or selected_layers[0] == 1 or selected_layers[-1] == layer_count
    ):
        raise ValueError("middle 必须是不接触首尾的连续官方层范围。")


def resolve_resnet18_layer_units(
    model: nn.Module,
    defense: str,
    protected_layers: str | None,
    protected_units: str | None,
) -> tuple[int, ...]:
    """把完整官方层选择映射为 122-unit 集合。"""
    groups = build_resnet18_layer_groups(model)
    if protected_layers is not None and protected_units is not None:
        raise ValueError("--protected-layers 与 --protected-units 不能同时使用。")
    if protected_layers is not None:
        selected_layers = parse_official_layer_selection(protected_layers, len(groups))
        _validate_layer_direction(defense, selected_layers, len(groups))
        return tuple(sorted(unit for index in selected_layers for unit in groups[index - 1].unit_indices))
    if protected_units is None:
        raise ValueError(f"{defense} 必须指定 --protected-layers 或 --protected-units。")

    selected_units = parse_unit_selection(protected_units, len(model.state_dict()))
    selected_set = set(selected_units)
    selected_layers = []
    for index, group in enumerate(groups, start=1):
        group_units = set(group.unit_indices)
        overlap = selected_set & group_units
        if overlap and overlap != group_units:
            raise ValueError(f"{defense} 对官方第 {index} 层只选择了部分 unit。")
        if overlap:
            selected_layers.append(index)
    covered = {unit for index in selected_layers for unit in groups[index - 1].unit_indices}
    if covered != selected_set:
        raise ValueError(f"{defense} 的 unit 未能完整映射到官方层。")
    selected_layer_tuple = tuple(selected_layers)
    _validate_layer_direction(defense, selected_layer_tuple, len(groups))
    return selected_units


def build_unit_selection(
    defense: str,
    victim_model: nn.Module,
    options: DefenseOptions,
) -> MaskSelection:
    if options.protected_scalars is not None:
        raise ValueError(f"{defense} 不接受 --protected-scalars。")
    unit_count = len(victim_model.state_dict())
    if defense in {"no_protection", "full_protection"}:
        if options.protected_layers is not None:
            raise ValueError(f"{defense} 不接受 --protected-layers。")
        selected_units = resolve_unit_selection(defense, options.protected_units, unit_count)
    elif defense == "custom":
        if options.protected_layers is not None:
            raise ValueError("custom 只接受 --protected-units。")
        selected_units = resolve_unit_selection(defense, options.protected_units, unit_count)
    elif options.architecture == "resnet18":
        selected_units = resolve_resnet18_layer_units(
            victim_model,
            defense,
            options.protected_layers,
            options.protected_units,
        )
    else:
        if options.protected_layers is not None:
            raise ValueError(f"{options.architecture} 尚未定义官方层注册表。")
        selected_units = resolve_unit_selection(defense, options.protected_units, unit_count)
    masks = build_unit_masks(victim_model, selected_units)

    head_weight_protected = bool(masks["last_linear.weight"].all())
    head_bias_protected = bool(masks["last_linear.bias"].all())
    if head_weight_protected != head_bias_protected:
        raise ValueError("分类头 unit 必须同时保护或同时暴露。")
    return MaskSelection(
        masks=masks,
        classifier_protected=head_weight_protected,
        head_mode="adapter" if head_weight_protected else "exposed",
    )
