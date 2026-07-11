#!/usr/bin/env python3
"""ResNet18 的 122-unit 注册表与官方 18 层映射。"""

from __future__ import annotations

import torch.nn as nn

from .base import LayerGroup, TensorUnit


RESNET18_LAYER_NAMES = (
    "layer01.conv1",
    "layer02.layer1.0.conv1",
    "layer03.layer1.0.conv2",
    "layer04.layer1.1.conv1",
    "layer05.layer1.1.conv2",
    "layer06.layer2.0.conv1",
    "layer07.layer2.0.conv2",
    "layer08.layer2.1.conv1",
    "layer09.layer2.1.conv2",
    "layer10.layer3.0.conv1",
    "layer11.layer3.0.conv2",
    "layer12.layer3.1.conv1",
    "layer13.layer3.1.conv2",
    "layer14.layer4.0.conv1",
    "layer15.layer4.0.conv2",
    "layer16.layer4.1.conv1",
    "layer17.layer4.1.conv2",
    "layer18.last_linear",
)


def _official_layer(state_name: str) -> int:
    if state_name.startswith(("conv1.", "bn1.")):
        return 1
    if state_name.startswith("last_linear."):
        return 18

    parts = state_name.split(".")
    if len(parts) < 4 or parts[0] not in {"layer1", "layer2", "layer3", "layer4"}:
        raise ValueError(f"无法把 ResNet18 状态映射到官方层：{state_name}")
    stage = int(parts[0][-1])
    block = int(parts[1])
    component = parts[2]
    stage_start = {1: 2, 2: 6, 3: 10, 4: 14}[stage]
    if component in {"conv1", "bn1"}:
        offset = block * 2
    elif component in {"conv2", "bn2"}:
        offset = block * 2 + 1
    elif component == "downsample" and block == 0 and stage > 1:
        offset = 0
    else:
        raise ValueError(f"无法把 ResNet18 状态映射到官方层：{state_name}")
    return stage_start + offset


def build_resnet18_tensor_units(model: nn.Module) -> list[TensorUnit]:
    """按 `state_dict` 顺序构造 ResNet18 的 122 个基础 unit。"""
    parameter_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    units = []
    for index, (state_name, value) in enumerate(model.state_dict().items()):
        if state_name in parameter_names:
            state_kind = "parameter"
        elif state_name in buffer_names:
            state_kind = "buffer"
        else:
            raise ValueError(f"状态既不是 parameter 也不是 buffer：{state_name}")
        units.append(
            TensorUnit(
                index=index,
                state_name=state_name,
                official_layer=_official_layer(state_name),
                state_kind=state_kind,
                trainable=state_name in parameter_names,
                numel=value.numel(),
            )
        )

    if len(units) != 122:
        raise ValueError(f"当前 ResNet18 应有 122 个 tensor unit，实际为 {len(units)}。")
    if [unit.index for unit in units] != list(range(122)):
        raise ValueError("ResNet18 tensor unit 索引不连续。")
    if {unit.official_layer for unit in units} != set(range(1, 19)):
        raise ValueError("ResNet18 tensor unit 未完整覆盖官方 18 层。")
    return units


def build_resnet18_layer_groups(model: nn.Module) -> list[LayerGroup]:
    """仅从 122-unit 注册表聚合 ResNet18 官方 18 层。"""
    units = build_resnet18_tensor_units(model)
    groups = []
    for layer_index, layer_name in enumerate(RESNET18_LAYER_NAMES, start=1):
        members = [unit for unit in units if unit.official_layer == layer_index]
        module_names = tuple(dict.fromkeys(unit.state_name.rsplit(".", 1)[0] for unit in members))
        associated_ops: tuple[str, ...] = ()
        if layer_index == 1:
            associated_ops = ("relu", "maxpool")
        elif layer_index == 18:
            associated_ops = ("avgpool",)
        groups.append(
            LayerGroup(
                name=layer_name,
                module_names=module_names,
                state_names=tuple(unit.state_name for unit in members),
                parameter_names=tuple(unit.state_name for unit in members if unit.trainable),
                unit_indices=tuple(unit.index for unit in members),
                associated_ops=associated_ops,
            )
        )
    return groups
