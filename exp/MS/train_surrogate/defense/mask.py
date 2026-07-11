#!/usr/bin/env python3
"""保护 unit 选择与掩码持久化。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn

from .base import LayerGroup
from .resnet18 import build_resnet18_layer_groups


UNIT_DEFENSES = {"no_protection", "full_protection", "shallow", "middle", "deep", "custom"}


def state_name(module_name: str, local_name: str) -> str:
    return f"{module_name}.{local_name}" if module_name else local_name


def direct_state_names(module_name: str, module: nn.Module) -> list[str]:
    parameters = [name for name, _ in module.named_parameters(recurse=False)]
    buffers = [name for name, _ in module.named_buffers(recurse=False)]
    return [state_name(module_name, name) for name in parameters + buffers]


def direct_parameter_names(module_name: str, module: nn.Module) -> list[str]:
    return [state_name(module_name, name) for name, _ in module.named_parameters(recurse=False)]


def build_layer_groups(model: nn.Module, architecture: str | None = None) -> list[LayerGroup]:
    """构造完整层；ResNet18 从 122-unit 注册表聚合官方 18 层。"""
    if architecture == "resnet18":
        return build_resnet18_layer_groups(model)

    groups: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            current = {
                "name": module_name,
                "module_names": [module_name],
                "state_names": direct_state_names(module_name, module),
                "parameter_names": direct_parameter_names(module_name, module),
            }
            groups.append(current)
        elif isinstance(module, nn.BatchNorm2d):
            if current is None:
                raise ValueError(f"BatchNorm {module_name} 前没有可归属的 Conv 层。")
            current["module_names"].append(module_name)  # type: ignore[union-attr]
            current["state_names"].extend(direct_state_names(module_name, module))  # type: ignore[union-attr]
            current["parameter_names"].extend(direct_parameter_names(module_name, module))  # type: ignore[union-attr]

    result = [
        LayerGroup(
            name=str(group["name"]),
            module_names=tuple(group["module_names"]),  # type: ignore[arg-type]
            state_names=tuple(group["state_names"]),  # type: ignore[arg-type]
            parameter_names=tuple(group["parameter_names"]),  # type: ignore[arg-type]
        )
        for group in groups
    ]
    grouped_parameters = {name for group in result for name in group.parameter_names}
    missing = sorted({name for name, _ in model.named_parameters()} - grouped_parameters)
    if missing:
        raise ValueError(f"以下参数未归入完整层：{missing}")
    return result


def parse_unit_selection(specification: str, unit_count: int) -> tuple[int, ...]:
    """解析 `0-50,60,72` 形式的闭区间 unit 选择表达式。"""
    specification = specification.strip().lower()
    if specification in {"", "none"}:
        return ()
    if specification == "all":
        return tuple(range(unit_count))

    selected: set[int] = set()
    for raw_token in specification.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError(f"unit 表达式包含空项：{specification}")
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
                raise ValueError(f"无效 unit 区间：{token}")
            start, end = (int(part.strip()) for part in parts)
            if start > end:
                raise ValueError(f"unit 区间起点大于终点：{token}")
            values = range(start, end + 1)
        elif token.isdigit():
            values = (int(token),)
        else:
            raise ValueError(f"无效 unit 索引：{token}")
        for index in values:
            if not 0 <= index < unit_count:
                raise ValueError(f"unit 索引越界：{index}，有效范围为 0-{unit_count - 1}")
            if index in selected:
                raise ValueError(f"unit 索引重复：{index}")
            selected.add(index)
    return tuple(sorted(selected))


def resolve_unit_selection(defense: str, specification: str | None, unit_count: int) -> tuple[int, ...]:
    """结合策略语义解析 unit；mask 而不是比例是最终保护定义。"""
    if defense not in UNIT_DEFENSES:
        raise ValueError(f"{defense} 不是 unit 保护策略。")
    if defense == "no_protection":
        if specification is not None and parse_unit_selection(specification, unit_count):
            raise ValueError("no_protection 不允许选择受保护 unit。")
        return ()
    if defense == "full_protection":
        if specification is not None and parse_unit_selection(specification, unit_count) != tuple(range(unit_count)):
            raise ValueError("full_protection 只能使用 all 或完整 unit 范围。")
        return tuple(range(unit_count))
    if specification is None:
        raise ValueError(f"{defense} 必须通过 --protected-units 明确指定 unit。")

    selected = parse_unit_selection(specification, unit_count)
    if not selected:
        raise ValueError(f"{defense} 至少需要选择一个 unit。")
    contiguous = selected == tuple(range(selected[0], selected[-1] + 1))
    if defense == "shallow" and (selected[0] != 0 or not contiguous):
        raise ValueError("shallow 必须是从 unit 0 开始的连续范围。")
    if defense == "deep" and (selected[-1] != unit_count - 1 or not contiguous):
        raise ValueError(f"deep 必须是到 unit {unit_count - 1} 结束的连续范围。")
    if defense == "middle" and (not contiguous or selected[0] == 0 or selected[-1] == unit_count - 1):
        raise ValueError("middle 必须是不接触首尾的连续 unit 范围。")
    return selected


def build_unit_masks(model: nn.Module, protected_units: tuple[int, ...]) -> dict[str, torch.Tensor]:
    """直接由 unit 索引生成保护掩码。"""
    protected = set(protected_units)
    return {
        name: torch.full_like(value, index in protected, dtype=torch.bool)
        for index, (name, value) in enumerate(model.state_dict().items())
    }


def _pack_mask(mask: torch.Tensor) -> torch.Tensor:
    flat = mask.detach().cpu().reshape(-1).to(torch.uint8)
    padding = (-flat.numel()) % 8
    if padding:
        flat = torch.cat((flat, torch.zeros(padding, dtype=torch.uint8)))
    shifts = torch.arange(8, dtype=torch.uint8)
    return (flat.reshape(-1, 8) * (1 << shifts)).sum(dim=1).to(torch.uint8)


def _unpack_mask(packed: torch.Tensor, shape: tuple[int, ...], numel: int) -> torch.Tensor:
    shifts = torch.arange(8, dtype=torch.uint8)
    flat = ((packed.to(torch.uint8).reshape(-1, 1) >> shifts) & 1).reshape(-1)[:numel]
    return flat.to(torch.bool).reshape(shape)


def build_protection_mask_payload(masks: dict[str, torch.Tensor]) -> dict[str, object]:
    """构造按 unit 顺序保存的紧凑保护掩码，True 表示状态不可见。"""
    units = []
    for index, (unit_name, mask) in enumerate(masks.items()):
        cpu_mask = mask.detach().cpu().to(torch.bool)
        unit: dict[str, object] = {
            "index": index,
            "state_name": unit_name,
            "shape": tuple(cpu_mask.shape),
            "numel": cpu_mask.numel(),
        }
        if cpu_mask.all():
            unit["mode"] = "all"
        elif not cpu_mask.any():
            unit["mode"] = "none"
        else:
            unit["mode"] = "partial"
            unit["packed"] = _pack_mask(cpu_mask)
        units.append(unit)
    return {"schema_version": 1, "unit_count": len(units), "units": units}


def protection_mask_sha256(masks: dict[str, torch.Tensor]) -> str:
    """计算与保存格式无关的稳定掩码摘要。"""
    digest = hashlib.sha256()
    payload = build_protection_mask_payload(masks)
    digest.update(b"TSDP-MS-protection-mask-v1\0")
    for unit in payload["units"]:  # type: ignore[index]
        name = str(unit["state_name"]).encode("utf-8")
        shape = tuple(unit["shape"])
        mode = str(unit["mode"])
        digest.update(int(unit["index"]).to_bytes(4, "little"))
        digest.update(len(name).to_bytes(4, "little"))
        digest.update(name)
        digest.update(repr(shape).encode("ascii"))
        digest.update(mode.encode("ascii"))
        if mode == "partial":
            packed = unit["packed"]
            assert isinstance(packed, torch.Tensor)
            digest.update(packed.numpy().tobytes())
    return digest.hexdigest()


def save_protection_mask(path: Path, masks: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_protection_mask_payload(masks), path)


def load_protection_mask(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"无法识别保护掩码格式：{path}")
    units = payload.get("units")
    if not isinstance(units, list) or payload.get("unit_count") != len(units):
        raise ValueError("保护掩码 unit 数量不一致。")

    masks = {}
    for expected_index, unit in enumerate(units):
        if unit.get("index") != expected_index:
            raise ValueError("保护掩码 unit 索引不连续。")
        shape = tuple(unit["shape"])
        mode = unit["mode"]
        if mode == "all":
            mask = torch.ones(shape, dtype=torch.bool)
        elif mode == "none":
            mask = torch.zeros(shape, dtype=torch.bool)
        elif mode == "partial":
            mask = _unpack_mask(unit["packed"], shape, int(unit["numel"]))
        else:
            raise ValueError(f"未知保护掩码模式：{mode}")
        masks[unit["state_name"]] = mask
    return masks
