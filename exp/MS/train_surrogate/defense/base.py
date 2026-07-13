#!/usr/bin/env python3
"""MS 保护策略的公共数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TensorUnit:
    """`state_dict` 中一个有稳定索引的状态张量。"""

    index: int
    state_name: str
    official_layer: int
    state_kind: str
    trainable: bool
    numel: int


@dataclass(frozen=True)
class LayerGroup:
    """由 tensor unit 聚合得到的完整层。"""

    name: str
    module_names: tuple[str, ...]
    state_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    unit_indices: tuple[int, ...] = ()
    associated_ops: tuple[str, ...] = ()


@dataclass(frozen=True)
class MaskSelection:
    """策略插件生成的 victim 空间保护掩码。"""

    masks: dict[str, torch.Tensor]
    classifier_protected: bool
    head_mode: str
    magnitude_eligible_count: int | None = None
    magnitude_protected_count: int | None = None


@dataclass(frozen=True)
class DefenseOptions:
    """由统一入口传给策略插件的选择参数。"""

    architecture: str
    protected_units: str | None
    protected_layers: str | None
    protected_scalars: int | None


@dataclass(frozen=True)
class ProtectionPlan:
    """一次 surrogate 初始化所采用的保护计划。"""

    defense: str
    tensor_unit_count: int
    protected_unit_count: int
    protection_mask_sha256: str
    classifier_protected: bool
    head_mode: str
    total_param_count: int
    protected_param_count: int
    magnitude_eligible_count: int | None = None
    magnitude_protected_count: int | None = None

    @property
    def protected_param_ratio(self) -> float:
        return self.protected_param_count / max(self.total_param_count, 1)

    def to_metadata(self) -> dict[str, object]:
        return {
            "defense": self.defense,
            "tensor_unit_count": self.tensor_unit_count,
            "protected_unit_count": self.protected_unit_count,
            "protection_mask_sha256": self.protection_mask_sha256,
            "classifier_protected": self.classifier_protected,
            "head_mode": self.head_mode,
            "total_param_count": self.total_param_count,
            "protected_param_count": self.protected_param_count,
            "protected_param_ratio": self.protected_param_ratio,
            "magnitude_eligible_count": self.magnitude_eligible_count,
            "magnitude_protected_count": self.magnitude_protected_count,
        }
