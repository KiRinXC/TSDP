#!/usr/bin/env python3
"""根据保护掩码初始化 surrogate，并实现暴露权重冻结。"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .base import DefenseOptions, ProtectionPlan
from .mask import direct_state_names, protection_mask_sha256
from .registry import build_mask_selection
from .resnet18 import build_resnet18_tensor_units


def _count_protected_parameters(
    model: nn.Module,
    masks: dict[str, torch.Tensor],
) -> tuple[int, int]:
    total = 0
    protected = 0
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        protected += int(masks[name].sum().item())
    return total, protected


def _copy_exposed_state(
    surrogate: nn.Module,
    victim_state: dict[str, torch.Tensor],
    victim_masks: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """复制可见 victim 状态，并返回 surrogate 冻结控制掩码。"""
    surrogate_state = surrogate.state_dict()
    trainable_masks: dict[str, torch.Tensor] = {}
    for name, current in surrogate_state.items():
        if name not in victim_state:
            trainable_masks[name] = torch.ones_like(current, dtype=torch.bool)
            continue
        if current.shape != victim_state[name].shape:
            raise ValueError(f"surrogate 与 victim 的状态形状不一致：{name}")
        protected = victim_masks[name]
        trainable_masks[name] = protected.clone()
        if protected.all():
            continue
        if protected.any():
            surrogate_state[name] = torch.where(protected, current, victim_state[name])
        else:
            surrogate_state[name] = victim_state[name].clone()
    surrogate.load_state_dict(surrogate_state)
    return trainable_masks


def _build_public_model(factory, factory_name: str, weight_path: Path, num_classes: int, adapter: bool):
    from models.imagenet import load_official_imagenet_weights

    model = factory(num_classes=1000)
    load_official_imagenet_weights(factory_name, model, str(weight_path), strict=True)
    if adapter:
        public_head = model.last_linear
        model.last_linear = nn.Sequential(public_head, nn.Linear(1000, num_classes))
    else:
        in_features = model.last_linear.in_features
        model.last_linear = nn.Linear(in_features, num_classes)
    return model


def initialize_surrogate(
    factory,
    factory_name: str,
    weight_path: Path,
    victim_model: nn.Module,
    num_classes: int,
    defense: str,
    protected_units: str | None,
    protected_layers: str | None,
    protected_scalars: int | None,
) -> tuple[nn.Module, ProtectionPlan, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """由策略插件生成掩码，再统一组合公开与 victim 权重。"""
    if factory_name == "resnet18":
        build_resnet18_tensor_units(victim_model)
    victim_state = {name: value.detach().cpu() for name, value in victim_model.state_dict().items()}

    rng_state = torch.get_rng_state()
    public_model = _build_public_model(factory, factory_name, weight_path, num_classes, adapter=False)
    selection = build_mask_selection(
        defense,
        victim_model,
        public_model,
        DefenseOptions(
            architecture=factory_name,
            protected_units=protected_units,
            protected_layers=protected_layers,
            protected_scalars=protected_scalars,
        ),
    )
    if selection.head_mode == "adapter":
        # 临时同形状 public 模型只用于统一策略接口，不应改变 adapter 的随机初始化。
        torch.set_rng_state(rng_state)
        surrogate = _build_public_model(factory, factory_name, weight_path, num_classes, adapter=True)
    else:
        surrogate = public_model
    trainable_masks = _copy_exposed_state(surrogate, victim_state, selection.masks)
    total_params, protected_params = _count_protected_parameters(victim_model, selection.masks)
    plan = ProtectionPlan(
        defense=defense,
        tensor_unit_count=len(selection.masks),
        protected_unit_count=sum(bool(mask.any()) for mask in selection.masks.values()),
        protection_mask_sha256=protection_mask_sha256(selection.masks),
        classifier_protected=selection.classifier_protected,
        head_mode=selection.head_mode,
        total_param_count=total_params,
        protected_param_count=protected_params,
        magnitude_eligible_count=selection.magnitude_eligible_count,
        magnitude_protected_count=selection.magnitude_protected_count,
    )
    return surrogate, plan, trainable_masks, selection.masks


class ExposureFreezer:
    """冻结已暴露的 victim 权重，包括逐标量掩码和 BN 运行状态。"""

    def __init__(self, model: nn.Module, trainable_masks: dict[str, torch.Tensor]):
        self.model = model
        self.parameter_anchors: dict[str, torch.Tensor] = {}
        self.parameter_masks: dict[str, torch.Tensor] = {}
        self.buffer_anchors: dict[str, torch.Tensor] = {}
        self.buffer_masks: dict[str, torch.Tensor] = {}
        self.frozen_bn_names: set[str] = set()

        for name, parameter in model.named_parameters():
            mask = trainable_masks[name].to(parameter.device)
            if mask.all():
                continue
            self.parameter_anchors[name] = parameter.detach().clone()
            self.parameter_masks[name] = mask
            if not mask.any():
                parameter.requires_grad_(False)
            else:
                parameter.register_hook(lambda grad, current=mask: grad * current.to(grad.dtype))

        for name, buffer in model.named_buffers():
            mask = trainable_masks[name].to(buffer.device)
            if mask.all():
                continue
            self.buffer_anchors[name] = buffer.detach().clone()
            self.buffer_masks[name] = mask

        for module_name, module in model.named_modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            names = direct_state_names(module_name, module)
            if names and all(not trainable_masks[name].any() for name in names):
                self.frozen_bn_names.add(module_name)

    def apply_train_mode(self) -> None:
        modules = dict(self.model.named_modules())
        for name in self.frozen_bn_names:
            modules[name].eval()

    @torch.no_grad()
    def restore(self) -> None:
        for name, parameter in self.model.named_parameters():
            if name not in self.parameter_anchors:
                continue
            mask = self.parameter_masks[name]
            anchor = self.parameter_anchors[name]
            parameter.copy_(torch.where(mask, parameter, anchor))
        for name, buffer in self.model.named_buffers():
            if name not in self.buffer_anchors:
                continue
            mask = self.buffer_masks[name]
            anchor = self.buffer_anchors[name]
            buffer.copy_(torch.where(mask, buffer, anchor))
