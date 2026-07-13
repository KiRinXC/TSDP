#!/usr/bin/env python3
"""正式 baseline 计划读取与 mask 一致性校验。"""

from __future__ import annotations

import json
from pathlib import Path

from defense.base import ProtectionPlan

from .config import ATTACK_PROTOCOL_VERSION, REPO_ROOT


PLANNED_DEFENSES = {"shallow", "middle", "deep", "large_weight"}
PLAN_PATH = REPO_ROOT / "exp" / "MS" / "train_surrogate" / "baseline.json"


def resolve_plan_configuration(
    *,
    plan_id: str | None,
    model_name: str,
    dataset_name: str,
    defense: str,
    protected_units: str | None,
    protected_layers: str | None,
    protected_scalars: int | None,
) -> dict[str, object] | None:
    if defense not in PLANNED_DEFENSES:
        if plan_id is not None:
            raise ValueError(f"{defense} 不接受 --plan-id。")
        return None
    if plan_id is None:
        raise ValueError(f"正式 {defense} baseline 必须指定 --plan-id。")
    if protected_units is not None:
        raise ValueError("计划内 baseline 不接受 --protected-units，必须使用清单中的层或标量预算。")

    manifest = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if manifest.get("attack_protocol") != ATTACK_PROTOCOL_VERSION:
        raise ValueError("baseline.json 的攻击协议版本与正式 runner 不一致。")
    if manifest.get("model") != model_name or manifest.get("dataset") != dataset_name:
        raise ValueError(f"baseline.json 不适用于 {model_name}+{dataset_name}。")
    configurations = [
        *manifest["layer_sweep"]["configurations"],
        *manifest["large_weight_sweep"]["configurations"],
    ]
    matches = [config for config in configurations if config["id"] == plan_id]
    if len(matches) != 1:
        raise ValueError(f"baseline.json 中找不到唯一 plan_id={plan_id}。")
    config = matches[0]
    if config["defense"] != defense:
        raise ValueError(f"plan_id={plan_id} 的策略是 {config['defense']}，不是 {defense}。")
    if config.get("protected_layers") != protected_layers:
        raise ValueError(
            f"plan_id={plan_id} 要求 --protected-layers {config.get('protected_layers')}。"
        )
    if config.get("protected_scalars") != protected_scalars:
        raise ValueError(
            f"plan_id={plan_id} 要求 --protected-scalars {config.get('protected_scalars')}。"
        )
    return config


def validate_built_plan(config: dict[str, object] | None, plan: ProtectionPlan) -> None:
    if config is None:
        return
    expected = {
        "protected_unit_count": plan.protected_unit_count,
        "protected_param_count": plan.protected_param_count,
        "classifier_protected": plan.classifier_protected,
        "head_mode": plan.head_mode,
        "protection_mask_sha256": plan.protection_mask_sha256,
    }
    mismatches = {
        name: (config.get(name), actual)
        for name, actual in expected.items()
        if config.get(name) != actual
    }
    if mismatches:
        raise ValueError(f"实际保护计划与 baseline.json 不一致：{mismatches}")
