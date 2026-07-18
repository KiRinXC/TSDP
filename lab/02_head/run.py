#!/usr/bin/env python3
"""按当前 MS 协议比较分类头结构与暴露权重训练方式。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.MS.train_surrogate.core.artifacts import sha256_file  # noqa: E402
from exp.MS.train_surrogate.core.config import resolve_device  # noqa: E402
from exp.MS.train_surrogate.core.data import build_victim  # noqa: E402
from exp.MS.train_surrogate.defense import (  # noqa: E402
    ExposureFreezer,
    build_resnet18_tensor_units,
    build_unit_masks,
    protection_mask_sha256,
    reset_surrogate_initialization,
)
from exp.MS.train_victim.common.trainer import configure_reproducibility  # noqa: E402
from lab.protocol import (  # noqa: E402
    evaluate_once,
    prepare_eval,
    prepare_soft_query,
    protocol_metadata,
    train_validation_best,
)
from models import imagenet as imagenet_models  # noqa: E402
from models.imagenet import load_official_imagenet_weights  # noqa: E402


EXPERIMENT = "02_head"
MODEL = "resnet18"
DATASET = "c100"
NUM_CLASSES = 100
BUDGET = 500
SEED = 42
CONFIGURATIONS = (
    ("replace_frozen", "replace", "frozen"),
    ("replace_finetune", "replace", "finetune"),
    ("adapter_frozen", "adapter", "frozen"),
    ("adapter_finetune", "adapter", "finetune"),
)
PROTECTION_FULL = "full_protection"
PROTECTION_RANDOM = "random_50"
BACKBONE_UNIT_COUNT = 120
CLASSIFIER_UNITS = (120, 121)
RANDOM_PROTECTED_BACKBONE_UNITS = 59


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对输入、随机保护集合和八组模型构造，不写结果。",
    )
    return parser.parse_args()


def build_random_plan(model: nn.Module, seed: int) -> dict[str, Any]:
    units = build_resnet18_tensor_units(model)
    if len(units) != 122:
        raise RuntimeError(f"ResNet18 unit 数量应为 122，实际为 {len(units)}。")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(BACKBONE_UNIT_COUNT, generator=generator).tolist()
    random_backbone = tuple(sorted(permutation[:RANDOM_PROTECTED_BACKBONE_UNITS]))
    protected_units = tuple(sorted((*random_backbone, *CLASSIFIER_UNITS)))
    masks = build_unit_masks(model, protected_units)
    exposed_units = tuple(
        index for index in range(len(units)) if index not in protected_units
    )
    return {
        "unit_count": len(units),
        "protected_unit_count": len(protected_units),
        "protected_fraction": len(protected_units) / len(units),
        "random_backbone_unit_count": len(random_backbone),
        "random_candidate_units": [0, BACKBONE_UNIT_COUNT - 1],
        "classifier_units": list(CLASSIFIER_UNITS),
        "classifier_policy": "fixed_protected_excluded_from_random_draw",
        "protected_units": list(protected_units),
        "exposed_units": list(exposed_units),
        "mask_sha256": protection_mask_sha256(masks),
    }


def make_task_model(
    head_mode: str,
    official_weight: Path,
    initialization_seed: int,
) -> tuple[nn.Module, nn.Module]:
    reset_surrogate_initialization(
        imagenet_models.resnet18,
        NUM_CLASSES,
        initialization_seed,
    )
    model = imagenet_models.resnet18(num_classes=1000)
    load_official_imagenet_weights(MODEL, model, official_weight, strict=True)
    if head_mode == "replace":
        model.last_linear = nn.Linear(model.last_linear.in_features, NUM_CLASSES)
        task_head = model.last_linear
    elif head_mode == "adapter":
        public_head = model.last_linear
        task_head = nn.Linear(public_head.out_features, NUM_CLASSES)
        model.last_linear = nn.Sequential(public_head, task_head)
    else:
        raise ValueError(f"未知分类头结构：{head_mode}")
    return model, task_head


@dataclass
class SurrogateSetup:
    model: nn.Module
    freezer: ExposureFreezer | None
    frozen_scope: str | None
    trainable_parameters: int
    total_parameters: int
    copied_parameter_elements: int


def _copy_exposed_backbone(
    model: nn.Module,
    victim_state: dict[str, torch.Tensor],
    protected_units: Iterable[int],
) -> tuple[dict[str, torch.Tensor], int]:
    protected = set(protected_units)
    victim_names = list(victim_state)
    if len(victim_names) != 122:
        raise RuntimeError(f"victim state 应有 122 个 unit，实际为 {len(victim_names)}。")
    model_state = model.state_dict()
    trainable_masks = {
        name: torch.ones_like(tensor, dtype=torch.bool)
        for name, tensor in model_state.items()
    }
    copied = 0
    parameter_names = dict(model.named_parameters())
    for unit_index, name in enumerate(victim_names[:BACKBONE_UNIT_COUNT]):
        if name not in model_state or model_state[name].shape != victim_state[name].shape:
            raise ValueError(f"public 与 victim 骨干状态不一致：{name}")
        if unit_index in protected:
            continue
        model_state[name] = victim_state[name].detach().clone()
        trainable_masks[name] = torch.zeros_like(model_state[name], dtype=torch.bool)
        if name in parameter_names:
            copied += model_state[name].numel()
    model.load_state_dict(model_state, strict=True)
    return trainable_masks, copied


def _full_frozen_masks(
    model: nn.Module,
    task_head: nn.Module,
) -> dict[str, torch.Tensor]:
    """冻结公开骨干、公开 1000 类头和 BN buffer，只放开任务头。"""

    task_parameter_ids = {id(parameter) for parameter in task_head.parameters()}
    trainable_parameter_names = {
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in task_parameter_ids
    }
    return {
        name: torch.full_like(
            value,
            name in trainable_parameter_names,
            dtype=torch.bool,
        )
        for name, value in model.state_dict().items()
    }


def build_surrogate(
    *,
    head_mode: str,
    training_mode: str,
    protection: str,
    victim_state: dict[str, torch.Tensor],
    protected_units: tuple[int, ...],
    official_weight: Path,
    initialization_seed: int,
    device: torch.device,
) -> SurrogateSetup:
    model, task_head = make_task_model(head_mode, official_weight, initialization_seed)
    freezer: ExposureFreezer | None = None
    frozen_scope: str | None = None
    copied_parameter_elements = 0

    if protection == PROTECTION_RANDOM:
        trainable_masks, copied_parameter_elements = _copy_exposed_backbone(
            model,
            victim_state,
            protected_units,
        )
        if training_mode == "frozen":
            frozen_scope = "stolen_victim_weights"
            model.to(device)
            freezer = ExposureFreezer(model, trainable_masks)
    elif protection == PROTECTION_FULL:
        if training_mode == "frozen":
            frozen_scope = "public_pretrained_weights_and_bn_state"
            model.to(device)
            freezer = ExposureFreezer(model, _full_frozen_masks(model, task_head))
    else:
        raise ValueError(f"未知保护范围：{protection}")

    if training_mode == "finetune":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif training_mode != "frozen":
        raise ValueError(f"未知训练方式：{training_mode}")

    model.to(device)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return SurrogateSetup(
        model=model,
        freezer=freezer,
        frozen_scope=frozen_scope,
        trainable_parameters=trainable,
        total_parameters=total,
        copied_parameter_elements=copied_parameter_elements,
    )


def write_history(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "protection",
        "configuration",
        "head_mode",
        "training_mode",
        "epoch",
        "learning_rate",
        "query_count",
        "query_loss_sum",
        "query_loss",
        "query_match_count",
        "query_match",
        "validation_count",
        "validation_loss",
        "validation_kl",
        "validation_match_count",
        "validation_match",
        "is_best",
    )
    with path.open("w", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(
            writer_file,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def clean_outputs(out_dir: Path) -> None:
    for filename in ("metrics.json", "history.tsv"):
        (out_dir / filename).unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    device = resolve_device(args.device)
    dataset_root = ROOT / "dataset" / "public"
    protocol_root = ROOT / "dataset" / "MS"
    victim_checkpoint = (
        ROOT / "weights" / "MS" / "victim" / MODEL / DATASET / "best.pth"
    )
    official_weight = ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    out_dir = ROOT / "results" / "lab" / EXPERIMENT

    configure_reproducibility(SEED, deterministic=True)
    query = prepare_soft_query(
        dataset=DATASET,
        model=MODEL,
        budget=BUDGET,
        seed=SEED,
        dataset_root=dataset_root,
        protocol_root=protocol_root,
    )
    victim, victim_metadata = build_victim(MODEL, NUM_CLASSES, victim_checkpoint)
    victim_sha256 = sha256_file(victim_checkpoint)
    expected_victim_sha256 = query.manifest.get("victim", {}).get("checkpoint_sha256")
    if expected_victim_sha256 and expected_victim_sha256 != victim_sha256:
        raise ValueError("victim best.pth 与 soft posterior 的来源 checkpoint 不一致。")
    victim_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in victim.state_dict().items()
    }
    random_plan = build_random_plan(victim, SEED)
    scopes = (
        (PROTECTION_FULL, tuple(range(122))),
        (PROTECTION_RANDOM, tuple(random_plan["protected_units"])),
    )

    for protection, protected_units in scopes:
        for name, head_mode, training_mode in CONFIGURATIONS:
            configure_reproducibility(SEED, deterministic=True)
            setup = build_surrogate(
                head_mode=head_mode,
                training_mode=training_mode,
                protection=protection,
                victim_state=victim_state,
                protected_units=protected_units,
                official_weight=official_weight,
                initialization_seed=SEED,
                device=device,
            )
            print(
                f"[MODEL/{protection}/{name}] "
                f"trainable={setup.trainable_parameters}/{setup.total_parameters} "
                f"frozen_scope={setup.frozen_scope}"
            )
            del setup.model, setup
    if args.dry_run:
        print("[INFO] dry-run 完成，未写入 Lab02 结果。")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    clean_outputs(out_dir)
    results: list[dict[str, object]] = []
    all_history: list[dict[str, object]] = []
    evaluation = None
    for protection, protected_units in scopes:
        for name, head_mode, training_mode in CONFIGURATIONS:
            configure_reproducibility(SEED, deterministic=True)
            setup = build_surrogate(
                head_mode=head_mode,
                training_mode=training_mode,
                protection=protection,
                victim_state=victim_state,
                protected_units=protected_units,
                official_weight=official_weight,
                initialization_seed=SEED,
                device=device,
            )
            selection, history = train_validation_best(
                setup.model,
                query,
                device=device,
                num_workers=args.num_workers,
                seed=SEED,
                freezer=setup.freezer,
            )
            if evaluation is None:
                evaluation = prepare_eval(
                    victim,
                    dataset=DATASET,
                    dataset_root=dataset_root,
                    protocol_root=protocol_root,
                    device=device,
                    num_workers=args.num_workers,
                    seed=SEED,
                )
            result_metrics = evaluate_once(setup.model, evaluation, device)
            results.append(
                {
                    "protection": protection,
                    "configuration": name,
                    "head_mode": head_mode,
                    "training_mode": training_mode,
                    "frozen_weight_source": setup.frozen_scope,
                    "trainable_parameters": setup.trainable_parameters,
                    "total_parameters": setup.total_parameters,
                    "copied_parameter_elements": setup.copied_parameter_elements,
                    "primary": {
                        "checkpoint": "best.pth",
                        "epoch": selection["epoch"],
                        "selection_metric": selection["metric"],
                    },
                    "selection": selection,
                    "result": result_metrics,
                }
            )
            all_history.extend(
                {
                    "protection": protection,
                    "configuration": name,
                    "head_mode": head_mode,
                    "training_mode": training_mode,
                    **row,
                }
                for row in history
            )
            print(
                f"[RESULT/{protection}/{name}] epoch={selection['epoch']} "
                f"accuracy={result_metrics['surrogate_acc']:.6f} "
                f"fidelity={result_metrics['fidelity']:.6f} "
                f"KL={result_metrics['posterior_kl']:.6f}"
            )
            del setup.model, setup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = {
        "schema_version": 3,
        "experiment": "head_and_stolen_weight_ablation",
        "protocol": "MS",
        **protocol_metadata(query),
        "dataset": DATASET,
        "model": MODEL,
        "seed": SEED,
        "randomization": {
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": SEED,
            "query_sampler_seed": SEED,
            "reset_before_each_surrogate_initialization": True,
        },
        "victim_weight": str(victim_checkpoint.relative_to(ROOT)),
        "victim_weight_sha256": victim_sha256,
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "official_weight": str(official_weight.relative_to(ROOT)),
        "official_weight_sha256": sha256_file(official_weight),
        "query_target_path": str(query.target_path.relative_to(ROOT)),
        "query_target_sha256": query.target_sha256,
        "protection_plans": {
            PROTECTION_FULL: {
                "protected_unit_count": 122,
                "protected_fraction": 1.0,
                "copied_victim_weights": False,
            },
            PROTECTION_RANDOM: random_plan,
        },
        "results": results,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_history(out_dir / "history.tsv", all_history)
    print(f"[INFO] 结果：{(out_dir / 'metrics.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
