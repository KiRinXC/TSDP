#!/usr/bin/env python3
"""MS surrogate 实验编排。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import (
    ATTACK_PROTOCOL_VERSION,
    MODEL_SPECS,
    NUM_CLASSES,
    REPO_ROOT,
    parse_args,
    resolve_device,
    validate_attack_configuration,
)

from common.trainer import (  # noqa: E402
    build_generator,
    configure_reproducibility,
    resolve_dataset_name,
    seed_worker,
)
from defense import initialize_surrogate, save_protection_mask  # noqa: E402
from models import imagenet as imagenet_models  # noqa: E402

from .artifacts import (
    display_path,
    make_artifact_id,
    make_run_id,
    save_checkpoint,
    sha256_file,
    update_index,
    write_history_row,
    write_json,
)
from .data import build_eval_dataset, build_query_dataset, build_victim, load_query_targets
from .engine import collect_eval_reference, evaluate_surrogate, train_one_epoch
from .planning import resolve_plan_configuration, validate_built_plan


def main() -> int:
    args = parse_args()
    validate_attack_configuration(args.defense, args.training_mode, args.label_mode)
    if args.epochs <= 0:
        raise ValueError("epochs 必须大于 0。")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("batch_size 和 eval_batch_size 必须大于 0。")
    dataset_name = resolve_dataset_name(args.dataset)
    model_name = args.model
    num_classes = NUM_CLASSES[dataset_name]
    plan_config = resolve_plan_configuration(
        plan_id=args.plan_id,
        model_name=model_name,
        dataset_name=dataset_name,
        defense=args.defense,
        protected_units=args.protected_units,
        protected_layers=args.protected_layers,
        protected_scalars=args.protected_scalars,
    )
    factory_name, weight_filename = MODEL_SPECS[model_name]
    factory: Callable[..., nn.Module] = getattr(imagenet_models, factory_name)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    protocol_root = Path(args.protocol_root).expanduser().resolve()
    victim_checkpoint = (
        Path(args.victim_checkpoint).expanduser().resolve()
        if args.victim_checkpoint
        else REPO_ROOT / "weights" / "MS" / "victim" / model_name / dataset_name / "best.pth"
    )
    if victim_checkpoint.name != "best.pth":
        raise ValueError("MS surrogate 只允许使用 victim 的 best.pth。")
    weight_path = (
        Path(args.weight_path).expanduser().resolve()
        if args.weight_path
        else REPO_ROOT / "weights" / "pre_train" / weight_filename
    )
    if not weight_path.is_file():
        raise FileNotFoundError(f"找不到官方预训练权重：{weight_path}")
    configure_reproducibility(args.seed, args.deterministic)
    query_indices, query_posteriors, query_labels, query_target_path, query_manifest = load_query_targets(
        protocol_root,
        dataset_name,
        model_name,
        args.budget,
        args.label_mode,
    )
    victim_model, victim_metadata = build_victim(model_name, num_classes, victim_checkpoint)
    surrogate_model, protection_plan, _, protection_masks = initialize_surrogate(
        factory=factory,
        factory_name=factory_name,
        weight_path=weight_path,
        victim_model=victim_model,
        num_classes=num_classes,
        defense=args.defense,
        protected_units=args.protected_units,
        protected_layers=args.protected_layers,
        protected_scalars=args.protected_scalars,
    )
    validate_built_plan(plan_config, protection_plan)
    query_dataset = build_query_dataset(
        dataset_name,
        dataset_root,
        query_indices,
        query_posteriors,
        query_labels,
    )
    eval_dataset = build_eval_dataset(dataset_name, dataset_root, protocol_root, args.eval_subset)

    victim_sha256 = sha256_file(victim_checkpoint)
    expected_sha256 = query_manifest.get("victim", {}).get("checkpoint_sha256")
    if expected_sha256 and expected_sha256 != victim_sha256:
        raise ValueError("当前 victim best.pth 与生成 query 标签时使用的 checkpoint 不一致。")
    query_target_config = {"posterior_sha256": sha256_file(query_target_path)}
    query_transform = "test" if args.label_mode == "soft" else "train"
    execution_mode = "identity" if args.defense == "no_protection" else "finetune"
    plan_run_config = (
        {
            "plan_id": args.plan_id,
            "protected_layer_count": plan_config.get("protected_layer_count"),
            "source_ratio": plan_config.get("source_ratio"),
        }
        if plan_config is not None
        else {}
    )
    run_config = {
        "schema_version": 2,
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "dataset": dataset_name,
        "model": model_name,
        "victim_checkpoint_sha256": victim_sha256,
        "official_weight_sha256": sha256_file(weight_path),
        **query_target_config,
        "query_transform": query_transform,
        "defense": args.defense,
        "tensor_unit_count": protection_plan.tensor_unit_count,
        "protection_mask_sha256": protection_plan.protection_mask_sha256,
        "protected_scalar_count": protection_plan.magnitude_protected_count,
        "head_mode": protection_plan.head_mode,
        **plan_run_config,
        "budget": args.budget,
        "training_mode": execution_mode,
        "label_mode": args.label_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "lr_step": args.lr_step,
        "lr_gamma": args.lr_gamma,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "eval_subset": args.eval_subset,
    }
    run_id = make_run_id(run_config)
    artifact_id = make_artifact_id(args.plan_id, args.defense, run_id)

    print(f"[INFO] run_id: {run_id}")
    print(f"[INFO] artifact_id: {artifact_id}")
    print(f"[INFO] 模型与数据集: {model_name}+{dataset_name}")
    print(f"[INFO] 保护策略: {args.defense}")
    print(f"[INFO] query budget: {args.budget}")
    print(f"[INFO] 标签模式: {args.label_mode}")
    print(f"[INFO] 攻击协议: {ATTACK_PROTOCOL_VERSION}")
    print(f"[INFO] 训练模式: {execution_mode}")
    print(f"[INFO] 分类头模式: {protection_plan.head_mode}")
    if args.plan_id is not None:
        print(f"[INFO] baseline plan_id: {args.plan_id}")
    if protection_plan.tensor_unit_count:
        print(f"[INFO] tensor unit 数量: {protection_plan.tensor_unit_count}")
        print(
            f"[INFO] 受保护 unit 数量: {protection_plan.protected_unit_count}/"
            f"{protection_plan.tensor_unit_count}"
        )
        print(f"[INFO] 保护掩码 SHA256: {protection_plan.protection_mask_sha256}")
    print(
        f"[INFO] victim 参数保护: {protection_plan.protected_param_count}/"
        f"{protection_plan.total_param_count} ({protection_plan.protected_param_ratio:.6f})"
    )
    print(f"[INFO] eval_ms 样本数: {len(eval_dataset)}")
    if args.dry_run:
        print("[INFO] dry-run 完成，未写入训练产物。")
        return 0

    weights_root = Path(args.weights_root).expanduser().resolve() / model_name / dataset_name
    results_root = Path(args.results_root).expanduser().resolve() / model_name / dataset_name
    weights_run = weights_root / artifact_id
    results_run = results_root / artifact_id
    if not args.overwrite and (weights_run.exists() or results_run.exists()):
        raise FileExistsError(
            f"artifact_id={artifact_id} 已存在；使用 --overwrite 重新运行。"
        )
    weights_run.mkdir(parents=True, exist_ok=True)
    results_run.mkdir(parents=True, exist_ok=True)
    protection_mask_path = weights_run / "protection_mask.pt"
    save_protection_mask(protection_mask_path, protection_masks)

    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    victim_model = victim_model.to(device)
    surrogate_model = surrogate_model.to(device)
    trainable_parameters = [parameter for parameter in surrogate_model.parameters() if parameter.requires_grad]
    optimizer = (
        torch.optim.SGD(
            trainable_parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        if trainable_parameters and args.defense != "no_protection"
        else None
    )
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        if optimizer is not None
        else None
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=build_generator(args.seed),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=build_generator(args.seed, offset=1),
    )
    reference = collect_eval_reference(victim_model, eval_loader, device)

    protection_metadata = {
        **protection_plan.to_metadata(),
        "mask_path": display_path(protection_mask_path),
    }
    query_target_params = {"posterior_path": str(query_target_path)}
    params = {
        **run_config,
        "artifact_id": artifact_id,
        "run_id": run_id,
        "device": str(device),
        "num_workers": args.num_workers,
        "eval_batch_size": args.eval_batch_size,
        "dataset_root": str(dataset_root),
        "protocol_root": str(protocol_root),
        "victim_checkpoint": str(victim_checkpoint),
        "official_weight": str(weight_path),
        **query_target_params,
        "weights_dir": str(weights_run),
        "results_dir": str(results_run),
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "protection": protection_metadata,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(weights_run / "params.json", params)
    history_path = weights_run / "train.log.tsv"
    write_history_row(history_path, {}, initialize=True)

    best_epoch = -1
    best_metrics: dict[str, int | float] | None = None
    end_metrics: dict[str, int | float] | None = None
    if optimizer is None:
        train_metrics = {
            "query_count": 0,
            "query_loss_sum": 0.0,
            "query_loss": 0.0,
            "query_match_count": 0,
            "query_match": 0.0,
        }
        end_metrics = evaluate_surrogate(surrogate_model, eval_loader, reference, device)
        best_metrics = end_metrics
        best_epoch = 0
        row = {"epoch": 0, "learning_rate": 0.0, **train_metrics, **end_metrics}
        write_history_row(history_path, row)
        save_checkpoint(
            weights_run / "best.pth", surrogate_model, None, None, 0,
            model_name, dataset_name, run_id, ATTACK_PROTOCOL_VERSION, best_metrics,
        )
    else:
        for epoch in range(1, args.epochs + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            train_metrics = train_one_epoch(
                surrogate_model,
                query_loader,
                optimizer,
                device,
                args.label_mode,
                epoch,
                args.epochs,
                None,
            )
            end_metrics = evaluate_surrogate(surrogate_model, eval_loader, reference, device)
            scheduler.step()
            row = {"epoch": epoch, "learning_rate": learning_rate, **train_metrics, **end_metrics}
            write_history_row(history_path, row)
            print(
                f"[EVAL] epoch={epoch:03d} surrogate_acc={end_metrics['surrogate_acc']:.6f} "
                f"fidelity={end_metrics['fidelity']:.6f} posterior_kl={end_metrics['posterior_kl']:.6f}"
            )
            if best_metrics is None or end_metrics["surrogate_acc"] > best_metrics["surrogate_acc"]:
                best_metrics = dict(end_metrics)
                best_epoch = epoch
                save_checkpoint(
                    weights_run / "best.pth", surrogate_model, optimizer, scheduler, epoch,
                    model_name, dataset_name, run_id, ATTACK_PROTOCOL_VERSION, best_metrics,
                )

    assert best_metrics is not None and end_metrics is not None
    end_epoch = args.epochs if optimizer is not None else 0
    save_checkpoint(
        weights_run / "end.pth", surrogate_model, optimizer, scheduler, end_epoch,
        model_name, dataset_name, run_id, ATTACK_PROTOCOL_VERSION, end_metrics,
    )

    metrics_payload = {
        "schema_version": 2,
        "protocol": "MS",
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "artifact_id": artifact_id,
        "run_id": run_id,
        "dataset": dataset_name,
        "victim_model": model_name,
        "query_budget": args.budget,
        "label_mode": args.label_mode,
        "query_transform": query_transform,
        "lr_step": args.lr_step,
        "training_mode": execution_mode,
        **plan_run_config,
        "protection": protection_metadata,
        "primary": {"checkpoint": "end.pth", "epoch": end_epoch},
        "diagnostic_best": {"metric": "surrogate_acc", "epoch": best_epoch},
        "best": best_metrics,
        "end": end_metrics,
    }
    metrics_path = results_run / "metrics.json"
    write_json(metrics_path, metrics_payload)
    index_row = {
        "artifact_id": artifact_id,
        "plan_id": args.plan_id or "",
        "run_id": run_id,
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "dataset": dataset_name,
        "victim_model": model_name,
        "defense": args.defense,
        "protected_layer_count": (
            plan_config.get("protected_layer_count", "") if plan_config is not None else ""
        ),
        "source_ratio": plan_config.get("source_ratio", "") if plan_config is not None else "",
        "training_mode": execution_mode,
        "label_mode": args.label_mode,
        "query_transform": query_transform,
        "query_budget": args.budget,
        "lr_step": args.lr_step,
        "protected_unit_count": protection_plan.protected_unit_count,
        "protection_mask_sha256": protection_plan.protection_mask_sha256,
        "protected_scalar_count": protection_plan.magnitude_protected_count,
        "protected_param_count": protection_plan.protected_param_count,
        "total_param_count": protection_plan.total_param_count,
        "protected_param_ratio": protection_plan.protected_param_ratio,
        "head_mode": protection_plan.head_mode,
        "primary_checkpoint": "end.pth",
        "primary_epoch": end_epoch,
        "best_epoch": best_epoch,
        **end_metrics,
        "metrics_path": display_path(metrics_path),
    }
    update_index(results_root / "metrics.tsv", index_row)
    print(f"[INFO] best checkpoint: {weights_run / 'best.pth'}")
    print(f"[INFO] end checkpoint: {weights_run / 'end.pth'}")
    print(f"[INFO] 原始指标: {metrics_path}")
    return 0
