#!/usr/bin/env python3
"""使用统一 MS 协议攻击 TEESlice defended victim。"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[4]
TRAIN_VICTIM_ROOT = REPO_ROOT / "exp" / "MS" / "train_victim"
TRAIN_SURROGATE_ROOT = REPO_ROOT / "exp" / "MS" / "train_surrogate"
for import_root in (REPO_ROOT, TRAIN_VICTIM_ROOT, TRAIN_SURROGATE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common.trainer import (  # noqa: E402
    build_generator,
    configure_reproducibility,
    resolve_dataset_name,
    seed_worker,
)
from core.artifacts import (  # noqa: E402
    display_path,
    make_run_id,
    save_checkpoint,
    sha256_file,
    update_index,
    write_history_row,
    write_json,
)
from core.config import ATTACK_PROTOCOL_VERSION  # noqa: E402
from core.data import build_eval_dataset, build_query_dataset, load_query_targets  # noqa: E402
from core.engine import collect_eval_reference, evaluate_surrogate, train_one_epoch  # noqa: E402
from models.teeslice import cifar_resnet18, teeslice_r18  # noqa: E402


QUERY_MODEL_ID = "teeslice_r18"
ARTIFACT_ID = "teeslice"
SOURCE_COMMIT = "93505cb3337ec8b89556ee29ffc598d31513aa5e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="攻击 TEESlice ResNet18+C100 defended victim。")
    parser.add_argument("model", choices=("resnet18",))
    parser.add_argument("dataset", choices=("c100",))
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--training-mode", required=True, choices=("finetune",))
    parser.add_argument("--label-mode", required=True, choices=("soft",))
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "dataset" / "public"))
    parser.add_argument("--protocol-root", default=str(REPO_ROOT / "dataset" / "MS"))
    parser.add_argument(
        "--victim-checkpoint",
        default=str(REPO_ROOT / "weights" / "MS" / "victim" / QUERY_MODEL_ID / "c100" / "best.pth"),
    )
    parser.add_argument(
        "--weight-path",
        default=str(REPO_ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"),
    )
    parser.add_argument("--weights-root", default=str(REPO_ROOT / "weights" / "MS" / "surrogate"))
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results" / "MS"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lr-step", type=int, default=60)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-subset", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境没有可用 CUDA 设备。")
    return device


def load_victim(path: Path, weight_path: Path):
    if path.name != "best.pth" or not path.is_file():
        raise FileNotFoundError(f"找不到 TEESlice victim best.pth：{path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("arch") != QUERY_MODEL_ID or checkpoint.get("defense") != "teeslice":
        raise ValueError(f"{path} 不是 TEESlice ResNet18 checkpoint。")
    model = teeslice_r18(num_classes=100, weight_path=weight_path)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if checkpoint.get("keep_flags") is None:
        raise ValueError(f"{path} 缺少 keep_flags。")
    model.set_keep_flags(checkpoint["keep_flags"])
    return model, checkpoint


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("epochs、batch_size 和 eval_batch_size 必须大于 0。")
    dataset = resolve_dataset_name(args.dataset)
    configure_reproducibility(args.seed, args.deterministic)
    device = resolve_device(args.device)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    protocol_root = Path(args.protocol_root).expanduser().resolve()
    victim_path = Path(args.victim_checkpoint).expanduser().resolve()
    weight_path = Path(args.weight_path).expanduser().resolve()

    query_indices, query_posteriors, query_labels, posterior_path, query_manifest = load_query_targets(
        protocol_root,
        dataset,
        QUERY_MODEL_ID,
        args.budget,
        args.label_mode,
    )
    victim, victim_checkpoint = load_victim(victim_path, weight_path)
    victim_sha256 = sha256_file(victim_path)
    expected_sha256 = query_manifest.get("victim", {}).get("checkpoint_sha256")
    if expected_sha256 != victim_sha256:
        raise ValueError("TEESlice query posterior 与当前 victim best.pth 不一致。")
    surrogate = cifar_resnet18(num_classes=100, weight_path=weight_path)
    query_dataset = build_query_dataset(dataset, dataset_root, query_indices, query_posteriors, query_labels)
    eval_dataset = build_eval_dataset(dataset, dataset_root, protocol_root, args.eval_subset)
    protection = {
        "strategy": "teeslice",
        "granularity": "private_slice",
        "source_commit": SOURCE_COMMIT,
        "private_proxy_count": int(victim.cost_summary()["active_proxy_count"]),
        "private_head_count": 1,
        "tensor_unit_count": 0,
        "protected_param_count": int(victim.cost_summary()["private_param_count"]),
        "protected_bn_buffer_count": int(victim.cost_summary()["private_bn_buffer_count"]),
        "paper_private_param_count": int(victim.cost_summary()["paper_private_param_count"]),
        "total_param_count": int(victim.cost_summary()["total_param_count"]),
        "protected_param_ratio": float(victim.cost_summary()["private_param_ratio"]),
        "protected_flops": int(victim.cost_summary()["private_flops"]),
        "paper_private_flops": int(victim.cost_summary()["paper_private_flops"]),
        "total_flops": int(victim.cost_summary()["total_flops"]),
        "protected_flops_ratio": float(victim.cost_summary()["private_flops_ratio"]),
        "head_mode": "replace",
        "public_backbone": "official_imagenet_cifar_resnet18",
        "protection_mask_sha256": "",
    }
    run_config = {
        "schema_version": 2,
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "dataset": dataset,
        "model": args.model,
        "victim_model": QUERY_MODEL_ID,
        "victim_checkpoint_sha256": victim_sha256,
        "official_weight_sha256": sha256_file(weight_path),
        "posterior_sha256": sha256_file(posterior_path),
        "query_transform": "test",
        "defense": "teeslice",
        "protected_param_count": protection["protected_param_count"],
        "protected_flops": protection["protected_flops"],
        "head_mode": "replace",
        "budget": args.budget,
        "training_mode": args.training_mode,
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
    print(f"[INFO] run_id={run_id} artifact_id={ARTIFACT_ID}")
    print(f"[INFO] TEESlice cost={protection}")
    print(f"[INFO] query={len(query_dataset)} eval_ms={len(eval_dataset)} device={device}")
    if args.dry_run:
        print("[INFO] dry-run 完成，未写入训练产物。")
        return 0

    weights_run = (
        Path(args.weights_root).expanduser().resolve() / args.model / dataset / ARTIFACT_ID
    )
    results_base = Path(args.results_root).expanduser().resolve() / args.model / dataset
    results_run = results_base / ARTIFACT_ID
    metrics_path = results_run / "metrics.json"
    if not args.overwrite and (weights_run.exists() or metrics_path.exists()):
        raise FileExistsError("TEESlice surrogate 产物已存在；使用 --overwrite 重新运行。")
    if args.overwrite:
        shutil.rmtree(weights_run, ignore_errors=True)
        metrics_path.unlink(missing_ok=True)
    weights_run.mkdir(parents=True, exist_ok=True)
    results_run.mkdir(parents=True, exist_ok=True)

    victim = victim.to(device)
    surrogate = surrogate.to(device)
    optimizer = torch.optim.SGD(
        surrogate.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step,
        gamma=args.lr_gamma,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=build_generator(args.seed),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=build_generator(args.seed, offset=1),
    )
    reference = collect_eval_reference(victim, eval_loader, device)
    params = {
        **run_config,
        "artifact_id": ARTIFACT_ID,
        "run_id": run_id,
        "device": str(device),
        "num_workers": args.num_workers,
        "dataset_root": str(dataset_root),
        "protocol_root": str(protocol_root),
        "victim_checkpoint": str(victim_path),
        "victim_checkpoint_epoch": victim_checkpoint.get("epoch"),
        "official_weight": str(weight_path),
        "posterior_path": str(posterior_path),
        "weights_dir": str(weights_run),
        "results_dir": str(results_run),
        "protection": protection,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(weights_run / "params.json", params)
    history_path = weights_run / "train.log.tsv"
    write_history_row(history_path, {}, initialize=True)

    best_metrics: dict[str, int | float] | None = None
    best_epoch = -1
    end_metrics: dict[str, int | float] | None = None
    for epoch in range(1, args.epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(
            surrogate,
            query_loader,
            optimizer,
            device,
            args.label_mode,
            epoch,
            args.epochs,
            None,
        )
        end_metrics = evaluate_surrogate(surrogate, eval_loader, reference, device)
        scheduler.step()
        write_history_row(
            history_path,
            {"epoch": epoch, "learning_rate": learning_rate, **train_metrics, **end_metrics},
        )
        print(
            f"[EVAL] epoch={epoch:03d} surrogate_acc={end_metrics['surrogate_acc']:.6f} "
            f"fidelity={end_metrics['fidelity']:.6f} posterior_kl={end_metrics['posterior_kl']:.6f}"
        )
        if best_metrics is None or end_metrics["surrogate_acc"] > best_metrics["surrogate_acc"]:
            best_metrics = dict(end_metrics)
            best_epoch = epoch
            save_checkpoint(
                weights_run / "best.pth",
                surrogate,
                optimizer,
                scheduler,
                epoch,
                "cifar_resnet18",
                dataset,
                run_id,
                ATTACK_PROTOCOL_VERSION,
                best_metrics,
            )
    assert best_metrics is not None and end_metrics is not None
    save_checkpoint(
        weights_run / "end.pth",
        surrogate,
        optimizer,
        scheduler,
        args.epochs,
        "cifar_resnet18",
        dataset,
        run_id,
        ATTACK_PROTOCOL_VERSION,
        end_metrics,
    )
    metrics_payload = {
        "schema_version": 2,
        "protocol": "MS",
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "artifact_id": ARTIFACT_ID,
        "run_id": run_id,
        "dataset": dataset,
        "victim_model": args.model,
        "defended_victim": QUERY_MODEL_ID,
        "query_budget": args.budget,
        "label_mode": args.label_mode,
        "query_transform": "test",
        "lr_step": args.lr_step,
        "training_mode": args.training_mode,
        "protection": protection,
        "primary": {"checkpoint": "end.pth", "epoch": args.epochs},
        "diagnostic_best": {"metric": "surrogate_acc", "epoch": best_epoch},
        "best": best_metrics,
        "end": end_metrics,
    }
    write_json(metrics_path, metrics_payload)
    update_index(
        results_base / "metrics.tsv",
        {
            "artifact_id": ARTIFACT_ID,
            "plan_id": "",
            "run_id": run_id,
            "attack_protocol": ATTACK_PROTOCOL_VERSION,
            "dataset": dataset,
            "victim_model": args.model,
            "defense": "teeslice",
            "protected_layer_count": "",
            "source_ratio": "",
            "training_mode": args.training_mode,
            "label_mode": args.label_mode,
            "query_transform": "test",
            "query_budget": args.budget,
            "lr_step": args.lr_step,
            "protected_unit_count": "",
            "protection_mask_sha256": "",
            "protected_scalar_count": "",
            "protected_param_count": protection["protected_param_count"],
            "total_param_count": protection["total_param_count"],
            "protected_param_ratio": protection["protected_param_ratio"],
            "head_mode": "replace",
            "primary_checkpoint": "end.pth",
            "primary_epoch": args.epochs,
            "best_epoch": best_epoch,
            **end_metrics,
            "metrics_path": display_path(metrics_path),
        },
    )
    print(f"[INFO] best checkpoint: {weights_run / 'best.pth'}")
    print(f"[INFO] end checkpoint: {weights_run / 'end.pth'}")
    print(f"[INFO] 原始指标: {metrics_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
