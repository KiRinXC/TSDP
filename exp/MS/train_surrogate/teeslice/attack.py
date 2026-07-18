#!/usr/bin/env python3
"""使用统一 MS 协议攻击 TEESlice defended victim。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
    INDEX_FIELDS,
    make_run_id,
    save_checkpoint,
    sha256_file,
    write_history_row,
    write_json,
)
from core.config import ATTACK_PROTOCOL_VERSION  # noqa: E402
from core.data import (  # noqa: E402
    build_eval_dataset,
    build_query_partition_datasets,
    load_query_targets,
    make_query_partition,
)
from core.engine import (  # noqa: E402
    collect_eval_reference,
    evaluate_surrogate,
    select_validation_best,
)
from models.teeslice import TEESliceResNet18, teeslice_r18  # noqa: E402


QUERY_MODEL_ID = "teeslice_r18"
ARTIFACT_ID = "teeslice"
SOURCE_COMMIT = "93505cb3337ec8b89556ee29ffc598d31513aa5e"
BLACKBOX_VISIBILITY = "blackbox_known_pruned_topology"
WHITEBOX_VISIBILITY = "whitebox_full_state"


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


def describe_topology(keep_flags) -> dict[str, object]:
    normalized = [[bool(value) for value in block] for block in keep_flags]
    if not normalized or any(not block or not block[0] for block in normalized):
        raise ValueError("TEESlice topology 必须包含且保留每个 block 的 main path。")
    canonical = json.dumps(normalized, ensure_ascii=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "keep_flags": normalized,
        "block_count": len(normalized),
        "active_proxy_count": sum(sum(block[1:]) for block in normalized),
        "topology_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def build_known_topology_surrogate(
    weight_path: Path,
    keep_flags,
    seed: int,
    deterministic: bool,
) -> TEESliceResNet18:
    """只复用公开拓扑和 backbone，重新初始化所有私有状态。"""
    configure_reproducibility(seed, deterministic)
    surrogate = teeslice_r18(num_classes=100, weight_path=weight_path)
    surrogate.set_keep_flags(keep_flags)
    for parameter in surrogate.parameters():
        parameter.requires_grad_(True)
    return surrogate


def validate_public_backbone(victim: TEESliceResNet18, surrogate: TEESliceResNet18) -> None:
    victim_public = victim.public_state_dict()
    surrogate_public = surrogate.public_state_dict()
    if victim_public.keys() != surrogate_public.keys():
        raise ValueError("victim 与 surrogate 的公开 backbone 字段不一致。")
    mismatched = [
        name
        for name in victim_public
        if not torch.equal(victim_public[name], surrogate_public[name])
    ]
    if mismatched:
        raise ValueError(f"victim 公开 backbone 已偏离官方初始化：{mismatched[0]}")


def remove_legacy_index_row(path: Path) -> None:
    if not path.is_file():
        return
    with path.open("r", newline="", encoding="utf-8") as reader_file:
        reader = csv.DictReader(reader_file, delimiter="\t")
        if reader.fieldnames != INDEX_FIELDS:
            path.unlink()
            return
        rows = [row for row in reader if row["artifact_id"] != ARTIFACT_ID]
    with path.open("w", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(writer_file, fieldnames=INDEX_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    expected_hyperparameters = {
        "epochs": 100,
        "batch_size": 64,
        "eval_batch_size": 128,
        "lr": 0.01,
        "momentum": 0.5,
        "weight_decay": 5e-4,
        "lr_step": 60,
        "lr_gamma": 0.1,
    }
    actual_hyperparameters = {
        name: getattr(args, name) for name in expected_hyperparameters
    }
    if actual_hyperparameters != expected_hyperparameters:
        raise ValueError(
            "TEESlice 正式攻击超参数不可临时覆盖："
            f"expected={expected_hyperparameters}, actual={actual_hyperparameters}"
        )
    if args.eval_subset is not None:
        raise ValueError("TEESlice 正式攻击不允许裁剪 eval_ms。")
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
    topology = describe_topology(victim_checkpoint["keep_flags"])
    surrogate = build_known_topology_surrogate(
        weight_path,
        victim_checkpoint["keep_flags"],
        args.seed,
        args.deterministic,
    )
    validate_public_backbone(victim, surrogate)
    if surrogate.active_proxy_count() != topology["active_proxy_count"]:
        raise ValueError("surrogate 活跃 proxy 数与公开剪枝拓扑不一致。")
    query_partition = make_query_partition(query_indices, seed=args.seed)
    query_train_dataset, query_validation_dataset = build_query_partition_datasets(
        dataset,
        dataset_root,
        query_indices,
        query_posteriors,
        query_labels,
        query_partition,
    )
    victim_cost = victim.cost_summary()
    protection = {
        "strategy": "teeslice",
        "granularity": "private_slice",
        "source_commit": SOURCE_COMMIT,
        "private_proxy_count": int(victim_cost["active_proxy_count"]),
        "private_head_count": 1,
        "tensor_unit_count": 0,
        "protected_param_count": int(victim_cost["private_param_count"]),
        "protected_bn_buffer_count": int(victim_cost["private_bn_buffer_count"]),
        "paper_private_param_count": int(victim_cost["paper_private_param_count"]),
        "total_param_count": int(victim_cost["total_param_count"]),
        "protected_param_ratio": float(victim_cost["private_param_ratio"]),
        "protected_flops": int(victim_cost["private_flops"]),
        "paper_private_flops": int(victim_cost["paper_private_flops"]),
        "total_flops": int(victim_cost["total_flops"]),
        "protected_flops_ratio": float(victim_cost["private_flops_ratio"]),
        "head_mode": "replace",
        "public_backbone": "official_imagenet_cifar_resnet18",
        "topology_sha256": topology["topology_sha256"],
    }
    run_config = {
        "schema_version": 2,
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "comparison_scope": "standalone_reproduction",
        "visibility": BLACKBOX_VISIBILITY,
        "whitebox_visibility": WHITEBOX_VISIBILITY,
        "dataset": dataset,
        "model": args.model,
        "victim_model": QUERY_MODEL_ID,
        "surrogate_model": QUERY_MODEL_ID,
        "surrogate_initialization": "official_backbone_fresh_private_state",
        "topology_sha256": topology["topology_sha256"],
        "victim_checkpoint_sha256": victim_sha256,
        "official_weight_sha256": sha256_file(weight_path),
        "posterior_sha256": sha256_file(posterior_path),
        "query_transform": "test",
        "defense": "teeslice",
        "protected_param_count": protection["protected_param_count"],
        "protected_flops": protection["protected_flops"],
        "head_mode": "replace",
        "budget": args.budget,
        "query_train_size": query_partition.train_size,
        "query_validation_size": query_partition.validation_size,
        "query_split_seed": args.seed,
        "query_split_seed_offset": query_partition.seed_offset,
        "query_train_ranks_sha256": query_partition.to_metadata()["train_ranks_sha256"],
        "query_validation_ranks_sha256": (
            query_partition.to_metadata()["validation_ranks_sha256"]
        ),
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
    print(
        f"[INFO] topology={topology['topology_sha256']} "
        f"active_proxy={topology['active_proxy_count']}"
    )
    print(
        f"[INFO] query={args.budget} train={query_partition.train_size} "
        f"validation={query_partition.validation_size} device={device}"
    )
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
    remove_legacy_index_row(results_base / "metrics.tsv")
    weights_run.mkdir(parents=True, exist_ok=True)
    results_run.mkdir(parents=True, exist_ok=True)

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
    query_train_loader = DataLoader(
        query_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=build_generator(args.seed),
    )
    query_validation_loader = DataLoader(
        query_validation_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=build_generator(args.seed, offset=1),
    )
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
        "topology": topology,
        "protection": protection,
        "surrogate_initialization_seed": args.seed,
        "query_sampler_seed": args.seed,
        "query_partition": query_partition.to_metadata(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(weights_run / "params.json", params)
    write_json(
        weights_run / "topology.json",
        {
            **topology,
            "source": "victim_checkpoint_keep_flags",
            "source_checkpoint_sha256": victim_sha256,
            "contains_private_values": False,
        },
    )
    history_path = weights_run / "train.log.tsv"
    write_history_row(history_path, {}, initialize=True)

    selection, history = select_validation_best(
        surrogate,
        query_train_loader,
        query_validation_loader,
        optimizer,
        scheduler,
        device,
        args.label_mode,
        args.epochs,
        query_partition.validation_size,
    )
    for row in history:
        write_history_row(history_path, row)
    selected_epoch = int(selection["epoch"])
    save_checkpoint(
        weights_run / "best.pth",
        surrogate,
        None,
        None,
        selected_epoch,
        QUERY_MODEL_ID,
        dataset,
        run_id,
        ATTACK_PROTOCOL_VERSION,
        selection,
    )

    # checkpoint 固定后才首次构造和迭代 eval_ms。
    eval_dataset = build_eval_dataset(dataset, dataset_root, protocol_root, None)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=build_generator(args.seed, offset=2),
    )
    victim = victim.to(device)
    reference = collect_eval_reference(victim, eval_loader, device)
    result_metrics = evaluate_surrogate(surrogate, eval_loader, reference, device)
    whitebox, _ = load_victim(victim_path, weight_path)
    whitebox = whitebox.to(device)
    whitebox_metrics = evaluate_surrogate(whitebox, eval_loader, reference, device)
    if whitebox_metrics["agreement_count"] != whitebox_metrics["eval_count"]:
        raise RuntimeError("完整状态白盒模型未能逐样本复现 victim 输出类别。")
    del whitebox

    metrics_payload = {
        "schema_version": 3,
        "protocol": "MS",
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "comparison_scope": "standalone_reproduction",
        "visibility": BLACKBOX_VISIBILITY,
        "whitebox_visibility": WHITEBOX_VISIBILITY,
        "artifact_id": ARTIFACT_ID,
        "run_id": run_id,
        "dataset": dataset,
        "victim_model": args.model,
        "defended_victim": QUERY_MODEL_ID,
        "surrogate_model": QUERY_MODEL_ID,
        "surrogate_initialization": "official_backbone_fresh_private_state",
        "topology": topology,
        "query_budget": args.budget,
        "query_partition": query_partition.to_metadata(),
        "query_sampler_seed": args.seed,
        "label_mode": args.label_mode,
        "query_transform": "test",
        "lr_step": args.lr_step,
        "training_mode": args.training_mode,
        "protection": protection,
        "primary": {
            "checkpoint": "best.pth",
            "epoch": selected_epoch,
            "selection_metric": selection["metric"],
        },
        "selection": selection,
        "result": {**result_metrics, "eval_passes": 1},
        "whitebox": {**whitebox_metrics, "eval_passes": 1},
    }
    write_json(metrics_path, metrics_payload)
    print(f"[INFO] best checkpoint: {weights_run / 'best.pth'}")
    print(f"[INFO] 原始指标: {metrics_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
