#!/usr/bin/env python3
"""训练并剪枝 TEESlice defended victim。"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[4]
TRAIN_VICTIM_ROOT = REPO_ROOT / "exp" / "MS" / "train_victim"
TRAIN_SURROGATE_ROOT = REPO_ROOT / "exp" / "MS" / "train_surrogate"
for import_root in (REPO_ROOT, TRAIN_VICTIM_ROOT, TRAIN_SURROGATE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common.trainer import (  # noqa: E402
    MS_EVAL_SOURCES,
    build_generator,
    build_public_split_dataset,
    build_transforms,
    configure_reproducibility,
    read_ms_split_indices,
    resolve_dataset_name,
    seed_worker,
)
from core.artifacts import INDEX_FIELDS, make_run_id, sha256_file  # noqa: E402
from models import imagenet as imagenet_models  # noqa: E402
from models.teeslice import (  # noqa: E402
    PUBLISHED_C100_R18_KEEP_FLAGS,
    PUBLISHED_C100_R18_TASK_FLOPS,
    PUBLISHED_C100_R18_TASK_PARAMS,
    CifarResNet18,
    TEESliceResNet18,
    cifar_resnet18,
    teeslice_r18,
)


MODEL_ID = "teeslice_r18"
SOURCE_COMMIT = "93505cb3337ec8b89556ee29ffc598d31513aa5e"
LOG_FIELDS = [
    "stage",
    "epoch",
    "learning_rate",
    "train_loss",
    "train_kd",
    "train_feature",
    "train_complexity",
    "train_accuracy",
    "internal_val_loss",
    "internal_val_accuracy",
    "internal_val_fidelity",
    "active_proxy_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 TEESlice ResNet18+C100 defended victim。")
    parser.add_argument("model", choices=("resnet18",), help="被保护的原始 victim 架构。")
    parser.add_argument("dataset", choices=("c100",), help="当前固化的数据集。")
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "dataset" / "public"))
    parser.add_argument("--protocol-root", default=str(REPO_ROOT / "dataset" / "MS"))
    parser.add_argument(
        "--victim-checkpoint",
        default=str(REPO_ROOT / "weights" / "MS" / "victim" / "resnet18" / "c100" / "best.pth"),
    )
    parser.add_argument(
        "--weight-path",
        default=str(REPO_ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "weights" / "MS" / "victim" / MODEL_ID / "c100"),
    )
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results" / "MS"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--full-epochs", type=int, default=40)
    parser.add_argument("--prune-epochs", type=int, default=20)
    parser.add_argument("--teacher-lr", type=float, default=0.01)
    parser.add_argument("--full-lr", type=float, default=0.1)
    parser.add_argument("--prune-lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--teacher-momentum", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=4e-4)
    parser.add_argument("--teacher-step", type=int, default=20)
    parser.add_argument("--full-step", type=int, default=30)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--max-skip", type=int, default=3)
    parser.add_argument("--complexity-coeff", type=float, default=0.3)
    parser.add_argument("--teacher-coeff", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--internal-val-ratio", type=float, default=0.1)
    parser.add_argument("--prune-tolerance", type=float, default=0.01)
    parser.add_argument("--prune-threshold", type=float, default=0.1)
    parser.add_argument("--initial-prune-fraction", type=float, default=0.5)
    parser.add_argument("--iterative-prune-fraction", type=float, default=0.05)
    parser.add_argument("--prune-interval", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--train-subset", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--val-subset", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--eval-subset", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
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


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def tensor_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def limit_indices(indices: list[int], limit: int | None) -> list[int]:
    if limit is None or limit <= 0 or limit >= len(indices):
        return indices
    return indices[:limit]


def build_datasets(args: argparse.Namespace):
    dataset = resolve_dataset_name(args.dataset)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    protocol_root = Path(args.protocol_root).expanduser().resolve()
    train_transform, test_transform = build_transforms(dataset)
    train_public = build_public_split_dataset(dataset, dataset_root, "train", train_transform)
    deterministic_train_public = build_public_split_dataset(dataset, dataset_root, "train", test_transform)
    eval_public = build_public_split_dataset(dataset, dataset_root, "test", test_transform)

    victim_indices = read_ms_split_indices(protocol_root, dataset, "victim_train", "official_train")
    shuffled = victim_indices[:]
    random.Random(args.seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * args.internal_val_ratio))
    val_indices = shuffled[:val_count]
    train_indices = shuffled[val_count:]
    train_indices = limit_indices(train_indices, args.train_subset)
    val_indices = limit_indices(val_indices, args.val_subset)
    eval_indices = read_ms_split_indices(
        protocol_root,
        dataset,
        "eval_ms",
        MS_EVAL_SOURCES[dataset],
    )
    eval_indices = limit_indices(eval_indices, args.eval_subset)
    return (
        Subset(train_public, train_indices),
        Subset(deterministic_train_public, val_indices),
        Subset(eval_public, eval_indices),
        train_indices,
        val_indices,
    )


def build_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    args: argparse.Namespace,
    offset: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=build_generator(args.seed, offset=offset),
    )


def load_original_victim(path: Path, device: torch.device) -> nn.Module:
    if path.name != "best.pth" or not path.is_file():
        raise FileNotFoundError(f"找不到普通 victim best.pth：{path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError("普通 victim checkpoint 缺少 state_dict。")
    model = imagenet_models.resnet18(num_classes=100)
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.to(device).eval()


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    student_log = functional.log_softmax(student_logits / temperature, dim=1)
    teacher_probability = functional.softmax(teacher_logits / temperature, dim=1)
    return functional.kl_div(student_log, teacher_probability, reduction="batchmean") * temperature**2


@torch.inference_mode()
def evaluate_against(
    model: nn.Module,
    reference: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    reference.eval()
    loss_sum = 0.0
    correct = 0
    agreement = 0
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        reference_logits = reference(images)
        loss_sum += float(functional.cross_entropy(logits, labels, reduction="sum").item())
        correct += int((logits.argmax(1) == labels).sum().item())
        agreement += int((logits.argmax(1) == reference_logits.argmax(1)).sum().item())
        count += labels.size(0)
    return {
        "count": count,
        "loss": loss_sum / max(count, 1),
        "accuracy": correct / max(count, 1),
        "fidelity": agreement / max(count, 1),
    }


def train_teacher_epoch(
    model: CifarResNet18,
    victim: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    model.train()
    victim.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    progress = tqdm(loader, desc=f"[TEACHER] {epoch:03d}/{total_epochs:03d}", dynamic_ncols=True)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            victim_logits = victim(images)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = kd_loss(logits, victim_logits, temperature)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        count += batch_size
        loss_sum += float(loss.item()) * batch_size
        correct += int((logits.argmax(1) == labels).sum().item())
        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / count:.4f}")
    return {"loss": loss_sum / count, "accuracy": correct / count}


def train_slice_epoch(
    model: TEESliceResNet18,
    teacher: CifarResNet18,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    teacher_coeff: float,
    complexity_coeff: float,
    epoch: int,
    total_epochs: int,
    stage: str,
) -> dict[str, float]:
    model.train()
    teacher.eval()
    sums = {"loss": 0.0, "kd": 0.0, "feature": 0.0, "complexity": 0.0}
    correct = 0
    count = 0
    progress = tqdm(loader, desc=f"[{stage.upper()}] {epoch:03d}/{total_epochs:03d}", dynamic_ncols=True)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            teacher_logits, teacher_features = teacher(images, return_features=True)
        optimizer.zero_grad(set_to_none=True)
        logits, features = model(images, return_features=True)
        kd = kd_loss(logits, teacher_logits, temperature)
        feature = sum(
            functional.mse_loss(student_feature, teacher_feature)
            for student_feature, teacher_feature in zip(features, teacher_features, strict=True)
        ) / len(features)
        complexity = model.expected_complexity() if complexity_coeff else logits.new_zeros(())
        loss = kd + teacher_coeff * feature + complexity_coeff * complexity
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        count += batch_size
        correct += int((logits.argmax(1) == labels).sum().item())
        for name, value in (
            ("loss", loss),
            ("kd", kd),
            ("feature", feature),
            ("complexity", complexity),
        ):
            sums[name] += float(value.item()) * batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / count:.4f}")
    return {**{name: value / count for name, value in sums.items()}, "accuracy": correct / count}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    stage: str,
    epoch: int,
    run_id: str,
    metrics: dict[str, object],
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> None:
    keep_flags = model.get_keep_flags() if isinstance(model, TEESliceResNet18) else None
    cost = model.cost_summary() if isinstance(model, TEESliceResNet18) else None
    torch.save(
        {
            "schema_version": 1,
            "protocol": "MS",
            "defense": "teeslice",
            "arch": MODEL_ID if isinstance(model, TEESliceResNet18) else "cifar_resnet18",
            "base_model": "resnet18",
            "dataset": "c100",
            "stage": stage,
            "epoch": epoch,
            "run_id": run_id,
            "state_dict": cpu_state_dict(model),
            "keep_flags": keep_flags,
            "cost": cost,
            "metrics": metrics,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        path,
    )


def append_log(path: Path, row: dict[str, object], initialize: bool = False) -> None:
    with path.open("w" if initialize else "a", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(writer_file, fieldnames=LOG_FIELDS, delimiter="\t", lineterminator="\n")
        if initialize:
            writer.writeheader()
        else:
            writer.writerow({name: row.get(name, "") for name in LOG_FIELDS})


def stage_row(
    stage: str,
    epoch: int,
    lr: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float | int],
    proxy_count: int | str = "",
) -> dict[str, object]:
    return {
        "stage": stage,
        "epoch": epoch,
        "learning_rate": lr,
        "train_loss": train_metrics.get("loss", ""),
        "train_kd": train_metrics.get("kd", train_metrics.get("loss", "")),
        "train_feature": train_metrics.get("feature", ""),
        "train_complexity": train_metrics.get("complexity", ""),
        "train_accuracy": train_metrics.get("accuracy", ""),
        "internal_val_loss": val_metrics["loss"],
        "internal_val_accuracy": val_metrics["accuracy"],
        "internal_val_fidelity": val_metrics["fidelity"],
        "active_proxy_count": proxy_count,
    }


def load_stage_model(
    checkpoint_path: Path,
    model: nn.Module,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if isinstance(model, TEESliceResNet18) and checkpoint.get("keep_flags") is not None:
        model.set_keep_flags(checkpoint["keep_flags"])
    return checkpoint


def remove_index_row(path: Path, artifact_id: str) -> None:
    if not path.is_file():
        return
    with path.open("r", newline="", encoding="utf-8") as reader_file:
        reader = csv.DictReader(reader_file, delimiter="\t")
        if reader.fieldnames != INDEX_FIELDS:
            raise ValueError(f"结果索引字段不兼容：{path}")
        rows = [row for row in reader if row["artifact_id"] != artifact_id]
    with path.open("w", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(writer_file, fieldnames=INDEX_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare_output(args: argparse.Namespace, out_dir: Path) -> None:
    repo_default = out_dir == (REPO_ROOT / "weights" / "MS" / "victim" / MODEL_ID / "c100")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"TEESlice victim 产物已存在：{out_dir}。使用 --overwrite 覆盖。")
    if not args.overwrite:
        return
    if out_dir.exists():
        shutil.rmtree(out_dir)
    if repo_default:
        shutil.rmtree(REPO_ROOT / "dataset" / "MS" / "c100" / MODEL_ID, ignore_errors=True)
        shutil.rmtree(REPO_ROOT / "weights" / "MS" / "surrogate" / "resnet18" / "c100" / "teeslice", ignore_errors=True)
        shutil.rmtree(REPO_ROOT / "results" / "MS" / "resnet18" / "c100" / "teeslice", ignore_errors=True)
        remove_index_row(REPO_ROOT / "results" / "MS" / "resnet18" / "c100" / "metrics.tsv", "teeslice")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.quick:
        args.teacher_epochs = args.full_epochs = args.prune_epochs = 1
        args.train_subset = args.train_subset or 128
        args.val_subset = args.val_subset or 64
        args.eval_subset = args.eval_subset or 64
        args.num_workers = 0
        args.prune_interval = 1
    for value, name in (
        (args.teacher_epochs, "teacher_epochs"),
        (args.full_epochs, "full_epochs"),
        (args.prune_epochs, "prune_epochs"),
        (args.batch_size, "batch_size"),
        (args.eval_batch_size, "eval_batch_size"),
    ):
        if value <= 0:
            raise ValueError(f"{name} 必须大于 0。")
    if not 0.0 < args.internal_val_ratio < 1.0:
        raise ValueError("internal_val_ratio 必须位于 (0, 1)。")

    configure_reproducibility(args.seed, args.deterministic)
    device = resolve_device(args.device)
    victim_path = Path(args.victim_checkpoint).expanduser().resolve()
    weight_path = Path(args.weight_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    trainset, valset, evalset, train_indices, val_indices = build_datasets(args)
    print(f"[INFO] TEESlice 数据：train={len(trainset)} internal_val={len(valset)} eval_ms={len(evalset)}")

    published_probe = teeslice_r18(100, weight_path, max_skip=args.max_skip)
    published_probe.set_keep_flags(PUBLISHED_C100_R18_KEEP_FLAGS)
    published_cost = published_probe.cost_summary()
    if published_cost["paper_private_param_count"] != PUBLISHED_C100_R18_TASK_PARAMS:
        raise RuntimeError("公开模型代码无法重建作者发布的 TEESlice task parameter。")
    if published_cost["paper_private_flops"] != PUBLISHED_C100_R18_TASK_FLOPS:
        raise RuntimeError("公开模型代码无法重建作者发布的 TEESlice task FLOPs。")
    del published_probe

    run_config = {
        "schema_version": 1,
        "protocol": "MS",
        "defense": "teeslice",
        "source_commit": SOURCE_COMMIT,
        "model": args.model,
        "defended_model": MODEL_ID,
        "dataset": args.dataset,
        "victim_checkpoint_sha256": sha256_file(victim_path),
        "official_weight_sha256": sha256_file(weight_path),
        "train_count": len(train_indices),
        "internal_val_count": len(val_indices),
        "teacher_epochs": args.teacher_epochs,
        "full_epochs": args.full_epochs,
        "prune_epochs": args.prune_epochs,
        "batch_size": args.batch_size,
        "teacher_lr": args.teacher_lr,
        "full_lr": args.full_lr,
        "prune_lr": args.prune_lr,
        "momentum": args.momentum,
        "teacher_momentum": args.teacher_momentum,
        "weight_decay": args.weight_decay,
        "max_skip": args.max_skip,
        "complexity_coeff": args.complexity_coeff,
        "teacher_coeff": args.teacher_coeff,
        "temperature": args.temperature,
        "prune_tolerance": args.prune_tolerance,
        "prune_threshold": args.prune_threshold,
        "initial_prune_fraction": args.initial_prune_fraction,
        "iterative_prune_fraction": args.iterative_prune_fraction,
        "prune_interval": args.prune_interval,
        "seed": args.seed,
        "deterministic": args.deterministic,
    }
    run_id = make_run_id(run_config)
    print(f"[INFO] run_id={run_id} device={device}")
    print(f"[INFO] 作者发布成本核对：params={PUBLISHED_C100_R18_TASK_PARAMS} FLOPs={PUBLISHED_C100_R18_TASK_FLOPS}")
    if args.dry_run:
        print("[INFO] dry-run 完成，未写入训练产物。")
        return 0

    prepare_output(args, out_dir)
    teacher_dir = out_dir / "teacher"
    full_dir = out_dir / "full"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log.tsv"
    append_log(log_path, {}, initialize=True)
    params = {
        **run_config,
        "run_id": run_id,
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "protocol_root": str(Path(args.protocol_root).expanduser().resolve()),
        "victim_checkpoint": str(victim_path),
        "official_weight": str(weight_path),
        "out_dir": str(out_dir),
        "internal_split": {
            "source": "victim_train",
            "train_count": len(train_indices),
            "validation_count": len(val_indices),
            "selection_seed": args.seed,
        },
        "published_reference": {
            "task_param_count": PUBLISHED_C100_R18_TASK_PARAMS,
            "task_flops": PUBLISHED_C100_R18_TASK_FLOPS,
            "keep_flags": PUBLISHED_C100_R18_KEEP_FLAGS,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out_dir / "params.json", params)

    train_loader = build_loader(trainset, args.batch_size, True, args, 0, device)
    val_loader = build_loader(valset, args.eval_batch_size, False, args, 1, device)
    eval_loader = build_loader(evalset, args.eval_batch_size, False, args, 2, device)
    original_victim = load_original_victim(victim_path, device)

    teacher = cifar_resnet18(100, weight_path).to(device)
    teacher_optimizer = torch.optim.SGD(
        teacher.parameters(),
        lr=args.teacher_lr,
        momentum=args.teacher_momentum,
        weight_decay=args.weight_decay,
    )
    teacher_scheduler = torch.optim.lr_scheduler.StepLR(
        teacher_optimizer,
        step_size=args.teacher_step,
        gamma=args.lr_gamma,
    )
    best_teacher_acc = -1.0
    for epoch in range(1, args.teacher_epochs + 1):
        lr = teacher_optimizer.param_groups[0]["lr"]
        train_metrics = train_teacher_epoch(
            teacher,
            original_victim,
            train_loader,
            teacher_optimizer,
            device,
            args.temperature,
            epoch,
            args.teacher_epochs,
        )
        val_metrics = evaluate_against(teacher, original_victim, val_loader, device)
        append_log(log_path, stage_row("teacher", epoch, lr, train_metrics, val_metrics))
        if float(val_metrics["accuracy"]) >= best_teacher_acc:
            best_teacher_acc = float(val_metrics["accuracy"])
            save_checkpoint(
                teacher_dir / "best.pth",
                teacher,
                "teacher",
                epoch,
                run_id,
                val_metrics,
                teacher_optimizer,
                teacher_scheduler,
            )
        teacher_scheduler.step()
    save_checkpoint(
        teacher_dir / "end.pth",
        teacher,
        "teacher",
        args.teacher_epochs,
        run_id,
        evaluate_against(teacher, original_victim, val_loader, device),
        teacher_optimizer,
        teacher_scheduler,
    )
    load_stage_model(teacher_dir / "best.pth", teacher)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    full_model = teeslice_r18(100, weight_path, max_skip=args.max_skip).to(device)
    public_hash_before = tensor_dict_sha256(full_model.public_state_dict())
    full_optimizer = torch.optim.SGD(
        [
            {
                "params": [parameter for block in full_model.blocks for parameter in block.proxies.parameters()],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [block.alpha for block in full_model.blocks],
                "weight_decay": 0.0,
            },
            {
                "params": list(full_model.last_linear.parameters()),
                "weight_decay": args.weight_decay,
            },
        ],
        lr=args.full_lr,
        momentum=args.momentum,
    )
    full_scheduler = torch.optim.lr_scheduler.StepLR(
        full_optimizer,
        step_size=args.full_step,
        gamma=args.lr_gamma,
    )
    best_full_acc = -1.0
    for epoch in range(1, args.full_epochs + 1):
        lr = full_optimizer.param_groups[0]["lr"]
        train_metrics = train_slice_epoch(
            full_model,
            teacher,
            train_loader,
            full_optimizer,
            device,
            args.temperature,
            args.teacher_coeff,
            args.complexity_coeff,
            epoch,
            args.full_epochs,
            "full",
        )
        val_metrics = evaluate_against(full_model, teacher, val_loader, device)
        append_log(
            log_path,
            stage_row("full", epoch, lr, train_metrics, val_metrics, full_model.active_proxy_count()),
        )
        if float(val_metrics["accuracy"]) >= best_full_acc:
            best_full_acc = float(val_metrics["accuracy"])
            save_checkpoint(
                full_dir / "best.pth",
                full_model,
                "full",
                epoch,
                run_id,
                val_metrics,
                full_optimizer,
                full_scheduler,
            )
        full_scheduler.step()
    save_checkpoint(
        full_dir / "end.pth",
        full_model,
        "full",
        args.full_epochs,
        run_id,
        evaluate_against(full_model, teacher, val_loader, device),
        full_optimizer,
        full_scheduler,
    )

    prune_model = teeslice_r18(100, weight_path, max_skip=args.max_skip).to(device)
    load_stage_model(full_dir / "best.pth", prune_model)
    full_baseline = evaluate_against(prune_model, teacher, val_loader, device)
    tolerance_accuracy = float(full_baseline["accuracy"]) * (1.0 - args.prune_tolerance)
    initial_removed = prune_model.initial_prune(args.prune_threshold, args.initial_prune_fraction)
    print(
        f"[PRUNE] full_val={full_baseline['accuracy']:.6f} tolerance={tolerance_accuracy:.6f} "
        f"initial_removed={len(initial_removed)} active={prune_model.active_proxy_count()}"
    )
    prune_optimizer = torch.optim.SGD(
        [
            {
                "params": [parameter for block in prune_model.blocks for parameter in block.proxies.parameters()],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [block.alpha for block in prune_model.blocks],
                "weight_decay": 0.0,
            },
            {
                "params": list(prune_model.last_linear.parameters()),
                "weight_decay": args.weight_decay,
            },
        ],
        lr=args.prune_lr,
        momentum=args.momentum,
    )
    prune_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(prune_optimizer, T_max=args.prune_epochs)
    valid_candidates: list[dict[str, object]] = []
    for epoch in range(1, args.prune_epochs + 1):
        lr = prune_optimizer.param_groups[0]["lr"]
        train_metrics = train_slice_epoch(
            prune_model,
            teacher,
            train_loader,
            prune_optimizer,
            device,
            args.temperature,
            args.teacher_coeff,
            0.0,
            epoch,
            args.prune_epochs,
            "prune",
        )
        val_metrics = evaluate_against(prune_model, teacher, val_loader, device)
        append_log(
            log_path,
            stage_row("prune", epoch, lr, train_metrics, val_metrics, prune_model.active_proxy_count()),
        )
        if epoch % args.prune_interval == 0 and float(val_metrics["accuracy"]) >= tolerance_accuracy:
            valid_candidates.append(
                {
                    "epoch": epoch,
                    "state_dict": cpu_state_dict(prune_model),
                    "keep_flags": prune_model.get_keep_flags(),
                    "metrics": dict(val_metrics),
                    "cost": prune_model.cost_summary(),
                }
            )
            removed = prune_model.iterative_prune(args.iterative_prune_fraction)
            print(
                f"[PRUNE] epoch={epoch} valid active={valid_candidates[-1]['cost']['active_proxy_count']} "
                f"next_removed={removed}"
            )
        prune_scheduler.step()

    end_metrics = evaluate_against(prune_model, teacher, val_loader, device)
    save_checkpoint(
        out_dir / "end.pth",
        prune_model,
        "prune",
        args.prune_epochs,
        run_id,
        end_metrics,
        prune_optimizer,
        prune_scheduler,
    )
    if valid_candidates:
        best_candidate = min(
            valid_candidates,
            key=lambda candidate: (
                int(candidate["cost"]["active_proxy_count"]),
                -float(candidate["metrics"]["accuracy"]),
            ),
        )
        prune_model.load_state_dict(best_candidate["state_dict"], strict=True)
        prune_model.set_keep_flags(best_candidate["keep_flags"])
        best_epoch = int(best_candidate["epoch"])
        best_internal_metrics = best_candidate["metrics"]
    else:
        print("[WARN] prune 阶段没有满足容忍阈值的候选，回退到 full/best.pth。")
        load_stage_model(full_dir / "best.pth", prune_model)
        best_epoch = 0
        best_internal_metrics = full_baseline
    save_checkpoint(
        out_dir / "best.pth",
        prune_model,
        "prune",
        best_epoch,
        run_id,
        best_internal_metrics,
        None,
        None,
    )

    public_hash_after = tensor_dict_sha256(prune_model.public_state_dict())
    if public_hash_after != public_hash_before:
        raise RuntimeError("TEESlice 公开 backbone 在训练期间发生变化。")
    final_cost = prune_model.cost_summary()
    eval_metrics = evaluate_against(prune_model, original_victim, eval_loader, device)
    published_flags = PUBLISHED_C100_R18_KEEP_FLAGS
    selected = {
        (block_index, proxy_index)
        for block_index, flags in enumerate(prune_model.get_keep_flags())
        for proxy_index, keep in enumerate(flags[1:])
        if keep
    }
    published_selected = {
        (block_index, proxy_index)
        for block_index, flags in enumerate(published_flags)
        for proxy_index, keep in enumerate(flags[1:])
        if keep
    }
    topology_overlap = len(selected & published_selected) / max(len(selected | published_selected), 1)
    victim_payload = {
        "schema_version": 1,
        "protocol": "MS",
        "defense": "teeslice",
        "model": "resnet18",
        "defended_model": MODEL_ID,
        "dataset": "c100",
        "run_id": run_id,
        "primary_checkpoint": "best.pth",
        "selection": {
            "split": "victim_train_internal_validation",
            "best_prune_epoch": best_epoch,
            "tolerance_accuracy": tolerance_accuracy,
            "internal_metrics": best_internal_metrics,
        },
        "cost": final_cost,
        "published_reference": {
            "task_param_count": PUBLISHED_C100_R18_TASK_PARAMS,
            "task_flops": PUBLISHED_C100_R18_TASK_FLOPS,
            "topology_jaccard": topology_overlap,
            "exact_topology_match": prune_model.get_keep_flags() == published_flags,
        },
        "eval_ms": eval_metrics,
        "public_state_sha256": public_hash_after,
        "victim_checkpoint_sha256": sha256_file(victim_path),
        "official_weight_sha256": sha256_file(weight_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    results_dir = Path(args.results_root).expanduser().resolve() / "resnet18" / "c100" / "teeslice"
    results_dir.mkdir(parents=True, exist_ok=True)
    write_json(results_dir / "victim.json", victim_payload)
    print(f"[INFO] TEESlice defended victim: {out_dir / 'best.pth'}")
    print(f"[INFO] eval_ms accuracy={eval_metrics['accuracy']:.6f} fidelity_to_original={eval_metrics['fidelity']:.6f}")
    print(f"[INFO] cost={final_cost}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
