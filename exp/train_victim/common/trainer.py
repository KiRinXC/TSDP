#!/usr/bin/env python3
"""victim 训练公共逻辑。"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets as tv_datasets
from torchvision import transforms
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import imagenet as imagenet_models  # noqa: E402
from models.imagenet import load_official_imagenet_weights  # noqa: E402

try:
    import numpy as np
except ImportError:
    np = None


DATASET_ALIASES = {
    "cifar10": "cifar10",
    "cifar_10": "cifar10",
    "cifar-10": "cifar10",
    "cifar100": "cifar100",
    "cifar_100": "cifar100",
    "cifar-100": "cifar100",
    "stl10": "stl10",
    "stl-10": "stl10",
    "tiny-imagenet-200": "tiny-imagenet-200",
    "tinyimagenet200": "tiny-imagenet-200",
    "tiny_imagenet200": "tiny-imagenet-200",
    "tiny_imagenet_200": "tiny-imagenet-200",
    "tinyimagenet": "tiny-imagenet-200",
}


@dataclass(frozen=True)
class ModelSpec:
    """描述一个 victim 模型入口。"""

    name: str
    display_name: str
    factory_name: str
    weight_filename: str


def resolve_dataset_name(name: str) -> str:
    """把外部输入统一成内部数据集名称。"""
    normalized = name.strip().lower()
    if normalized not in DATASET_ALIASES:
        valid = ", ".join(sorted(DATASET_ALIASES))
        raise ValueError(f"不支持的数据集：{name}。可选值：{valid}")
    return DATASET_ALIASES[normalized]


def default_weight_path(spec: ModelSpec) -> Path:
    """返回该模型对应的官方 ImageNet 权重路径。"""
    return REPO_ROOT / "weights" / "pre_train" / spec.weight_filename


def configure_reproducibility(seed: int | None, deterministic: bool) -> None:
    """配置随机种子和确定性选项。"""
    if seed is None or seed < 0:
        if deterministic:
            set_deterministic_mode()
        return

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        set_deterministic_mode()


def set_deterministic_mode() -> None:
    """打开 PyTorch 尽量确定性的执行路径。"""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """固定 DataLoader worker 中的 Python 和 NumPy 随机状态。"""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if np is not None:
        np.random.seed(worker_seed)


def build_generator(seed: int | None, offset: int = 0) -> torch.Generator | None:
    """给 DataLoader 构造可复现的随机数生成器。"""
    if seed is None or seed < 0:
        return None

    generator = torch.Generator()
    generator.manual_seed(seed + offset)
    return generator


def build_transforms(dataset_name: str):
    """按当前实验协议构造训练和测试增强。"""
    if dataset_name in {"cifar10", "cifar100"}:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        return train_transform, test_transform

    if dataset_name == "stl10":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        train_transform = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.RandomCrop(128, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        return train_transform, test_transform

    if dataset_name == "tiny-imagenet-200":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        return train_transform, test_transform

    raise ValueError(f"未知的数据集：{dataset_name}")


def build_datasets(dataset_name: str, dataset_root: Path):
    """构造训练集和测试集，并返回类别数。"""
    train_transform, test_transform = build_transforms(dataset_name)

    if dataset_name == "cifar10":
        root = dataset_root / "cifar10"
        trainset = tv_datasets.CIFAR10(root=str(root), train=True, download=False, transform=train_transform)
        testset = tv_datasets.CIFAR10(root=str(root), train=False, download=False, transform=test_transform)
        return trainset, testset, 10

    if dataset_name == "cifar100":
        root = dataset_root / "cifar100"
        trainset = tv_datasets.CIFAR100(root=str(root), train=True, download=False, transform=train_transform)
        testset = tv_datasets.CIFAR100(root=str(root), train=False, download=False, transform=test_transform)
        return trainset, testset, 100

    if dataset_name == "stl10":
        root = dataset_root / "stl10"
        trainset = tv_datasets.STL10(root=str(root), split="train", download=False, transform=train_transform)
        testset = tv_datasets.STL10(root=str(root), split="test", download=False, transform=test_transform)
        return trainset, testset, 10

    if dataset_name == "tiny-imagenet-200":
        root = dataset_root / "tiny-imagenet-200"
        train_root = root / "train"
        eval_root = root / "val2"
        if not eval_root.exists():
            eval_root = root / "val"
        trainset = tv_datasets.ImageFolder(root=str(train_root), transform=train_transform)
        testset = tv_datasets.ImageFolder(root=str(eval_root), transform=test_transform)
        return trainset, testset, len(trainset.classes)

    raise ValueError(f"未知的数据集：{dataset_name}")


def subset_dataset(dataset, limit: int | None, seed: int | None):
    """按样本数截取子集，用于快速训练。"""
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset

    indices = list(range(len(dataset)))
    rng = random.Random(seed if seed is not None and seed >= 0 else None)
    rng.shuffle(indices)
    return Subset(dataset, indices[:limit])


def build_model(spec: ModelSpec, num_classes: int, weight_path: Path | None, use_pretrained: bool) -> nn.Module:
    """先加载官方 ImageNet 权重，再替换最后的分类头。"""
    factory: Callable[..., nn.Module] = getattr(imagenet_models, spec.factory_name)

    if not use_pretrained:
        return factory(num_classes=num_classes)

    if weight_path is None:
        raise ValueError("使用预训练权重时必须提供 weight_path。")
    if not weight_path.is_file():
        raise FileNotFoundError(f"找不到官方预训练权重：{weight_path}")

    model = factory(num_classes=1000)
    load_official_imagenet_weights(spec.factory_name, model, str(weight_path), strict=True)
    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    log_interval: int,
) -> tuple[float, float]:
    """完成一个训练 epoch。"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"[TRAIN] epoch={epoch:03d}/{total_epochs:03d}",
        dynamic_ncols=True,
        leave=True,
    )
    for batch_idx, (inputs, targets) in enumerate(progress, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()

        if batch_idx % log_interval == 0 or batch_idx == len(loader):
            acc = 100.0 * total_correct / max(total_samples, 1)
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2f}%")

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = 100.0 * total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """在测试集上评估模型。"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = 100.0 * total_correct / max(total_samples, 1)
    print(f"[TEST] epoch={epoch:03d}/{total_epochs:03d} loss={avg_loss:.4f} acc={avg_acc:.2f}%")
    return avg_loss, avg_acc


def save_checkpoint(
    out_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_acc: float,
    spec: ModelSpec,
) -> None:
    """保存训练 checkpoint。"""
    checkpoint = {
        "epoch": epoch,
        "arch": spec.name,
        "state_dict": model.state_dict(),
        "best_acc": best_acc,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "created_on": str(datetime.now()),
    }
    torch.save(checkpoint, out_dir / "checkpoint.pth.tar")


def parse_args(spec: ModelSpec) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=f"训练受害者模型 {spec.display_name}")
    parser.add_argument("dataset", nargs="?", default=None, help="数据集名称")
    parser.add_argument("--dataset", dest="dataset_flag", default=None, help="数据集名称")
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "dataset" / "public"), help="公开数据集根目录")
    parser.add_argument("--weight-path", default=None, help="官方 ImageNet 预训练权重")
    parser.add_argument("--out-dir", default=None, help="输出目录")
    parser.add_argument("--device", default="auto", help="运行设备：auto / cpu / cuda / cuda:0")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.1, help="学习率")
    parser.add_argument("--momentum", type=float, default=0.5, help="SGD 动量")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="权重衰减")
    parser.add_argument("--lr-step", type=int, default=60, help="学习率衰减步长")
    parser.add_argument("--lr-gamma", type=float, default=0.1, help="学习率衰减系数")
    parser.add_argument("--num-workers", type=int, default=10, help="数据加载线程数")
    parser.add_argument("--log-interval", type=int, default=100, help="训练日志间隔")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，-1 表示不固定")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否打开确定性训练设置，默认打开",
    )
    parser.add_argument("--resume", default=None, help="恢复训练的 checkpoint")
    parser.add_argument("--no-pretrained", action="store_true", help="不加载 ImageNet 预训练权重")
    parser.add_argument("--quick", action="store_true", help="快速训练模式")
    parser.add_argument("--train-subset", type=int, default=None, help="训练子集大小")
    parser.add_argument("--test-subset", type=int, default=None, help="测试子集大小")
    parser.add_argument("--dry-run", action="store_true", help="只检查数据和模型，不进入训练")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    """把用户输入的设备名转换成 torch.device。"""
    normalized = name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用的 CUDA 设备。")
    return torch.device(normalized)


def apply_quick_mode(args: argparse.Namespace) -> None:
    """把命令行切换成短训练配置。"""
    if not args.quick:
        return

    args.epochs = min(args.epochs, 1)
    if args.train_subset is None:
        args.train_subset = 512
    if args.test_subset is None:
        args.test_subset = 512
    args.num_workers = 0


def train_main(spec: ModelSpec) -> None:
    """执行指定 victim 模型的训练流程。"""
    args = parse_args(spec)
    apply_quick_mode(args)

    dataset_name = resolve_dataset_name(args.dataset_flag or args.dataset or "cifar10")
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else REPO_ROOT / "weights" / "victim" / spec.name / dataset_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_reproducibility(args.seed, args.deterministic)
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    weight_path = Path(args.weight_path).expanduser().resolve() if args.weight_path else default_weight_path(spec)

    trainset, testset, num_classes = build_datasets(dataset_name, dataset_root)
    trainset = subset_dataset(trainset, args.train_subset, args.seed)
    testset = subset_dataset(testset, args.test_subset, args.seed)
    model = build_model(spec, num_classes, weight_path, use_pretrained=not args.no_pretrained).to(device)

    print(f"[INFO] 模型: {spec.display_name}")
    print(f"[INFO] 数据集: {dataset_name}")
    print(f"[INFO] 训练样本数: {len(trainset)}")
    print(f"[INFO] 测试样本数: {len(testset)}")
    print(f"[INFO] 类别数: {num_classes}")
    print(f"[INFO] 输出目录: {out_dir}")
    print(f"[INFO] 设备: {device}")
    print(f"[INFO] deterministic: {args.deterministic}")

    if args.dry_run:
        print("[INFO] dry-run 模式结束，没有开始训练。")
        return

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if args.seed is not None and args.seed >= 0 else None,
        generator=build_generator(args.seed),
    )
    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if args.seed is not None and args.seed >= 0 else None,
        generator=build_generator(args.seed, offset=1),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step,
        gamma=args.lr_gamma,
    )

    start_epoch = 1
    best_acc = -1.0
    if args.resume:
        checkpoint_path = Path(args.resume).expanduser().resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_acc = float(checkpoint.get("best_acc", -1.0))
        print(f"[INFO] 已恢复 checkpoint: {checkpoint_path}")
        print(f"[INFO] 从第 {start_epoch} 轮继续训练")

    log_path = out_dir / "train.log.tsv"
    if start_epoch == 1 or not log_path.exists():
        with log_path.open("w", encoding="utf-8") as writer:
            writer.write("run_id\tepoch\tsplit\tloss\taccuracy\tbest_accuracy\n")
    run_id = str(datetime.now())

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            log_interval=args.log_interval,
        )
        test_loss, test_acc = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        scheduler.step()

        is_best = test_acc >= best_acc
        if is_best:
            save_checkpoint(
                out_dir=out_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_acc=test_acc,
                spec=spec,
            )
        best_acc = max(best_acc, test_acc)

        with log_path.open("a", encoding="utf-8") as writer:
            writer.write(f"{run_id}\t{epoch}\ttrain\t{train_loss:.6f}\t{train_acc:.4f}\t{best_acc:.4f}\n")
            writer.write(f"{run_id}\t{epoch}\ttest\t{test_loss:.6f}\t{test_acc:.4f}\t{best_acc:.4f}\n")

    torch.save(model.state_dict(), out_dir / "target.pth")

    params = vars(args).copy()
    params["dataset"] = dataset_name
    params["dataset_root"] = str(dataset_root)
    params["out_dir"] = str(out_dir)
    params["device"] = str(device)
    params["num_classes"] = num_classes
    params["model_name"] = spec.name
    params["display_name"] = spec.display_name
    params["weight_path"] = str(weight_path)
    params["deterministic"] = args.deterministic
    params["created_on"] = str(datetime.now())
    with (out_dir / "params.json").open("w", encoding="utf-8") as writer:
        json.dump(params, writer, ensure_ascii=False, indent=2)

    print("[INFO] 训练完成。")
