#!/usr/bin/env python3
"""Compare classifier-head and freezing choices for two MS protection settings."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TRAIN_SURROGATE_ROOT = ROOT / "exp" / "MS" / "train_surrogate"
if str(TRAIN_SURROGATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_SURROGATE_ROOT))

from exp.MS.train_surrogate.core.data import (  # noqa: E402
    build_eval_dataset,
    build_query_dataset,
    build_victim,
    load_query_targets,
)
from exp.MS.train_surrogate.defense import (  # noqa: E402
    ExposureFreezer,
    build_resnet18_tensor_units,
    build_unit_masks,
    protection_mask_sha256,
)
from models import imagenet as imagenet_models  # noqa: E402
from models.imagenet import load_official_imagenet_weights  # noqa: E402


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
    parser.add_argument(
        "--dataset-root", type=Path, default=ROOT / "dataset" / "public"
    )
    parser.add_argument("--protocol-root", type=Path, default=ROOT / "dataset" / "MS")
    parser.add_argument(
        "--victim-weight",
        type=Path,
        default=ROOT / "weights" / "MS" / "victim" / "resnet18" / "c100" / "best.pth",
    )
    parser.add_argument(
        "--official-weight",
        type=Path,
        default=ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "results" / "lab" / "02_head"
    )
    parser.add_argument(
        "--scope",
        choices=("full", "random", "all"),
        default="all",
        help="运行全保护、随机保护或二者全部配置",
    )
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_random_plan(model: nn.Module, seed: int) -> dict[str, Any]:
    units = build_resnet18_tensor_units(model)
    if len(units) != 122:
        raise RuntimeError(f"ResNet18 unit 数量应为 122，实际为 {len(units)}")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(BACKBONE_UNIT_COUNT, generator=generator).tolist()
    random_backbone = tuple(sorted(permutation[:RANDOM_PROTECTED_BACKBONE_UNITS]))
    protected_units = tuple(sorted((*random_backbone, *CLASSIFIER_UNITS)))
    masks = build_unit_masks(model, protected_units)
    exposed_units = tuple(index for index in range(len(units)) if index not in protected_units)
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


def make_task_model(head_mode: str, official_weight: Path) -> tuple[nn.Module, nn.Module]:
    model = imagenet_models.resnet18(num_classes=1000)
    load_official_imagenet_weights("resnet18", model, official_weight, strict=True)
    if head_mode == "replace":
        model.last_linear = nn.Linear(model.last_linear.in_features, 100)
        task_head = model.last_linear
    elif head_mode == "adapter":
        public_head = model.last_linear
        task_head = nn.Linear(public_head.out_features, 100)
        model.last_linear = nn.Sequential(public_head, task_head)
    else:
        raise ValueError(f"Unsupported head mode: {head_mode}")
    return model, task_head


@dataclass
class SurrogateSetup:
    model: nn.Module
    task_head: nn.Module
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
        raise RuntimeError(f"受害者模型状态应包含 122 个 unit，实际为 {len(victim_names)}")

    model_state = model.state_dict()
    trainable_masks = {name: torch.ones_like(tensor, dtype=torch.bool) for name, tensor in model_state.items()}
    copied = 0
    for unit_index, name in enumerate(victim_names[:BACKBONE_UNIT_COUNT]):
        if name not in model_state:
            raise KeyError(f"代理模型缺少骨干状态: {name}")
        if model_state[name].shape != victim_state[name].shape:
            raise ValueError(f"骨干状态形状不一致: {name}")
        if unit_index in protected:
            continue
        model_state[name] = victim_state[name].detach().clone()
        trainable_masks[name] = torch.zeros_like(model_state[name], dtype=torch.bool)
        if name in dict(model.named_parameters()):
            copied += model_state[name].numel()
    model.load_state_dict(model_state, strict=True)
    return trainable_masks, copied


def build_surrogate(
    *,
    head_mode: str,
    training_mode: str,
    protection: str,
    victim_state: dict[str, torch.Tensor],
    protected_units: tuple[int, ...],
    official_weight: Path,
    device: torch.device,
) -> SurrogateSetup:
    model, task_head = make_task_model(head_mode, official_weight)
    freezer: ExposureFreezer | None = None
    trainable_masks: dict[str, torch.Tensor] | None = None
    frozen_scope: str | None = None
    copied_parameter_elements = 0

    if protection == PROTECTION_RANDOM:
        trainable_masks, copied_parameter_elements = _copy_exposed_backbone(
            model, victim_state, protected_units
        )
        if training_mode == "frozen":
            frozen_scope = "stolen_victim_weights"
    elif protection == PROTECTION_FULL:
        if training_mode == "frozen":
            for parameter in model.parameters():
                parameter.requires_grad = False
            for parameter in task_head.parameters():
                parameter.requires_grad = True
            frozen_scope = "public_pretrained_weights"
    else:
        raise ValueError(f"Unsupported protection: {protection}")

    if training_mode == "finetune":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif training_mode != "frozen":
        raise ValueError(f"Unsupported training mode: {training_mode}")

    model.to(device)
    if frozen_scope == "stolen_victim_weights":
        assert trainable_masks is not None
        freezer = ExposureFreezer(model, trainable_masks)

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return SurrogateSetup(
        model=model,
        task_head=task_head,
        freezer=freezer,
        frozen_scope=frozen_scope,
        trainable_parameters=trainable,
        total_parameters=total,
        copied_parameter_elements=copied_parameter_elements,
    )


def train_one_epoch(
    setup: SurrogateSetup,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model = setup.model
    if setup.frozen_scope == "public_pretrained_weights":
        model.eval()
        setup.task_head.train()
    else:
        model.train()
        if setup.freezer is not None:
            setup.freezer.apply_train_mode()

    loss_sum = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if setup.freezer is not None:
            setup.freezer.restore()
        loss_sum += float(loss.item()) * labels.size(0)
        correct += int(logits.argmax(dim=1).eq(labels).sum().item())
        total += labels.size(0)
    return loss_sum / total, correct / total


@torch.inference_mode()
def collect_reference(
    victim: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    victim.eval()
    logits_parts: list[torch.Tensor] = []
    labels_parts: list[torch.Tensor] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits_parts.append(victim(images).cpu())
        labels_parts.append(labels.cpu())
    return torch.cat(logits_parts), torch.cat(labels_parts)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    victim_logits: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    for images, _ in loader:
        logits_parts.append(model(images.to(device, non_blocking=True)).cpu())
    surrogate_logits = torch.cat(logits_parts)
    surrogate_predictions = surrogate_logits.argmax(dim=1)
    victim_predictions = victim_logits.argmax(dim=1)
    victim_probabilities = torch.softmax(victim_logits, dim=1)
    surrogate_log_probabilities = torch.log_softmax(surrogate_logits, dim=1)
    kl = torch.nn.functional.kl_div(
        surrogate_log_probabilities, victim_probabilities, reduction="batchmean"
    )
    return {
        "accuracy": float(surrogate_predictions.eq(labels).float().mean().item()),
        "fidelity": float(surrogate_predictions.eq(victim_predictions).float().mean().item()),
        "kl_divergence_victim_to_surrogate": float(kl.item()),
    }


def run_configuration(
    *,
    args: argparse.Namespace,
    protection: str,
    name: str,
    head_mode: str,
    training_mode: str,
    victim_state: dict[str, torch.Tensor],
    protected_units: tuple[int, ...],
    train_loader: DataLoader,
    eval_loader: DataLoader,
    victim_logits: torch.Tensor,
    eval_labels: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_seed(args.seed)
    if train_loader.generator is not None:
        train_loader.generator.manual_seed(args.seed)
    setup = build_surrogate(
        head_mode=head_mode,
        training_mode=training_mode,
        protection=protection,
        victim_state=victim_state,
        protected_units=protected_units,
        official_weight=args.official_weight,
        device=device,
    )
    optimizer = torch.optim.SGD(
        (parameter for parameter in setup.model.parameters() if parameter.requires_grad),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            setup, train_loader, optimizer, criterion, device
        )
        metrics = evaluate(setup.model, eval_loader, victim_logits, eval_labels, device)
        row = {
            "protection": protection,
            "configuration": name,
            "head_mode": head_mode,
            "training_mode": training_mode,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            **metrics,
        }
        history.append(row)
        if best is None or metrics["accuracy"] > best["accuracy"]:
            best = {"epoch": epoch, **metrics}
        scheduler.step()
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"[{protection}/{name}] epoch={epoch:03d} "
                f"train_acc={train_accuracy:.4f} acc={metrics['accuracy']:.4f} "
                f"fidelity={metrics['fidelity']:.4f} kl={metrics['kl_divergence_victim_to_surrogate']:.4f}"
            )

    assert best is not None
    result = {
        "protection": protection,
        "configuration": name,
        "head_mode": head_mode,
        "training_mode": training_mode,
        "frozen_weight_source": setup.frozen_scope,
        "trainable_parameters": setup.trainable_parameters,
        "total_parameters": setup.total_parameters,
        "copied_parameter_elements": setup.copied_parameter_elements,
        "best": best,
        "end": history[-1],
    }
    del setup.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, history


def load_existing_full_results(out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics_path = out_dir / "metrics.json"
    history_path = out_dir / "history.tsv"
    if not metrics_path.exists() or not history_path.exists():
        raise FileNotFoundError("--scope random 需要已有的全保护 metrics.json 和 history.tsv")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for original in payload.get("results", []):
        row = dict(original)
        row.setdefault("protection", PROTECTION_FULL)
        if row["protection"] != PROTECTION_FULL:
            continue
        if "pretrained_mode" in row and "training_mode" not in row:
            row["training_mode"] = row.pop("pretrained_mode")
        if "trainable_param_count" in row:
            row["trainable_parameters"] = row.pop("trainable_param_count")
        if "total_param_count" in row:
            row["total_parameters"] = row.pop("total_param_count")
        best_epoch = int(row.pop("best_epoch", row["best"].get("epoch", 0)))
        for metric_name, epoch in (("best", best_epoch), ("end", int(payload.get("epochs", 0)))):
            metrics = row[metric_name]
            if "accuracy" not in metrics:
                row[metric_name] = {
                    "epoch": epoch,
                    "accuracy": metrics["surrogate_acc"],
                    "fidelity": metrics["fidelity"],
                    "kl_divergence_victim_to_surrogate": metrics["posterior_kl"],
                    "victim_accuracy": metrics["victim_acc"],
                    "eval_count": metrics["eval_count"],
                }
        row.setdefault(
            "frozen_weight_source",
            "public_pretrained_weights" if row["training_mode"] == "frozen" else None,
        )
        row.setdefault("copied_parameter_elements", 0)
        results.append(row)

    history = read_existing_history(history_path, PROTECTION_FULL)
    if len(results) != len(CONFIGURATIONS):
        raise RuntimeError("已有 metrics.json 不包含完整的四组全保护结果")
    return results, history


def normalize_history_row(row: dict[str, Any], protection: str) -> dict[str, Any]:
    if "pretrained_mode" in row:
        row["training_mode"] = row.pop("pretrained_mode")
    return {
        "protection": protection,
        "configuration": row["configuration"],
        "head_mode": row["head_mode"],
        "training_mode": row["training_mode"],
        "epoch": row["epoch"],
        "lr": row.get("lr", row.get("learning_rate")),
        "train_loss": row.get("train_loss", row.get("query_loss")),
        "train_accuracy": row.get("train_accuracy", row.get("query_match")),
        "accuracy": row.get("accuracy", row.get("surrogate_acc")),
        "fidelity": row["fidelity"],
        "kl_divergence_victim_to_surrogate": row.get(
            "kl_divergence_victim_to_surrogate", row.get("posterior_kl")
        ),
    }


def read_existing_history(path: Path, protection: str) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []

    rows: list[dict[str, Any]] = []
    if lines[0].startswith("configuration,"):
        comma_header = next(csv.reader([lines[0].split("\t", 1)[0]]))
        tab_header = lines[0].split("\t")
        for line in lines[1:]:
            comma_part = line.split("\t", 1)[0]
            if comma_part and protection == PROTECTION_FULL:
                values = next(csv.reader([comma_part]))
                if len(values) == len(comma_header):
                    rows.append(
                        normalize_history_row(dict(zip(comma_header, values)), PROTECTION_FULL)
                    )
                continue
            tab_values = line.split("\t")
            tab_row = dict(zip(tab_header, tab_values))
            if tab_row.get("protection") != protection:
                continue
            rows.append(normalize_history_row(tab_row, protection))
        return rows

    reader = csv.DictReader(lines, delimiter="\t")
    for original in reader:
        row_protection = original.get("protection", PROTECTION_FULL)
        if row_protection == protection:
            rows.append(normalize_history_row(dict(original), row_protection))
    return rows


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    query_indices, _, query_labels, query_target_path, _ = load_query_targets(
        args.protocol_root, "c100", "resnet18", args.budget, "hard"
    )
    query_dataset = build_query_dataset(
        "c100", args.dataset_root, query_indices, None, query_labels
    )
    eval_dataset = build_eval_dataset("c100", args.dataset_root, args.protocol_root, None)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    victim, _ = build_victim("resnet18", 100, args.victim_weight)
    victim_state = {name: tensor.detach().cpu().clone() for name, tensor in victim.state_dict().items()}
    random_plan = build_random_plan(victim, args.seed)
    victim.to(device)
    victim_logits, eval_labels = collect_reference(victim, eval_loader, device)
    del victim
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.scope == "random":
        results, all_history = load_existing_full_results(args.out_dir)
    else:
        results = []
        all_history = []

    scopes = []
    if args.scope in {"full", "all"}:
        scopes.append((PROTECTION_FULL, tuple(range(122))))
    if args.scope in {"random", "all"}:
        scopes.append((PROTECTION_RANDOM, tuple(random_plan["protected_units"])))

    for protection, protected_units in scopes:
        for name, head_mode, training_mode in CONFIGURATIONS:
            result, history = run_configuration(
                args=args,
                protection=protection,
                name=name,
                head_mode=head_mode,
                training_mode=training_mode,
                victim_state=victim_state,
                protected_units=protected_units,
                train_loader=train_loader,
                eval_loader=eval_loader,
                victim_logits=victim_logits,
                eval_labels=eval_labels,
                device=device,
            )
            results.append(result)
            all_history.extend(history)

    payload = {
        "schema_version": 2,
        "experiment": "head_and_stolen_weight_ablation",
        "dataset": "c100",
        "model": "resnet18",
        "query_budget": len(query_indices),
        "query_target": "hard",
        "epochs": args.epochs,
        "seed": args.seed,
        "victim_weight": str(args.victim_weight),
        "official_weight": str(args.official_weight),
        "query_target_path": str(query_target_path),
        "protection_plans": {
            PROTECTION_FULL: {
                "protected_unit_count": 122,
                "protected_fraction": 1.0,
                "copied_victim_weights": False,
            },
            PROTECTION_RANDOM: random_plan,
        },
        "results": results,
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_history(args.out_dir / "history.tsv", all_history)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
