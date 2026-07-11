#!/usr/bin/env python3
"""surrogate 训练与原始 MS 指标评估。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from defense import ExposureFreezer


@dataclass(frozen=True)
class EvalReference:
    targets: torch.Tensor
    victim_predictions: torch.Tensor
    victim_posteriors: torch.Tensor
    victim_correct: int


def distillation_loss(
    logits: torch.Tensor,
    posteriors: torch.Tensor | None,
    labels: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "hard":
        return functional.cross_entropy(logits, labels)
    if posteriors is None:
        raise ValueError("soft 标签模式缺少 posterior。")
    return -(posteriors * functional.log_softmax(logits, dim=1)).sum(dim=1).mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_mode: str,
    epoch: int,
    total_epochs: int,
    freezer: ExposureFreezer | None,
) -> dict[str, int | float]:
    model.train()
    if freezer is not None:
        freezer.apply_train_mode()
    loss_sum = 0.0
    match_count = 0
    sample_count = 0
    progress = tqdm(loader, desc=f"[TRAIN] {epoch:03d}/{total_epochs:03d}", dynamic_ncols=True)
    for batch in progress:
        if label_mode == "hard":
            images, labels = batch
            posteriors = None
        else:
            images, posteriors, labels = batch
            posteriors = posteriors.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = distillation_loss(logits, posteriors, labels, label_mode)
        loss.backward()
        optimizer.step()
        if freezer is not None:
            freezer.restore()

        batch_size = labels.size(0)
        sample_count += batch_size
        loss_sum += float(loss.item()) * batch_size
        match_count += int((logits.argmax(dim=1) == labels).sum().item())
        progress.set_postfix(loss=f"{loss.item():.4f}", match=f"{match_count / sample_count:.4f}")
    return {
        "query_count": sample_count,
        "query_loss_sum": loss_sum,
        "query_loss": loss_sum / max(sample_count, 1),
        "query_match_count": match_count,
        "query_match": match_count / max(sample_count, 1),
    }


@torch.no_grad()
def collect_eval_reference(model: nn.Module, loader: DataLoader, device: torch.device) -> EvalReference:
    model.eval()
    targets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    posteriors: list[torch.Tensor] = []
    correct = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        probability = functional.softmax(logits, dim=1)
        prediction = logits.argmax(dim=1)
        correct += int((prediction == labels).sum().item())
        targets.append(labels.cpu())
        predictions.append(prediction.cpu())
        posteriors.append(probability.cpu())
    return EvalReference(
        targets=torch.cat(targets),
        victim_predictions=torch.cat(predictions),
        victim_posteriors=torch.cat(posteriors),
        victim_correct=correct,
    )


@torch.no_grad()
def evaluate_surrogate(
    model: nn.Module,
    loader: DataLoader,
    reference: EvalReference,
    device: torch.device,
) -> dict[str, int | float]:
    model.eval()
    surrogate_correct = 0
    agreement_count = 0
    kl_sum = 0.0
    offset = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        prediction = logits.argmax(dim=1)
        batch_size = labels.size(0)
        victim_predictions = reference.victim_predictions[offset : offset + batch_size].to(device)
        victim_posteriors = reference.victim_posteriors[offset : offset + batch_size].to(device)
        surrogate_correct += int((prediction == labels).sum().item())
        agreement_count += int((prediction == victim_predictions).sum().item())
        batch_kl = float(
            functional.kl_div(
                functional.log_softmax(logits, dim=1),
                victim_posteriors,
                reduction="sum",
            ).item()
        )
        kl_sum += max(batch_kl, 0.0)
        offset += batch_size

    eval_count = reference.targets.numel()
    if offset != eval_count:
        raise ValueError(f"surrogate 评估数量 {offset} 与 victim reference {eval_count} 不一致。")
    return {
        "eval_count": eval_count,
        "victim_correct": reference.victim_correct,
        "surrogate_correct": surrogate_correct,
        "agreement_count": agreement_count,
        "victim_acc": reference.victim_correct / eval_count,
        "surrogate_acc": surrogate_correct / eval_count,
        "fidelity": agreement_count / eval_count,
        "posterior_kl_sum": kl_sum,
        "posterior_kl": kl_sum / eval_count,
    }
