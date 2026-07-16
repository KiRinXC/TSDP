#!/usr/bin/env python3
"""在 temp 中验证 ResNet18+C100 的 Attack-Recoverability Cut。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_VICTIM_ROOT = REPO_ROOT / "exp" / "MS" / "train_victim"
TRAIN_SURROGATE_ROOT = REPO_ROOT / "exp" / "MS" / "train_surrogate"
for search_root in (REPO_ROOT, TRAIN_VICTIM_ROOT, TRAIN_SURROGATE_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from common.trainer import (  # noqa: E402
    build_generator,
    build_public_split_dataset,
    build_transforms,
    configure_reproducibility,
    read_ms_split_indices,
    seed_worker,
)
from core.artifacts import sha256_file  # noqa: E402
from core.config import ATTACK_PROTOCOL_VERSION, resolve_device  # noqa: E402
from core.data import (  # noqa: E402
    build_eval_dataset,
    build_query_dataset,
    build_victim,
    load_query_targets,
)
from core.engine import (  # noqa: E402
    collect_eval_reference,
    evaluate_surrogate,
    train_one_epoch,
)
from defense import (  # noqa: E402
    build_public_model as build_seeded_public_model,
    load_protection_mask,
    protection_mask_sha256,
    save_protection_mask,
)
from models import imagenet as imagenet_models  # noqa: E402


MODEL = "resnet18"
DATASET = "c100"
NUM_CLASSES = 100
SEED = 42
QUERY_BUDGET = 500
DISCOVERY_QUERY_SIZE = 500
DISCOVERY_HOLDOUT_SIZE = 4096
TARGET_RATIO = 0.08
SELECTION_EPOCHS = 100
FINAL_EPOCHS = 100
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
NUM_WORKERS = 4
SHADOW_LR = 0.01
MOMENTUM = 0.5
WEIGHT_DECAY = 5e-4
LR_STEP = 60
LR_GAMMA = 0.1
GATE_LR = 0.02
GATE_TEMPERATURE_START = 2.0
GATE_TEMPERATURE_END = 0.1
BINARY_REGULARIZER = 0.01
HEAD_PREFIX = "last_linear."


@dataclass(frozen=True)
class FeatureGroup:
    """一个图对齐 feature 节点及其所有 producer。"""

    index: int
    name: str
    producers: tuple[tuple[str, str], ...]
    channels: int
    block_size: int
    block_count: int


@dataclass(frozen=True)
class ChannelBlock:
    """一个候选通道块。"""

    index: int
    group_index: int
    group_name: str
    block_index: int
    channel_start: int
    channel_end: int
    param_cost: int


class PosteriorDataset(Dataset):
    """把固定 source index 与 victim posterior 绑定。"""

    def __init__(self, public_dataset, source_indices: list[int], posteriors: torch.Tensor):
        if len(source_indices) != len(posteriors):
            raise ValueError("discovery 索引与 posterior 数量不一致。")
        self.public_dataset = public_dataset
        self.source_indices = source_indices
        self.posteriors = posteriors.float().cpu()
        self.labels = self.posteriors.argmax(dim=1)

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, index: int):
        image, _ = self.public_dataset[self.source_indices[index]]
        return image, self.posteriors[index], self.labels[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 ARC 通道块保护选择。")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--dry-run", action="store_true", help="只核对协议，不训练或写产物。")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 temp/output。")
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="只消费已固定的 selection.json 与 mask.pt，恢复最终 MS。",
    )
    return parser.parse_args()


def build_public_model() -> nn.Module:
    weight_path = REPO_ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    return build_seeded_public_model(
        imagenet_models.resnet18,
        MODEL,
        weight_path,
        NUM_CLASSES,
        initialization_seed=SEED,
    )


def closest_power_two_block_size(channels: int, target_blocks: int = 16) -> int:
    candidates = [2**power for power in range(int(math.log2(channels)) + 1) if channels % (2**power) == 0]
    eligible = [value for value in candidates if value >= 2]
    if not eligible:
        raise ValueError(f"通道数 {channels} 没有满足要求的块大小。")
    return min(eligible, key=lambda value: (abs(channels // value - target_blocks), value))


def build_feature_groups(model: nn.Module) -> list[FeatureGroup]:
    modules = dict(model.named_modules())
    specifications: list[tuple[str, tuple[tuple[str, str], ...]]] = [
        ("stem", (("conv1", "bn1"),)),
    ]
    for stage in range(1, 5):
        layer = getattr(model, f"layer{stage}")
        for block_index in range(len(layer)):
            prefix = f"layer{stage}.{block_index}"
            specifications.append((f"{prefix}.inner", ((f"{prefix}.conv1", f"{prefix}.bn1"),)))
            output_producers = [(f"{prefix}.conv2", f"{prefix}.bn2")]
            block = layer[block_index]
            if block.downsample is not None:
                output_producers.append((f"{prefix}.downsample.0", f"{prefix}.downsample.1"))
            specifications.append((f"{prefix}.output", tuple(output_producers)))

    groups: list[FeatureGroup] = []
    for index, (name, producers) in enumerate(specifications):
        widths = []
        for conv_name, bn_name in producers:
            conv = modules.get(conv_name)
            bn = modules.get(bn_name)
            if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
                raise TypeError(f"图节点 {name} 的 producer 类型不正确：{conv_name}/{bn_name}")
            if conv.out_channels != bn.num_features:
                raise ValueError(f"producer 通道数不一致：{conv_name}/{bn_name}")
            widths.append(conv.out_channels)
        if len(set(widths)) != 1:
            raise ValueError(f"共享门 producer 的输出宽度不一致：{name} -> {widths}")
        channels = widths[0]
        block_size = closest_power_two_block_size(channels)
        groups.append(
            FeatureGroup(
                index=index,
                name=name,
                producers=producers,
                channels=channels,
                block_size=block_size,
                block_count=channels // block_size,
            )
        )
    if len(groups) != 17 or any(group.block_count != 16 for group in groups):
        raise ValueError("ResNet18 图块必须是 17 组且每组 16 块。")
    return groups


def producer_parameter_names(group: FeatureGroup, modules: dict[str, nn.Module]) -> set[str]:
    names: set[str] = set()
    for conv_name, bn_name in group.producers:
        for module_name in (conv_name, bn_name):
            module = modules[module_name]
            for local_name, _ in module.named_parameters(recurse=False):
                names.add(f"{module_name}.{local_name}")
    return names


def block_parameter_cost(
    group: FeatureGroup,
    start: int,
    end: int,
    parameters: dict[str, nn.Parameter],
) -> int:
    total = 0
    for conv_name, bn_name in group.producers:
        for parameter_name in (f"{conv_name}.weight", f"{conv_name}.bias", f"{bn_name}.weight", f"{bn_name}.bias"):
            parameter = parameters.get(parameter_name)
            if parameter is None:
                continue
            if parameter.ndim == 0 or parameter.shape[0] != group.channels:
                raise ValueError(f"参数不能按 producer 输出通道切分：{parameter_name}")
            total += parameter[start:end].numel()
    return total


def build_channel_blocks(model: nn.Module, groups: list[FeatureGroup]) -> list[ChannelBlock]:
    parameters = dict(model.named_parameters())
    blocks: list[ChannelBlock] = []
    for group in groups:
        for local_index in range(group.block_count):
            start = local_index * group.block_size
            end = start + group.block_size
            blocks.append(
                ChannelBlock(
                    index=len(blocks),
                    group_index=group.index,
                    group_name=group.name,
                    block_index=local_index,
                    channel_start=start,
                    channel_end=end,
                    param_cost=block_parameter_cost(group, start, end, parameters),
                )
            )
    return blocks


def validate_parameter_coverage(model: nn.Module, groups: list[FeatureGroup], blocks: list[ChannelBlock]) -> dict[str, int]:
    modules = dict(model.named_modules())
    all_parameters = dict(model.named_parameters())
    grouped = set().union(*(producer_parameter_names(group, modules) for group in groups))
    head = {name for name in all_parameters if name.startswith(HEAD_PREFIX)}
    missing = sorted(set(all_parameters) - grouped - head)
    overlap = sorted(grouped & head)
    if missing or overlap:
        raise ValueError(f"参数覆盖不完整：missing={missing}, overlap={overlap}")
    candidate_count = sum(block.param_cost for block in blocks)
    grouped_count = sum(all_parameters[name].numel() for name in grouped)
    if candidate_count != grouped_count:
        raise ValueError(f"块成本 {candidate_count} 与 producer 参数 {grouped_count} 不一致。")
    total_count = sum(parameter.numel() for parameter in all_parameters.values())
    head_count = sum(all_parameters[name].numel() for name in head)
    if candidate_count + head_count != total_count:
        raise ValueError("候选块与分类头没有覆盖全部 trainable parameter。")
    target_count = math.floor(total_count * TARGET_RATIO)
    if head_count >= target_count:
        raise ValueError("分类头已经超过总保护预算。")
    return {
        "total_param_count": total_count,
        "head_param_count": head_count,
        "candidate_param_count": candidate_count,
        "target_param_count": target_count,
        "channel_budget": target_count - head_count,
    }


def digest_indices(indices: Iterable[int]) -> str:
    digest = hashlib.sha256()
    digest.update(b"TSDP-ARC-source-indices-v1\0")
    for index in indices:
        digest.update(int(index).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def build_discovery_indices(protocol_root: Path) -> tuple[list[int], list[int], list[int]]:
    victim_indices = read_ms_split_indices(protocol_root, DATASET, "victim_train", "official_train")
    final_query_indices = read_ms_split_indices(protocol_root, DATASET, "query_pool_ms", "official_train")
    final_set = set(final_query_indices)
    remaining = [index for index in victim_indices if index not in final_set]
    required = DISCOVERY_QUERY_SIZE + DISCOVERY_HOLDOUT_SIZE
    if len(remaining) < required:
        raise ValueError("排除正式 query 后没有足够 discovery 样本。")
    generator = random.Random(SEED)
    generator.shuffle(remaining)
    discovery_query = remaining[:DISCOVERY_QUERY_SIZE]
    discovery_holdout = remaining[DISCOVERY_QUERY_SIZE:required]
    if set(discovery_query) & set(discovery_holdout):
        raise ValueError("discovery query 与 holdout 重叠。")
    if (set(discovery_query) | set(discovery_holdout)) & final_set:
        raise ValueError("discovery 集合与正式 query_pool_ms 重叠。")
    return discovery_query, discovery_holdout, final_query_indices


def group_state_map(model: nn.Module, groups: list[FeatureGroup]) -> dict[str, int]:
    modules = dict(model.named_modules())
    mapping: dict[str, int] = {}
    for group in groups:
        for conv_name, bn_name in group.producers:
            for module_name in (conv_name, bn_name):
                module = modules[module_name]
                for local_name, value in list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False)):
                    state_name = f"{module_name}.{local_name}"
                    if local_name == "num_batches_tracked":
                        continue
                    if value.ndim == 0 or value.shape[0] != group.channels:
                        raise ValueError(f"状态不能按输出通道映射：{state_name}")
                    if state_name in mapping:
                        raise ValueError(f"状态被多个图节点重复映射：{state_name}")
                    mapping[state_name] = group.index
    return mapping


class AttackRecoverabilityModel(nn.Module):
    """以可微保护门控制 victim-public 初始化，并训练内部攻击增量。"""

    def __init__(self, public_model: nn.Module, victim_model: nn.Module, groups: list[FeatureGroup], blocks: list[ChannelBlock]):
        super().__init__()
        self.template = public_model
        self.template.eval()
        for parameter in self.template.parameters():
            parameter.requires_grad_(False)

        self.parameter_names = [name for name, _ in self.template.named_parameters()]
        self.buffer_names = [name for name, _ in self.template.named_buffers()]
        victim_parameters = dict(victim_model.named_parameters())
        victim_buffers = dict(victim_model.named_buffers())
        self.public_parameters = {
            name: parameter.detach().clone() for name, parameter in self.template.named_parameters()
        }
        self.victim_parameters = {
            name: victim_parameters[name].detach().clone() for name in self.parameter_names
        }
        self.public_buffers = {
            name: buffer.detach().clone() for name, buffer in self.template.named_buffers()
        }
        self.victim_buffers = {
            name: victim_buffers[name].detach().clone() for name in self.buffer_names
        }
        self.attack_deltas = nn.ParameterList(
            [nn.Parameter(torch.zeros_like(self.public_parameters[name])) for name in self.parameter_names]
        )
        self.gate_logits = nn.ParameterList(
            [nn.Parameter(torch.zeros(group.block_count, device=self.public_parameters[self.parameter_names[0]].device)) for group in groups]
        )
        self.groups = groups
        self.blocks = blocks
        self.state_groups = group_state_map(self.template, groups)
        self.costs = [
            torch.tensor(
                [block.param_cost for block in blocks if block.group_index == group.index],
                dtype=torch.float32,
                device=self.gate_logits[group.index].device,
            )
            for group in groups
        ]
        if [len(cost) for cost in self.costs] != [group.block_count for group in groups]:
            raise ValueError("门成本与图组块数量不一致。")

    def probabilities(self, temperature: float, detach: bool = False) -> list[torch.Tensor]:
        values = [torch.sigmoid(logit / temperature) for logit in self.gate_logits]
        return [value.detach() for value in values] if detach else values

    @torch.no_grad()
    def project_budget(self, channel_budget: int, temperature: float) -> None:
        total_cost = sum(float(cost.sum().item()) for cost in self.costs)
        if channel_budget >= total_cost:
            return
        low = -100.0
        high = 100.0
        for _ in range(80):
            shift = (low + high) / 2.0
            protected = sum(
                (torch.sigmoid((logit - shift) / temperature) * cost).sum()
                for logit, cost in zip(self.gate_logits, self.costs)
            )
            if float(protected.item()) > channel_budget:
                low = shift
            else:
                high = shift
        shift = (low + high) / 2.0
        for logit in self.gate_logits:
            logit.sub_(shift)

    def soft_protected_cost(self, probabilities: list[torch.Tensor]) -> torch.Tensor:
        return sum((probability * cost).sum() for probability, cost in zip(probabilities, self.costs))

    def binary_penalty(self, probabilities: list[torch.Tensor]) -> torch.Tensor:
        numerator = sum(
            (probability * (1.0 - probability) * cost).sum()
            for probability, cost in zip(probabilities, self.costs)
        )
        denominator = sum(cost.sum() for cost in self.costs)
        return numerator / denominator

    def channel_probabilities(self, probabilities: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            probability.repeat_interleave(group.block_size)
            for probability, group in zip(probabilities, self.groups)
        ]

    def protection_tensor(
        self,
        state_name: str,
        reference: torch.Tensor,
        channels: list[torch.Tensor],
    ) -> torch.Tensor:
        if state_name.startswith(HEAD_PREFIX):
            return torch.ones_like(reference, dtype=reference.dtype)
        group_index = self.state_groups.get(state_name)
        if group_index is None:
            return torch.zeros_like(reference, dtype=reference.dtype)
        vector = channels[group_index].to(reference.dtype)
        shape = (vector.numel(),) + (1,) * (reference.ndim - 1)
        return vector.reshape(shape).expand_as(reference)

    def effective_parameters(
        self,
        probabilities: list[torch.Tensor],
        detach_attack: bool,
    ) -> dict[str, torch.Tensor]:
        channels = self.channel_probabilities(probabilities)
        effective: dict[str, torch.Tensor] = {}
        for index, name in enumerate(self.parameter_names):
            public = self.public_parameters[name]
            victim = self.victim_parameters[name]
            protected = self.protection_tensor(name, public, channels)
            delta = self.attack_deltas[index].detach() if detach_attack else self.attack_deltas[index]
            effective[name] = public + (1.0 - protected) * (victim - public) + delta
        return effective

    def effective_buffers(self, probabilities: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        channels = self.channel_probabilities(probabilities)
        effective: dict[str, torch.Tensor] = {}
        for name in self.buffer_names:
            public = self.public_buffers[name]
            victim = self.victim_buffers[name]
            if public.is_floating_point():
                # PyTorch BatchNorm 不支持对 running_mean/var 求梯度；运行状态仍按当前
                # 门混合并在硬掩码中保护，只在可微选择时停止这条梯度路径。
                protected = self.protection_tensor(name, public, channels).detach()
                effective[name] = public + (1.0 - protected) * (victim - public)
            else:
                effective[name] = victim
        return effective

    def forward_with_probabilities(
        self,
        images: torch.Tensor,
        probabilities: list[torch.Tensor],
        detach_attack: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        parameters = self.effective_parameters(probabilities, detach_attack)
        buffers = self.effective_buffers(probabilities)
        logits = functional_call(self.template, (parameters, buffers), (images,), strict=True)
        return logits, parameters


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def temperature_at(epoch: int) -> float:
    if SELECTION_EPOCHS == 1:
        return GATE_TEMPERATURE_END
    progress = (epoch - 1) / (SELECTION_EPOCHS - 1)
    return GATE_TEMPERATURE_START * (
        GATE_TEMPERATURE_END / GATE_TEMPERATURE_START
    ) ** progress


@torch.no_grad()
def collect_posteriors(
    victim: nn.Module,
    public_dataset,
    indices: list[int],
    device: torch.device,
    num_workers: int,
    generator_offset: int,
) -> torch.Tensor:
    dataset = torch.utils.data.Subset(public_dataset, indices)
    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=build_generator(SEED, offset=generator_offset),
    )
    victim.eval()
    chunks = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        chunks.append(F.softmax(victim(images), dim=1).cpu())
    result = torch.cat(chunks)
    if result.shape != (len(indices), NUM_CLASSES):
        raise ValueError("victim discovery posterior 形状不正确。")
    return result


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(writer_file, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_selection(
    arc: AttackRecoverabilityModel,
    query_dataset: PosteriorDataset,
    holdout_dataset: PosteriorDataset,
    channel_budget: int,
    device: torch.device,
    num_workers: int,
    output_root: Path,
) -> tuple[list[dict[str, object]], list[torch.Tensor]]:
    query_loader = DataLoader(
        query_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=build_generator(SEED, offset=20),
    )
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=build_generator(SEED, offset=21),
    )
    attack_optimizer = torch.optim.SGD(
        arc.attack_deltas,
        lr=SHADOW_LR,
        momentum=MOMENTUM,
    )
    attack_scheduler = torch.optim.lr_scheduler.StepLR(
        attack_optimizer,
        step_size=LR_STEP,
        gamma=LR_GAMMA,
    )
    gate_optimizer = torch.optim.Adam(arc.gate_logits, lr=GATE_LR)
    history: list[dict[str, object]] = []
    holdout_iterator = iter(holdout_loader)

    for epoch in range(1, SELECTION_EPOCHS + 1):
        temperature = temperature_at(epoch)
        arc.project_budget(channel_budget, temperature)
        query_loss_sum = 0.0
        query_match = 0
        query_count = 0
        holdout_loss_sum = 0.0
        holdout_match = 0
        holdout_count = 0

        for images, targets, labels in query_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            attack_optimizer.zero_grad(set_to_none=True)
            attack_probabilities = arc.probabilities(temperature, detach=True)
            logits, effective_parameters = arc.forward_with_probabilities(
                images,
                attack_probabilities,
                detach_attack=False,
            )
            distillation = soft_cross_entropy(logits, targets)
            decay = 0.5 * WEIGHT_DECAY * sum(parameter.square().sum() for parameter in effective_parameters.values())
            attack_loss = distillation + decay
            attack_loss.backward()
            attack_optimizer.step()

            batch_size = images.size(0)
            query_count += batch_size
            query_loss_sum += float(distillation.detach().item()) * batch_size
            query_match += int((logits.detach().argmax(dim=1) == labels).sum().item())

            try:
                holdout_images, holdout_targets, holdout_labels = next(holdout_iterator)
            except StopIteration:
                holdout_iterator = iter(holdout_loader)
                holdout_images, holdout_targets, holdout_labels = next(holdout_iterator)
            holdout_images = holdout_images.to(device, non_blocking=True)
            holdout_targets = holdout_targets.to(device, non_blocking=True)
            holdout_labels = holdout_labels.to(device, non_blocking=True)

            gate_optimizer.zero_grad(set_to_none=True)
            gate_probabilities = arc.probabilities(temperature, detach=False)
            holdout_logits, _ = arc.forward_with_probabilities(
                holdout_images,
                gate_probabilities,
                detach_attack=True,
            )
            holdout_loss = soft_cross_entropy(holdout_logits, holdout_targets)
            binary_penalty = arc.binary_penalty(gate_probabilities)
            gate_loss = -holdout_loss + BINARY_REGULARIZER * binary_penalty
            gate_loss.backward()
            gate_optimizer.step()
            arc.project_budget(channel_budget, temperature)

            holdout_batch_size = holdout_images.size(0)
            holdout_count += holdout_batch_size
            holdout_loss_sum += float(holdout_loss.detach().item()) * holdout_batch_size
            holdout_match += int(
                (holdout_logits.detach().argmax(dim=1) == holdout_labels).sum().item()
            )

        learning_rate = attack_optimizer.param_groups[0]["lr"]
        attack_scheduler.step()
        probabilities = arc.probabilities(temperature, detach=True)
        soft_cost = float(arc.soft_protected_cost(probabilities).item())
        binary_value = float(arc.binary_penalty(probabilities).item())
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "soft_protected_param_count": soft_cost,
            "binary_penalty": binary_value,
            "query_loss": query_loss_sum / query_count,
            "query_match": query_match / query_count,
            "holdout_loss": holdout_loss_sum / holdout_count,
            "holdout_match": holdout_match / holdout_count,
        }
        history.append(row)
        print(
            f"[SELECT] {epoch:03d}/{SELECTION_EPOCHS:03d} "
            f"T={temperature:.4f} query={row['query_loss']:.4f}/{row['query_match']:.4f} "
            f"holdout={row['holdout_loss']:.4f}/{row['holdout_match']:.4f} "
            f"soft_cost={soft_cost:.1f} binary={binary_value:.6f}"
        )
        write_tsv(
            output_root / "selection.tsv",
            history,
            list(row),
        )

    final_probabilities = arc.probabilities(GATE_TEMPERATURE_END, detach=True)
    return history, final_probabilities


def harden_blocks(
    groups: list[FeatureGroup],
    blocks: list[ChannelBlock],
    probabilities: list[torch.Tensor],
    channel_budget: int,
) -> tuple[list[int], list[torch.Tensor], int]:
    scored = []
    for group in groups:
        values = probabilities[group.index].detach().cpu().tolist()
        group_blocks = [block for block in blocks if block.group_index == group.index]
        for block, score in zip(group_blocks, values):
            scored.append((float(score), block))
    scored.sort(key=lambda item: (-item[0], item[1].index))
    selected: list[int] = []
    protected = 0
    for _, block in scored:
        if protected + block.param_cost <= channel_budget:
            selected.append(block.index)
            protected += block.param_cost
    selected_set = set(selected)
    hard_probabilities = [
        torch.tensor(
            [
                1.0 if block.index in selected_set else 0.0
                for block in blocks
                if block.group_index == group.index
            ],
            device=probabilities[group.index].device,
        )
        for group in groups
    ]
    return sorted(selected), hard_probabilities, protected


def build_hard_masks(
    model: nn.Module,
    groups: list[FeatureGroup],
    blocks: list[ChannelBlock],
    selected_indices: list[int],
) -> dict[str, torch.Tensor]:
    masks = {
        name: torch.zeros_like(value, dtype=torch.bool)
        for name, value in model.state_dict().items()
    }
    for name in masks:
        if name.startswith(HEAD_PREFIX):
            masks[name].fill_(True)
    modules = dict(model.named_modules())
    selected = set(selected_indices)
    for block in blocks:
        if block.index not in selected:
            continue
        group = groups[block.group_index]
        for conv_name, bn_name in group.producers:
            for module_name in (conv_name, bn_name):
                module = modules[module_name]
                for local_name, _ in list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False)):
                    if local_name == "num_batches_tracked":
                        continue
                    name = f"{module_name}.{local_name}"
                    masks[name][block.channel_start:block.channel_end] = True
    return masks


@torch.no_grad()
def evaluate_internal_shadow(
    arc: AttackRecoverabilityModel,
    probabilities: list[torch.Tensor],
    dataset: PosteriorDataset,
    device: torch.device,
    num_workers: int,
) -> dict[str, float | int]:
    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=build_generator(SEED, offset=22),
    )
    loss_sum = 0.0
    match = 0
    count = 0
    for images, targets, labels in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, _ = arc.forward_with_probabilities(images, probabilities, detach_attack=True)
        loss = soft_cross_entropy(logits, targets)
        count += images.size(0)
        loss_sum += float(loss.item()) * images.size(0)
        match += int((logits.argmax(dim=1) == labels).sum().item())
    return {"count": count, "soft_cross_entropy": loss_sum / count, "agreement": match / count}


def combine_surrogate_state(
    public_model: nn.Module,
    victim_model: nn.Module,
    masks: dict[str, torch.Tensor],
) -> None:
    public_state = public_model.state_dict()
    victim_state = victim_model.state_dict()
    combined = {}
    for name, public in public_state.items():
        victim = victim_state[name]
        mask = masks[name].to(public.device)
        if public.shape != victim.shape or public.shape != mask.shape:
            raise ValueError(f"组合状态形状不一致：{name}")
        combined[name] = torch.where(mask, public, victim)
    public_model.load_state_dict(combined, strict=True)


def save_end_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, metrics: dict[str, object]) -> None:
    torch.save(
        {
            "schema_version": 2,
            "protocol": "MS",
            "attack_protocol": ATTACK_PROTOCOL_VERSION,
            "arch": MODEL,
            "dataset": DATASET,
            "artifact_id": "temp_arc",
            "epoch": FINAL_EPOCHS,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        path,
    )


def run_final_attack(
    masks: dict[str, torch.Tensor],
    device: torch.device,
    num_workers: int,
    output_root: Path,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, str]]:
    dataset_root = REPO_ROOT / "dataset" / "public"
    protocol_root = REPO_ROOT / "dataset" / "MS"
    victim_path = REPO_ROOT / "weights" / "MS" / "victim" / MODEL / DATASET / "best.pth"
    official_path = REPO_ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"

    configure_reproducibility(SEED, deterministic=True)
    query_indices, posteriors, labels, posterior_path, query_manifest = load_query_targets(
        protocol_root, DATASET, MODEL, QUERY_BUDGET, "soft"
    )
    victim, _ = build_victim(MODEL, NUM_CLASSES, victim_path)
    expected_victim_hash = query_manifest.get("victim", {}).get("checkpoint_sha256")
    actual_victim_hash = sha256_file(victim_path)
    if expected_victim_hash and expected_victim_hash != actual_victim_hash:
        raise ValueError("正式 posterior 与当前 victim best.pth 不一致。")
    surrogate = build_public_model()
    combine_surrogate_state(surrogate, victim, masks)

    query_dataset = build_query_dataset(
        DATASET,
        dataset_root,
        query_indices,
        posteriors,
        labels,
        input_transform="test",
    )
    eval_dataset = build_eval_dataset(DATASET, dataset_root, protocol_root, None)
    query_loader = DataLoader(
        query_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=build_generator(SEED),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=build_generator(SEED, offset=1),
    )
    victim = victim.to(device)
    surrogate = surrogate.to(device)
    reference = collect_eval_reference(victim, eval_loader, device)
    optimizer = torch.optim.SGD(
        surrogate.parameters(),
        lr=SHADOW_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)
    history: list[dict[str, object]] = []
    for epoch in range(1, FINAL_EPOCHS + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(
            surrogate,
            query_loader,
            optimizer,
            device,
            "soft",
            epoch,
            FINAL_EPOCHS,
            None,
        )
        scheduler.step()
        row = {"epoch": epoch, "learning_rate": learning_rate, **train_metrics}
        history.append(row)
        write_tsv(output_root / "attack.tsv", history, list(row))
    end_metrics = evaluate_surrogate(surrogate, eval_loader, reference, device)
    save_end_checkpoint(output_root / "end.pth", surrogate, optimizer, scheduler, end_metrics)
    hashes = {
        "victim_checkpoint_sha256": actual_victim_hash,
        "official_weight_sha256": sha256_file(official_path),
        "posterior_sha256": sha256_file(posterior_path),
    }
    return end_metrics, history, hashes


def load_reference_metrics() -> dict[str, dict[str, object]]:
    root = REPO_ROOT / "results" / "MS" / MODEL / DATASET
    references = {}
    for name in ("no_protection", "head_only", "tensorshield", "full_protection"):
        path = root / name / "metrics.json"
        with path.open("r", encoding="utf-8") as reader:
            payload = json.load(reader)
        references[name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(path),
            "protected_param_ratio": payload.get("protection", {}).get("protected_param_ratio"),
            "end": payload["end"],
        }
    return references


def block_payload(block: ChannelBlock, score: float, selected: bool) -> dict[str, object]:
    return {
        "index": block.index,
        "group_index": block.group_index,
        "group_name": block.group_name,
        "block_index": block.block_index,
        "channel_start": block.channel_start,
        "channel_end": block.channel_end,
        "param_cost": block.param_cost,
        "score": score,
        "selected": selected,
    }


def write_final_metrics(
    output_root: Path,
    selection_payload: dict[str, object],
    end_metrics: dict[str, object],
    attack_history: list[dict[str, object]],
    input_hashes: dict[str, str],
) -> dict[str, object]:
    budget = selection_payload["budget"]
    if not isinstance(budget, dict):
        raise ValueError("selection.json 缺少预算信息。")
    protected_param_count = int(budget["protected_param_count"])
    total_param_count = int(budget["total_param_count"])
    selected_block_count = int(selection_payload["selected_block_count"])
    mask_hash = str(selection_payload["protection_mask_sha256"])
    internal_metrics = selection_payload["internal_hard_holdout"]
    metrics_payload = {
        "schema_version": 1,
        "experiment": "attack_recoverability_cut",
        "scope": "temporary_validation",
        "protocol": "MS",
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "model": MODEL,
        "dataset": DATASET,
        "query_budget": QUERY_BUDGET,
        "label_mode": "soft",
        "query_transform": "test",
        "training_mode": "finetune_all_parameters",
        "randomization": {
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": SEED,
            "query_sampler_seed": SEED,
            "reset_before_each_surrogate_initialization": True,
        },
        "primary": {"checkpoint": "end.pth", "epoch": FINAL_EPOCHS},
        "protection": {
            "strategy": "attack_recoverability_cut",
            "head_mode": "protected_weight_and_bias",
            "selected_block_count": selected_block_count,
            "protection_mask_sha256": mask_hash,
            "protected_param_count": protected_param_count,
            "total_param_count": total_param_count,
            "protected_param_ratio": protected_param_count / total_param_count,
        },
        "end": end_metrics,
        "selection_internal_hard_holdout": internal_metrics,
        "references": load_reference_metrics(),
        "inputs": input_hashes,
        "attack_end": attack_history[-1],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with (output_root / "metrics.json").open("w", encoding="utf-8") as writer:
        json.dump(metrics_payload, writer, ensure_ascii=False, indent=2)
        writer.write("\n")
    print(
        f"[RESULT] ARC ratio={protected_param_count / total_param_count:.6f} "
        f"acc={end_metrics['surrogate_acc']:.6f} fidelity={end_metrics['fidelity']:.6f} "
        f"KL={end_metrics['posterior_kl']:.6f}"
    )
    return metrics_payload


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能为负数。")
    dataset_root = REPO_ROOT / "dataset" / "public"
    protocol_root = REPO_ROOT / "dataset" / "MS"
    victim_path = REPO_ROOT / "weights" / "MS" / "victim" / MODEL / DATASET / "best.pth"
    official_path = REPO_ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    for path in (victim_path, official_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    configure_reproducibility(SEED, deterministic=True)
    victim, victim_metadata = build_victim(MODEL, NUM_CLASSES, victim_path)
    public = build_public_model()
    groups = build_feature_groups(public)
    blocks = build_channel_blocks(public, groups)
    counts = validate_parameter_coverage(public, groups, blocks)
    discovery_query, discovery_holdout, final_query = build_discovery_indices(protocol_root)

    print(f"[CHECK] 图节点={len(groups)}，候选块={len(blocks)}")
    print(
        f"[CHECK] 参数={counts['total_param_count']}，分类头={counts['head_param_count']}，"
        f"8% 上限={counts['target_param_count']}，通道块预算={counts['channel_budget']}"
    )
    print(
        f"[CHECK] discovery query/holdout/final query="
        f"{len(discovery_query)}/{len(discovery_holdout)}/{len(final_query)}，三者无重叠"
    )
    print(f"[CHECK] victim best.pth sha256={sha256_file(victim_path)}")
    print(f"[CHECK] official weight sha256={sha256_file(official_path)}")
    if args.dry_run:
        print("[CHECK] dry-run 完成；没有创建或修改 temp/output。")
        return 0

    output_root = REPO_ROOT / "temp" / "output"
    if args.final_only:
        selection_path = output_root / "selection.json"
        mask_path = output_root / "mask.pt"
        if not selection_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError("--final-only 需要已有 selection.json 和 mask.pt。")
        with selection_path.open("r", encoding="utf-8") as reader:
            selection_payload = json.load(reader)
        masks = load_protection_mask(mask_path)
        actual_mask_hash = protection_mask_sha256(masks)
        if actual_mask_hash != selection_payload.get("protection_mask_sha256"):
            raise ValueError("mask.pt 与 selection.json 的掩码摘要不一致。")
        protected_param_count = sum(
            int(masks[name].sum().item()) for name, _ in public.named_parameters()
        )
        if protected_param_count != selection_payload.get("budget", {}).get("protected_param_count"):
            raise ValueError("mask.pt 的保护参数数与 selection.json 不一致。")
        device = resolve_device(args.device)
        print(f"[INFO] final-only device={device}，复用固定掩码 {actual_mask_hash}")
        end_metrics, attack_history, input_hashes = run_final_attack(
            masks, device, args.num_workers, output_root
        )
        write_final_metrics(
            output_root, selection_payload, end_metrics, attack_history, input_hashes
        )
        return 0

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError("temp/output 已有产物；如需重跑请使用 --overwrite。")
    output_root.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for child in output_root.iterdir():
            if child.is_file():
                child.unlink()
            else:
                raise ValueError(f"不自动删除非文件产物：{child}")

    device = resolve_device(args.device)
    print(f"[INFO] device={device}")
    _, test_transform = build_transforms(DATASET)
    discovery_public = build_public_split_dataset(DATASET, dataset_root, "train", test_transform)
    victim = victim.to(device)
    query_posteriors = collect_posteriors(
        victim, discovery_public, discovery_query, device, args.num_workers, generator_offset=10
    )
    holdout_posteriors = collect_posteriors(
        victim, discovery_public, discovery_holdout, device, args.num_workers, generator_offset=11
    )
    query_dataset = PosteriorDataset(discovery_public, discovery_query, query_posteriors)
    holdout_dataset = PosteriorDataset(discovery_public, discovery_holdout, holdout_posteriors)
    public = public.to(device)
    arc = AttackRecoverabilityModel(public, victim, groups, blocks).to(device)
    del victim
    if device.type == "cuda":
        torch.cuda.empty_cache()

    history, probabilities = run_selection(
        arc,
        query_dataset,
        holdout_dataset,
        counts["channel_budget"],
        device,
        args.num_workers,
        output_root,
    )
    selected, hard_probabilities, selected_channel_params = harden_blocks(
        groups, blocks, probabilities, counts["channel_budget"]
    )
    internal_metrics = evaluate_internal_shadow(
        arc, hard_probabilities, holdout_dataset, device, args.num_workers
    )
    masks = build_hard_masks(arc.template, groups, blocks, selected)
    protected_param_count = sum(
        int(masks[name].sum().item()) for name, _ in arc.template.named_parameters()
    )
    expected_protected = counts["head_param_count"] + selected_channel_params
    if protected_param_count != expected_protected:
        raise ValueError(
            f"硬掩码参数数 {protected_param_count} 与块成本 {expected_protected} 不一致。"
        )
    if protected_param_count > counts["target_param_count"]:
        raise ValueError("硬化后的保护参数超过 8% 上限。")
    mask_hash = protection_mask_sha256(masks)
    save_protection_mask(output_root / "mask.pt", masks)

    flattened_scores = {
        block.index: float(probabilities[block.group_index][block.block_index].item())
        for block in blocks
    }
    selection_payload = {
        "schema_version": 1,
        "experiment": "attack_recoverability_cut",
        "scope": "temporary_validation",
        "model": MODEL,
        "dataset": DATASET,
        "seed": SEED,
        "randomization": {
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": SEED,
            "query_sampler_seed": SEED,
            "reset_before_each_surrogate_initialization": True,
        },
        "victim_checkpoint": str(victim_path.relative_to(REPO_ROOT)),
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "victim_checkpoint_sha256": sha256_file(victim_path),
        "official_weight": str(official_path.relative_to(REPO_ROOT)),
        "official_weight_sha256": sha256_file(official_path),
        "discovery": {
            "query_size": len(discovery_query),
            "query_indices": discovery_query,
            "query_indices_sha256": digest_indices(discovery_query),
            "holdout_size": len(discovery_holdout),
            "holdout_indices": discovery_holdout,
            "holdout_indices_sha256": digest_indices(discovery_holdout),
            "excluded_final_query_size": len(final_query),
            "excluded_final_query_indices_sha256": digest_indices(final_query),
            "formal_query_posteriors_used_for_selection": False,
            "formal_eval_used_for_selection": False,
        },
        "objective": {
            "attack_loss": "soft_cross_entropy",
            "gate_loss": "negative_holdout_soft_cross_entropy_plus_binary_regularizer",
            "first_order": True,
            "selection_epochs": SELECTION_EPOCHS,
            "shadow_lr": SHADOW_LR,
            "shadow_momentum": MOMENTUM,
            "shadow_weight_decay": WEIGHT_DECAY,
            "gate_lr": GATE_LR,
            "temperature_start": GATE_TEMPERATURE_START,
            "temperature_end": GATE_TEMPERATURE_END,
            "binary_regularizer": BINARY_REGULARIZER,
        },
        "budget": {
            **counts,
            "target_ratio": TARGET_RATIO,
            "selected_channel_param_count": selected_channel_params,
            "protected_param_count": protected_param_count,
            "protected_param_ratio": protected_param_count / counts["total_param_count"],
        },
        "groups": [
            {
                "index": group.index,
                "name": group.name,
                "producers": [list(producer) for producer in group.producers],
                "channels": group.channels,
                "block_size": group.block_size,
                "block_count": group.block_count,
            }
            for group in groups
        ],
        "blocks": [
            block_payload(block, flattened_scores[block.index], block.index in set(selected))
            for block in blocks
        ],
        "selected_block_indices": selected,
        "selected_block_count": len(selected),
        "internal_hard_holdout": internal_metrics,
        "protection_mask_sha256": mask_hash,
        "selection_end": history[-1],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with (output_root / "selection.json").open("w", encoding="utf-8") as writer:
        json.dump(selection_payload, writer, ensure_ascii=False, indent=2)
        writer.write("\n")
    print(
        f"[SELECT] hard blocks={len(selected)}，protected={protected_param_count}/"
        f"{counts['total_param_count']} ({protected_param_count / counts['total_param_count']:.6f})，"
        f"internal holdout={internal_metrics}"
    )

    del arc
    if device.type == "cuda":
        torch.cuda.empty_cache()
    end_metrics, attack_history, input_hashes = run_final_attack(
        masks, device, args.num_workers, output_root
    )
    write_final_metrics(
        output_root, selection_payload, end_metrics, attack_history, input_hashes
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
