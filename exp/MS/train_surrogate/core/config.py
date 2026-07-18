#!/usr/bin/env python3
"""MS surrogate 的路径、模型与命令行配置。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
TRAIN_SURROGATE_ROOT = REPO_ROOT / "exp" / "MS" / "train_surrogate"
TRAIN_VICTIM_ROOT = REPO_ROOT / "exp" / "MS" / "train_victim"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRAIN_SURROGATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_SURROGATE_ROOT))
if str(TRAIN_VICTIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_VICTIM_ROOT))

from defense import DEFENSES  # noqa: E402


NUM_CLASSES = {"c10": 10, "c100": 100, "s10": 10, "t200": 200}
ATTACK_PROTOCOL_VERSION = "soft_query_validation_best_v1"
HARD_BLACKBOX_ATTACK_PROTOCOL_VERSION = "hard_query_validation_best_v1"
FORMAL_EPOCHS = 100
FORMAL_BATCH_SIZE = 64
FORMAL_EVAL_BATCH_SIZE = 128
FORMAL_LEARNING_RATE = 0.01
FORMAL_MOMENTUM = 0.5
FORMAL_WEIGHT_DECAY = 5e-4
FORMAL_LR_STEP = 60
FORMAL_LR_GAMMA = 0.1
MODEL_SPECS = {
    "resnet18": ("resnet18", "resnet18-5c106cde.pth"),
    "resnet50": ("resnet50", "resnet50-19c8e357.pth"),
    "vgg16_bn": ("vgg16_bn", "vgg16_bn-6c64b313.pth"),
    "mobilenetv2": ("mobilenetv2", "mobilenet_v2-b0353104.pth"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练并评估 MS surrogate。")
    parser.add_argument("model", choices=sorted(MODEL_SPECS), help="victim/surrogate 模型。")
    parser.add_argument("dataset", help="数据集 id：c10/c100/s10/t200。")
    parser.add_argument("--defense", required=True, choices=DEFENSES, help="参数保护 baseline。")
    parser.add_argument("--plan-id", default=None, help="baseline.json 中的固定配置 id。")
    parser.add_argument("--budget", required=True, type=int, help="使用 query_pool_ms 的前缀长度。")
    parser.add_argument(
        "--protected-units",
        default=None,
        help="闭区间 unit 表达式，例如 0-50、100-121、3,6,9；no/full 可省略。",
    )
    parser.add_argument(
        "--protected-layers",
        default=None,
        help="1-based 官方完整层表达式，例如 1-3、8-11、16-18。",
    )
    parser.add_argument(
        "--protected-scalars",
        type=int,
        default=None,
        help="large_weight 保护的绝对标量数量。",
    )
    parser.add_argument(
        "--training-mode",
        required=True,
        choices=("finetune",),
        help="正式协议固定为全部参数共同微调。",
    )
    parser.add_argument(
        "--label-mode",
        choices=("soft", "hard"),
        default="soft",
        help="部分保护与 soft 全保护使用 soft；正式 label-only 黑盒使用 hard。",
    )
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "dataset" / "public"), help="公开数据集根目录。")
    parser.add_argument("--protocol-root", default=str(REPO_ROOT / "dataset" / "MS"), help="MS 协议根目录。")
    parser.add_argument("--victim-checkpoint", default=None, help="victim best.pth 路径。")
    parser.add_argument("--weight-path", default=None, help="官方 ImageNet 预训练权重。")
    parser.add_argument(
        "--weights-root",
        default=str(REPO_ROOT / "weights" / "MS" / "surrogate"),
        help="surrogate 权重根目录。",
    )
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results" / "MS"), help="MS 原始指标根目录。")
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS, help="训练轮数。")
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE, help="训练 batch size。")
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=FORMAL_EVAL_BATCH_SIZE,
        help="validation/eval batch size。",
    )
    parser.add_argument("--lr", type=float, default=FORMAL_LEARNING_RATE, help="SGD 学习率。")
    parser.add_argument("--momentum", type=float, default=FORMAL_MOMENTUM, help="SGD 动量。")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=FORMAL_WEIGHT_DECAY,
        help="权重衰减。",
    )
    parser.add_argument("--lr-step", type=int, default=FORMAL_LR_STEP, help="学习率衰减步长。")
    parser.add_argument("--lr-gamma", type=float, default=FORMAL_LR_GAMMA, help="学习率衰减系数。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用确定性训练。",
    )
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--eval-subset", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="完成输入与保护计划检查后退出。")
    parser.add_argument("--overwrite", action="store_true", help="覆盖相同 artifact_id 的已有产物。")
    return parser.parse_args()


def resolve_attack_protocol(label_mode: str) -> str:
    if label_mode == "soft":
        return ATTACK_PROTOCOL_VERSION
    if label_mode == "hard":
        return HARD_BLACKBOX_ATTACK_PROTOCOL_VERSION
    raise ValueError(f"未知 label mode：{label_mode}")


def validate_attack_configuration(
    defense: str,
    training_mode: str,
    label_mode: str,
    model_name: str,
    dataset_name: str,
) -> None:
    if training_mode != "finetune":
        raise ValueError("正式 MS 协议只允许 --training-mode finetune。")
    if label_mode == "soft":
        return
    if label_mode != "hard":
        raise ValueError(f"未知 label mode：{label_mode}")
    if defense != "full_protection":
        raise ValueError("hard-label 正式对比只允许 --defense full_protection。")


def validate_formal_hyperparameters(args: argparse.Namespace) -> None:
    expected = {
        "epochs": FORMAL_EPOCHS,
        "batch_size": FORMAL_BATCH_SIZE,
        "eval_batch_size": FORMAL_EVAL_BATCH_SIZE,
        "lr": FORMAL_LEARNING_RATE,
        "momentum": FORMAL_MOMENTUM,
        "weight_decay": FORMAL_WEIGHT_DECAY,
        "lr_step": FORMAL_LR_STEP,
        "lr_gamma": FORMAL_LR_GAMMA,
    }
    actual = {name: getattr(args, name) for name in expected}
    if actual != expected:
        raise ValueError(f"正式 MS 超参数不可临时覆盖：expected={expected}, actual={actual}")
    if args.eval_subset is not None:
        raise ValueError("正式 MS 不允许裁剪 eval_ms。")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境没有可用 CUDA 设备。")
    return device
