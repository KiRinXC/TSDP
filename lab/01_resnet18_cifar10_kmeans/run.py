#!/usr/bin/env python3
"""用 ImageNet 预训练 ResNet18 特征对 CIFAR-10 做无监督 KMeans 探测。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets as tv_datasets
from torchvision import transforms
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import imagenet as imagenet_models  # noqa: E402
from models.imagenet import load_official_imagenet_weights  # noqa: E402


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="提取 ImageNet 预训练 ResNet18 特征，不用 CIFAR-10 标签训练，使用 KMeans 检查 10 类样本是否自然可分。"
    )
    parser.add_argument(
        "--dataset-root",
        default=str(REPO_ROOT / "dataset" / "public" / "cifar10"),
        help="torchvision CIFAR10 数据根目录",
    )
    parser.add_argument(
        "--weight-path",
        default=str(REPO_ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"),
        help="官方 torchvision ResNet18 ImageNet-1k 权重路径",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "results" / "lab" / "01_resnet18_cifar10_kmeans"),
        help="输出目录",
    )
    parser.add_argument("--device", default="auto", help="运行设备：auto / cpu / cuda / cuda:0")
    parser.add_argument("--batch-size", type=int, default=256, help="特征提取 batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 个样本，用于快速检查")
    parser.add_argument("--seed", type=int, default=42, help="KMeans 初始化随机种子")
    parser.add_argument("--kmeans-iters", type=int, default=100, help="KMeans 最大迭代次数")
    parser.add_argument("--kmeans-restarts", type=int, default=10, help="KMeans 随机重启次数")
    parser.add_argument("--tol", type=float, default=1e-4, help="KMeans 中心移动收敛阈值")
    return parser.parse_args()


def configure_reproducibility(seed: int) -> None:
    """固定随机过程，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    """把用户输入的设备名转换成 torch.device。"""
    normalized = name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用的 CUDA 设备。")
    return torch.device(normalized)


def build_transform() -> transforms.Compose:
    """构造 ImageNet 预训练模型的标准验证预处理。"""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_dataset(dataset_root: Path, limit: int | None):
    """读取 CIFAR-10 测试集。"""
    dataset = tv_datasets.CIFAR10(
        root=str(dataset_root),
        train=False,
        download=False,
        transform=build_transform(),
    )
    if limit is not None and limit > 0 and limit < len(dataset):
        dataset = Subset(dataset, list(range(limit)))
    return dataset


def build_model(weight_path: Path, device: torch.device) -> torch.nn.Module:
    """加载保持 ImageNet 权重的 ResNet18，用其倒数第二层特征做聚类。"""
    if not weight_path.is_file():
        raise FileNotFoundError(f"找不到 ResNet18 预训练权重：{weight_path}")

    model = imagenet_models.resnet18(num_classes=1000)
    load_official_imagenet_weights("resnet18", model, str(weight_path), strict=True)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def extract_features(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """提取 ResNet18 平均池化后的 512 维特征。"""
    features = []
    labels = []

    for inputs, targets in tqdm(loader, desc="[FEATURE]", dynamic_ncols=True):
        inputs = inputs.to(device, non_blocking=True)
        feature_maps = model.features(inputs)
        pooled = model.avgpool(feature_maps)
        batch_features = torch.flatten(pooled, 1)
        features.append(batch_features.cpu().numpy())
        labels.append(targets.numpy())

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def standardize_features(features: np.ndarray) -> np.ndarray:
    """按维度标准化特征，避免尺度差异主导聚类。"""
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return ((features - mean) / np.maximum(std, 1e-12)).astype(np.float32)


def kmeans(
    features: np.ndarray,
    n_clusters: int,
    seed: int,
    max_iters: int,
    restarts: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """使用 NumPy 实现 KMeans，避免引入额外依赖。"""
    rng = np.random.default_rng(seed)
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_inertia = float("inf")
    n_samples = features.shape[0]

    for restart in range(restarts):
        init_indices = rng.choice(n_samples, size=n_clusters, replace=False)
        centers = features[init_indices].copy()
        labels = np.zeros(n_samples, dtype=np.int64)

        for _ in range(max_iters):
            distances = squared_euclidean(features, centers)
            labels = distances.argmin(axis=1)

            new_centers = centers.copy()
            for cluster_index in range(n_clusters):
                members = features[labels == cluster_index]
                if len(members) == 0:
                    farthest_index = distances.min(axis=1).argmax()
                    new_centers[cluster_index] = features[farthest_index]
                else:
                    new_centers[cluster_index] = members.mean(axis=0)

            center_shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            if center_shift < tol:
                break

        final_distances = squared_euclidean(features, centers)
        labels = final_distances.argmin(axis=1)
        inertia = float(final_distances[np.arange(n_samples), labels].sum())
        print(f"[KMEANS] restart={restart + 1}/{restarts} inertia={inertia:.2f}")
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    if best_labels is None or best_centers is None:
        raise RuntimeError("KMeans 没有产生有效结果。")
    return best_labels, best_centers, best_inertia


def squared_euclidean(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """计算样本到中心的平方欧氏距离。"""
    feature_norm = np.sum(features * features, axis=1, keepdims=True)
    center_norm = np.sum(centers * centers, axis=1, keepdims=True).T
    distances = feature_norm + center_norm - 2.0 * features @ centers.T
    return np.maximum(distances, 0.0)


def build_cluster_label_matrix(cluster_ids: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """统计行=cluster、列=真实标签的计数矩阵。"""
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for cluster_index, label_index in zip(cluster_ids, labels):
        matrix[int(cluster_index), int(label_index)] += 1
    return matrix


def best_cluster_to_label_mapping(cluster_label_matrix: np.ndarray) -> tuple[dict[int, int], int]:
    """用 bitmask 动态规划求 cluster 到 label 的一对一最佳匹配。"""
    n_clusters, n_labels = cluster_label_matrix.shape
    if n_clusters != n_labels:
        raise ValueError("当前匹配实现要求 cluster 数和 label 数相同。")

    memo: dict[tuple[int, int], tuple[int, list[int]]] = {}

    def solve(cluster_index: int, used_labels_mask: int) -> tuple[int, list[int]]:
        key = (cluster_index, used_labels_mask)
        if key in memo:
            return memo[key]
        if cluster_index == n_clusters:
            return 0, []

        best_score = -1
        best_path: list[int] = []
        for label_index in range(n_labels):
            if used_labels_mask & (1 << label_index):
                continue
            rest_score, rest_path = solve(cluster_index + 1, used_labels_mask | (1 << label_index))
            score = int(cluster_label_matrix[cluster_index, label_index]) + rest_score
            if score > best_score:
                best_score = score
                best_path = [label_index] + rest_path

        memo[key] = (best_score, best_path)
        return memo[key]

    score, path = solve(0, 0)
    return {cluster_index: label_index for cluster_index, label_index in enumerate(path)}, score


def greedy_cluster_to_label_mapping(cluster_label_matrix: np.ndarray) -> tuple[dict[int, int], int]:
    """逐 cluster 选择样本数最多的真实标签，允许多个 cluster 对应同一标签。"""
    mapping = {}
    score = 0
    for cluster_index, row in enumerate(cluster_label_matrix):
        label_index = int(row.argmax())
        mapping[cluster_index] = label_index
        score += int(row[label_index])
    return mapping, score


def aligned_confusion_matrix(
    cluster_ids: np.ndarray,
    labels: np.ndarray,
    cluster_to_label: dict[int, int],
    n_classes: int,
) -> np.ndarray:
    """构造行=真实标签、列=匹配后预测标签的混淆矩阵。"""
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for cluster_index, true_index in zip(cluster_ids, labels):
        pred_index = cluster_to_label[int(cluster_index)]
        matrix[int(true_index), pred_index] += 1
    return matrix


def normalized_mutual_information(cluster_ids: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """计算 NMI，用于评估聚类与真实标签的一致性。"""
    contingency = build_cluster_label_matrix(cluster_ids, labels, n_classes).astype(np.float64)
    total = contingency.sum()
    if total <= 0:
        return 0.0

    pi = contingency.sum(axis=1)
    pj = contingency.sum(axis=0)
    nonzero = contingency > 0
    mutual_info = float(
        np.sum((contingency[nonzero] / total) * np.log((contingency[nonzero] * total) / (pi[:, None] * pj[None, :])[nonzero]))
    )

    cluster_probs = pi[pi > 0] / total
    label_probs = pj[pj > 0] / total
    cluster_entropy = float(-np.sum(cluster_probs * np.log(cluster_probs)))
    label_entropy = float(-np.sum(label_probs * np.log(label_probs)))
    denom = (cluster_entropy + label_entropy) / 2.0
    return mutual_info / denom if denom > 0 else 0.0


def plot_confusion_matrix(path: Path, matrix: np.ndarray, title: str, accuracy: float, nmi: float) -> None:
    """绘制并保存混淆矩阵图片。"""
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(f"{title}\nACC={accuracy * 100:.2f}%  NMI={nmi:.3f}")
    ax.set_xlabel("matched cluster label")
    ax.set_ylabel("true CIFAR-10 label")
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels([str(index) for index in range(10)])
    ax.set_yticklabels([f"{index}:{name}" for index, name in enumerate(CIFAR10_CLASSES)])

    threshold = matrix.max() * 0.6 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            color = "white" if value > threshold else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    """运行实验。"""
    args = parse_args()
    configure_reproducibility(args.seed)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    weight_path = Path(args.weight_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"

    dataset = build_dataset(dataset_root, args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    model = build_model(weight_path, device)

    print(f"[INFO] 数据集: {dataset_root}")
    print(f"[INFO] 样本数: {len(dataset)}")
    print(f"[INFO] 权重: {weight_path}")
    print("[INFO] 方法: 提取 512 维特征，KMeans(k=10)，评估阶段再用标签做最佳匹配")
    print(f"[INFO] 随机种子: {args.seed}")
    print(f"[INFO] 输出目录: {out_dir}")
    print(f"[INFO] 设备: {device}")

    raw_features, labels = extract_features(model=model, loader=loader, device=device)
    features = standardize_features(raw_features)
    cluster_ids, _centers, inertia = kmeans(
        features=features,
        n_clusters=10,
        seed=args.seed,
        max_iters=args.kmeans_iters,
        restarts=args.kmeans_restarts,
        tol=args.tol,
    )

    cluster_label_matrix = build_cluster_label_matrix(cluster_ids, labels, n_classes=10)
    optimal_cluster_to_label, optimal_correct = best_cluster_to_label_mapping(cluster_label_matrix)
    greedy_cluster_to_label, greedy_correct = greedy_cluster_to_label_mapping(cluster_label_matrix)
    optimal_confusion_matrix = aligned_confusion_matrix(cluster_ids, labels, optimal_cluster_to_label, n_classes=10)
    greedy_confusion_matrix = aligned_confusion_matrix(cluster_ids, labels, greedy_cluster_to_label, n_classes=10)
    optimal_accuracy = optimal_correct / max(len(labels), 1)
    greedy_accuracy = greedy_correct / max(len(labels), 1)
    nmi = normalized_mutual_information(cluster_ids, labels, n_classes=10)

    optimal_figure_path = out_dir / "confusion_matrix_optimal.png"
    greedy_figure_path = out_dir / "confusion_matrix_greedy.png"
    plot_confusion_matrix(
        optimal_figure_path,
        optimal_confusion_matrix,
        "Global One-to-One Matching",
        optimal_accuracy,
        nmi,
    )
    plot_confusion_matrix(
        greedy_figure_path,
        greedy_confusion_matrix,
        "Greedy Majority Mapping",
        greedy_accuracy,
        nmi,
    )

    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "01_resnet18_cifar10_kmeans",
        "model": "resnet18",
        "pretraining": "imagenet-1k",
        "method": "feature_kmeans",
        "uses_cifar10_labels_for_training": False,
        "uses_cifar10_labels_for_cluster_matching": True,
        "cifar10_split": "test",
        "dataset_root": str(dataset_root),
        "weight_path": str(weight_path),
        "num_samples": int(len(labels)),
        "feature_dim": int(features.shape[1]),
        "num_clusters": 10,
        "seed": args.seed,
        "kmeans_iters": args.kmeans_iters,
        "kmeans_restarts": args.kmeans_restarts,
        "kmeans_inertia": inertia,
        "nmi": nmi,
        "optimal_one_to_one": {
            "description": "全局一对一最佳匹配；每个 cluster 只能对应一个标签，每个标签也只能被一个 cluster 使用。",
            "cluster_to_label": {str(key): int(value) for key, value in optimal_cluster_to_label.items()},
            "matched_accuracy": optimal_accuracy,
            "matched_accuracy_percent": 100.0 * optimal_accuracy,
            "confusion_matrix_png": str(optimal_figure_path),
        },
        "greedy_majority": {
            "description": "逐 cluster 选择该 cluster 内样本数最多的真实标签；允许多个 cluster 映射到同一个标签。",
            "cluster_to_label": {str(key): int(value) for key, value in greedy_cluster_to_label.items()},
            "matched_accuracy": greedy_accuracy,
            "matched_accuracy_percent": 100.0 * greedy_accuracy,
            "confusion_matrix_png": str(greedy_figure_path),
        },
        "cifar10_classes": CIFAR10_CLASSES,
        "note": "KMeans 聚类过程没有使用 CIFAR-10 标签；标签只在评估阶段用于 cluster 到类别的映射和绘制混淆矩阵。",
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as writer:
        json.dump(metrics, writer, ensure_ascii=False, indent=2)
        writer.write("\n")

    print(f"[RESULT] optimal_one_to_one_accuracy={100.0 * optimal_accuracy:.2f}% ({optimal_correct}/{len(labels)})")
    print(f"[RESULT] greedy_majority_accuracy={100.0 * greedy_accuracy:.2f}% ({greedy_correct}/{len(labels)})")
    print(f"[RESULT] nmi={nmi:.4f}")
    print(f"[RESULT] optimal_confusion_matrix_png={optimal_figure_path}")
    print(f"[RESULT] greedy_confusion_matrix_png={greedy_figure_path}")


if __name__ == "__main__":
    main()
