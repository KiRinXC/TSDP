#!/usr/bin/env python3
"""用 ImageNet 预训练 ResNet18 特征对 CIFAR-100 做无监督 KMeans 探测。"""

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


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 100


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="提取 ImageNet 预训练 ResNet18 特征，不用 CIFAR-100 标签训练，使用 KMeans 检查 100 类样本是否自然可分。"
    )
    parser.add_argument(
        "--dataset-root",
        default=str(REPO_ROOT / "dataset" / "public" / "c100"),
        help="torchvision CIFAR100 数据根目录",
    )
    parser.add_argument(
        "--weight-path",
        default=str(REPO_ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"),
        help="官方 torchvision ResNet18 ImageNet-1k 权重路径",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "results" / "lab" / "01_kmeans"),
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
    """读取 CIFAR-100 测试集。"""
    dataset = tv_datasets.CIFAR100(
        root=str(dataset_root),
        train=False,
        download=False,
        transform=build_transform(),
    )
    class_names = list(dataset.classes)
    selected_dataset = dataset
    if limit is not None and limit > 0 and limit < len(dataset):
        selected_dataset = Subset(dataset, list(range(limit)))
    return selected_dataset, class_names


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
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """使用 Torch 实现 KMeans，在 GPU 上处理 CIFAR-100 的 100 个聚类。"""
    rng = np.random.default_rng(seed)
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_inertia = float("inf")
    feature_tensor = torch.from_numpy(features).to(device)
    n_samples, feature_dim = feature_tensor.shape

    for restart in range(restarts):
        init_indices = rng.choice(n_samples, size=n_clusters, replace=False)
        centers = feature_tensor[torch.from_numpy(init_indices).to(device)].clone()

        for _ in range(max_iters):
            distances = squared_euclidean(feature_tensor, centers)
            labels = distances.argmin(dim=1)
            counts = torch.bincount(labels, minlength=n_clusters)
            sums = torch.zeros((n_clusters, feature_dim), device=device, dtype=feature_tensor.dtype)
            sums.index_add_(0, labels, feature_tensor)
            new_centers = sums / counts.clamp_min(1).unsqueeze(1)
            empty = counts == 0
            if empty.any():
                farthest = distances.min(dim=1).values.topk(int(empty.sum().item())).indices
                new_centers[empty] = feature_tensor[farthest]

            center_shift = float(torch.linalg.vector_norm(new_centers - centers).item())
            centers = new_centers
            if center_shift < tol:
                break

        final_distances = squared_euclidean(feature_tensor, centers)
        labels = final_distances.argmin(dim=1)
        inertia = float(final_distances.gather(1, labels.unsqueeze(1)).sum().item())
        print(f"[KMEANS] restart={restart + 1}/{restarts} inertia={inertia:.2f}")
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.cpu().numpy()
            best_centers = centers.cpu().numpy()

    if best_labels is None or best_centers is None:
        raise RuntimeError("KMeans 没有产生有效结果。")
    return best_labels, best_centers, best_inertia


def squared_euclidean(features: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """计算样本到中心的平方欧氏距离。"""
    feature_norm = torch.sum(features * features, dim=1, keepdim=True)
    center_norm = torch.sum(centers * centers, dim=1).unsqueeze(0)
    distances = feature_norm + center_norm - 2.0 * features @ centers.T
    return distances.clamp_min_(0.0)


def build_cluster_label_matrix(cluster_ids: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """统计行=cluster、列=真实标签的计数矩阵。"""
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for cluster_index, label_index in zip(cluster_ids, labels):
        matrix[int(cluster_index), int(label_index)] += 1
    return matrix


def best_cluster_to_label_mapping(cluster_label_matrix: np.ndarray) -> tuple[dict[int, int], int]:
    """使用 O(n^3) Hungarian 算法求 cluster 到 label 的一对一最佳匹配。"""
    n_clusters, n_labels = cluster_label_matrix.shape
    if n_clusters != n_labels:
        raise ValueError("当前匹配实现要求 cluster 数和 label 数相同。")
    cost = cluster_label_matrix.max() - cluster_label_matrix
    u = np.zeros(n_clusters + 1, dtype=np.float64)
    v = np.zeros(n_labels + 1, dtype=np.float64)
    matched_row = np.zeros(n_labels + 1, dtype=np.int64)
    previous_col = np.zeros(n_labels + 1, dtype=np.int64)
    for row in range(1, n_clusters + 1):
        matched_row[0] = row
        col0 = 0
        min_value = np.full(n_labels + 1, np.inf)
        used = np.zeros(n_labels + 1, dtype=bool)
        while True:
            used[col0] = True
            row0 = matched_row[col0]
            delta = np.inf
            col1 = 0
            for col in range(1, n_labels + 1):
                if used[col]:
                    continue
                current = float(cost[row0 - 1, col - 1]) - u[row0] - v[col]
                if current < min_value[col]:
                    min_value[col] = current
                    previous_col[col] = col0
                if min_value[col] < delta:
                    delta = min_value[col]
                    col1 = col
            for col in range(n_labels + 1):
                if used[col]:
                    u[matched_row[col]] += delta
                    v[col] -= delta
                else:
                    min_value[col] -= delta
            col0 = col1
            if matched_row[col0] == 0:
                break
        while True:
            col1 = previous_col[col0]
            matched_row[col0] = matched_row[col1]
            col0 = col1
            if col0 == 0:
                break

    mapping = {int(matched_row[col] - 1): col - 1 for col in range(1, n_labels + 1)}
    score = sum(int(cluster_label_matrix[row, col]) for row, col in mapping.items())
    return mapping, score


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
    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(f"{title}\nACC={accuracy * 100:.2f}%  NMI={nmi:.3f}")
    ax.set_xlabel("matched cluster label")
    ax.set_ylabel("true CIFAR-100 label")
    ticks = np.arange(0, matrix.shape[0], 10)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(index) for index in ticks])
    ax.set_yticklabels([str(index) for index in ticks])

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

    dataset, class_names = build_dataset(dataset_root, args.limit)
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
    print("[INFO] 方法: 提取 512 维特征，KMeans(k=100)，评估阶段再用标签做最佳匹配")
    print(f"[INFO] 随机种子: {args.seed}")
    print(f"[INFO] 输出目录: {out_dir}")
    print(f"[INFO] 设备: {device}")

    raw_features, labels = extract_features(model=model, loader=loader, device=device)
    features = standardize_features(raw_features)
    cluster_ids, _centers, inertia = kmeans(
        features=features,
        n_clusters=NUM_CLASSES,
        seed=args.seed,
        max_iters=args.kmeans_iters,
        restarts=args.kmeans_restarts,
        tol=args.tol,
        device=device,
    )

    cluster_label_matrix = build_cluster_label_matrix(cluster_ids, labels, n_classes=NUM_CLASSES)
    optimal_cluster_to_label, optimal_correct = best_cluster_to_label_mapping(cluster_label_matrix)
    greedy_cluster_to_label, greedy_correct = greedy_cluster_to_label_mapping(cluster_label_matrix)
    optimal_confusion_matrix = aligned_confusion_matrix(cluster_ids, labels, optimal_cluster_to_label, n_classes=NUM_CLASSES)
    greedy_confusion_matrix = aligned_confusion_matrix(cluster_ids, labels, greedy_cluster_to_label, n_classes=NUM_CLASSES)
    optimal_accuracy = optimal_correct / max(len(labels), 1)
    greedy_accuracy = greedy_correct / max(len(labels), 1)
    nmi = normalized_mutual_information(cluster_ids, labels, n_classes=NUM_CLASSES)

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
        "experiment": "01_kmeans",
        "model": "resnet18",
        "pretraining": "imagenet-1k",
        "method": "feature_kmeans",
        "dataset": "c100",
        "uses_labels_for_training": False,
        "uses_labels_for_cluster_matching": True,
        "public_split": "test",
        "dataset_root": str(dataset_root),
        "weight_path": str(weight_path),
        "num_samples": int(len(labels)),
        "feature_dim": int(features.shape[1]),
        "num_clusters": NUM_CLASSES,
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
        "class_names": class_names,
        "note": "KMeans 聚类过程没有使用 CIFAR-100 标签；标签只在评估阶段用于 cluster 到类别的映射和绘制混淆矩阵。",
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
