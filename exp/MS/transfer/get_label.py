#!/usr/bin/env python3
"""Query a best victim model and persist canonical MS pseudo labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_VICTIM_ROOT = REPO_ROOT / "exp" / "MS" / "train_victim"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRAIN_VICTIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_VICTIM_ROOT))

from common.trainer import build_public_split_dataset, build_transforms, resolve_dataset_name  # noqa: E402
from models import imagenet as imagenet_models  # noqa: E402


MODEL_FACTORIES = {
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "vgg16_bn": "vgg16_bn",
    "mobilenetv2": "mobilenetv2",
}
NUM_CLASSES = {"c10": 10, "c100": 100, "s10": 10, "t200": 200}
LABEL_FIELDS = [
    "query_rank",
    "record_id",
    "source_split",
    "source_index",
    "global_index",
    "pseudo_label",
    "confidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 best victim 权重生成 canonical MS 伪标签。")
    parser.add_argument("model", choices=sorted(MODEL_FACTORIES), help="victim 模型名。")
    parser.add_argument("dataset", help="数据集 id：c10/c100/s10/t200。")
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "dataset" / "public"), help="公开数据集根目录。")
    parser.add_argument("--protocol-root", default=str(REPO_ROOT / "dataset" / "MS"), help="MS 协议根目录。")
    parser.add_argument(
        "--weights-root",
        default=str(REPO_ROOT / "weights" / "MS" / "victim"),
        help="victim 权重根目录。",
    )
    parser.add_argument("--checkpoint", default=None, help="可选的 best.pth 显式路径。")
    parser.add_argument("--batch-size", type=int, default=128, help="推理 batch size。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 labels.tsv、posteriors.pt 和 manifest.json。")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境没有可用 CUDA 设备。")
    return device


def read_query_rows(splits_path: Path) -> list[dict[str, str]]:
    if not splits_path.is_file():
        raise FileNotFoundError(f"找不到 MS 划分文件：{splits_path}")
    required = {"record_id", "split", "source_split", "source_index", "global_index", "query_rank"}
    with splits_path.open("r", newline="", encoding="utf-8") as reader_file:
        reader = csv.DictReader(reader_file, delimiter="\t")
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{splits_path} 缺少字段：{sorted(missing)}")
        rows = [row for row in reader if row["split"] == "query_pool_ms"]
    if not rows:
        raise ValueError(f"{splits_path} 不包含 query_pool_ms")
    if any(row["source_split"] != "official_train" for row in rows):
        raise ValueError("query_pool_ms 必须来自 official_train。")
    try:
        rows.sort(key=lambda row: int(row["query_rank"]))
        ranks = [int(row["query_rank"]) for row in rows]
        source_indices = [int(row["source_index"]) for row in rows]
    except ValueError as exc:
        raise ValueError(f"query_pool_ms 的索引或 query_rank 非法：{exc}") from exc
    if ranks != list(range(len(rows))):
        raise ValueError("query_pool_ms 的 query_rank 必须连续且从 0 开始。")
    if len(source_indices) != len(set(source_indices)):
        raise ValueError("query_pool_ms 包含重复 source_index。")
    return rows


def checkpoint_path(args: argparse.Namespace, dataset: str) -> Path:
    path = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else Path(args.weights_root).expanduser().resolve() / args.model / dataset / "best.pth"
    )
    if path.name != "best.pth":
        raise ValueError("伪标签只能使用最佳验证权重 best.pth。")
    if not path.is_file():
        raise FileNotFoundError(f"找不到 victim 最佳权重：{path}")
    return path


def load_model(model_name: str, num_classes: int, path: Path, device: torch.device):
    factory = getattr(imagenet_models, MODEL_FACTORIES[model_name])
    model = factory(num_classes=num_classes)
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"{path} 不包含模型 state_dict。")
    normalized = {
        (key.removeprefix("module.") if key.startswith("module.") else key): value for key, value in state_dict.items()
    }
    model.load_state_dict(normalized, strict=True)
    return model.to(device).eval()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        while chunk := reader.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, str]],
    pseudo_labels: list[int],
    confidences: list[float],
    posteriors: torch.Tensor,
    model_name: str,
    dataset: str,
    checkpoint: Path,
    splits_path: Path,
) -> None:
    labels_path = output_dir / "labels.tsv"
    with labels_path.open("w", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(writer_file, fieldnames=LABEL_FIELDS, delimiter="\t")
        writer.writeheader()
        for row, label, confidence in zip(rows, pseudo_labels, confidences, strict=True):
            writer.writerow(
                {
                    "query_rank": row["query_rank"],
                    "record_id": row["record_id"],
                    "source_split": row["source_split"],
                    "source_index": row["source_index"],
                    "global_index": row["global_index"],
                    "pseudo_label": label,
                    "confidence": f"{confidence:.8f}",
                }
            )
    posteriors_path = output_dir / "posteriors.pt"
    torch.save(
        {
            "schema_version": 1,
            "protocol": "MS",
            "dataset": dataset,
            "model": model_name,
            "query_split": "query_pool_ms",
            "posteriors": posteriors,
            "pseudo_labels": torch.tensor(pseudo_labels, dtype=torch.long),
        },
        posteriors_path,
    )
    manifest = {
        "schema_version": 1,
        "protocol": "MS",
        "dataset": dataset,
        "model": model_name,
        "query": {
            "split": "query_pool_ms",
            "count": len(rows),
            "splits_path": relative_to_repo(splits_path),
        },
        "victim": {
            "checkpoint": relative_to_repo(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
        },
        "outputs": {"labels": "labels.tsv", "posteriors": "posteriors.pt"},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size 必须为正数，num-workers 不能为负数。")
    dataset = resolve_dataset_name(args.dataset)
    protocol_root = Path(args.protocol_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    splits_path = protocol_root / dataset / "splits.tsv"
    query_rows = read_query_rows(splits_path)
    query_indices = [int(row["source_index"]) for row in query_rows]
    checkpoint = checkpoint_path(args, dataset)
    output_dir = protocol_root / dataset / args.model
    outputs = [output_dir / name for name in ("labels.tsv", "posteriors.pt", "manifest.json")]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"伪标签产物已存在：{existing[0]}。使用 --overwrite 重新生成。")
    if args.overwrite:
        for path in outputs:
            path.unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, test_transform = build_transforms(dataset)
    train_dataset = build_public_split_dataset(dataset, dataset_root, "train", test_transform)
    if any(index < 0 or index >= len(train_dataset) for index in query_indices):
        raise ValueError("query_pool_ms 包含超出公开训练集范围的索引。")
    loader = DataLoader(
        Subset(train_dataset, query_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=resolve_device(args.device).type == "cuda",
    )
    device = resolve_device(args.device)
    model = load_model(args.model, NUM_CLASSES[dataset], checkpoint, device)
    posterior_batches: list[torch.Tensor] = []
    pseudo_labels: list[int] = []
    confidences: list[float] = []
    for inputs, _ in loader:
        logits = model(inputs.to(device, non_blocking=True))
        probabilities = functional.softmax(logits, dim=1)
        confidence, labels = probabilities.max(dim=1)
        posterior_batches.append(probabilities.cpu())
        pseudo_labels.extend(labels.cpu().tolist())
        confidences.extend(confidence.cpu().tolist())
    posteriors = torch.cat(posterior_batches, dim=0)
    if len(pseudo_labels) != len(query_rows) or posteriors.shape != (len(query_rows), NUM_CLASSES[dataset]):
        raise RuntimeError("伪标签数量或 posterior 形状与 query_pool_ms 不一致。")
    write_outputs(output_dir, query_rows, pseudo_labels, confidences, posteriors, args.model, dataset, checkpoint, splits_path)
    print(
        f"[INFO] {args.model}+{dataset}: 写入 {len(query_rows)} 条 query_pool_ms 伪标签至 {output_dir} "
        f"(checkpoint={checkpoint})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
