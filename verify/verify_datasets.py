#!/usr/bin/env python3
"""验证本地公开数据集目录结构和样本数量。"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset" / "public"


EXPECTED_ARCHIVE_MD5 = {
    "cifar-10-python.tar.gz": "c58f30108f718f92721af3b95e74349a",
    "cifar-100-python.tar.gz": "eb9058c3a382ffc7106e4002c42a8d85",
    "stl10_binary.tar.gz": "91f7769df0f17e558f3565bffb0c7dfb",
    "tiny-imagenet-200.zip": "90528d7ca1a48142e341f4ef8d21d0de",
}


def md5sum(path: Path) -> str:
    """按块计算 MD5，避免一次性把大压缩包读入内存。"""
    digest = hashlib.md5()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle(path: Path):
    """读取 CIFAR 官方 Python pickle 文件。"""
    with path.open("rb") as reader:
        return pickle.load(reader, encoding="latin1")


class Verifier:
    """收集所有检查错误，尽量一次性报告完整问题列表。"""

    def __init__(self, dataset_root: Path, check_archives: bool) -> None:
        self.dataset_root = dataset_root
        self.check_archives = check_archives
        self.errors: list[str] = []

    def ok(self, message: str) -> None:
        print(f"[OK] {message}")

    def fail(self, message: str) -> None:
        self.errors.append(message)
        print(f"[FAIL] {message}")

    def require_dir(self, path: Path) -> bool:
        if path.is_dir():
            return True
        self.fail(f"missing directory: {path}")
        return False

    def require_file(self, path: Path) -> bool:
        if path.is_file():
            return True
        self.fail(f"missing file: {path}")
        return False

    def expect_equal(self, label: str, actual: int | str, expected: int | str) -> None:
        if actual == expected:
            self.ok(f"{label}: {actual}")
        else:
            self.fail(f"{label}: expected {expected}, got {actual}")

    def verify_archives(self) -> None:
        if not self.check_archives:
            self.ok("archive MD5 checks skipped")
            return

        archive_root = self.dataset_root / "_archives"
        if not self.require_dir(archive_root):
            return

        # 压缩包校验用于发现下载中断、代理返回错误页等问题。
        for filename, expected in EXPECTED_ARCHIVE_MD5.items():
            archive = archive_root / filename
            if not self.require_file(archive):
                continue
            actual = md5sum(archive)
            self.expect_equal(f"archive md5 {filename}", actual, expected)

    def verify_cifar10(self) -> None:
        root = self.dataset_root / "cifar10" / "cifar-10-batches-py"
        if not self.require_dir(root):
            return

        required_files = [
            "batches.meta",
            "data_batch_1",
            "data_batch_2",
            "data_batch_3",
            "data_batch_4",
            "data_batch_5",
            "test_batch",
        ]
        if not all(self.require_file(root / name) for name in required_files):
            return

        # CIFAR-10 的五个 data_batch 文件每个包含 10000 张训练图像。
        train_count = 0
        sample_width = None
        for index in range(1, 6):
            batch = load_pickle(root / f"data_batch_{index}")
            train_count += len(batch["data"])
            if sample_width is None and len(batch["data"]) > 0:
                sample_width = len(batch["data"][0])

        test_batch = load_pickle(root / "test_batch")
        meta = load_pickle(root / "batches.meta")

        self.expect_equal("CIFAR-10 train images", train_count, 50000)
        self.expect_equal("CIFAR-10 test images", len(test_batch["data"]), 10000)
        self.expect_equal("CIFAR-10 classes", len(meta["label_names"]), 10)
        self.expect_equal("CIFAR-10 flattened image width", sample_width, 3 * 32 * 32)

    def verify_cifar100(self) -> None:
        root = self.dataset_root / "cifar100" / "cifar-100-python"
        if not self.require_dir(root):
            return

        required_files = ["train", "test", "meta"]
        if not all(self.require_file(root / name) for name in required_files):
            return

        train = load_pickle(root / "train")
        test = load_pickle(root / "test")
        meta = load_pickle(root / "meta")
        sample_width = len(train["data"][0]) if len(train["data"]) else None

        self.expect_equal("CIFAR-100 train images", len(train["data"]), 50000)
        self.expect_equal("CIFAR-100 test images", len(test["data"]), 10000)
        self.expect_equal("CIFAR-100 fine classes", len(meta["fine_label_names"]), 100)
        self.expect_equal("CIFAR-100 coarse classes", len(meta["coarse_label_names"]), 20)
        self.expect_equal("CIFAR-100 flattened image width", sample_width, 3 * 32 * 32)

    def verify_stl10(self) -> None:
        root = self.dataset_root / "stl10" / "stl10_binary"
        if not self.require_dir(root):
            return

        required_files = [
            "class_names.txt",
            "fold_indices.txt",
            "test_X.bin",
            "test_y.bin",
            "train_X.bin",
            "train_y.bin",
            "unlabeled_X.bin",
        ]
        if not all(self.require_file(root / name) for name in required_files):
            return

        bytes_per_image = 3 * 96 * 96

        # STL-10 标签是单字节标签，因此 y 文件大小等于标签数量。
        train_images = (root / "train_X.bin").stat().st_size // bytes_per_image
        test_images = (root / "test_X.bin").stat().st_size // bytes_per_image
        unlabeled_images = (root / "unlabeled_X.bin").stat().st_size // bytes_per_image
        train_labels = (root / "train_y.bin").stat().st_size
        test_labels = (root / "test_y.bin").stat().st_size

        with (root / "class_names.txt").open("r", encoding="utf-8") as reader:
            classes = [line.strip() for line in reader if line.strip()]

        self.expect_equal("STL-10 train images", train_images, 5000)
        self.expect_equal("STL-10 train labels", train_labels, 5000)
        self.expect_equal("STL-10 test images", test_images, 8000)
        self.expect_equal("STL-10 test labels", test_labels, 8000)
        self.expect_equal("STL-10 unlabeled images", unlabeled_images, 100000)
        self.expect_equal("STL-10 classes", len(classes), 10)

    def verify_tiny_imagenet(self) -> None:
        root = self.dataset_root / "tiny-imagenet-200"
        if not self.require_dir(root):
            return

        train_root = root / "train"
        val_root = root / "val"
        val2_root = root / "val2"
        test_images_root = root / "test" / "images"

        required_dirs = [train_root, val_root, test_images_root]
        required_files = [root / "wnids.txt", root / "words.txt"]
        if not all(self.require_dir(path) for path in required_dirs):
            return
        if not all(self.require_file(path) for path in required_files):
            return

        train_classes = sorted(path for path in train_root.iterdir() if path.is_dir())
        val_classes = sorted(path for path in val_root.iterdir() if path.is_dir())
        train_images = self.count_images(train_root)
        val_images = self.count_images(val_root)
        test_images = self.count_images(test_images_root)

        self.expect_equal("Tiny-ImageNet train classes", len(train_classes), 200)
        self.expect_equal("Tiny-ImageNet train images", train_images, 100000)
        self.expect_equal("Tiny-ImageNet val classes", len(val_classes), 200)
        self.expect_equal("Tiny-ImageNet val images", val_images, 10000)
        self.expect_equal("Tiny-ImageNet test images", test_images, 10000)

        with (root / "wnids.txt").open("r", encoding="utf-8") as reader:
            wnids = [line.strip() for line in reader if line.strip()]
        self.expect_equal("Tiny-ImageNet wnids", len(wnids), 200)

        # 不同实验脚本可能使用 val 或 val2，因此这里要求 val2 可用。
        if val2_root.exists():
            val2_images = self.count_images(val2_root)
            self.expect_equal("Tiny-ImageNet val2 images", val2_images, 10000)
            if val2_root.is_symlink():
                self.expect_equal("Tiny-ImageNet val2 symlink target", val2_root.readlink().as_posix(), "val")
            else:
                self.ok("Tiny-ImageNet val2 exists as a real directory")
        else:
            self.fail(f"missing val2 compatibility path: {val2_root}")

        # 原始 train/<class>/images 目录应该已经被拍平成 train/<class>/*.JPEG。
        nested_train_images_dirs = list(train_root.glob("*/images"))
        self.expect_equal("Tiny-ImageNet nested train images dirs", len(nested_train_images_dirs), 0)

    @staticmethod
    def count_images(root: Path) -> int:
        return sum(1 for path in root.rglob("*.JPEG") if path.is_file())

    def run(self) -> int:
        if not self.require_dir(self.dataset_root):
            return 1

        self.verify_archives()
        self.verify_cifar10()
        self.verify_cifar100()
        self.verify_stl10()
        self.verify_tiny_imagenet()

        if self.errors:
            print()
            print(f"Dataset verification failed with {len(self.errors)} error(s).")
            return 1

        print()
        print("Dataset verification passed.")
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify public datasets.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"公开数据集根目录，默认：{DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--skip-archives",
        action="store_true",
        help="skip archive MD5 checks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verifier = Verifier(
        dataset_root=args.dataset_root.resolve(),
        check_archives=not args.skip_archives,
    )
    return verifier.run()


if __name__ == "__main__":
    sys.exit(main())
