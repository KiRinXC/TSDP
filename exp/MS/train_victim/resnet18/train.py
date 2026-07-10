#!/usr/bin/env python3
"""训练 ResNet18 受害者模型。"""

from __future__ import annotations

import sys
from pathlib import Path


TRAIN_VICTIM_ROOT = Path(__file__).resolve().parents[1]
if str(TRAIN_VICTIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_VICTIM_ROOT))

from common.trainer import ModelSpec, train_main  # noqa: E402


SPEC = ModelSpec(
    name="resnet18",
    display_name="ResNet18",
    factory_name="resnet18",
    weight_filename="resnet18-5c106cde.pth",
)


if __name__ == "__main__":
    train_main(SPEC)
