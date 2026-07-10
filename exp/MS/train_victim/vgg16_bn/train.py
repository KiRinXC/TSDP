#!/usr/bin/env python3
"""训练 VGG16_BN 受害者模型。"""

from __future__ import annotations

import sys
from pathlib import Path


TRAIN_VICTIM_ROOT = Path(__file__).resolve().parents[1]
if str(TRAIN_VICTIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_VICTIM_ROOT))

from common.trainer import ModelSpec, train_main  # noqa: E402


SPEC = ModelSpec(
    name="vgg16_bn",
    display_name="VGG16_BN",
    factory_name="vgg16_bn",
    weight_filename="vgg16_bn-6c64b313.pth",
)


if __name__ == "__main__":
    train_main(SPEC)
