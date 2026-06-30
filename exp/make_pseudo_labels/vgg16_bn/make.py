#!/usr/bin/env python3
"""使用 VGG16_BN victim 模型构造伪标签数据集。"""

from __future__ import annotations

import sys
from pathlib import Path


MAKE_PSEUDO_LABELS_ROOT = Path(__file__).resolve().parents[1]
if str(MAKE_PSEUDO_LABELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MAKE_PSEUDO_LABELS_ROOT))

from common.labeler import ModelSpec, pseudo_label_main  # noqa: E402


SPEC = ModelSpec(
    name="vgg16_bn",
    display_name="VGG16_BN",
    factory_name="vgg16_bn",
)


if __name__ == "__main__":
    pseudo_label_main(SPEC)
