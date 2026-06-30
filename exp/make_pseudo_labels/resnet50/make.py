#!/usr/bin/env python3
"""使用 ResNet50 victim 模型构造伪标签数据集。"""

from __future__ import annotations

import sys
from pathlib import Path


MAKE_PSEUDO_LABELS_ROOT = Path(__file__).resolve().parents[1]
if str(MAKE_PSEUDO_LABELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MAKE_PSEUDO_LABELS_ROOT))

from common.labeler import ModelSpec, pseudo_label_main  # noqa: E402


SPEC = ModelSpec(
    name="resnet50",
    display_name="ResNet50",
    factory_name="resnet50",
)


if __name__ == "__main__":
    pseudo_label_main(SPEC)
