#!/usr/bin/env python3
"""从 PG01 派生按特征图元素数归一化的残差乘积分数。"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playground.normalize import run_normalization  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_normalization("feature"))
