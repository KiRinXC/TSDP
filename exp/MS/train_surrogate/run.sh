#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${HOME}/venvs/dl-py310-torch210-cu121/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  printf '错误: 找不到 TSDP 唯一 Python 环境: %s\n' "${PYTHON}" >&2
  exit 1
fi

if [[ $# -lt 2 ]]; then
  printf '用法: %s <model> <dataset> [train.py 参数...]\n' "$0" >&2
  exit 2
fi

MODEL="$1"
DATASET="$2"
shift 2

exec "${PYTHON}" "${SCRIPT_DIR}/train.py" "${MODEL}" "${DATASET}" "$@"
