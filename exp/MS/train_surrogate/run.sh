#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 2 ]]; then
  printf '用法: %s <model> <dataset> [train.py 参数...]\n' "$0" >&2
  exit 2
fi

MODEL="$1"
DATASET="$2"
shift 2

exec python3 "${SCRIPT_DIR}/train.py" "${MODEL}" "${DATASET}" "$@"
