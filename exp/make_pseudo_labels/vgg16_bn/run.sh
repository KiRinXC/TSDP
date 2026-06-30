#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DATASET="${1:-${DATASET:-cifar10}}"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  shift
fi

MODEL_SCRIPT="${SCRIPT_DIR}/make.py"
MODEL_NAME="vgg16_bn"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/dataset/public}"
DERIVED_ROOT="${DERIVED_ROOT:-${REPO_ROOT}/dataset/derived}"
PSEUDO_LABEL_ROOT="${PSEUDO_LABEL_ROOT:-${REPO_ROOT}/dataset/pseudo_labels}"
VICTIM_WEIGHT_PATH="${VICTIM_WEIGHT_PATH:-${REPO_ROOT}/weights/victim/${MODEL_NAME}/${DATASET}/target.pth}"

CMD=(
  python3 "${MODEL_SCRIPT}"
  --dataset "${DATASET}"
  --split "${SPLIT:-eval}"
  --dataset-root "${DATASET_ROOT}"
  --derived-root "${DERIVED_ROOT}"
  --pseudo-label-root "${PSEUDO_LABEL_ROOT}"
  --victim-weight-path "${VICTIM_WEIGHT_PATH}"
  --batch-size "${BATCH_SIZE:-256}"
  --num-workers "${NUM_WORKERS:-4}"
  --device "${DEVICE:-auto}"
  --seed "${SEED:-42}"
)

if [[ -n "${QUERY_MANIFEST:-}" ]]; then
  CMD+=(--query-manifest "${QUERY_MANIFEST}")
fi
if [[ -n "${OUT_DIR:-}" ]]; then
  CMD+=(--out-dir "${OUT_DIR}")
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  CMD+=(--force)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  CMD+=(--dry-run)
fi
CMD+=("$@")

printf '将要执行的命令:'
for arg in "${CMD[@]}"; do
  printf ' %q' "${arg}"
done
printf '\n'

exec "${CMD[@]}"
