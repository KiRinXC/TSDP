#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

DATASET="${1:-${DATASET:-c10}}"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  shift
fi

MODEL_SCRIPT="${SCRIPT_DIR}/train.py"
MODEL_NAME="resnet18"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/dataset/public}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/weights/MS/victim/${MODEL_NAME}/${DATASET}}"
WEIGHT_PATH="${WEIGHT_PATH:-${REPO_ROOT}/weights/pre_train/resnet18-5c106cde.pth}"

CMD=(
  python3 "${MODEL_SCRIPT}"
  --dataset "${DATASET}"
  --dataset-root "${DATASET_ROOT}"
  --out-dir "${OUT_DIR}"
  --weight-path "${WEIGHT_PATH}"
  --epochs "${EPOCHS:-100}"
  --batch-size "${BATCH_SIZE:-64}"
  --lr "${LR:-0.1}"
  --momentum "${MOMENTUM:-0.5}"
  --weight-decay "${WEIGHT_DECAY:-0.0005}"
  --lr-step "${LR_STEP:-60}"
  --lr-gamma "${LR_GAMMA:-0.1}"
  --num-workers "${NUM_WORKERS:-10}"
  --device "${DEVICE:-auto}"
  --seed "${SEED:-42}"
)

if [[ "${QUICK:-0}" == "1" ]]; then
  CMD+=(--quick)
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
