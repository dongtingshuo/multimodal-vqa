#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${REPO_ROOT}"
python scripts/validate_vqa_data.py --root "${DATA_ROOT}" --sample-images 20 --strict-full
python -m autodl.preflight \
  --config "${CONFIG_PATH}" \
  --checkpoint "${RUN_DIR}/latest.pt" \
  --data-root "${DATA_ROOT}" \
  --answer-vocab "${ANSWER_VOCAB}" \
  --require-data

python -u train.py \
  --config "${CONFIG_PATH}" \
  --device cuda \
  --data-root "${DATA_ROOT}" \
  --answer-vocab-path "${ANSWER_VOCAB}" \
  --checkpoint-dir "${RUN_DIR}" \
  --resume "${RUN_DIR}/latest.pt" \
  --epochs 10 \
  --no-wandb

python evaluate.py \
  --config "${CONFIG_PATH}" \
  --checkpoint "${RUN_DIR}/best.pt" \
  --device cuda \
  --data-root "${DATA_ROOT}" \
  --predictions-output "${RUN_DIR}/val_predictions.json"

cp "${ANSWER_VOCAB}" "${RUN_DIR}/answer_vocab.json"
tar -czf "${ARTIFACT_DIR}/vilt-seed42-autodl-results.tar.gz" -C "${RUN_DIR}" .
echo "Pipeline complete: ${ARTIFACT_DIR}/vilt-seed42-autodl-results.tar.gz"
