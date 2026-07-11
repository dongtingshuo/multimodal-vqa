#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${REPO_ROOT}"
python scripts/prepare_vqa_data.py \
  --root "${DATA_ROOT}" \
  --answer-vocab-path "${ANSWER_VOCAB}" \
  --answer-vocab-size 3000 \
  --full-coco-images

python scripts/validate_vqa_data.py \
  --root "${DATA_ROOT}" \
  --sample-images 100 \
  --strict-full

if [[ "${KEEP_DATA_ARCHIVES:-0}" != "1" ]]; then
  find "${DATA_ROOT}/downloads" -maxdepth 1 -type f -name '*.zip' -delete
fi

echo "Data preparation and strict validation complete."
echo "Next: bash autodl/start_resume.sh"
