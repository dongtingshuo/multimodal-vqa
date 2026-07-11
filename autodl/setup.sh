#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

echo "Repository: ${REPO_ROOT}"
echo "Work root: ${WORK_ROOT}"
nvidia-smi
python - <<'PY'
import sys
import torch
import torchvision

if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"Expected Python 3.12, found {sys.version.split()[0]}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__} torchvision={torchvision.__version__} cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
PY

python -m pip install --no-cache-dir -r "${SCRIPT_DIR}/requirements.txt"

if [[ ! -f "${RESUME_SOURCE}/latest.pt" ]]; then
  echo "Missing packaged checkpoint: ${RESUME_SOURCE}/latest.pt" >&2
  exit 1
fi
if [[ ! -f "${WORK_ROOT}/resume/answer_vocab.json" ]]; then
  echo "Missing packaged answer vocabulary" >&2
  exit 1
fi

cp "${WORK_ROOT}/resume/answer_vocab.json" "${ANSWER_VOCAB}"
for name in latest.pt training_history.csv config.snapshot.json run_metadata.json training_curves.png; do
  if [[ -f "${RESUME_SOURCE}/${name}" && ! -f "${RUN_DIR}/${name}" ]]; then
    cp "${RESUME_SOURCE}/${name}" "${RUN_DIR}/${name}"
  fi
done
if [[ ! -f "${RUN_DIR}/best.pt" ]]; then
  cp "${RUN_DIR}/latest.pt" "${RUN_DIR}/best.pt"
fi

cd "${REPO_ROOT}"
python -m autodl.preflight \
  --config "${CONFIG_PATH}" \
  --checkpoint "${RUN_DIR}/latest.pt" \
  --data-root "${DATA_ROOT}" \
  --answer-vocab "${ANSWER_VOCAB}"

python - <<'PY'
from transformers import AutoTokenizer, ViltModel, ViltProcessor

AutoTokenizer.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
ViltModel.from_pretrained("dandelin/vilt-b32-mlm-itm")
print("Hugging Face ViLT assets cached")
PY

echo "Setup complete. Next: bash autodl/prepare_data.sh"
