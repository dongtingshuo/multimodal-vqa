#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -d "${REPO_ROOT}/../resume" ]]; then
  DEFAULT_WORK_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
else
  DEFAULT_WORK_ROOT="/root/autodl-tmp/multimodal-vqa-autodl"
fi

WORK_ROOT="${AUTODL_WORK_ROOT:-${DEFAULT_WORK_ROOT}}"
DATA_ROOT="${WORK_ROOT}/data/vqa"
ANSWER_VOCAB="${WORK_ROOT}/answer_vocab.json"
RESUME_SOURCE="${WORK_ROOT}/resume/vilt-seed42"
RUN_DIR="${WORK_ROOT}/runs/vilt-seed42"
ARTIFACT_DIR="${WORK_ROOT}/artifacts"
LOG_DIR="${WORK_ROOT}/logs"
CONFIG_PATH="${REPO_ROOT}/autodl/configs/vilt_resume_4090d.yaml"

export HF_HOME="${HF_HOME:-${WORK_ROOT}/cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="disabled"

mkdir -p "${RUN_DIR}" "${ARTIFACT_DIR}" "${LOG_DIR}" "${HF_HOME}"
