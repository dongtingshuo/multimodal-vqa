#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PID_FILE="${WORK_ROOT}/train.pid"
LOG_FILE="${LOG_DIR}/vilt-seed42.log"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "status=running pid=$(cat "${PID_FILE}")"
else
  echo "status=stopped"
fi
nvidia-smi
if [[ -f "${LOG_FILE}" ]]; then
  tail -n 80 "${LOG_FILE}"
else
  echo "Log file has not been created: ${LOG_FILE}"
fi
