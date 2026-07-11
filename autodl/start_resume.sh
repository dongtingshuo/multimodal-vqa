#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PID_FILE="${WORK_ROOT}/train.pid"
LOG_FILE="${LOG_DIR}/vilt-seed42.log"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if kill -0 "${existing_pid}" 2>/dev/null; then
    echo "Training is already running with PID ${existing_pid}."
    exit 1
  fi
  rm -f "${PID_FILE}"
fi

nohup setsid bash "${SCRIPT_DIR}/run_pipeline.sh" >>"${LOG_FILE}" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${PID_FILE}"
echo "Started AutoDL continuation pipeline with PID ${pid}."
echo "Log: ${LOG_FILE}"
echo "Monitor: bash autodl/status.sh"
