#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PID_FILE="${WORK_ROOT}/train.pid"
if [[ ! -f "${PID_FILE}" ]]; then
  echo "No training PID file found."
  exit 0
fi

pid="$(cat "${PID_FILE}")"
if kill -0 "${pid}" 2>/dev/null; then
  kill -TERM -- "-${pid}"
  echo "Sent SIGTERM to pipeline PID ${pid}. A partial epoch is not checkpointed."
else
  echo "Pipeline PID ${pid} is not running."
fi
rm -f "${PID_FILE}"
