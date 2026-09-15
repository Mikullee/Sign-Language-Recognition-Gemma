#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "錯誤：尚未安裝。請先執行 ./scripts/setup_mac_auto_trigger.sh" >&2
  exit 1
fi

.venv/bin/python scripts/verify_mac_auto_trigger_package.py
exec .venv/bin/python -m webservice.server \
  --host 127.0.0.1 \
  --port "${PORT:-8642}" \
  --http \
  --trigger-config configs/auto_trigger_knee_web_live.json
