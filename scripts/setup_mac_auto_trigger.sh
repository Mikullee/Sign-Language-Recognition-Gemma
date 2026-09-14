#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "錯誤：這個安裝腳本只供 macOS 使用。" >&2
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "錯誤：此套件固定的 MediaPipe 版本只支援 Apple Silicon（arm64），不支援 Intel Mac。" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "錯誤：找不到 python3；請先安裝 Python 3.10 以上版本。" >&2
  exit 1
fi

python3 - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 15)):
    raise SystemExit("錯誤：需要 Python 3.10–3.14。")
PY

python3 scripts/verify_mac_auto_trigger_package.py
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-transformer.txt
.venv/bin/python scripts/verify_mac_auto_trigger_package.py
.venv/bin/python - <<'PY'
from recognition.transformer.recognizer import Knee42TransformerRecognizer
recognizer = Knee42TransformerRecognizer("artifacts/realtime/best_current")
if len(recognizer.labels) != 42:
    raise SystemExit("錯誤：模型不是 42 類。")
print("模型自我檢查通過：42 類。")
PY

echo "安裝完成。執行 ./scripts/run_mac_auto_trigger.sh 啟動。"
