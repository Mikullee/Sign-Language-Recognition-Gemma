# Quick Start

## 1. Clone

```powershell
git clone <your-repo-url>
cd Sign-Language-Recognition-Gemma-preview
```

## 2. Install

### Conda

```powershell
conda env create -f environment.yml
conda activate slr_preview
```

### Pip

```powershell
pip install -r requirements.txt
```

## 3. Check required assets

確認這些檔案存在：

- `models/hand_landmarker.task`
- `models/pose_landmarker.task`
- `artifacts/realtime/best_current/best_model.pt`
- `artifacts/realtime/best_current/label_map_v1.json`
- `artifacts/realtime/best_current/train_summary_v1.json`
- `artifacts/realtime/best_current/launch_summary.json`
- `artifacts/realtime/best_current/fixed_sentence_templates_daily30.csv`

## 4. Run realtime demo

### Webcam

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_realtime_daily30_sentence.ps1 --source 0 --trigger-mode manual --save-log
```

### MP4

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_realtime_daily30_sentence.ps1 --source "C:\path\to\demo.mp4" --trigger-mode manual --save-log
```

## 5. Optional environment overrides

- `SLR_PYTHON`: 指定 PowerShell wrapper 要用哪個 Python
- `SLR_MODELS_DIR`: 覆蓋 `models/` 位置
- `SLR_RUNTIME_BUNDLE_DIR`: 覆蓋 runtime bundle 位置
- `SLR_RESULTS_DIR`: 覆蓋 log 與輸出位置
- `SLR_DISABLE_REMOTE_FETCH=1`: 禁止 fallback 遠端抓取 artifact
