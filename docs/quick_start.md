# Quick Start

## Windows 可攜版

1. 解壓 `SignLanguageRecognition-v0.1.0-windows-x64.zip`。
2. 啟動 iVCam，確認手機畫面已出現在 Windows。
3. 執行 `start_ivcam.cmd`。
4. 站在鏡頭前，讓上半身與雙手腕完整入鏡，雙手自然垂放身側。
5. 完成一句後將雙手放回身側；約 0.5 秒確認後會顯示結果並準備下一句。

Auto 模式不使用 Space 切段。`Q` 離開、`R` 重設、`S` 立即儲存日誌。

若 iVCam 不是攝影機 0，可編輯 `app_config.json` 的 `source`，或執行：

```powershell
SignLanguageRecognition.exe --source 1 --backend dshow
```

## 原始碼執行

```powershell
git clone https://github.com/Mikullee/Sign-Language-Recognition-Gemma.git
cd Sign-Language-Recognition-Gemma
conda create -n knee42 python=3.12 -y
conda activate knee42
python -m pip install -r requirements-transformer.txt
```

現行 42 類 Transformer 的 bundle 已附在 `artifacts/realtime/best_current/`，
可先確認它通過完整性驗證：

```powershell
python -c "from recognition.transformer.recognizer import Knee42TransformerRecognizer as R; r = R('artifacts/realtime/best_current'); print(len(r.labels), '類載入成功')"
```

### Legacy 27 類即時推論

```powershell
conda env create -f environment.yml
conda activate slr_runtime
python -m recognition.realtime.realtime_infer_daily30_sentence
```

CLI 會覆寫 `app_config.json`：

```powershell
python -m recognition.realtime.realtime_infer_daily30_sentence `
  --source 1 `
  --backend dshow `
  --trigger-mode auto `
  --save-log
```

手動切段仍可用 `--trigger-mode manual`，此模式才會顯示 Space 操作提示。

## 重新校準自動起訖

先將三支驗收影片放入 `data/videos/auto_trigger/`，或修改 CSV 中的相對路徑：

```powershell
python -m recognition.evaluation.eval_auto_trigger_boundaries `
  --annotations data/annotations/auto_trigger_three_videos.csv
```

評估會輸出逐片指標、摘要、最佳設定與 debug 影片。分類結果只用來比較人工與自動裁切，不參與切段參數選擇。
