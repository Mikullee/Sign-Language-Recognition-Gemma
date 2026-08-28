# Quick Start

## 安裝

```powershell
git clone https://github.com/Mikullee/Sign-Language-Recognition-Gemma.git
cd Sign-Language-Recognition-Gemma
conda create -n knee42 python=3.12 -y
conda activate knee42
python -m pip install -r requirements-transformer.txt
```

42 類 Transformer 的 bundle 已附在 `artifacts/realtime/best_current/`，
先確認它通過完整性驗證：

```powershell
python -c "from recognition.transformer.recognizer import Knee42TransformerRecognizer as R; r = R('artifacts/realtime/best_current'); print(len(r.labels), '類載入成功')"
```

再取得兩個 MediaPipe `.task`：

```powershell
python scripts/fetch_mediapipe_models.py
```

它們不隨 repository 散布（Google 的模型資產有自己的條款，見
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)），腳本會從官方位置下載
並核對 SHA-256——**pose 必須是 lite 版**，抓錯不會報錯只會讓準確率下降。

## 用網頁測試站

```powershell
python -m webservice.server --port 8642
```

瀏覽器開 `https://<主機>:8642`——**必須 https**，攝影機 API 只在安全來源啟用。
第一次會跳自簽憑證警告，進階 → 繼續前往。

- **攝影機**：按住空白鍵錄 1–3 秒，放開出結果。MediaPipe 在瀏覽器端跑，畫面不外傳。
- **上傳影片**：≤ 200 MB／≤ 180 秒，自動切段，每段給 top-3。
- 比劃時**兩邊肩膀都要在畫面內**（要靠雙肩做正規化）。

攝影機模式另需 MediaPipe 的網頁資產，設定見 [`webservice/README.md`](../webservice/README.md)。

## 相機即時辨識

```powershell
python -m recognition.transformer.realtime
```

站到鏡頭前、上半身與雙手腕完整入鏡、雙手自然垂放身側。系統會先用前一秒校準靜止基準
（畫面會顯示 `calibrating` → `calibrated`），之後**每一句的起訖都自動判定**，不必按鍵。
一句結束就印出 top-3，接著自動回到待命。`Ctrl-C` 結束。

指定攝影機用 `--camera 1`；沒有畫面預覽需求加 `--headless`。

## 用 CLI 辨識單支影片

```powershell
python scripts/analyze_knee42_video.py <影片路徑>
```

自拍鏡像的影片加 `--selfie-flip`——必須在偵測前把畫面轉正，事後修座標不等價。

## Legacy 42 類 BiGRU

`recognition/realtime/knee42_ivcam.py` 需要自備 v11 bundle
（Release `v1.0.0-v13`）與兩個 MediaPipe `.task`，`--bundle` 為必填：

```powershell
python -m recognition.realtime.knee42_ivcam --bundle <v11-bundle-dir>
```

> **沒有 Windows 可攜版。** 原本的可攜版打包的是 27 類 daily30 app，
> 該子系統已於 v12 移除，而現行 Transformer 還沒有對應的 PyInstaller 設定。
> 即時辨識本身可以從原始碼執行，見上面「相機即時辨識」。

## 重新校準自動起訖

先將三支驗收影片放入 `data/videos/auto_trigger/`，或修改 CSV 中的相對路徑：

```powershell
python -m recognition.evaluation.eval_auto_trigger_boundaries `
  --annotations data/annotations/auto_trigger_three_videos.csv
```

輸出逐片指標、摘要、最佳設定與 debug 影片。加 `--install-config <path>`
才會在全數通過時把校準結果寫出去；預設不寫，避免覆蓋被 provenance 雜湊鎖住的設定。
