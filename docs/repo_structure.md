# Repo Structure

## 根目錄

- `README.md`: GitHub 首頁主說明
- `.gitignore`: 忽略影片、模型、結果與本地暫存

## recognition

放辨識端主成果：
- `transformer/` — **現行辨識路徑**（42 類 Transformer，657 維輸入）
- `realtime/` — 即時擷取、auto-trigger、共用幀層前處理、legacy BiGRU 即時推論
- `training/` — 訓練
- `evaluation/` — 評估
- `inference/` — 共用 inference helpers

## webservice

瀏覽器測試站(HTTPS,標準函式庫實作):
- `server.py` — 靜態檔 + `/predict` + 影片工作佇列
- `static/index.html` — 單頁介面,MediaPipe 在瀏覽器端執行

## artifacts

放即時辨識載入的模型 bundle：
- `realtime/best_current/` — 42 類 Transformer bundle（唯一的 runtime bundle）

細節見 [`artifacts/README.md`](../artifacts/README.md)。

## generation

放生成端正式位置與整合說明：
- pipeline
- prompts
- docs

## data

放 metadata，不放影片：
- annotations
- manifests
- 資料政策說明

## scripts

只放少量入口腳本：
- 即時辨識啟動
- Google Drive 同步

## docs

放教授、組員、GitHub 訪客最需要的說明：
- 專案狀態
- 辨識端說明
- 生成端說明
- repo 結構
- 分支策略
- 變更紀錄
