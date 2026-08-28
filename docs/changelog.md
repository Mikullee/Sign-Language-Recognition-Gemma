# Changelog

## v12 模型

- 辨識核心由 2 層 BiGRU 改為 4 層 Transformer encoder，輸入自 `64 × 438` 改為 `64 × 657`
  （位置／速度／加速度三通道，不再串接遮罩、不再套用 train-only standardizer）
- 新增 `recognition/transformer/`，與 legacy BiGRU 路徑共用同一份幀層合約
- `artifacts/realtime/best_current/` 改放 42 類 Transformer bundle；
  原先誤置於此的 27 類 daily30 bundle 移至 `artifacts/legacy/daily30_27class/`
- `PreviewPaths` 新增 `legacy_bundle_dir`，兩條路徑的 bundle 不再互相覆蓋
- 新增 `model_card.json`：載明發布權重以全部四位簽者訓練、因此沒有保留測試分數，
  並標註 checkpoint 內的 `val_macro_mixed = 1.0` 不可作為準確率引用
- 新增 `scripts/aggregate_knee42_loso_runs.py`，README 的成績表全部由原始訓練 log 重算
- 新增 `scripts/build_knee42_transformer_bundle.py`，bundle 可重建並比對雜湊
- 移植後的推論路徑與上游參考實作在 300 筆特徵樣本上逐位元一致
- 測試由 175 增為 193 項

## v0.1.0

- 正式支援 27 個固定句型，移除未受模型支援的 `T09 我聽不懂`
- 新增共用 auto-trigger 狀態機與秒制參數
- 加入 pre-roll、可見雙手腕、身體兩側、低動作與滑動投票句尾判斷
- 加入三支標註影片的離線校準、邊界評估、分類比較與 debug 影片
- Auto 模式改為站姿操作提示，可在冷卻後連續辨識下一句
- 移除遠端主機、帳號、密碼、SSH 自動下載與遠端執行參數
- runtime bundle、MediaPipe 模型與自動切段設定改為完全離線
- 新增 `app_config.json`，CLI 明確參數優先
- 新增 Windows x64 PyInstaller `onedir` 可攜版與 iVCam 啟動腳本
- 新增敏感資訊、27 類一致性、frozen 資源路徑與打包內容測試
