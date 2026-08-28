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
- 新增 `recognition/transformer/landmarks.py` 與 `segmentation.py`:離線影片辨識,
  直接驅動 MediaPipe Tasks,不再需要第三方的 SignAvatar 套件
- 對齊訓練抽取契約:`RunningMode.IMAGE`、不翻轉畫面、handedness 原樣採用。
  修正了原本誤用 `RunningMode.VIDEO` 的問題
- 釐清 `left_shoulder_x` 是正規化健檢而非鏡像偵測;需要轉正的影片改在
  送進 MediaPipe 前處理(`selfie_flip`)
- 新增 `webservice/`:瀏覽器測試站,攝影機模式的 MediaPipe 在使用者端執行,
  只有骨架座標會送到伺服器。自寫串流 multipart 解析,不依賴 Python 3.13 已移除的 `cgi`
- 新增 `recognition/training/knee42_transformer.py` 與訓練 CLI,
  直接沿用部署端的特徵管線與模型定義,避免訓練與推論分岔
- `models/README.md` 記載 `.task` 檔的 SHA-256,並註明必須用 lite 而非 full
- **移除整個 27 類 daily30 子系統**(23 個檔案):即時推論、訓練、評估、
  inference helpers、27 類 bundle、Windows 打包與進入點。理由是該模型
  `best_dev_top1` 僅 0.418,類別集合與 42 類 Knee42 無關,留著只會讓
  「哪一顆才是模型」持續混淆
- **副作用:Windows 可攜版停止**。其進入點就是 27 類 app,Transformer 的
  沒有對應的 PyInstaller 設定;即時辨識本身改由 `recognition.transformer.realtime`
  從原始碼執行
- `daily30_sentence_model_utils.py` 更名為 `bigru_sentence_model.py`——
  它其實是 legacy **Knee42** BiGRU 的分類器,原檔名有誤導性
- auto-trigger 邊界校準工具保留,但拿掉綁死 daily30 的分類對照;
  225 維 trigger 向量改由共用的 `knee42_preprocessing` 提供(已驗證兩者輸出完全相同)。
  **`install_best_config_if_passed` 的門檻因此變寬**:不再檢查分類回歸,只看邊界是否全數通過
- 測試由 175 增為 221 項

### 相機即時辨識(Transformer)

- 新增 `recognition/transformer/realtime.py`:把既有的 auto-trigger 狀態機接到
  Transformer 辨識器。兩邊都沒有改動——狀態機的門檻是在真實錄影上校準過的,
  辨識器是已驗證的 bundle 載入路徑;缺的只是中間那條線
- 段落緩衝存的是帶 NaN 的肩寬正規化座標,正好就是 Transformer 的輸入合約,
  遮罩在此不使用(內插與重取樣都在 `materialize_sequence` 裡)
- 實測:靜止 → 比劃 → 靜止的影片會走完
  `IDLE_BLANK → SIGNING_ACTIVE → END_CONFIRM → 輸出 → IDLE_BLANK`,
  邊界落在 2.52–4.27 秒(實際比劃 2.50–4.23),pre-roll 有把起手收進來

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
