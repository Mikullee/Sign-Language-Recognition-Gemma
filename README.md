# Sign-Language-Recognition-Gemma

台灣手語辨識與生成整合專題。現階段主成果是固定句型的即時辨識系統，並保留後續接生成端的正式模組位置。

## 目前成果

- 已完成 28 個固定常用句子的辨識流程整理
- 多數固定句型可穩定辨識，並可用 webcam 或影片做即時測試
- 已有 daily30 sentence BiGRU 訓練、評估、即時推論工具
- 已做 gloss 版本實驗，但目前效果仍弱於固定句型版本
- 正在優化自動開始 / 結束切段，目標是取代手動按空白鍵

## 系統流程

`影片或即時輸入 -> MediaPipe pose/hand landmarks -> 特徵處理 -> BiGRU 句子分類 -> 即時輸出 / 後續生成`

## 模組結構

- [`recognition/`](./recognition/README.md): 辨識端主體，包含訓練、評估、即時推論與 legacy baseline
- [`generation/`](./generation/README.md): 生成端正式位置，保留未來銜接辨識結果的流程與文件
- [`data/`](./data/README_data.md): metadata、annotations、manifest 與資料政策
- [`docs/`](./docs/repo_structure.md): 專案狀態、模組說明、分支策略、repo 結構
- [`scripts/`](./scripts/): 少量入口腳本與同步工具

## 資料管理方式

- 正式影片資料放在 Google Drive，不放進 GitHub
- GitHub 僅保留 manifest、annotations、資料說明與同步腳本
- `dev-agent-sync` 僅作為歷史資料分支，不再當主要資料來源

## 安裝環境

### Conda

```powershell
conda env create -f environment.yml
conda activate slr_preview
```

### 或 pip

```powershell
pip install -r requirements.txt
```

## 快速啟動

clone 下來後，預設需要這些檔案已在 repo 內：

- `models/hand_landmarker.task`
- `models/pose_landmarker.task`
- `artifacts/realtime/best_current/` 內的即時辨識 runtime bundle

這個 preview repo 目前已整理成使用 repo-relative 預設路徑，不再依賴原作者本機的硬編碼資料夾。

### 即時辨識

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_realtime_daily30_sentence.ps1 --source 0 --trigger-mode manual --save-log
```

### 影片測試

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_realtime_daily30_sentence.ps1 --source "C:\path\to\demo.mp4" --trigger-mode manual --save-log
```

### 資料同步

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sync_drive.ps1 -Mode download
```

## Clone 後可用條件

若要 clone 後直接使用，至少需要：

1. 安裝 `requirements.txt` 或 `environment.yml` 內的依賴
2. 保留 `models/` 內的 MediaPipe `.task` 檔
3. 保留 `artifacts/realtime/best_current/` 內的 sentence runtime bundle
4. 從 repo 根目錄執行 `scripts/run_realtime_daily30_sentence.ps1`

若你要換自己的 Python，可設定環境變數：

```powershell
$env:SLR_PYTHON="C:\path\to\python.exe"
```

## 分支策略

- `main`: 穩定展示版與整合版
- `feature/recognition-*`: 辨識端功能開發
- `feature/generation-*`: 生成端功能開發
- `dev-agent-sync`: 歷史資料同步分支，保留但不再作為正式主流程

## 目前限制

- gloss 層級辨識目前仍不如固定句型版本穩定
- 即時系統的 auto-trigger 仍在調整，桌面遮擋與邊界切段還有改進空間
- 這個 preview repo 著重展示結構與說明，不代表目前所有腳本都已完成最終整理

## 相關文件

- [`docs/project_status.md`](./docs/project_status.md)
- [`docs/recognition_overview.md`](./docs/recognition_overview.md)
- [`docs/generation_overview.md`](./docs/generation_overview.md)
- [`docs/repo_structure.md`](./docs/repo_structure.md)
- [`docs/branch_strategy.md`](./docs/branch_strategy.md)
- [`docs/changelog.md`](./docs/changelog.md)
