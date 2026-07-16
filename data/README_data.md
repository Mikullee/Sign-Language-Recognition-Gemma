# Data Management

## 原則

- 正式影片資料放在 Google Drive
- GitHub 不放原始影片、模型權重、訓練輸出結果
- GitHub 只保留 metadata、annotations、manifest、資料政策與同步腳本

## 目前資料類型

- `annotations/`: 標註與說明檔
- `manifests/`: 資料索引與 split/label metadata

## 分支與歷史

- `dev-agent-sync` 僅為歷史資料同步分支
- 正式資料來源不再依賴 GitHub branch 內的影片檔

## 推薦工作流

1. 在 Google Drive 維護影片資料
2. 在 GitHub 更新 manifest 與 annotation
3. 透過 `scripts/sync_drive.ps1` 做資料同步
4. 只提交 metadata、文件與程式碼
