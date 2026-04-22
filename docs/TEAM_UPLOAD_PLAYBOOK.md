# 團隊上傳操作手冊

本文件定義「檔案放哪裡、哪些內容可以 push」，讓組員可以平行工作且不互相卡住。

## 分支職責

1. `dev-agent-sync`
- 用途：收新拍的 raw sequence 影片
- 可變更內容：
  - `data/videos/raw_sequences/**`
  - `data/annotations/sequence_recording_manifest_300.csv`
- 不要把 approved 資料 push 到這裡

2. `main`
- 用途：穩定可訓練資料
- 可變更內容：
  - `data/videos/approved_sequences/**`
  - `data/annotations/sequence_review_checklist_300.csv`
  - `data/annotations/sequence_annotations_300_template.csv`
- 只用 PR 合併，不直推

## 資料夾放置規則

1. 新錄製（尚未審片）
- `data/videos/raw_sequences/<sequence_id>/`

2. 審片通過
- `data/videos/approved_sequences/<sequence_id>/`

3. 需要重錄
- `data/videos/rerecord_needed/<sequence_id>/`

## 批次作業流程

1. 錄影組
- 影片上傳到 `raw_sequences`
- 更新 `sequence_recording_manifest_300.csv` 的錄製狀態

2. 審片組
- 在 `sequence_review_checklist_300.csv` 記錄審核結果
- 通過影片移到 `approved_sequences`
- 不通過影片移到 `rerecord_needed`

3. 標註組
- 在 `sequence_annotations_300_template.csv` 填 `start_sec` / `end_sec`

4. 推送組
- 一個批次開一個 PR
- PR 需包含證據連結與變更範圍

## 命名規則

1. Sequence 影片命名格式
- `SEQ##_X_##.mp4`
- 範例：`SEQ01_A_03.mp4`

2. 禁止空白與隨機後綴
- 避免檔名如 `final2`、`new`、`copy`

## Commit 格式

請用結構化前綴：
- `data: add raw batch ...`
- `data: promote approved batch ...`
- `anno: update timestamp annotation for batch ...`
- `docs: update playbook/template ...`

## Push 前最小檢查清單

1. 確認分支正確（`dev-agent-sync` 或由 `main` 切出的功能分支）
2. 確認只改到預期資料夾
3. 確認沒有暫存檔被 stage
4. 確認 commit 訊息符合規範
5. push 並開 PR
