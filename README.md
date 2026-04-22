# 手語辨識專案（dev-agent-sync）

## 目前進度（2026-04）

1. `300 部 sequence` 已完成錄製（raw）。
2. 目前重點不是再盲拍，而是：
- 審片（通過/重錄）
- 標註時間（start/end）
- 針對低 token gloss 做補錄

## 分支定位

- `dev-agent-sync`：放「工作中資料」（raw 與同步中的資料）
- `main`：放「穩定可訓練資料」（approved + annotation）

## 當前策略

### Phase 1（先做）
- 優先補低 token gloss（先補短板）
- 每批補完後重訓 BiGRU 並跑固定評估

### Phase 2（Phase 1 達標後）
- 接 `Qwen3.5-9B` 做 gloss 到可讀句子化

## 組員資料放置規則

1. 新拍影片（未審）
- `data/videos/raw_sequences/<sequence_id>/`

2. 審片通過
- `data/videos/approved_sequences/<sequence_id>/`

3. 需要重錄
- `data/videos/rerecord_needed/<sequence_id>/`

4. 必填標註檔
- `data/annotations/sequence_recording_manifest_300.csv`
- `data/annotations/sequence_review_checklist_300.csv`
- `data/annotations/sequence_annotations_300_template.csv`

## 先讀文件

1. `docs/TEAM_UPLOAD_PLAYBOOK.md`
2. `docs/WORKFLOW_RULES.md`
3. `docs/WEEKLY_GOALS.md`

## 提交規範（簡版）

1. 不直接 push `main`
2. 一次只提交一個批次
3. commit 訊息用結構化前綴：
- `data: ...`
- `anno: ...`
- `docs: ...`
