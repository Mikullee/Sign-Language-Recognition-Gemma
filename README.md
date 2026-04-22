# 手語辨識專案（Sign-Language-Recognition-Gemma）

## Raw 影片位置

目前上傳的 raw sequence 影片主要維護在 `dev-agent-sync` 分支。

如果你在 `main` 看到 `data/videos/raw_sequences` 幾乎是空的，請切到：
- [dev-agent-sync 分支](https://github.com/Mikullee/Sign-Language-Recognition-Gemma/tree/dev-agent-sync)

## 專案重點

目前專案有兩條主線：
1. sequence 資料蒐集與標註協作
2. BiGRU baseline 訓練與推論成果

目前策略：
- Phase 1：先把低 token gloss 的辨識效果補齊，並以可量化指標驗收
- Phase 2：Phase 1 達標後，接上 `Qwen3.5-9B` 做 gloss 到可讀句子的轉換

## 先看這裡

請依序閱讀：
1. [docs/WORKFLOW_RULES.md](./docs/WORKFLOW_RULES.md)
2. [docs/TEAM_UPLOAD_PLAYBOOK.md](./docs/TEAM_UPLOAD_PLAYBOOK.md)
3. [docs/WEEKLY_GOALS.md](./docs/WEEKLY_GOALS.md)
4. [docs/COMMIT_CONVENTION.md](./docs/COMMIT_CONVENTION.md)
5. [docs/TEAM_TASKS.md](./docs/TEAM_TASKS.md)
6. [docs/FILE_GUIDE.md](./docs/FILE_GUIDE.md)
7. [baseline_gru/README.md](./baseline_gru/README.md)

## Repository 結構

```text
docs/
  WORKFLOW_RULES.md
  WEEKLY_GOALS.md
  COMMIT_CONVENTION.md
  TEAM_TASKS.md
  FILE_GUIDE.md

data/
  annotations/
  videos/
    raw_sequences/
    approved_sequences/
    rerecord_needed/

baseline_gru/
  README.md
  scripts/
  models/
  results/
```
