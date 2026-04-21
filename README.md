# Sign-Language-Recognition-Gemma

## Raw Video Location

Uploaded raw sequence videos are maintained in branch `dev-agent-sync`.

If `data/videos/raw_sequences` looks empty on `main`, switch to:
- [dev-agent-sync branch](https://github.com/Mikullee/Sign-Language-Recognition-Gemma/tree/dev-agent-sync)

## Project Focus

This repository currently has two parallel tracks:
1. Sequence data collection and annotation collaboration
2. BiGRU baseline training/inference artifacts

Current strategy:
- Phase 1: improve low-token gloss recognition with measurable model metrics
- Phase 2: after Phase 1 target is met, connect `Qwen3.5-9B` for gloss-to-readable sentence generation

## Start Here

Read these files in order:
1. [docs/WORKFLOW_RULES.md](./docs/WORKFLOW_RULES.md)
2. [docs/WEEKLY_GOALS.md](./docs/WEEKLY_GOALS.md)
3. [docs/COMMIT_CONVENTION.md](./docs/COMMIT_CONVENTION.md)
4. [docs/TEAM_TASKS.md](./docs/TEAM_TASKS.md)
5. [docs/FILE_GUIDE.md](./docs/FILE_GUIDE.md)
6. [baseline_gru/README.md](./baseline_gru/README.md)

## Repository Structure

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
