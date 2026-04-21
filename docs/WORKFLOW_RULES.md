# Workflow Rules

## 1) Branch Policy

- Do not push directly to `main`.
- Use feature branches:
  - `data/*` for data/annotation updates
  - `exp/*` for training/experiment updates
  - `docs/*` for documentation updates
  - `fix/*` for bug fixes

## 2) Pull Request Policy

- Every change must go through a PR.
- One PR should have one clear purpose.
- PR title format:
  - `data: ...`
  - `exp: ...`
  - `docs: ...`
  - `fix: ...`

## 3) Required PR Content

- What changed
- Why this change is needed
- Evidence (metric, sample output, CSV/video path)
- Risk and rollback note

## 4) Main Branch Protection (GitHub Settings)

Recommended settings for `main`:
- Require a pull request before merging
- Require at least 1 approving review
- Dismiss stale approvals when new commits are pushed
- Restrict who can push to matching branches

## 5) Phase Reminder

- Phase 1 first: prioritize low-token gloss recognition improvement with measurable metrics.
- Start Phase 2 only after Phase 1 target is met.
- Phase 2 scope: only gloss-to-readable sentence generation via `Qwen3.5-9B`.
