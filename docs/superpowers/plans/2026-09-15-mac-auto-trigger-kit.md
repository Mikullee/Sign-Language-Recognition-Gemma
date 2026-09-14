# Knee42 Mac Auto-Trigger Kit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish one reproducible private GitHub prerelease ZIP that lets an Apple Silicon Mac install, verify, and launch the current Knee42 Web auto-trigger candidate with its model, runtime assets, and annotated boundary videos.

**Architecture:** Keep large third-party assets and videos out of Git history. A Python builder exports tracked files from the current commit, validates and injects local runtime assets and benchmark videos, then writes a deterministic ZIP plus SHA-256. Two small shell scripts install a local virtual environment and start the loopback-only Web service.

**Tech Stack:** Python 3.10+, standard library, Bash/zsh, Git, pytest, GitHub CLI.

---

### Task 1: Specify the package contract

**Files:**
- Create: `tests/test_mac_auto_trigger_package.py`
- Create: `scripts/build_mac_auto_trigger_package.py`

- [ ] Write tests requiring a package manifest, the shipped model bundle, MediaPipe Python/browser assets, the annotation CSV, and all three named MP4 files.
- [ ] Run `python -m pytest tests/test_mac_auto_trigger_package.py -q` and confirm failure because the builder is absent.
- [ ] Implement the minimum builder API and input validation.
- [ ] Re-run the focused test and confirm it passes.

### Task 2: Add Mac setup and launch scripts

**Files:**
- Create: `scripts/setup_mac_auto_trigger.sh`
- Create: `scripts/run_mac_auto_trigger.sh`
- Modify: `tests/test_mac_auto_trigger_package.py`

- [ ] Add failing tests for macOS/ARM64 guards, local `.venv`, dependency installation, package verification, loopback binding, HTTP mode, and the Web trigger config.
- [ ] Implement the scripts with fail-fast error messages.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Document handoff status

**Files:**
- Create: `docs/mac_auto_trigger_handoff.md`
- Modify: `README.md`
- Modify: `tests/test_mac_auto_trigger_package.py`

- [ ] Add a failing documentation-contract test for install, start, benchmark paths, limitations, and no-retraining guidance.
- [ ] Write the handoff guide and link it from the README.
- [ ] Run the focused tests and confirm they pass.

### Task 4: Build and inspect the Release artifact

**Files:**
- Generate: `release/Knee42-Mac-AutoTrigger-<commit>.zip`
- Generate: `release/Knee42-Mac-AutoTrigger-<commit>.zip.sha256`

- [ ] Run the builder with the repository `models/`, `webservice/vendor/mediapipe/`, and the local annotated-video source directory as inputs.
- [ ] Inspect the archive listing and run the builder's verification mode against the ZIP.
- [ ] Extract into a temporary directory and verify the model manifest, 42 labels, assets, CSV paths, and video names.

### Task 5: Verify and publish

**Files:**
- Modify: `docs/mac_auto_trigger_handoff.md` only if verification discovers a correction.

- [ ] Run the focused package tests, release-safety tests, and complete `python -m pytest tests -q`.
- [ ] Run `git diff --check` and inspect `git status`.
- [ ] Commit the package tooling and documentation.
- [ ] Push `fix/web-auto-trigger-live-tuning`.
- [ ] Create a prerelease in the private reproduction-data repository for the exact public-code commit and upload the ZIP plus SHA-256.
- [ ] Verify the remote release metadata and downloadable asset sizes.
