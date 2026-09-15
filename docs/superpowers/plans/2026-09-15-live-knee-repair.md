# Live knee calibration repair implementation plan

> **For agentic workers:** Use the executing-plans workflow for this tightly coupled repair; review before delivery.

**Goal:** Repair rejected stationary knee rests without modifying the 42-class model, recognition features, or archived v13 modules.

**Architecture:** Keep raw geometry/visibility immediate. Compute a separate causal rest-motion measure from a 0.16-second window (including excursion, not only endpoint displacement). Spatially associate hand detections with pose wrists in the trigger-only copy. Accept a visible palm or wrist in a bounded knee neighborhood for foreshortened seated views. Maintain loss, chest-pause, timeout and rearm safety checks. Diagnostics use separate live fields, not a rewritten paragraph.

**Tech stack:** Existing NumPy controller, Python pytest, vanilla browser JavaScript, Node tests.

Scope approval: user requested the previously discussed calibration, tracking-jitter and diagnostic repairs: “請修正到我可以使用”. Local delivery first. No new session, model training, raw-video recording, or upload.

## 1. Trigger repair

- [x] Add `tests/test_web_knee_stability.py`: stationary coordinate noise across 10/15/17.5/30 FPS calibrates and ends, while continuous motion and excursions do not; hand-label swaps do not create motion; seated palm rest accepted; chest/lost wrists rejected; features untouched.
- [x] Run `python -m pytest tests/test_web_knee_stability.py -q` and retain expected failures.
- [x] Implement the trigger-only normalization and rest-motion window in `recognition/transformer/knee_geometry.py`, integrate/reset it in `live_trigger.py`, and keep the existing instantaneous score for departure.
- [x] Run new tests plus `test_web_knee_gate.py`, `test_web_trigger_timing.py`, `test_web_knee_stream.py`, `test_web_auto_trigger.py`.

## 2. Live diagnostics

- [x] Extend `tests/web_diagnostics_flicker.test.mjs` to require independent immediately updated status, motion, reference fields and no “算不出來” for absent calibration.
- [x] Run failing Node tests; split `index.html` diagnostics into fixed rows with numeric slots and no duplicate state prose, maintaining every response update.
- [x] Report specific calibration blockers and hold progress from `/stream`; test the API fields. Do not imply Pose visibility or raw hand count equals trusted two-wrist detection.

## 3. Verification and delivery

- [x] Run all related Python and browser contract tests, diff checks and an independent code review.
- [x] Restart the known local server with the corrected code; verify health and served UI version. Leave local scalar diagnostics available, bounded and private.
- [ ] Ask user to refresh and perform one real knee-rest / sign / return cycle. Do not claim real usability solely from synthetic tests.

Alternatives considered: increasing instantaneous thresholds alone cannot address geometry and introduces false endings; bypassing knee calibration violates the agreed protocol. Both rejected. No new UI refresh timer or artificial stationary frames.
