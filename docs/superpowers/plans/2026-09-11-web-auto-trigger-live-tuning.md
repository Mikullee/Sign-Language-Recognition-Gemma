# Knee42 Web Auto-Trigger Live Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current Transformer v12 Web camera mode end segments reliably, refuse immediate post-timeout retriggers, expose actionable rest diagnostics, and support fast field-test iteration without changing the recognition model.

**Architecture:** Keep `AutoTriggerEngine` as the single state-machine implementation shared by CLI and Web. Add an opt-in re-arm state that collects a stable rest window after every finalized segment and refreshes the session rest reference before returning to idle. Use a separately named Web field config so each threshold change is auditable; the browser only renders server decisions and never duplicates trigger logic.

**Tech Stack:** Python 3.12, NumPy, MediaPipe landmarks, Transformer v12 recognizer, standard-library HTTP server, vanilla JavaScript, pytest/unittest.

---

## File map

- Modify `recognition/realtime/auto_trigger.py`: add the re-arm state, stable-sample predicate, reference refresh, and diagnostics.
- Modify `recognition/realtime/knee42_controllers.py`: expose whether the controller is waiting for re-arm.
- Create `configs/auto_trigger_knee_web_live.json`: field-test candidate with five-second safety timeout and adaptive re-arm enabled.
- Modify `webservice/server.py`: return reference/re-arm diagnostic fields and load the candidate config when explicitly requested.
- Modify `webservice/static/index.html`: display the new re-arm state and precise reason when rest distance cannot be computed.
- Modify `tests/test_auto_trigger.py`: state-machine unit tests for timeout, re-arm, reference refresh, and missing-hand behavior.
- Modify `tests/test_webservice.py`: response contract and per-session diagnostics tests.
- Modify `docs/auto_trigger_field_notes.md`: record the candidate and measured field results.

### Task 1: Lock the current baseline

**Files:**
- Read: `docs/auto_trigger_field_notes.md`
- Test: `tests/test_auto_trigger.py`
- Test: `tests/test_knee42_transformer_realtime.py`
- Test: `tests/test_webservice.py`

- [ ] **Step 1: Run the focused baseline tests**

Run:

```powershell
python -m pytest tests/test_auto_trigger.py tests/test_knee42_transformer_realtime.py tests/test_webservice.py -q
```

Expected: all selected tests pass before any production-code change.

- [ ] **Step 2: Verify runtime assets and health endpoint prerequisites**

Run:

```powershell
python scripts/fetch_mediapipe_models.py
python -c "from recognition.transformer.recognizer import Knee42TransformerRecognizer; r=Knee42TransformerRecognizer('artifacts/realtime/best_current'); print(len(r.labels))"
```

Expected: MediaPipe assets verify and the model prints `42`.

### Task 2: Add an opt-in post-segment re-arm gate

**Files:**
- Modify: `recognition/realtime/auto_trigger.py:12-100`
- Modify: `recognition/realtime/auto_trigger.py:415-710`
- Test: `tests/test_auto_trigger.py`

- [ ] **Step 1: Write failing configuration and state tests**

Add these imports and tests:

```python
from recognition.realtime.auto_trigger import SEGMENT_STATE_REARMING

def test_adaptive_rearm_settings_validate():
    with self.assertRaisesRegex(ValueError, "Adaptive re-arm hold"):
        AutoTriggerConfig(adaptive_rearm_hold_sec=-0.1)

def test_timeout_requires_stable_rest_before_another_start():
    config = AutoTriggerConfig(
        start_motion_threshold=0.01,
        start_hold_sec=0.1,
        pre_roll_sec=0.0,
        max_segment_sec=0.5,
        cooldown_sec=0.1,
        knee_geometry_enabled=False,
        reference_rest_enabled=True,
        reference_seed_sec=0.2,
        adaptive_rearm_enabled=True,
        adaptive_rearm_hold_sec=0.2,
        adaptive_rearm_requires_knee_rest=False,
    )
    engine = AutoTriggerEngine(config)
    rest_frame = body_frame(hands_visible=True, hands_on_knees=True)
    moving_frame = body_frame(hands_visible=True, hands_on_knees=False)
    moving_frame[99:] += 0.05
    previous = None
    for timestamp in (0.0, 0.1, 0.2):
        observed = analyze_frame_vector(previous, rest_frame, config)
        engine.update(rest_frame, observed, timestamp)
        previous = rest_frame
    event = None
    for timestamp in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        observed = analyze_frame_vector(previous, moving_frame, config)
        event = engine.update(moving_frame, observed, timestamp)
        previous = moving_frame
    self.assertIsNotNone(event)
    self.assertEqual(event.reason, "timeout_finalize")
    for timestamp in (0.9, 1.0, 1.1):
        observed = analyze_frame_vector(previous, moving_frame, config)
        self.assertIsNone(engine.update(moving_frame, observed, timestamp))
        previous = moving_frame
    self.assertEqual(engine.state, SEGMENT_STATE_REARMING)
```

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```powershell
python -m pytest tests/test_auto_trigger.py -q
```

Expected: failure because `SEGMENT_STATE_REARMING` and adaptive re-arm fields do not yet exist.

- [ ] **Step 3: Implement the minimal re-arm state**

Add the public state and configuration fields:

```python
SEGMENT_STATE_REARMING = "REARMING"

adaptive_rearm_enabled: bool = False
adaptive_rearm_hold_sec: float = 0.50
adaptive_rearm_requires_knee_rest: bool = False
```

Add a stable sample predicate that requires a usable full and wrist signature, valid torso/wrists, two explicit hands, and motion no greater than `reference_seed_motion_threshold`. While `REARMING`, accumulate only consecutive safe samples. Once the configured hold is reached, replace both references with the median of the collected signatures and transition to `IDLE_BLANK`. Any unsafe sample clears only the candidate window, not the existing reference. The core transition is:

```python
def _complete_rearm(self) -> None:
    if not self._rearm_ready:
        return
    self._set_rest_reference(self._rearm_signatures, self._rearm_wrist_signatures)
    self.state = SEGMENT_STATE_IDLE
    self._pre_roll.clear()
    self._active_start_sec = None
    self._clear_rearm_window()
```

At the end of cooldown use:

```python
if self.config.adaptive_rearm_enabled:
    self.state = SEGMENT_STATE_REARMING
    self._update_rearm(timestamp_sec, analysis)
    return None
self.state = SEGMENT_STATE_IDLE
```

- [ ] **Step 4: Run state-machine tests and confirm GREEN**

Run:

```powershell
python -m pytest tests/test_auto_trigger.py -q
```

Expected: all tests pass, including no immediate retrigger after timeout and refreshed reference after stable rest.

- [ ] **Step 5: Commit the re-arm state**

```powershell
git add recognition/realtime/auto_trigger.py tests/test_auto_trigger.py
git commit -m "fix(trigger): require stable rest before rearming"
```

### Task 3: Create a separately auditable Web field configuration

**Files:**
- Create: `configs/auto_trigger_knee_web_live.json`
- Modify: `tests/test_knee42_auto_trigger_integration.py`

- [ ] **Step 1: Write a failing config-contract test**

```python
def test_web_live_config_is_fail_safe_and_rearms():
    config = load_auto_trigger_config(ROOT / "configs" / "auto_trigger_knee_web_live.json")
    self.assertEqual(config.max_segment_sec, 5.0)
    self.assertTrue(config.adaptive_rearm_enabled)
    self.assertEqual(config.adaptive_rearm_hold_sec, 0.5)
    self.assertFalse(config.adaptive_rearm_requires_knee_rest)
```

- [ ] **Step 2: Run the contract test and confirm RED**

Run:

```powershell
python -m pytest tests/test_knee42_auto_trigger_integration.py -q
```

Expected: failure because the candidate config does not exist.

- [ ] **Step 3: Add the candidate config**

Start from `configs/auto_trigger_knee_v1.json` and change only:

```json
{
  "max_segment_sec": 5.0,
  "adaptive_rearm_enabled": true,
  "adaptive_rearm_hold_sec": 0.5,
  "adaptive_rearm_requires_knee_rest": false
}
```

Keep `reference_rest_distance_threshold` at `0.18` until observed field values justify changing it.

- [ ] **Step 4: Run the config tests and confirm GREEN**

```powershell
python -m pytest tests/test_knee42_auto_trigger_integration.py -q
```

- [ ] **Step 5: Commit the candidate config**

```powershell
git add configs/auto_trigger_knee_web_live.json tests/test_knee42_auto_trigger_integration.py
git commit -m "feat(trigger): add web field tuning profile"
```

### Task 4: Expose re-arm and signature diagnostics to the Web UI

**Files:**
- Modify: `recognition/realtime/knee42_controllers.py:126-220`
- Modify: `webservice/server.py:249-380`
- Modify: `webservice/static/index.html:475-530`
- Modify: `tests/test_webservice.py`

- [ ] **Step 1: Write failing response-contract tests**

Require `/stream` success payloads to include:

```python
{
    "state": "REARMING",
    "rest_distance": None,
    "rest_threshold": 0.18,
    "rest_signature_status": "missing_hand",
    "reference_revision": 1,
}
```

The exact state varies by fixture; the contract requirement is that every field exists and is JSON serializable.

- [ ] **Step 2: Run Web tests and confirm RED**

```powershell
python -m pytest tests/test_webservice.py -q
```

Expected: missing response fields.

- [ ] **Step 3: Add state-machine diagnostics**

Track a monotonically increasing `reference_revision` every time `_set_rest_reference()` succeeds. Add a read-only signature-status helper and return both fields from `stream_payload()`:

```python
def rest_signature_status(self, analysis: AutoFrameAnalysis | None) -> str:
    if self._rest_reference_signature is None:
        return "uncalibrated"
    if analysis is None or not analysis.torso_valid:
        return "missing_pose"
    if analysis.explicit_hands_detected == 0:
        return "missing_both_hands"
    if analysis.explicit_hands_detected == 1:
        return "missing_hand"
    return "ready" if analysis.rest_signature is not None else "missing_pose"

"rest_signature_status": controller.engine.rest_signature_status(last_analysis),
"reference_revision": controller.engine.reference_revision,
```

The landmark payload does not currently preserve which anatomical hand is absent, so the public contract uses `missing_hand` rather than inventing left/right certainty.

- [ ] **Step 4: Render the diagnostics without duplicating decisions**

Add `REARMING: "請回到休息姿勢，重新校準中…"` to `AUTO_STATE_TEXT`. Replace the fixed “需要雙手” message with the server-provided signature status. The browser must not calculate rest distance or decide state transitions:

```javascript
const REST_STATUS_TEXT = {
  uncalibrated: "尚未完成初始校準",
  missing_pose: "姿態關鍵點不足",
  missing_both_hands: "未偵測到雙手",
  missing_hand: "僅偵測到單手",
  ready: "靜止特徵可計算",
};
```

- [ ] **Step 5: Run Web and focused regression tests**

```powershell
python -m pytest tests/test_webservice.py tests/test_auto_trigger.py tests/test_knee42_transformer_realtime.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit diagnostics**

```powershell
git add recognition/realtime/auto_trigger.py recognition/realtime/knee42_controllers.py webservice/server.py webservice/static/index.html tests/test_webservice.py
git commit -m "feat(web): expose trigger rearm diagnostics"
```

### Task 5: Start the live field loop

**Files:**
- Modify after measurement: `configs/auto_trigger_knee_web_live.json`
- Modify after measurement: `docs/auto_trigger_field_notes.md`

- [ ] **Step 1: Start the Web Service with the field config**

Use the existing `--trigger-config` server option and start explicitly with:

```powershell
python -m webservice.server --port 8642 --trigger-config configs/auto_trigger_knee_web_live.json
```

Expected: `/health` returns `ok: true`, 42 classes, and the browser opens the HTTPS test page.

- [ ] **Step 2: Run a fixed first field set**

Use ten attempts in this order and report the spoken label with each attempt: `你好、謝謝、再見、可以、不可以、我知道、我不知道、早安、晚安、對不起`.

For each attempt record: result index, expected phrase, displayed top-1, start/end, finalize reason, rest distance/status, and whether the boundary looked correct.

- [ ] **Step 3: Apply only the evidence-selected threshold change**

- If stable resting distance is consistently above `0.18` but at or below `0.28`, change only `reference_rest_distance_threshold` to `0.28`.
- If distance is uncomputable, do not widen the threshold; fix the missing-hand/fallback path instead.
- If rest finalizes but is noisy, tune only `end_hold_sec` or `end_rest_vote_ratio`, not both in one round.

- [ ] **Step 4: Repeat the same ten attempts**

Keep the candidate only if timeout count, immediate retriggers, and visible boundary errors do not regress.

- [ ] **Step 5: Commit the measured candidate and notes**

```powershell
git add configs/auto_trigger_knee_web_live.json docs/auto_trigger_field_notes.md
git commit -m "test(trigger): record web live tuning results"
```

### Task 6: Full verification and handoff

**Files:**
- Modify: `README.md`
- Modify: `webservice/README.md`
- Modify: `docs/auto_trigger_field_notes.md`

- [ ] **Step 1: Document the field profile and limitations**

Document the exact launch command, expected rest posture, meaning of `REARMING`, and the rule that timeout segments are segmentation failures rather than model evidence.

- [ ] **Step 2: Run the complete test suite**

```powershell
python -m pytest tests -q
```

Expected: all tests pass with no failures or errors.

- [ ] **Step 3: Verify the diff and repository cleanliness**

```powershell
git diff origin/main...HEAD --check
git status --short --branch
```

Expected: no whitespace errors; only intentional commits ahead of `origin/main`.

- [ ] **Step 4: Commit documentation**

```powershell
git add README.md webservice/README.md docs/auto_trigger_field_notes.md
git commit -m "docs: explain web trigger rearm workflow"
```

- [ ] **Step 5: Push and open a PR only after field acceptance**

```powershell
git push -u origin fix/web-auto-trigger-live-tuning
gh pr create --base main --head fix/web-auto-trigger-live-tuning --title "fix: stabilize Knee42 web auto-trigger rearming" --body-file docs/superpowers/specs/2026-09-11-web-auto-trigger-live-tuning-design.md
```

Expected: a reviewable PR; no new Release is created until the PR is merged and a clean-clone launch is verified.
