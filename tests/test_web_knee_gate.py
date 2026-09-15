"""Protocol tests use synthetic geometry, not evidence of real-person accuracy."""
from dataclasses import replace

import numpy as np
import pytest

from recognition.transformer.live_trigger import WebAutoKnee42Controller, WebAutoTriggerConfig


def skeleton(*, hand_y=0.78, knee_visible=True, shift=0.0, scale=1.0, hands=True):
    pose = np.zeros((33, 3), dtype=np.float32)
    for i, xy in {11: (.35, .25), 12: (.65, .25), 13: (.35, .45),
                  14: (.65, .45), 15: (.4, hand_y), 16: (.6, hand_y),
                  23: (.4, .52), 24: (.6, .52), 25: (.4, .85), 26: (.6, .85)}.items():
        pose[i] = [*xy, .01]
    left = np.tile(pose[15], (21, 1)) if hands else np.zeros((21, 3))
    right = np.tile(pose[16], (21, 1)) if hands else np.zeros((21, 3))
    vector = np.concatenate([pose.ravel(), left.ravel(), right.ravel()]).astype(np.float32)
    points = vector.reshape(-1, 3)
    valid = np.any(points != 0, axis=1)
    points[valid, :2] = (points[valid, :2] - .5) * scale + .5
    points[valid, 0] += shift
    visibility = np.ones(33, dtype=np.float32)
    if not knee_visible:
        visibility[[25, 26]] = .05
    return vector, visibility


def config(**kwargs):
    return WebAutoTriggerConfig(**dict({
        "knee_gate_enabled": True, "reference_rest_enabled": True,
        "reference_seed_sec": 1.0, "adaptive_rearm_enabled": True,
        "adaptive_rearm_requires_knee_rest": True,
        "knee_min_thigh_progress_ratio": .35,
        "start_motion_threshold": .35, "blank_motion_threshold": .20,
        "reference_seed_motion_threshold": .15,
        "max_segment_sec": 5.0,
    }, **kwargs))


def feed(controller, t, **kwargs):
    vector, visibility = skeleton(**kwargs)
    return controller.add_observation(t, vector, t, pose_visibility=visibility)


def seed(controller):
    for t in np.arange(0, 1.11, .1):
        feed(controller, float(t))
    assert controller.calibrated


@pytest.mark.parametrize("case", ["chest", "invisible_knees", "out_of_frame", "no_visibility", "missing_wrists"])
def test_initial_calibration_rejects_untrusted_geometry(case):
    c = WebAutoKnee42Controller(config())
    for t in np.arange(0, 1.5, .1):
        v, vis = skeleton(hand_y=.4 if case == "chest" else .78, knee_visible=case != "invisible_knees")
        if case == "out_of_frame":
            v.reshape(-1, 3)[[25, 26], 1] = 1.1
        if case == "missing_wrists":
            v[99:] = 0
            vis[[15, 16]] = 0
        c.add_observation(float(t), v, t, pose_visibility=None if case == "no_visibility" else vis)
    assert not c.calibrated
    assert c.engine.reference_revision == 0


@pytest.mark.parametrize("fps", [10, 15, 17.5, 30])
def test_full_knee_cycle_and_exactly_one_refresh(fps):
    c = WebAutoKnee42Controller(config())
    events = []
    for t in np.arange(0, 4.8, 1 / fps):
        event = feed(c, float(t), hand_y=.35 if 1.3 <= t < 2.8 else .78)
        if event.segment:
            events.append(event)
    assert len(events) == 1
    assert events[0].infer
    assert events[0].segment.reason == "visible_rest_finalize"
    assert abs(events[0].segment.clip_end_sec - 2.8) <= .2
    assert c.engine.reference_revision == 2
    assert c.state == "IDLE_BLANK"


def test_chest_pause_never_reseeds_or_ends_and_timeout_does_not_infer():
    c = WebAutoKnee42Controller(config(max_segment_sec=2))
    seed(c)
    events = [feed(c, float(t), hand_y=.35) for t in np.arange(1.2, 4.5, .1)]
    failures = [e for e in events if e.segment]
    assert len(failures) == 1
    assert failures[0].segment.reason == "timeout_finalize"
    assert not failures[0].infer
    assert c.engine.reference_revision == 1
    assert c.state == "REARMING"


def test_short_knee_occlusion_can_end_but_cannot_refresh_reference():
    c = WebAutoKnee42Controller(config())
    seed(c)
    for t in np.arange(1.2, 2.5, .1):
        feed(c, float(t), hand_y=.35)
    events = [feed(c, float(t), knee_visible=False) for t in np.arange(2.5, 4.0, .1)]
    endings = [e for e in events if e.segment]
    assert len(endings) == 1 and endings[0].infer
    assert endings[0].segment.reason == "reference_rest_finalize"
    assert c.engine.reference_revision == 1
    assert c.state == "REARMING"
    for t in np.arange(4.0, 5.0, .1):
        feed(c, float(t))
    assert c.engine.reference_revision == 2


def test_expired_fallback_or_body_move_never_counts_as_rest():
    c = WebAutoKnee42Controller(config())
    seed(c)
    for t in np.arange(1.2, 2.5, .1):
        feed(c, float(t), hand_y=.35, knee_visible=False)
    for t in np.arange(2.5, 3.3, .1):
        assert feed(c, float(t), knee_visible=False).segment is None
    feed(c, 3.4, knee_visible=False, shift=.2)
    assert not c.engine._is_rest_candidate(c.last_analysis)
    assert c.engine.reference_revision == 1


def test_a_timestamp_gap_does_not_supply_rest_evidence():
    c = WebAutoKnee42Controller(config())
    feed(c, 0)
    feed(c, 1.1)
    assert not c.calibrated
    assert c.engine.reference_revision == 0


def test_eof_never_manufactures_confirmation():
    c = WebAutoKnee42Controller(config())
    seed(c)
    for t in np.arange(1.2, 2.5, .1):
        feed(c, float(t), hand_y=.35)
    feed(c, 2.5)
    feed(c, 2.6)
    last = c._last_timestamp
    event = c.finalize_video_eof(frame_interval_sec=.1)
    assert not event.infer
    assert event.message == "incomplete_eof"
    assert c._last_timestamp == last


def test_reference_geometry_is_position_and_scale_relative():
    original = WebAutoKnee42Controller(config())
    transformed = WebAutoKnee42Controller(config())
    for t in np.arange(0, 1.2, .1):
        feed(original, float(t))
        feed(transformed, float(t), shift=.03, scale=.85)
    assert original.calibrated and transformed.calibrated
    np.testing.assert_allclose(original.engine._rest_wrist_reference_signature,
                               transformed.engine._rest_wrist_reference_signature, atol=1e-5)


def test_long_tracking_gap_returns_failure_without_accessing_expired_features():
    c = WebAutoKnee42Controller(config())
    seed(c)
    for t in np.arange(1.2, 2.6, .1):
        feed(c, float(t), hand_y=.35)
    event = feed(c, 20, hand_y=.35)
    assert event.message == "tracking_gap" and not event.infer
    assert event.segment.reason == "tracking_gap"


def test_relocation_restarts_rearm_hold_at_new_body_position():
    c = WebAutoKnee42Controller(config())
    seed(c)
    for t in np.arange(1.2, 2.5, .1):
        feed(c, float(t), hand_y=.35)
    for i in range(25, 37):
        feed(c, i / 10)
    feed(c, 3.7, shift=.2)
    feed(c, 3.8, shift=.2)
    assert c.engine.reference_revision == 1
    for t in np.arange(3.9, 4.5, .1):
        feed(c, float(t), shift=.2)
    assert c.engine.reference_revision == 2


def test_strict_config_defaults_cannot_calibrate_chest():
    c = WebAutoKnee42Controller(WebAutoTriggerConfig(knee_gate_enabled=True,
        reference_rest_enabled=True, adaptive_rearm_enabled=True,
        adaptive_rearm_requires_knee_rest=True))
    for t in np.arange(0, 1.3, .1):
        feed(c, float(t), hand_y=.4)
    assert not c.calibrated


def test_stable_lower_thigh_rest_does_not_start_due_to_reference_distance():
    c = WebAutoKnee42Controller(config())
    seed(c)
    # Both new wrists still lie on the lower thighs, but depart the signature.
    for i in range(12, 23):
        feed(c, i / 10, hand_y=.7)
    assert c.state == "IDLE_BLANK"
    assert c.engine.reference_revision == 1


@pytest.mark.parametrize("fps", [10, 15, 17.5, 30])
@pytest.mark.parametrize("speed", [.75, 1, 1.25])
@pytest.mark.parametrize("scale,shift", [(1, 0), (.85, .04)])
@pytest.mark.parametrize("settings", ["unit", "shipped"])
def test_two_sentences_rearm_once_each_under_speed_position_and_clock_jitter(fps, speed, scale, shift, settings):
    from pathlib import Path
    from recognition.transformer.live_trigger import load_web_trigger_config
    selected = (config() if settings == "unit" else load_web_trigger_config(
        Path(__file__).resolve().parents[1] / "configs/auto_trigger_knee_web_live.json"))
    c = WebAutoKnee42Controller(selected)
    events = []
    for i, t in enumerate(np.arange(0, 8 / speed, 1 / fps)):
        source_t = t * speed
        signing = 1.6 <= source_t < 3 or 5 <= source_t < 6.2
        timestamp = float(t + .003 * np.sin(i * 2.399)) if i else 0
        event = feed(c, timestamp, hand_y=.35 if signing else .78, scale=scale, shift=shift)
        if event.segment:
            events.append(event)
    assert len(events) == 2 and all(e.infer for e in events)
    assert c.engine.reference_revision == 3
    for event, (start, end) in zip(events, [(1.6 / speed, 3 / speed), (5 / speed, 6.2 / speed)]):
        assert abs(event.segment.clip_start_sec - start) <= .3
        assert abs(event.segment.clip_end_sec - end) <= .3


def test_lost_wrists_break_rearm_even_after_readiness_latched():
    c = WebAutoKnee42Controller(config(cooldown_sec=1.5))
    seed(c)
    for i in range(12, 26):
        feed(c, i / 10, hand_y=.35)
    for i in range(26, 43):
        feed(c, i / 10)
    assert c.engine._rearm_ready
    values, vis = skeleton(hands=False)
    vis[[15, 16]] = 0
    c.add_observation(4.3, values, 4.3, pose_visibility=vis)
    assert not c.engine._rearm_ready and c.engine.reference_revision == 1
