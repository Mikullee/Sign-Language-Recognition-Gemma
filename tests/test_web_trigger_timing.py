"""Time-based boundary regressions (never duplicate frames to finish a hold)."""
import numpy as np
import pytest

from recognition.realtime.auto_trigger import SEGMENT_STATE_ACTIVE, FrameSample
from recognition.transformer.live_trigger import WebAutoTriggerConfig, WebAutoTriggerEngine
from tests.test_web_auto_trigger import observed


@pytest.mark.parametrize("fps", [10, 15, 17.5, 30, 1 / 0.13])
def test_end_confirmation_uses_elapsed_time_not_exact_frame_grid(fps):
    engine = WebAutoTriggerEngine(WebAutoTriggerConfig(end_hold_sec=0.5))
    engine.state = SEGMENT_STATE_ACTIVE
    engine.clip_start_sec = -1.0
    engine.segment_samples = [FrameSample(-1.0, np.ones(225))]
    rest = observed(rest=True, motion=0, palm_value=0, wrist_value=0)
    events = []
    for index in range(int(fps) + 2):
        event = engine.update(np.ones(225), rest, index / fps)
        if event is not None:
            events.append(event)
    assert len(events) == 1
    assert events[0].reason == "visible_rest_finalize"
    assert events[0].clip_end_sec == 0
    assert 0.5 - 1e-8 <= events[0].finalize_sec <= 0.5 + 1 / fps + 1e-8


def test_end_confirmation_cannot_finish_on_an_active_frame():
    engine = WebAutoTriggerEngine(WebAutoTriggerConfig(end_hold_sec=0.5))
    engine.state = SEGMENT_STATE_ACTIVE
    engine.clip_start_sec = -1
    engine.segment_samples = [FrameSample(-1, np.ones(225))]
    for t in np.arange(0, 0.5, 0.1):
        engine.update(np.ones(225), observed(rest=True, motion=0, palm_value=0, wrist_value=0), t)
    assert engine.update(np.ones(225), observed(rest=False, motion=0.1, palm_value=1, wrist_value=1), 0.5) is None


def test_jitter_preserves_first_observed_continuous_rest_boundary():
    engine = WebAutoTriggerEngine(WebAutoTriggerConfig(end_hold_sec=.5))
    engine.state = SEGMENT_STATE_ACTIVE
    engine.clip_start_sec = -1
    engine.segment_samples = [FrameSample(-1, np.ones(225))]
    event = None
    for t in [0, .24, .49, .74]:
        event = engine.update(np.ones(225), observed(rest=True, motion=0, palm_value=0, wrist_value=0), t)
    assert event is not None
    assert event.clip_end_sec == 0


def test_abandoned_rest_onset_does_not_cut_intervening_activity():
    engine = WebAutoTriggerEngine(WebAutoTriggerConfig(end_hold_sec=.5))
    engine.state = SEGMENT_STATE_ACTIVE
    engine.clip_start_sec = -1
    engine.segment_samples = [FrameSample(-1, np.ones(225))]
    events = []
    for i in range(11):
        rest = i < 2 or i >= 5
        event = engine.update(np.ones(225), observed(rest=rest, motion=0 if rest else .1,
            palm_value=0 if rest else 1, wrist_value=0 if rest else 1), i / 10)
        if event:
            events.append(event)
    assert len(events) == 1
    assert events[0].clip_end_sec == .5
