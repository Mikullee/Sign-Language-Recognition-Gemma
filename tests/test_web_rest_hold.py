"""State regressions for observed live failure patterns, not accuracy tests."""
from dataclasses import replace

import numpy as np
import pytest

from recognition.realtime.auto_trigger import FrameSample
from recognition.transformer.live_trigger import WebAutoTriggerEngine
from tests.test_web_auto_trigger import observed
from tests.test_web_knee_gate import config


def rest(motion=.10, **changes):
    return replace(observed(rest=True, motion=motion, palm_value=0, wrist_value=0), **changes)


def update(engine, t, analysis):
    return engine.update(np.ones(225), analysis, float(t))


def active():
    e = WebAutoTriggerEngine(config())
    e.state = 'SIGNING_ACTIVE'
    e.clip_start_sec = 0
    e.segment_samples = [FrameSample(0, np.ones(225))]
    return e


@pytest.mark.parametrize('fps', [10, 15, 17.5, 30])
def test_initial_hold_survives_one_small_motion_spike_without_counting_it(fps):
    e = WebAutoTriggerEngine(config())
    calibrated_at = None
    for i, t in enumerate(np.arange(0, 1.5, 1/fps)):
        update(e, t, rest(.19 if i == round(.5*fps) else .10))
        if e.calibrated and calibrated_at is None:
            calibrated_at = t
    assert calibrated_at is not None and 1 < calibrated_at <= 1.3
    assert e.reference_revision == 1 and e.state == 'IDLE_BLANK'


def test_persistent_borderline_motion_never_seeds_a_reference():
    e = WebAutoTriggerEngine(config())
    for i in range(90):
        update(e, i/30, rest(.10 if i%3 == 0 else .19))
    assert not e.calibrated


@pytest.mark.parametrize('bad', [rest(.5), rest(wrists_detected=False),
    rest(knee_landmarks_valid=False), rest(hands_on_knees=False)])
def test_real_motion_or_untrusted_geometry_still_resets_initial_hold(bad):
    e = WebAutoTriggerEngine(config())
    for i in range(21):
        update(e, i/30, bad if i == 20 else rest())
    for i in range(21, 40):
        update(e, i/30, rest())
    assert not e.calibrated


def test_rearm_survives_small_spike_and_updates_reference_once():
    e = WebAutoTriggerEngine(config())
    e.state = 'REARMING'
    for i in range(30):
        update(e, i/30, rest(.19 if i == 7 else .10))
        if i == 19:
            assert e.reference_revision == 1
    assert e.reference_revision == 1 and e.state == 'IDLE_BLANK'


def test_end_hold_survives_motion_crossing_point_two():
    e = active()
    events = []
    for i in range(24):
        event = update(e, 2+i/30, rest(.203 if i == 7 else .19))
        if event:
            events.append(event)
    assert len(events) == 1
    assert events[0].clip_end_sec == 2
    assert 2.5 < events[0].finalize_sec < 2.7


@pytest.mark.parametrize('onset', [4.7,4.8])
@pytest.mark.parametrize('fps', [10,15,17.5,30])
def test_pending_return_can_finish_after_deadline(onset, fps):
    e = active()
    events = []
    for t in np.arange(onset,5.65,1/fps):
        event = update(e,t,rest())
        if event:
            events.append(event)
    assert len(events) == 1 and events[0].reason == 'visible_rest_finalize'
    assert events[0].clip_end_sec == onset
    assert 5 < events[0].finalize_sec <= onset+.5+1/fps+1e-8


def test_return_starting_after_deadline_does_not_get_grace():
    e = active()
    event = update(e,5.01,rest())
    assert event.reason == 'timeout_finalize'


@pytest.mark.parametrize('bad', [rest(.6),rest(wrists_detected=False),rest(hands_on_knees=False)])
def test_grace_cannot_finish_on_lost_or_moving_evidence(bad):
    e = active()
    for t in [4.8,4.9]:
        assert update(e,t,rest()) is None
    assert update(e,5.01,bad).reason == 'timeout_finalize'


def test_knee_motion_without_departure_cannot_start_a_false_short_sentence():
    e = WebAutoTriggerEngine(config())
    for t in np.arange(0,1.2,.1):
        update(e,t,rest())
    for t in np.arange(1.2,1.8,.1):
        update(e,t,rest(.7))
    assert e.state == 'IDLE_BLANK'


def test_continuous_good_tail_is_never_delayed_by_earlier_tolerated_noise():
    e = WebAutoTriggerEngine(config())
    e.state = 'REARMING'
    for i in range(31):
        # Noisy prefix, then exactly .5 seconds continuously stable.
        motion = .19 if i < 15 and i%3 == 1 else .1
        update(e,i/30,rest(motion))
    assert e.reference_revision == 1
    assert len(e._rest_wrist_reference_signature) == 6


def test_expired_confirmation_cannot_restart_after_a_long_soft_gap():
    e = active()
    for t,m in [(4.80,.1),(4.90,.24)]:
        assert update(e,t,rest(m)) is None
    event = update(e,5.01,rest())
    assert event is not None and event.reason == 'timeout_finalize'


def test_body_scale_change_starts_fresh_end_confirmation():
    from tests.test_web_knee_gate import feed, seed, skeleton
    from recognition.transformer.live_trigger import WebAutoKnee42Controller
    c = WebAutoKnee42Controller(config())
    seed(c)
    for i in range(12,26):
        feed(c,i/10,hand_y=.35)
    for i in range(26,30):
        feed(c,i/10)
    assert c.engine._end_onset_sec is not None
    events=[]
    for i in range(30,40):
        v,vis=skeleton()
        p=v.reshape(75,3)
        valid=np.any(p,axis=1)
        p[valid,:2]=(p[valid,:2]-[.5,.25])*.75+[.5,.25]
        event=c.add_observation(i/10,v,i,pose_visibility=vis)
        if event.segment:
            events.append(event)
    assert len(events)==1
    assert events[0].segment.clip_end_sec >= 3.0
    assert events[0].segment.finalize_sec >= 3.5
