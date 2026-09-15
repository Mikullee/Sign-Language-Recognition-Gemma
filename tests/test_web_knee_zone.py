"""Generic geometry/duration regressions, not signer or phrase accuracy."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from recognition.transformer.live_trigger import WebAutoKnee42Controller, load_web_trigger_config
from tests.test_web_knee_gate import config, skeleton, feed, seed

ROOT=Path(__file__).resolve().parents[1]


def borderline_seated_frame(i):
    v,vis=skeleton(hand_y=.4795)
    p=v.reshape(75,3)
    p[[23,24],1]=.52+(.003 if i%2 else -.003)
    p[[25,26],1]=.58
    return v,vis


@pytest.mark.parametrize('fps',[10,15,17.5,30])
def test_small_raw_geometry_flips_do_not_prevent_initial_calibration(fps):
    c=WebAutoKnee42Controller(config())
    states=[]
    for i,t in enumerate(np.arange(0,1.7,1/fps)):
        v,vis=borderline_seated_frame(i)
        event=c.add_observation(float(t),v,t,pose_visibility=vis)
        states.append(c.state)
        assert not event.infer
    assert c.calibrated
    assert c.engine.reference_revision==1
    assert set(states)=={'IDLE_BLANK'}


def test_hysteresis_cannot_establish_a_zone_without_initial_strict_evidence():
    c=WebAutoKnee42Controller(config())
    v,vis=borderline_seated_frame(1)
    for i in range(60):
        c.add_observation(i/30,v,i,pose_visibility=vis)
    assert not c.calibrated


@pytest.mark.parametrize('hand_y',[.35,.45])
def test_chest_or_abdomen_pause_is_not_a_knee_return(hand_y):
    c=WebAutoKnee42Controller(config())
    seed(c)
    events=[feed(c,i/10,hand_y=hand_y) for i in range(12,65)]
    assert not any(e.infer for e in events)
    assert c.engine.reference_revision==1


@pytest.mark.parametrize('duration',[6.,8.])
def test_shipped_web_accepts_a_complete_long_gesture(duration):
    cfg=load_web_trigger_config(ROOT/'configs/auto_trigger_knee_web_live.json')
    c=WebAutoKnee42Controller(cfg)
    events=[]
    for t in np.arange(0,2+duration+2,1/15):
        event=feed(c,float(t),hand_y=.35 if 2<=t<2+duration else .78)
        if event.segment:
            events.append(event)
    assert len(events)==1 and events[0].infer
    assert events[0].segment.clip_end_sec-events[0].segment.clip_start_sec > duration-.2
    assert c.engine.reference_revision==2


def test_shipped_web_still_rejects_an_unfinished_gesture_at_its_hard_limit():
    cfg=load_web_trigger_config(ROOT/'configs/auto_trigger_knee_web_live.json')
    c=WebAutoKnee42Controller(cfg)
    events=[]
    for t in np.arange(0,2+cfg.max_segment_sec+2,.1):
        event=feed(c,float(t),hand_y=.35 if t>=2 else .78)
        if event.segment:
            events.append(event)
    assert len(events)==1 and events[0].message=='timeout_finalize' and not events[0].infer


def test_stream_explains_current_segment_elapsed_and_configured_limit():
    from types import SimpleNamespace
    from tests.test_web_knee_stream import payload_frame
    from webservice.server import stream_payload
    service=SimpleNamespace(trigger_config=config(max_segment_sec=10),recognizer=None)
    result=stream_payload(service,{'session':'duration-diagnostic','reset':True,
        'frames':[payload_frame(float(t),hand_y=.35 if t>=1.3 else .78) for t in np.arange(0,3,.1)]})
    assert result.get('segment_limit_sec')==10
    assert 1 < result.get('segment_elapsed_sec',0) < 2


@pytest.mark.parametrize('break_kind',['knees','wrists','gap','relocation','reset'])
def test_a_geometry_latch_does_not_bridge_untrusted_tracking_or_relocation(break_kind):
    c=WebAutoKnee42Controller(config())
    v,vis=borderline_seated_frame(0)
    for i in range(3):
        c.add_observation(i/30,v,i,pose_visibility=vis)
    if break_kind=='reset':
        c.reset()
    elif break_kind in {'knees','wrists'}:
        missing,visibility=borderline_seated_frame(1)
        if break_kind=='knees':
            visibility[[25,26]]=0
        else:
            visibility[[15,16]]=0
            missing[99:]=0
        c.add_observation(.1,missing,0,pose_visibility=visibility)
    for i in range(60):
        v,vis=borderline_seated_frame(1)
        if break_kind=='relocation':
            p=v.reshape(75,3)
            valid=np.any(p,axis=1)
            p[valid,:2]=(p[valid,:2]-.5)*1.3+.5
        t=(.5 if break_kind=='gap' else .2)+i/30
        c.add_observation(t,v,i,pose_visibility=vis)
    assert not c.calibrated
    assert not c.last_analysis.hands_on_knees


@pytest.mark.parametrize('fps',[10,15,17.5,30])
def test_borderline_seated_two_sentence_cycle_uses_one_reference_per_return(fps):
    c=WebAutoKnee42Controller(config(max_segment_sec=10))
    events=[]
    for i,t in enumerate(np.arange(0,8,1/fps)):
        v,vis=borderline_seated_frame(i)
        if 2<=t<3.3 or 5<=t<6.3:
            v.reshape(75,3)[[15,16]+list(range(33,75)),1]=.30
        e=c.add_observation(float(t),v,i,pose_visibility=vis)
        if e.segment:
            events.append(e)
    assert len(events)==2 and all(e.infer for e in events)
    assert c.engine.reference_revision==3
