"""Synthetic regressions, not claims about real-person accuracy."""
import numpy as np
import pytest

from recognition.transformer.live_trigger import WebAutoKnee42Controller
from tests.test_web_knee_gate import config, skeleton


def noisy_frame(i, *, hand_y=.78):
    vector, vis = skeleton(hand_y=hand_y)
    points = vector.reshape(75, 3)
    # Bounded 0.6%-image peak-to-peak detector jitter, not new signers/data.
    offset = .003 if i % 2 else -.003
    points[[13,14,15,16], 0] += offset
    points[33:, 0] += offset
    return vector, vis


@pytest.mark.parametrize('fps', [10,15,17.5,30])
def test_small_tracking_jitter_calibrates_and_completes_a_sentence(fps):
    c = WebAutoKnee42Controller(config())
    events = []
    features = {}
    for i,t in enumerate(np.arange(0,6,1/fps)):
        v,vis = noisy_frame(i,hand_y=.35 if 2.5<=t<3.8 else .78)
        original = v.copy()
        feature = object()
        features[float(t)] = feature
        event = c.add_observation(float(t),v,feature,pose_visibility=vis)
        np.testing.assert_array_equal(v,original)
        if event.infer:
            events.append(event)
            assert all(any(item is f for f in features.values()) for item in event.features)
    assert c.calibrated
    assert len(events)==1
    assert events[0].segment.reason=='visible_rest_finalize'
    assert abs(events[0].segment.clip_end_sec-3.8)<=.3


def test_handedness_label_swaps_do_not_become_physical_motion():
    c = WebAutoKnee42Controller(config())
    for i in range(65):
        v,vis = skeleton()
        p = v.reshape(75,3)
        if i % 2:
            p[33:54],p[54:75] = p[54:75].copy(),p[33:54].copy()
        c.add_observation(i/30,v,i,pose_visibility=vis)
    assert c.calibrated
    assert c.state=='IDLE_BLANK'


def test_one_hand_cannot_count_twice_when_pose_wrist_is_missing():
    c = WebAutoKnee42Controller(config())
    v,vis=skeleton()
    p=v.reshape(75,3)
    p[33:54]=p[54:75].copy()  # right hand mislabeled left
    p[54:75]=0
    vis[15]=0
    for i in range(45):
        c.add_observation(i/30,v,i,pose_visibility=vis)
    assert not c.last_analysis.wrists_detected
    assert not c.calibrated


def test_visible_palm_on_seated_knee_is_not_rejected_for_wrist_height():
    c=WebAutoKnee42Controller(config())
    v,vis=skeleton()
    p=v.reshape(75,3)
    for idx,xy in {15:(.39,.53),16:(.61,.53),23:(.4,.6),24:(.6,.6),
                   25:(.45,.68),26:(.55,.68)}.items():
        p[idx]=[*xy,.01]
    for start,pose,palm in [(33,15,(.43,.66)),(54,16,(.57,.66))]:
        p[start:start+21]=[*palm,.01]
        p[start]=p[pose]
    for i in range(65):
        c.add_observation(i/30,v,i,pose_visibility=vis)
    assert c.calibrated


@pytest.mark.parametrize('movement', ['drift','oscillation','chest'])
def test_smoothing_never_turns_real_movement_or_chest_into_calibration(movement):
    c=WebAutoKnee42Controller(config())
    for i,t in enumerate(np.arange(0,1.6,1/30)):
        y=.35 if movement=='chest' else (.75+.09*t if movement=='drift' else .78+.025*np.sin(2*np.pi*5*t))
        v,vis=skeleton(hand_y=y)
        c.add_observation(float(t),v,i,pose_visibility=vis)
    assert not c.calibrated


def test_steady_rest_after_reset_does_not_inherit_previous_motion():
    c=WebAutoKnee42Controller(config())
    for i in range(20):
        v,vis=skeleton(hand_y=.35 if i%2 else .78)
        c.add_observation(i/30,v,i,pose_visibility=vis)
    c.reset()
    for i in range(50):
        v,vis=skeleton()
        c.add_observation(i/30,v,i,pose_visibility=vis)
    assert c.calibrated


def test_switching_wrist_sources_cannot_erase_continuous_motion():
    c=WebAutoKnee42Controller(config())
    for i in range(65):
        y=.78+.04*np.sin(2*np.pi*2*i/30)
        v,vis=skeleton(hand_y=y)
        if i%3==0:
            v[99:]=0
        else:
            vis[[15,16]]=0
        c.add_observation(i/30,v,i,pose_visibility=vis)
    assert not c.calibrated


@pytest.mark.parametrize('fps',[5,6])
def test_slow_observed_frames_still_allow_static_rest(fps):
    c=WebAutoKnee42Controller(config())
    for i in range(fps*3):
        v,vis=skeleton()
        c.add_observation(i/fps,v,i,pose_visibility=vis)
    assert c.calibrated


@pytest.mark.parametrize('moving_part',['wrist','fingers'])
def test_intermittent_hands_do_not_hide_motion_behind_static_pose(moving_part):
    c=WebAutoKnee42Controller(config())
    for i in range(65):
        v,vis=skeleton()
        p=v.reshape(75,3)
        y=.78+.04*np.sin(2*np.pi*2*i/30)
        for start in (33,54):
            p[start+(moving_part=='fingers'):start+21,1]=y
        if i%3==0:
            v[99:]=0
        c.add_observation(i/30,v,i,pose_visibility=vis)
    assert not c.calibrated
