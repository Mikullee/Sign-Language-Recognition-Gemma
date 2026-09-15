from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from recognition.transformer.landmarks import observation_from_frame
from recognition.transformer.live_trigger import WebAutoKnee42Controller
from tests.test_web_knee_gate import config, skeleton
from webservice.server import frames_from_payload, stream_payload


class Recognizer:
    def predict(self, sequence, topk):
        assert sequence.shape[1] == 219
        return [("K42_01", "test", .9)]


def payload_frame(t, **kwargs):
    values, visibility = skeleton(**kwargs)
    return {"timestamp": t, "pose": {"landmarks": values[:99].reshape(33, 3).tolist(),
                                       "visibility": visibility.tolist()},
            "hands": [{"handedness": side, "landmarks": values[start:start + 63].reshape(21, 3).tolist()}
                      for side, start in (("Left", 99), ("Right", 162))]}


def test_visibility_roundtrip_does_not_change_recognition_values():
    raw = payload_frame(0)
    tracked = frames_from_payload([raw])[0]
    np.testing.assert_equal(tracked.pose_visibility, raw["pose"]["visibility"])
    before = observation_from_frame(tracked).recognition_values
    raw["pose"]["visibility"] = [0] * 33
    after = observation_from_frame(frames_from_payload([raw])[0]).recognition_values
    np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("batch", [1, 7, 200])
def test_web_batch_and_direct_controller_share_exact_boundaries(batch):
    raw = [payload_frame(float(t), hand_y=.35 if 1.3 <= t < 2.8 else .78)
           for t in np.arange(0, 4.8, 1 / 17.5)]
    c = WebAutoKnee42Controller(config())
    expected = []
    for frame in frames_from_payload(raw):
        obs = observation_from_frame(frame)
        event = c.add_observation(frame.timestamp, obs.trigger_values,
                                  (obs.recognition_values, obs.recognition_mask),
                                  pose_visibility=frame.pose_visibility)
        if event.segment:
            expected.append(asdict(event.segment))
    service = SimpleNamespace(trigger_config=config(), recognizer=Recognizer())
    actual = []
    for start in range(0, len(raw), batch):
        result = stream_payload(service, {"session": f"parity-{batch}", "reset": start == 0,
                                          "frames": raw[start:start + batch]})
        actual.extend(item["segment"] for item in result["events"])
    assert actual == expected
    assert len(actual) == 1
    assert result["reference_revision"] == 2


def test_stream_diagnostics_preserve_actual_motion_and_failure():
    service = SimpleNamespace(trigger_config=config(max_segment_sec=1.5), recognizer=None)
    raw = [payload_frame(float(t), hand_y=.35 if t >= 1.3 else .78)
           for t in np.arange(0, 1.41, .1)]
    result = stream_payload(service, {"session": "motion", "reset": True, "frames": raw})
    # Deliberately stop on the first movement frame, not a recomputed zero delta.
    result = stream_payload(service, {"session": "motion2", "reset": True, "frames": raw[:-1]})
    assert result["motion_score"] > 0
    result = stream_payload(service, {"session": "timeout", "reset": True,
        "frames": [payload_frame(float(t), hand_y=.35 if t >= 1.3 else .78)
                   for t in np.arange(0, 4, .1)]})
    assert result["results"] == []
    assert result["failure_count"] == 1
    assert result["events"][0]["segment"]["reason"] == "timeout_finalize"


def test_web_default_requires_real_knee_calibration():
    from pathlib import Path
    from recognition.transformer.live_trigger import load_web_trigger_config
    c = load_web_trigger_config(Path(__file__).resolve().parents[1] / "configs/auto_trigger_knee_web_live.json")
    assert c.knee_gate_enabled
    assert c.adaptive_rearm_requires_knee_rest
    assert c.blank_motion_threshold < c.start_motion_threshold


def test_missing_knee_diagnostics_explain_why_calibration_is_waiting():
    service = SimpleNamespace(trigger_config=config(), recognizer=None)
    result = stream_payload(service, {"session": "missing-knee-reason", "reset": True,
        "frames": [payload_frame(0, knee_visible=False)]})
    assert result["rest_signature_status"] == "waiting_visible_knees"


def test_ui_uses_rest_gate_not_distance_alone_for_green_confirmation():
    from pathlib import Path
    page = (Path(__file__).resolve().parents[1] / "webservice/static/index.html").read_text(encoding="utf-8")
    assert "const ok = j.rest_candidate === true" in page
