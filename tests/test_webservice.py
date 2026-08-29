from __future__ import annotations

import io
import unittest

import numpy as np

from pathlib import Path

from recognition.transformer.landmarks import HAND_LANDMARKS, POSE_LANDMARKS
from webservice.server import (
    _boundary_of,
    _points,
    frames_from_payload,
    read_multipart_file,
)


ROOT = Path(__file__).resolve().parents[1]


def _multipart(filename: str, payload: bytes, boundary: bytes = b"----abc123") -> tuple[bytes, str]:
    body = b"".join(
        [
            b"--", boundary, b"\r\n",
            b'Content-Disposition: form-data; name="file"; filename="', filename.encode(), b'"\r\n',
            b"Content-Type: video/mp4\r\n\r\n",
            payload,
            b"\r\n--", boundary, b"--\r\n",
        ]
    )
    return body, f"multipart/form-data; boundary={boundary.decode()}"


class MultipartTests(unittest.TestCase):
    def test_boundary_is_read_from_the_content_type(self):
        self.assertEqual(_boundary_of("multipart/form-data; boundary=xyz"), b"xyz")
        self.assertEqual(_boundary_of('multipart/form-data; boundary="xyz"'), b"xyz")

    def test_a_content_type_without_a_boundary_is_rejected(self):
        with self.assertRaises(ValueError):
            _boundary_of("multipart/form-data")

    def test_the_file_part_round_trips_exactly(self):
        payload = bytes(range(256)) * 40
        body, content_type = _multipart("clip.mp4", payload)
        destination = io.BytesIO()
        filename = read_multipart_file(io.BytesIO(body), len(body), content_type, destination)
        self.assertEqual(filename, "clip.mp4")
        self.assertEqual(destination.getvalue(), payload)

    def test_a_payload_larger_than_one_chunk_round_trips(self):
        payload = b"\x00\x01\x02\x03" * 400_000
        body, content_type = _multipart("big.mp4", payload)
        destination = io.BytesIO()
        read_multipart_file(io.BytesIO(body), len(body), content_type, destination)
        self.assertEqual(destination.getvalue(), payload)

    def test_a_body_with_no_file_field_is_rejected(self):
        body = b'------abc123\r\nContent-Disposition: form-data; name="note"\r\n\r\nhi\r\n------abc123--\r\n'
        with self.assertRaises(ValueError):
            read_multipart_file(
                io.BytesIO(body), len(body), "multipart/form-data; boundary=----abc123", io.BytesIO()
            )


class PredictPayloadTests(unittest.TestCase):
    def _frame(self, with_hands: bool = True) -> dict:
        entry = {
            "timestamp": 0.5,
            "pose": np.zeros((POSE_LANDMARKS, 3)).tolist(),
        }
        if with_hands:
            entry["hands"] = [
                {"handedness": "Right", "landmarks": np.zeros((HAND_LANDMARKS, 3)).tolist()}
            ]
        return entry

    def test_frames_and_handedness_survive_the_round_trip(self):
        frames = frames_from_payload([self._frame(), self._frame()])
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].timestamp, 0.5)
        self.assertIn("Right", frames[0].hands)
        self.assertEqual(frames[0].pose.shape, (POSE_LANDMARKS, 3))

    def test_a_frame_without_hands_is_kept_rather_than_dropped(self):
        frames = frames_from_payload([self._frame(with_hands=False)])
        self.assertEqual(len(frames), 1)
        self.assertFalse(frames[0].has_hands)

    def test_landmarks_of_the_wrong_shape_are_ignored_not_trusted(self):
        frames = frames_from_payload(
            [{"pose": [[0.0, 0.0, 0.0]], "hands": [{"handedness": "Left", "landmarks": [[0, 0, 0]]}]}]
        )
        self.assertIsNone(frames[0].pose)
        self.assertEqual(frames[0].hands, {})

    def test_an_unknown_handedness_label_is_ignored(self):
        frames = frames_from_payload(
            [{"hands": [{"handedness": "Middle", "landmarks": np.zeros((HAND_LANDMARKS, 3)).tolist()}]}]
        )
        self.assertEqual(frames[0].hands, {})

    def test_non_dict_entries_are_skipped(self):
        self.assertEqual(len(frames_from_payload(["nope", 42, self._frame()])), 1)


class BrowserLandmarkShapeTests(unittest.TestCase):
    """The page forwards MediaPipe output as-is, wrappers included."""

    def test_mediapipe_wraps_pose_in_a_landmarks_key(self):
        wrapped = {"landmarks": np.zeros((POSE_LANDMARKS, 3)).tolist()}
        parsed = _points(wrapped, POSE_LANDMARKS)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.shape, (POSE_LANDMARKS, 3))

    def test_a_bare_list_still_works(self):
        bare = np.zeros((HAND_LANDMARKS, 3)).tolist()
        self.assertIsNotNone(_points(bare, HAND_LANDMARKS))

    def test_a_real_browser_frame_yields_a_pose(self):
        """Regression: treating only the bare form as valid dropped every pose."""
        frame = {
            "pose": {"landmarks": np.zeros((POSE_LANDMARKS, 3)).tolist()},
            "hands": [
                {
                    "handedness": "Right",
                    "score": 0.98,
                    "landmarks": np.zeros((HAND_LANDMARKS, 3)).tolist(),
                }
            ],
        }
        parsed = frames_from_payload([frame])
        self.assertIsNotNone(parsed[0].pose)
        self.assertIn("Right", parsed[0].hands)

    def test_malformed_landmarks_are_ignored_not_raised(self):
        self.assertIsNone(_points({"landmarks": "nope"}, POSE_LANDMARKS))
        self.assertIsNone(_points({"nothing": 1}, POSE_LANDMARKS))
        self.assertIsNone(_points([[0, 0]], POSE_LANDMARKS))


class StreamSessionTests(unittest.TestCase):
    """Auto mode keeps one calibrated state machine per browser tab."""

    def test_each_session_gets_its_own_state_machine(self):
        from recognition.realtime.auto_trigger import load_auto_trigger_config
        from webservice.server import _STREAMS, _stream_session

        config = load_auto_trigger_config(ROOT / "configs" / "auto_trigger_knee_v1.json")
        first = _stream_session("tab-a", config, reset=True)
        second = _stream_session("tab-b", config, reset=True)
        self.assertIsNot(first["controller"], second["controller"])
        self.assertIn("tab-a", _STREAMS)
        self.assertIn("tab-b", _STREAMS)

    def test_reset_replaces_the_machine_rather_than_reusing_it(self):
        from recognition.realtime.auto_trigger import load_auto_trigger_config
        from webservice.server import _stream_session

        config = load_auto_trigger_config(ROOT / "configs" / "auto_trigger_knee_v1.json")
        original = _stream_session("tab-c", config, reset=True)["controller"]
        same = _stream_session("tab-c", config, reset=False)["controller"]
        fresh = _stream_session("tab-c", config, reset=True)["controller"]
        self.assertIs(same, original)
        self.assertIsNot(fresh, original)

    def test_expired_sessions_are_swept(self):
        from recognition.realtime.auto_trigger import load_auto_trigger_config
        from webservice.server import _STREAMS, _stream_session, _sweep_streams

        config = load_auto_trigger_config(ROOT / "configs" / "auto_trigger_knee_v1.json")
        _stream_session("tab-old", config, reset=True)
        _STREAMS["tab-old"]["seen"] = 0.0
        _sweep_streams()
        self.assertNotIn("tab-old", _STREAMS)


class PageContractTests(unittest.TestCase):
    """The response must carry every field the page reads.

    Regression: /predict returned the analysis shape but not `top5`, so the page
    threw `Cannot read properties of undefined (reading '0')` -- which its own
    catch block reported as "cannot reach the server", sending the diagnosis in
    entirely the wrong direction. The field list is parsed from the page rather
    than hard-coded, so adding a `d.something` in the UI fails here until the
    server provides it.
    """

    PAGE = ROOT / "webservice" / "static" / "index.html"

    def _fields_read_by(self, function_name: str) -> set[str]:
        import re

        source = self.PAGE.read_text(encoding="utf-8")
        start = source.index(f"function {function_name}(")
        body = source[start : source.index("\n}", start)]
        return set(re.findall(r"\bd\.([a-z_0-9]+)", body))

    def test_predict_returns_every_field_the_camera_view_reads(self):
        import json
        from types import SimpleNamespace

        from recognition.transformer.recognizer import Knee42TransformerRecognizer
        from webservice.server import predict_payload

        wanted = self._fields_read_by("renderCam")
        self.assertIn("top5", wanted, "the page stopped reading top5; update this test")

        config = SimpleNamespace(
            recognizer=Knee42TransformerRecognizer(
                ROOT / "artifacts" / "realtime" / "best_current"
            )
        )
        frame = {
            "pose": {"landmarks": [[0.5, 0.4, 0.0]] * POSE_LANDMARKS},
            "hands": [
                {"handedness": "Right", "landmarks": [[0.4, 0.6, 0.0]] * HAND_LANDMARKS}
            ],
        }
        for index, entry in enumerate([dict(frame, timestamp=i / 30) for i in range(30)]):
            pass
        payload = {"frames": [dict(frame, timestamp=i / 30) for i in range(30)]}
        response = predict_payload(config, payload)
        response["ok"] = True  # the handler adds this before sending

        # message is only read in the !d.ok branch, so a success response
        # legitimately omits it.
        wanted -= {"message"}
        missing = sorted(field for field in wanted if field not in response)
        self.assertEqual(missing, [], f"page reads fields /predict never sends: {missing}")
        json.dumps(response, ensure_ascii=False)  # must survive serialization

    def test_top5_is_ranked_and_carries_what_the_bars_need(self):
        from types import SimpleNamespace

        from recognition.transformer.recognizer import Knee42TransformerRecognizer
        from webservice.server import predict_payload

        config = SimpleNamespace(
            recognizer=Knee42TransformerRecognizer(
                ROOT / "artifacts" / "realtime" / "best_current"
            )
        )
        frame = {
            "pose": {"landmarks": [[0.5, 0.4, 0.0]] * POSE_LANDMARKS},
            "hands": [
                {"handedness": "Right", "landmarks": [[0.4, 0.6, 0.0]] * HAND_LANDMARKS}
            ],
        }
        top5 = predict_payload(
            config, {"frames": [dict(frame, timestamp=i / 30) for i in range(30)]}
        )["top5"]

        self.assertEqual(len(top5), 5)
        for item in top5:
            self.assertEqual(set(item), {"label", "text", "prob"})
        probabilities = [item["prob"] for item in top5]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))


if __name__ == "__main__":
    unittest.main()
