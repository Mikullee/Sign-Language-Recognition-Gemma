from __future__ import annotations

import io
import unittest

import numpy as np

from recognition.transformer.landmarks import HAND_LANDMARKS, POSE_LANDMARKS
from webservice.server import (
    _boundary_of,
    frames_from_payload,
    read_multipart_file,
)


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


if __name__ == "__main__":
    unittest.main()
