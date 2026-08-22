from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from recognition.realtime import knee42_capture


class VideoOrientationTests(unittest.TestCase):
    def test_rotation_metadata_canonicalizes_0_90_180_270_degrees(self):
        frame = np.repeat(
            np.arange(1, 7, dtype=np.uint8).reshape(2, 3, 1),
            3,
            axis=2,
        )
        rotate = getattr(knee42_capture, "apply_video_transform", lambda *_args, **_kwargs: None)

        expected = {
            0: np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
            90: np.array([[4, 1], [5, 2], [6, 3]], dtype=np.uint8),
            180: np.array([[6, 5, 4], [3, 2, 1]], dtype=np.uint8),
            270: np.array([[3, 6], [2, 5], [1, 4]], dtype=np.uint8),
        }
        for degrees, wanted in expected.items():
            with self.subTest(degrees=degrees):
                actual = rotate(frame, degrees, horizontal_mirror=False)
                self.assertIsNotNone(actual)
                np.testing.assert_array_equal(actual[:, :, 0], wanted)

    def test_horizontal_mirror_is_explicit_and_separate_from_rotation(self):
        frame = np.repeat(
            np.arange(1, 7, dtype=np.uint8).reshape(2, 3, 1),
            3,
            axis=2,
        )

        actual = knee42_capture.apply_video_transform(
            frame,
            0,
            horizontal_mirror=True,
        )

        np.testing.assert_array_equal(
            actual[:, :, 0],
            np.array([[3, 2, 1], [6, 5, 4]], dtype=np.uint8),
        )

    def test_video_source_reads_rotation_metadata_and_transforms_every_frame(self):
        frame = np.repeat(
            np.arange(1, 7, dtype=np.uint8).reshape(2, 3, 1),
            3,
            axis=2,
        )

        class Capture:
            def __init__(self):
                self.frames = [frame]
                self.set_calls = []

            def isOpened(self): return True
            def release(self): return None
            def read(self): return (True, self.frames.pop(0)) if self.frames else (False, None)
            def get(self, prop): return 90.0 if prop == CV2.CAP_PROP_ORIENTATION_META else 25.0
            def set(self, prop, value): self.set_calls.append((prop, value)); return True

        class CV2:
            CAP_PROP_FPS = 5
            CAP_PROP_ORIENTATION_META = 48
            CAP_PROP_ORIENTATION_AUTO = 49
            capture = Capture()

            @staticmethod
            def VideoCapture(_source): return CV2.capture

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rotated.mp4"
            path.write_bytes(b"fixture")
            source = knee42_capture.open_video(path, cv2_module=CV2)
            ok, actual = source.read()

        self.assertTrue(ok)
        self.assertEqual(getattr(source, "rotation_degrees", None), 90.0)
        self.assertFalse(getattr(source, "horizontal_mirror", True))
        np.testing.assert_array_equal(
            actual[:, :, 0],
            np.array([[4, 1], [5, 2], [6, 3]], dtype=np.uint8),
        )
        self.assertIn((CV2.CAP_PROP_ORIENTATION_AUTO, 0), CV2.capture.set_calls)


if __name__ == "__main__":
    unittest.main()
