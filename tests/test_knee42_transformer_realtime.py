from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from recognition.realtime.auto_trigger import load_auto_trigger_config
from recognition.realtime.knee42_preprocessing import LANDMARK_DIM, MODEL_INPUT_DIM
from recognition.transformer.realtime import (
    DEFAULT_TRIGGER_CONFIG,
    MIN_SEGMENT_FRAMES,
    Detectors,
    Recognition,
    recognize_stream,
    sequence_from_features,
)


ROOT = Path(__file__).resolve().parents[1]


class SequenceAssemblyTests(unittest.TestCase):
    def test_features_stack_into_the_219_wide_transformer_input(self):
        features = [
            (np.full(LANDMARK_DIM, index, dtype=np.float32), np.ones(LANDMARK_DIM, bool))
            for index in range(7)
        ]
        sequence = sequence_from_features(features)
        self.assertEqual(sequence.shape, (7, LANDMARK_DIM))
        self.assertEqual(sequence.dtype, np.float32)

    def test_the_mask_is_dropped_and_nan_gaps_survive(self):
        values = np.full(LANDMARK_DIM, 0.5, dtype=np.float32)
        values[3] = np.nan
        sequence = sequence_from_features([(values, np.isfinite(values))])
        self.assertEqual(sequence.shape[1], LANDMARK_DIM)
        self.assertTrue(np.isnan(sequence[0, 3]))

    def test_the_trigger_view_is_wider_than_the_recognition_view(self):
        """225 trigger values keep the knees; the 219 recognition values do not."""
        self.assertEqual(LANDMARK_DIM, 219)
        self.assertEqual(MODEL_INPUT_DIM, 438)


class _FakeSource:
    """A capture source that replays prepared frames, like a camera would."""

    def __init__(self, frames: list[np.ndarray], fps: float = 30.0):
        self._frames = frames
        self._index = 0
        self.fps = fps
        self.status = "fake"
        self.released = False

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self):
        self.released = True


class _FakeDetectors:
    """Return a scripted observation per frame, bypassing MediaPipe."""

    def __init__(self, observations):
        self._observations = observations
        self._index = 0

    def observe(self, _frame):
        observation = self._observations[min(self._index, len(self._observations) - 1)]
        self._index += 1
        return observation


class _FakeRecognizer:
    labels = ["K42_01"]

    def __init__(self):
        self.calls: list[np.ndarray] = []

    def predict(self, sequence, topk=3):
        self.calls.append(sequence)
        return [("K42_12", "可以", 0.9)][:topk]


class StreamWiringTests(unittest.TestCase):
    def test_a_stream_with_no_segments_returns_nothing_and_still_reads_every_frame(self):
        from recognition.realtime.knee42_preprocessing import FrameObservation

        still = FrameObservation(
            trigger_values=np.zeros(225, dtype=np.float32),
            recognition_values=np.zeros(LANDMARK_DIM, dtype=np.float32),
            recognition_mask=np.ones(LANDMARK_DIM, bool),
        )
        source = _FakeSource([np.zeros((4, 4, 3), np.uint8)] * 40)
        recognizer = _FakeRecognizer()
        results, raw = recognize_stream(
            source,
            _FakeDetectors([still]),
            recognizer,
            load_auto_trigger_config(ROOT / DEFAULT_TRIGGER_CONFIG),
            frame_step=2,
        )
        self.assertEqual(results, [])
        self.assertEqual(raw, 40)
        self.assertEqual(recognizer.calls, [])

    def test_frame_step_controls_how_often_detection_runs(self):
        from recognition.realtime.knee42_preprocessing import FrameObservation

        observation = FrameObservation(
            trigger_values=np.zeros(225, dtype=np.float32),
            recognition_values=np.zeros(LANDMARK_DIM, dtype=np.float32),
            recognition_mask=np.ones(LANDMARK_DIM, bool),
        )
        detectors = _FakeDetectors([observation])
        recognize_stream(
            _FakeSource([np.zeros((4, 4, 3), np.uint8)] * 20),
            detectors,
            _FakeRecognizer(),
            load_auto_trigger_config(ROOT / DEFAULT_TRIGGER_CONFIG),
            frame_step=4,
        )
        self.assertEqual(detectors._index, 5)

    def test_max_frames_stops_the_stream_early(self):
        from recognition.realtime.knee42_preprocessing import FrameObservation

        observation = FrameObservation(
            trigger_values=np.zeros(225, dtype=np.float32),
            recognition_values=np.zeros(LANDMARK_DIM, dtype=np.float32),
            recognition_mask=np.ones(LANDMARK_DIM, bool),
        )
        _, raw = recognize_stream(
            _FakeSource([np.zeros((4, 4, 3), np.uint8)] * 100),
            _FakeDetectors([observation]),
            _FakeRecognizer(),
            load_auto_trigger_config(ROOT / DEFAULT_TRIGGER_CONFIG),
            frame_step=2,
            max_frames=10,
        )
        self.assertEqual(raw, 11)


class RecognitionFormattingTests(unittest.TestCase):
    def test_a_classified_segment_reports_its_boundaries_and_ranking(self):
        text = Recognition(
            index=1, start_sec=2.5, end_sec=4.25, frames=105,
            reason="reference_rest_finalize",
            top=[("K42_12", "可以", 0.62), ("K42_20", "我要看醫生", 0.03)],
        ).format()
        self.assertIn("2.50", text)
        self.assertIn("4.25", text)
        self.assertIn("可以 0.62", text)
        self.assertIn("reference_rest_finalize", text)

    def test_a_segment_too_short_to_classify_says_so_rather_than_guessing(self):
        text = Recognition(
            index=2, start_sec=0.0, end_sec=0.1, frames=2, reason="short", top=[]
        ).format()
        self.assertIn("too short", text)

    def test_duration_never_goes_negative(self):
        self.assertEqual(
            Recognition(index=1, start_sec=5.0, end_sec=1.0, frames=1, reason="x", top=[]).duration_sec,
            0.0,
        )


class DetectorGuardTests(unittest.TestCase):
    def test_a_missing_model_names_the_fetch_script(self):
        with self.assertRaises(FileNotFoundError) as caught:
            Detectors(Path("nope/hand.task"), Path("nope/pose.task"))
        self.assertIn("fetch_mediapipe_models.py", str(caught.exception))

    def test_a_segment_shorter_than_the_floor_is_not_classified(self):
        self.assertGreater(MIN_SEGMENT_FRAMES, 1)


if __name__ == "__main__":
    unittest.main()
