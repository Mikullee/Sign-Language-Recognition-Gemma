from __future__ import annotations

import unittest

import numpy as np

from recognition.realtime.knee42_preprocessing import LANDMARK_DIM, POSE_KEEP
from recognition.transformer.landmarks import (
    HAND_LANDMARKS,
    LEFT_SHOULDER,
    LEFT_SHOULDER_X,
    POSE_LANDMARKS,
    RIGHT_SHOULDER,
    TrackedFrame,
    convention_check,
    frame_to_219,
    frames_to_sequence,
    left_shoulder_x_mean,
)
from recognition.transformer.segmentation import (
    analyze_frames,
    motion_energy,
    segment_frames,
)


def _pose(left_x: float = 0.6, right_x: float = 0.4) -> np.ndarray:
    """A minimal upright pose: the signer's left shoulder to the right of frame."""
    pose = np.zeros((POSE_LANDMARKS, 3), dtype=np.float32)
    pose[:, 0] = 0.5
    pose[:, 1] = np.linspace(0.2, 0.9, POSE_LANDMARKS)
    pose[LEFT_SHOULDER] = (left_x, 0.4, 0.0)
    pose[RIGHT_SHOULDER] = (right_x, 0.4, 0.0)
    return pose


def _hand(x: float, y: float) -> np.ndarray:
    hand = np.zeros((HAND_LANDMARKS, 3), dtype=np.float32)
    hand[:, 0] = x
    hand[:, 1] = y
    return hand


def _frame(index: int, timestamp: float, *, hands: dict | None = None, pose=None) -> TrackedFrame:
    return TrackedFrame(
        index=index,
        timestamp=timestamp,
        pose=_pose() if pose is None else pose,
        hands=hands or {},
    )


class FrameConversionTests(unittest.TestCase):
    def test_a_frame_becomes_219_shoulder_normalized_values(self):
        values = frame_to_219(_frame(0, 0.0, hands={"Right": _hand(0.3, 0.6)}), mirrored=False)
        self.assertIsNotNone(values)
        self.assertEqual(values.shape, (LANDMARK_DIM,))

    def test_left_shoulder_lands_on_positive_x_like_the_training_cache(self):
        values = frame_to_219(_frame(0, 0.0), mirrored=False)
        self.assertAlmostEqual(float(values[LEFT_SHOULDER_X]), 0.5, places=3)
        measured, ok = convention_check(values[None])
        self.assertTrue(ok)
        self.assertGreater(measured, 0)

    def test_mirroring_flips_the_convention(self):
        values = frame_to_219(_frame(0, 0.0), mirrored=True)
        self.assertAlmostEqual(float(values[LEFT_SHOULDER_X]), -0.5, places=3)
        self.assertFalse(convention_check(values[None])[1])

    def test_a_frame_without_both_shoulders_is_dropped(self):
        pose = _pose()
        pose[RIGHT_SHOULDER] = np.nan
        self.assertIsNone(frame_to_219(_frame(0, 0.0, pose=pose), mirrored=False))

    def test_an_undetected_hand_stays_nan_rather_than_zero(self):
        values = frame_to_219(_frame(0, 0.0, hands={"Right": _hand(0.3, 0.6)}), mirrored=False)
        left_block = values[len(POSE_KEEP) * 3 : len(POSE_KEEP) * 3 + HAND_LANDMARKS * 3]
        self.assertTrue(np.all(np.isnan(left_block)))

    def test_frames_to_sequence_can_drop_or_keep_gaps(self):
        broken = _pose()
        broken[LEFT_SHOULDER] = np.nan
        frames = [_frame(0, 0.0), _frame(1, 0.1, pose=broken), _frame(2, 0.2)]
        self.assertEqual(len(frames_to_sequence(frames, mirrored=False)), 2)
        kept = frames_to_sequence(frames, mirrored=False, keep_gaps=True)
        self.assertEqual(len(kept), 3)
        self.assertIsNone(kept[1])

    def test_left_shoulder_x_mean_rejects_a_wrong_width(self):
        with self.assertRaises(ValueError):
            left_shoulder_x_mean(np.zeros((3, LANDMARK_DIM - 1), dtype=np.float32))


class SegmentationTests(unittest.TestCase):
    def test_motion_energy_is_zero_without_hands(self):
        frames = [_frame(index, index * 0.1) for index in range(5)]
        self.assertEqual(motion_energy(frames), [0.0] * 5)

    def test_a_hand_entering_the_frame_counts_as_motion(self):
        frames = [_frame(0, 0.0), _frame(1, 0.1, hands={"Right": _hand(0.3, 0.6)})]
        self.assertGreater(motion_energy(frames)[1], 0.0)

    def test_a_moving_wrist_produces_speed_in_units_per_second(self):
        frames = [
            _frame(0, 0.0, hands={"Right": _hand(0.30, 0.6)}),
            _frame(1, 0.5, hands={"Right": _hand(0.40, 0.6)}),
        ]
        self.assertAlmostEqual(motion_energy(frames)[1], 0.2, places=4)

    def test_a_pause_closes_a_segment(self):
        frames = []
        for index in range(12):
            timestamp = index * 0.1
            moving = index < 6
            position = 0.3 + (0.05 * index if moving else 0.55)
            frames.append(_frame(index, timestamp, hands={"Right": _hand(position, 0.6)}))
        for index in range(12, 24):
            frames.append(_frame(index, index * 0.1, hands={"Right": _hand(0.55, 0.6)}))
        segments = segment_frames(frames, motion_threshold=0.1, min_duration=0.2, pause=0.4)
        self.assertEqual(len(segments), 1)
        start, end = segments[0]
        self.assertLess(start, end)

    def test_no_frames_means_no_segments(self):
        self.assertEqual(segment_frames([]), [])


class _StubRecognizer:
    labels = ["K42_01", "K42_02"]

    def predict(self, matrix, topk=3):
        self.last = matrix
        return [("K42_01", "你好", 0.9), ("K42_02", "早安", 0.1)][:topk]


class AnalyzeFramesTests(unittest.TestCase):
    def test_a_clip_with_no_pause_still_returns_a_whole_result(self):
        frames = [
            _frame(index, index * 0.05, hands={"Right": _hand(0.3 + 0.01 * index, 0.6)})
            for index in range(20)
        ]
        result = analyze_frames(frames, _StubRecognizer(), topk=2)
        self.assertIsNotNone(result["whole"])
        self.assertEqual(result["whole"]["top"][0]["label"], "K42_01")
        self.assertTrue(result["convention_ok"])
        self.assertEqual(result["n_valid"], 20)

    def test_too_few_usable_frames_reports_a_message_instead_of_guessing(self):
        broken = _pose()
        broken[LEFT_SHOULDER] = np.nan
        frames = [_frame(index, index * 0.1, pose=broken) for index in range(10)]
        result = analyze_frames(frames, _StubRecognizer())
        self.assertIsNone(result["whole"])
        self.assertIn("shoulders", result["message"])

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze_frames([], _StubRecognizer())


if __name__ == "__main__":
    unittest.main()
