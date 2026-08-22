from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from recognition.realtime.knee42_preprocessing import (
    LANDMARK_DIM,
    POSE_KEEP,
    flatten_landmarks,
    landmarks_from_results,
    materialize_sequence,
    normalize_frame,
    observation_from_results,
)
from recognition.training.train_knee42_bigru import CACHE_VERSION, Knee42Dataset


def landmarks(count: int, offset: float) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            x=np.float32(offset + index * 10 + 1),
            y=np.float32(offset + index * 10 + 2),
            z=np.float32(offset + index * 10 + 3),
        )
        for index in range(count)
    ]


class Knee42PreprocessingTests(unittest.TestCase):
    def test_training_uses_upright_v2_feature_cache(self):
        self.assertEqual(CACHE_VERSION, "knee42_features_upright_v2")

    def test_pose_25_26_are_removed_and_output_order_is_219(self):
        pose = landmarks(33, 0.0)
        left = landmarks(21, 1_000.0)
        right = landmarks(21, 2_000.0)

        values, mask = flatten_landmarks(pose, left, right)

        expected = np.asarray(
            [
                coordinate
                for index in POSE_KEEP
                for coordinate in (index * 10 + 1, index * 10 + 2, index * 10 + 3)
            ]
            + [
                coordinate
                for offset in (1_000, 2_000)
                for index in range(21)
                for coordinate in (
                    offset + index * 10 + 1,
                    offset + index * 10 + 2,
                    offset + index * 10 + 3,
                )
            ],
            dtype=np.float32,
        )
        self.assertEqual(LANDMARK_DIM, 219)
        self.assertEqual(values.shape, (219,))
        np.testing.assert_array_equal(values, expected)
        np.testing.assert_array_equal(mask, np.ones(219, dtype=np.bool_))
        self.assertNotIn(251.0, values.tolist())
        self.assertNotIn(261.0, values.tolist())

    def test_missing_landmark_group_stays_nan_with_false_mask(self):
        values, mask = flatten_landmarks(landmarks(33, 0.0), None, None)

        self.assertEqual(int(mask.sum()), len(POSE_KEEP) * 3)
        self.assertTrue(np.isnan(values[len(POSE_KEEP) * 3 :]).all())
        self.assertFalse(mask[len(POSE_KEEP) * 3 :].any())

    def test_mediapipe_results_are_mapped_by_handedness_not_detection_order(self):
        right = landmarks(21, 2_000.0)
        left = landmarks(21, 1_000.0)
        hand_result = SimpleNamespace(
            handedness=[
                [SimpleNamespace(category_name="Right")],
                [SimpleNamespace(category_name="Left")],
            ],
            hand_landmarks=[right, left],
        )
        pose_result = SimpleNamespace(pose_landmarks=[landmarks(33, 0.0)])

        values, mask = landmarks_from_results(hand_result, pose_result)

        left_start = len(POSE_KEEP) * 3
        right_start = left_start + 21 * 3
        self.assertEqual(values[left_start], 1_001.0)
        self.assertEqual(values[right_start], 2_001.0)
        self.assertTrue(mask.all())

    def test_observation_preserves_full_pose_and_handed_display_arrays(self):
        right = landmarks(21, 2_000.0)
        left = landmarks(21, 1_000.0)
        hand_result = SimpleNamespace(
            handedness=[
                [SimpleNamespace(category_name="Right")],
                [SimpleNamespace(category_name="Left")],
            ],
            hand_landmarks=[right, left],
        )
        pose_result = SimpleNamespace(pose_landmarks=[landmarks(33, 0.0)])

        observation = observation_from_results(hand_result, pose_result)

        self.assertEqual(observation.display_pose.shape, (33, 3))
        self.assertEqual(observation.display_left_hand.shape, (21, 3))
        self.assertEqual(observation.display_right_hand.shape, (21, 3))
        self.assertEqual(observation.display_pose[25, 0], 251.0)
        self.assertEqual(observation.display_left_hand[0, 0], 1_001.0)
        self.assertEqual(observation.display_right_hand[0, 0], 2_001.0)

    def test_observation_uses_nan_display_arrays_for_missing_hands(self):
        observation = observation_from_results(
            None,
            SimpleNamespace(pose_landmarks=[landmarks(33, 0.0)]),
        )

        self.assertTrue(np.isnan(observation.display_left_hand).all())
        self.assertTrue(np.isnan(observation.display_right_hand).all())

    def test_shoulder_normalization_matches_training_contract(self):
        pose = landmarks(33, 0.0)
        pose[11] = SimpleNamespace(x=1.0, y=2.0, z=3.0)
        pose[12] = SimpleNamespace(x=3.0, y=2.0, z=3.0)
        values, mask = flatten_landmarks(pose, None, None)

        normalized = normalize_frame(values, mask).reshape(-1, 3)
        pose_position = {source: target for target, source in enumerate(POSE_KEEP)}

        np.testing.assert_allclose(
            normalized[pose_position[11]], [-0.5, 0.0, 0.0], atol=1e-7
        )
        np.testing.assert_allclose(
            normalized[pose_position[12]], [0.5, 0.0, 0.0], atol=1e-7
        )
        self.assertTrue(np.isnan(normalized[len(POSE_KEEP) :]).all())

    def test_materialization_is_equivalent_to_training_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frame_count = 7
            values = np.arange(frame_count * LANDMARK_DIM, dtype=np.float32).reshape(
                frame_count, LANDMARK_DIM
            )
            mask = np.ones_like(values, dtype=np.bool_)
            values[2, 8] = np.nan
            mask[2, 8] = False
            mean = np.linspace(-1.0, 1.0, LANDMARK_DIM, dtype=np.float32)
            std = np.linspace(0.5, 2.0, LANDMARK_DIM, dtype=np.float32)
            np.savez_compressed(
                root / "sample.npz",
                cache_version=np.asarray(CACHE_VERSION),
                values=values,
                mask=mask,
            )
            rows = [
                {
                    "sample_id": "sample",
                    "label_id": "K42_01",
                    "display_text": "sample",
                    "source": "fixture",
                    "signer_id": "H",
                    "split": "dev",
                }
            ]
            dataset = Knee42Dataset(
                rows,
                root,
                {"K42_01": 0},
                mean,
                std,
                sequence_length=64,
            )

            expected = dataset[0][0].numpy()
            actual = materialize_sequence(values, mask, mean, std, sequence_length=64)

            self.assertEqual(actual.shape, (64, 438))
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
            sampled_indices = np.rint(np.linspace(0, frame_count - 1, 64)).astype(np.int64)
            sampled_mask = mask[sampled_indices]
            self.assertTrue(np.all(actual[:, :LANDMARK_DIM][~sampled_mask] == 0.0))
            np.testing.assert_array_equal(
                actual[:, LANDMARK_DIM:], sampled_mask.astype(np.float32)
            )


if __name__ == "__main__":
    unittest.main()
