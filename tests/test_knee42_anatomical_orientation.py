from __future__ import annotations

import importlib
import inspect
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from recognition.realtime.knee42_preprocessing import (
    LANDMARK_DIM,
    POSE_KEEP,
    landmarks_from_results,
    materialize_sequence,
    observation_from_results,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "knee42_anatomical_frame.json"


def _orientation_module():
    return importlib.import_module("recognition.realtime.knee42_orientation")


def _landmarks(first_x: float, count: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            x=np.float32(first_x + index * 10.0),
            y=np.float32(index + 0.25),
            z=np.float32(index + 0.5),
        )
        for index in range(count)
    ]


def _pose(marker_data: dict[str, float | int]) -> list[SimpleNamespace]:
    pose = _landmarks(1.0, 33)
    pose[int(marker_data["pose_left_index"])] = SimpleNamespace(
        x=np.float32(marker_data["pose_left_x"]), y=np.float32(11.25), z=np.float32(11.5)
    )
    pose[int(marker_data["pose_right_index"])] = SimpleNamespace(
        x=np.float32(marker_data["pose_right_x"]), y=np.float32(12.25), z=np.float32(12.5)
    )
    return pose


def _results_from_scenario(fixture: dict, scenario_name: str):
    scenario = fixture["scenarios"][scenario_name]
    hand_result = SimpleNamespace(
        handedness=[
            [SimpleNamespace(category_name=item["mediapipe_label"])]
            for item in scenario["detections"]
        ],
        hand_landmarks=[
            _landmarks(float(item["first_x"]), 21) for item in scenario["detections"]
        ],
    )
    pose_result = SimpleNamespace(
        pose_landmarks=[_pose(fixture["anatomical_markers"])]
    )
    return scenario, hand_result, pose_result


class Knee42AnatomicalOrientationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_strict_parsers_and_structured_orientation_description(self):
        orientation = _orientation_module()

        self.assertFalse(orientation.MirrorMode.parse("off").enabled)
        self.assertTrue(orientation.MirrorMode.parse("on").enabled)
        declared = orientation.InputOrientation(
            rotation="auto",
            input_mirror=False,
            display_mirror=False,
        )
        self.assertEqual(
            declared.description,
            {
                "rotation": "auto",
                "input_mirror": False,
                "display_mirror": False,
            },
        )

        for invalid in ("ON", "true", "1", True, None):
            with self.subTest(invalid_mirror=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    orientation.MirrorMode.parse(invalid)
        for invalid in (-90, 360, 90.0, "90", None, True):
            with self.subTest(invalid_rotation=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    orientation.InputOrientation(invalid, False, False)

    def test_rotation_resolution_is_source_explicit(self):
        orientation = _orientation_module()

        self.assertEqual(
            orientation.resolve_rotation("auto", source_kind="video", metadata_rotation=90),
            90,
        )
        self.assertEqual(
            orientation.resolve_rotation("auto", source_kind="camera", metadata_rotation=270),
            0,
        )
        self.assertEqual(
            orientation.resolve_rotation(270, source_kind="video", metadata_rotation=90),
            270,
        )
        for kwargs in (
            {"rotation": "auto", "source_kind": "video", "metadata_rotation": 45},
            {"rotation": "auto", "source_kind": "file", "metadata_rotation": 0},
            {"rotation": -90, "source_kind": "video", "metadata_rotation": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    orientation.resolve_rotation(**kwargs)

    def test_anatomical_hand_slot_truth_table_and_unknown_label(self):
        orientation = _orientation_module()

        truth_table = {
            ("Left", True): "left",
            ("Right", True): "right",
            ("Left", False): "right",
            ("Right", False): "left",
        }
        for (label, pixels_mirrored), expected in truth_table.items():
            with self.subTest(label=label, pixels_mirrored=pixels_mirrored):
                self.assertEqual(
                    orientation.anatomical_hand_slot(
                        label,
                        pixels_mirrored=pixels_mirrored,
                    ),
                    expected,
                )
        for invalid in ("Unknown", "left", "", None):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    orientation.anatomical_hand_slot(invalid, pixels_mirrored=False)

    def test_unmirrored_fixture_swaps_labels_into_anatomical_slots(self):
        scenario, hand_result, pose_result = _results_from_scenario(
            self.fixture,
            "unmirrored_pixels",
        )
        # During the RED run this deliberately exercises the old direct-label behavior;
        # once the orientation API exists it exercises the declared unmirrored policy.
        kwargs = (
            {"pixels_mirrored": scenario["pixels_mirrored"]}
            if "pixels_mirrored" in inspect.signature(landmarks_from_results).parameters
            else {}
        )

        values, mask = landmarks_from_results(hand_result, pose_result, **kwargs)

        left_start = len(POSE_KEEP) * 3
        right_start = left_start + 21 * 3
        markers = self.fixture["anatomical_markers"]
        self.assertEqual(values[left_start], markers["left_hand_first_x"])
        self.assertEqual(values[right_start], markers["right_hand_first_x"])
        self.assertTrue(mask.all())

    def test_mirrored_and_unmirrored_scenarios_have_identical_anatomical_features(self):
        observations = []
        markers = self.fixture["anatomical_markers"]
        for scenario_name in ("mirrored_pixels", "unmirrored_pixels"):
            scenario, hand_result, pose_result = _results_from_scenario(
                self.fixture,
                scenario_name,
            )
            observation = observation_from_results(
                hand_result,
                pose_result,
                pixels_mirrored=scenario["pixels_mirrored"],
            )
            observations.append(observation)
            self.assertEqual(
                observation.display_pose[int(markers["pose_left_index"]), 0],
                markers["pose_left_x"],
            )
            self.assertEqual(
                observation.display_pose[int(markers["pose_right_index"]), 0],
                markers["pose_right_x"],
            )

        np.testing.assert_array_equal(
            observations[0].recognition_values,
            observations[1].recognition_values,
        )
        np.testing.assert_array_equal(
            observations[0].trigger_values,
            observations[1].trigger_values,
        )

    def test_display_mirror_setting_cannot_change_materialized_model_tensor(self):
        orientation = _orientation_module()
        scenario, hand_result, pose_result = _results_from_scenario(
            self.fixture,
            "unmirrored_pixels",
        )
        tensors = []
        for display_mirror in (False, True):
            declared = orientation.InputOrientation(
                rotation="auto",
                input_mirror=scenario["pixels_mirrored"],
                display_mirror=display_mirror,
            )
            observation = observation_from_results(
                hand_result,
                pose_result,
                pixels_mirrored=declared.input_mirror,
            )
            values = np.repeat(observation.recognition_values[None, :], 3, axis=0)
            masks = np.repeat(observation.recognition_mask[None, :], 3, axis=0)
            tensors.append(
                materialize_sequence(
                    values,
                    masks,
                    np.zeros(LANDMARK_DIM, dtype=np.float32),
                    np.ones(LANDMARK_DIM, dtype=np.float32),
                )
            )

        self.assertEqual(tensors[0].shape, (64, 438))
        np.testing.assert_array_equal(tensors[0], tensors[1])


if __name__ == "__main__":
    unittest.main()
