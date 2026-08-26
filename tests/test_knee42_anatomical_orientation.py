from __future__ import annotations

import importlib
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


def _hand_landmarks(detection: dict, count: int = 21) -> list[SimpleNamespace]:
    first_x = float(detection["first_x"])
    first_z = float(detection["first_z"])
    return [
        SimpleNamespace(
            x=np.float32(first_x + index * 0.001),
            y=np.float32(0.40 + index * 0.005),
            z=np.float32(first_z + index * 0.001),
        )
        for index in range(count)
    ]


def _pose(marker_data: dict[str, float | int]) -> list[SimpleNamespace]:
    pose = [
        SimpleNamespace(
            x=np.float32(0.20 + index * 0.01),
            y=np.float32(0.25 + index * 0.005),
            z=np.float32(index * 0.001),
        )
        for index in range(33)
    ]
    pose[int(marker_data["pose_left_index"])] = SimpleNamespace(
        x=np.float32(0.31),
        y=np.float32(marker_data["pose_left_y"]),
        z=np.float32(0.011),
    )
    pose[int(marker_data["pose_right_index"])] = SimpleNamespace(
        x=np.float32(0.32),
        y=np.float32(marker_data["pose_right_y"]),
        z=np.float32(0.012),
    )
    return pose


def _results_from_case(fixture: dict, scenario_name: str, case_id: str):
    scenario = fixture["scenarios"][scenario_name]
    case = next(item for item in scenario["cases"] if item["case_id"] == case_id)
    hand_result = SimpleNamespace(
        handedness=[
            [SimpleNamespace(category_name=item["mediapipe_label"])]
            for item in case["detections"]
        ],
        hand_landmarks=[
            _hand_landmarks(item) for item in case["detections"]
        ],
    )
    pose_result = SimpleNamespace(
        pose_landmarks=[_pose(fixture["anatomical_markers"])]
    )
    return scenario, case, hand_result, pose_result


class Knee42AnatomicalOrientationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_fixture_is_adversarial_to_fixed_or_mirror_aware_screen_x_rules(self):
        markers = self.fixture["anatomical_markers"]
        self.assertEqual(markers["hand_identity_axis"], "z")
        self.assertIn(markers["pose_identity_axis"], {"y", "z"})
        scenario_cases = {}
        for scenario_name in ("mirrored_pixels", "unmirrored_pixels"):
            scenario = self.fixture["scenarios"][scenario_name]
            self.assertIn(
                "cases",
                scenario,
                f"{scenario_name} must contain crossed-hand adversarial cases",
            )
            cases = scenario["cases"]
            self.assertGreaterEqual(len(cases), 2)
            left_is_lower_x = set()
            for case in cases:
                detections = {
                    item["anatomical_hand"]: item for item in case["detections"]
                }
                self.assertEqual(set(detections), {"left", "right"})
                left_x = float(detections["left"]["first_x"])
                right_x = float(detections["right"]["first_x"])
                self.assertTrue(np.isfinite(left_x) and 0.0 <= left_x <= 1.0)
                self.assertTrue(np.isfinite(right_x) and 0.0 <= right_x <= 1.0)
                self.assertNotEqual(left_x, right_x)
                left_is_lower_x.add(left_x < right_x)
                self.assertAlmostEqual(
                    float(detections["left"]["first_z"]),
                    float(markers["left_hand_first_z"]),
                )
                self.assertAlmostEqual(
                    float(detections["right"]["first_z"]),
                    float(markers["right_hand_first_z"]),
                )
                expected_labels = (
                    {"left": "Left", "right": "Right"}
                    if scenario["pixels_mirrored"]
                    else {"left": "Right", "right": "Left"}
                )
                self.assertEqual(
                    detections["left"]["mediapipe_label"],
                    expected_labels["left"],
                )
                self.assertEqual(
                    detections["right"]["mediapipe_label"],
                    expected_labels["right"],
                )
            self.assertEqual(
                left_is_lower_x,
                {False, True},
                f"{scenario_name} must place anatomical left on both screen sides",
            )
            scenario_cases[scenario_name] = cases

        # Even a rule that changes between mirrored and unmirrored inputs must
        # fail at least one crossed-hand case for every min/max combination.
        for mirrored_rule in ("min", "max"):
            for unmirrored_rule in ("min", "max"):
                all_correct = True
                for scenario_name, rule in (
                    ("mirrored_pixels", mirrored_rule),
                    ("unmirrored_pixels", unmirrored_rule),
                ):
                    for case in scenario_cases[scenario_name]:
                        chooser = min if rule == "min" else max
                        guessed = chooser(
                            case["detections"],
                            key=lambda item: float(item["first_x"]),
                        )
                        all_correct &= guessed["anatomical_hand"] == "left"
                self.assertFalse(
                    all_correct,
                    f"mirror-aware x rule unexpectedly succeeded: {mirrored_rule}/{unmirrored_rule}",
                )

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
        markers = self.fixture["anatomical_markers"]
        scenario = self.fixture["scenarios"]["unmirrored_pixels"]
        for fixture_case in scenario["cases"]:
            _, case, hand_result, pose_result = _results_from_case(
                self.fixture,
                "unmirrored_pixels",
                fixture_case["case_id"],
            )
            values, mask = landmarks_from_results(
                hand_result,
                pose_result,
                pixels_mirrored=scenario["pixels_mirrored"],
            )

            left_start = len(POSE_KEEP) * 3
            right_start = left_start + 21 * 3
            self.assertAlmostEqual(
                float(values[left_start + 2]),
                float(markers["left_hand_first_z"]),
                places=6,
            )
            self.assertAlmostEqual(
                float(values[right_start + 2]),
                float(markers["right_hand_first_z"]),
                places=6,
            )
            detections = {
                item["anatomical_hand"]: item for item in case["detections"]
            }
            self.assertAlmostEqual(values[left_start], detections["left"]["first_x"])
            self.assertAlmostEqual(values[right_start], detections["right"]["first_x"])
            self.assertTrue(mask.all())

    def test_mirrored_and_unmirrored_scenarios_have_identical_anatomical_features(self):
        markers = self.fixture["anatomical_markers"]
        case_ids = [
            item["case_id"]
            for item in self.fixture["scenarios"]["mirrored_pixels"]["cases"]
        ]
        for case_id in case_ids:
            observations = []
            for scenario_name in ("mirrored_pixels", "unmirrored_pixels"):
                scenario, _, hand_result, pose_result = _results_from_case(
                    self.fixture,
                    scenario_name,
                    case_id,
                )
                observation = observation_from_results(
                    hand_result,
                    pose_result,
                    pixels_mirrored=scenario["pixels_mirrored"],
                )
                observations.append(observation)
                self.assertAlmostEqual(
                    float(observation.display_pose[int(markers["pose_left_index"]), 1]),
                    float(markers["pose_left_y"]),
                    places=6,
                )
                self.assertAlmostEqual(
                    float(observation.display_pose[int(markers["pose_right_index"]), 1]),
                    float(markers["pose_right_y"]),
                    places=6,
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
        scenario, _, hand_result, pose_result = _results_from_case(
            self.fixture,
            "unmirrored_pixels",
            "anatomical_left_higher_x",
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
