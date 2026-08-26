from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from recognition.realtime.auto_trigger import AutoFrameAnalysis, AutoTriggerConfig
from recognition.realtime.knee42_controllers import AutoKnee42Controller
from recognition.realtime.knee42_preprocessing import observation_from_results


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "packaging" / "knee42_ivcam"
ARCHIVED_TRIGGER_SHA256 = "0092136a14a859c7a11aa0e5df9d0920a37471e8da09718828eb773d00d6fdc7"
ARCHIVED_CONTROLLER_SHA256 = "969f164356fa0abb532d924e08714bca781cba2eb0fc0fef658f577f8e9ec12b"
ARCHIVED_CONFIG_SHA256 = "d21f64f4f45f343964a532c5525ff5a4ce5669c9dcbc288cc7b65ccdf62ef728"
ARCHIVED_ZIP_SHA256 = "d31da3a2075321304cc595657417bf810eb52ee83ff5057a09aec9a2218f4f3c"


def sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def landmarks(count: int, offset: float) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            x=np.float32(offset + index * 10 + 1),
            y=np.float32(offset + index * 10 + 2),
            z=np.float32(offset + index * 10 + 3),
        )
        for index in range(count)
    ]


def frame_analysis(_previous, current, _config) -> AutoFrameAnalysis:
    code = int(current[0])
    rest = code == 0
    motion = 0.0 if rest else 0.08
    return AutoFrameAnalysis(
        visible_rest_blank=rest,
        hidden_rest_blank=False,
        torso_motion_score=motion,
        hand_motion_score=motion,
        effective_motion_score=motion,
        hands_on_knees=rest,
        knee_landmarks_valid=True,
        wrists_detected=True,
        torso_valid=True,
        explicit_hands_detected=2,
        wrist_source_left="hand",
        wrist_source_right="hand",
    )


class AutoTriggerProvenanceTests(unittest.TestCase):
    def test_archived_trigger_source_and_config_are_exactly_locked(self):
        source = ROOT / "recognition" / "realtime" / "auto_trigger.py"
        controller = ROOT / "recognition" / "realtime" / "knee42_controllers.py"
        config = TEMPLATE / "auto_trigger_knee_ivcam_local.json"

        self.assertEqual(sha256(source), ARCHIVED_TRIGGER_SHA256)
        self.assertEqual(sha256(controller), ARCHIVED_CONTROLLER_SHA256)
        self.assertEqual(sha256(config), ARCHIVED_CONFIG_SHA256)
        payload = json.loads(config.read_text(encoding="utf-8"))
        self.assertFalse(payload["knee_geometry_enabled"])
        self.assertFalse(payload["temporal_classifier_enabled"])
        self.assertTrue(payload["reference_rest_enabled"])

    def test_provenance_binds_supplied_zip_trigger_and_config(self):
        payload = json.loads(
            (TEMPLATE / "auto_trigger_provenance.json").read_text(encoding="utf-8")
        )

        self.assertEqual(payload["source_zip_sha256"], ARCHIVED_ZIP_SHA256)
        self.assertEqual(payload["auto_trigger_source_sha256"], ARCHIVED_TRIGGER_SHA256)
        self.assertEqual(
            payload["auto_trigger_controller_sha256"],
            ARCHIVED_CONTROLLER_SHA256,
        )
        self.assertEqual(payload["auto_trigger_config_sha256"], ARCHIVED_CONFIG_SHA256)
        self.assertFalse(payload["temporal_classifier_enabled"])


class DualObservationTests(unittest.TestCase):
    def test_same_results_create_full_trigger_and_kneeless_recognition_paths(self):
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

        self.assertEqual(observation.trigger_values.shape, (225,))
        self.assertEqual(observation.recognition_values.shape, (219,))
        self.assertEqual(observation.recognition_mask.shape, (219,))
        self.assertEqual(observation.trigger_values[25 * 3], 251.0)
        self.assertEqual(observation.trigger_values[26 * 3], 261.0)
        self.assertNotIn(251.0, observation.recognition_values.tolist())
        self.assertNotIn(261.0, observation.recognition_values.tolist())
        self.assertEqual(observation.trigger_values[99], 1_001.0)
        self.assertEqual(observation.trigger_values[162], 2_001.0)

    def test_missing_hands_are_zero_for_trigger_and_nan_masked_for_recognition(self):
        pose_result = SimpleNamespace(pose_landmarks=[landmarks(33, 0.0)])

        observation = observation_from_results(None, pose_result)

        self.assertTrue(np.all(observation.trigger_values[99:] == 0.0))
        hand_start = 31 * 3
        self.assertTrue(np.isnan(observation.recognition_values[hand_start:]).all())
        self.assertFalse(observation.recognition_mask[hand_start:].any())


class AutoKnee42ControllerTests(unittest.TestCase):
    def test_toggle_clears_auto_state_and_enables_manual_space_recording(self):
        controller = AutoKnee42Controller(
            AutoTriggerConfig(reference_rest_enabled=False),
            analysis_fn=frame_analysis,
        )
        trigger = np.ones(225, dtype=np.float32)
        controller.add_observation(0.0, trigger, ("auto", "auto"))

        event = controller.toggle_mode()

        self.assertEqual(controller.mode, "manual")
        self.assertEqual(event.state, "idle")
        self.assertEqual(controller.buffered_observations, 0)
        self.assertEqual(controller.on_space().state, "recording")
        controller.add_observation(0.1, trigger, ("manual", "manual"))
        stopped = controller.on_space()
        self.assertTrue(stopped.infer)
        self.assertEqual(stopped.features, (("manual", "manual"),))

        controller.toggle_mode()
        self.assertEqual(controller.mode, "auto")
        self.assertEqual(controller.state, "IDLE_BLANK")

    def test_engine_selected_timestamps_map_to_recognition_features_once(self):
        config = AutoTriggerConfig(
            start_hold_sec=0.10,
            pre_roll_sec=0.10,
            end_hold_sec=0.20,
            end_rest_vote_ratio=0.75,
            end_safety_tail_sec=0.15,
            min_segment_sec=0.10,
            cooldown_sec=0.20,
            knee_geometry_enabled=False,
            reference_rest_enabled=False,
        )
        controller = AutoKnee42Controller(config, analysis_fn=frame_analysis)
        events = []
        for timestamp, code in (
            (0.0, 0),
            (0.1, 1),
            (0.2, 1),
            (0.3, 1),
            (0.4, 0),
            (0.5, 0),
            (0.6, 0),
        ):
            trigger = np.full(225, code, dtype=np.float32)
            feature = (f"values-{timestamp}", f"mask-{timestamp}")
            event = controller.add_observation(timestamp, trigger, feature)
            if event.infer:
                events.append(event)

        self.assertEqual(len(events), 1)
        self.assertEqual(
            events[0].features,
            (
                ("values-0.0", "mask-0.0"),
                ("values-0.1", "mask-0.1"),
                ("values-0.2", "mask-0.2"),
                ("values-0.3", "mask-0.3"),
            ),
        )
        self.assertEqual(events[0].message, "visible_rest_finalize")
        self.assertIsNotNone(events[0].segment)
        self.assertAlmostEqual(events[0].segment.clip_start_sec, 0.0)
        self.assertAlmostEqual(events[0].segment.clip_end_sec, 0.4)
        self.assertAlmostEqual(events[0].segment.finalize_sec, 0.6)
        self.assertEqual(events[0].segment.reason, "visible_rest_finalize")
        self.assertEqual(events[0].segment.boundary_policy, "low_motion_anchor_v1")
        self.assertAlmostEqual(events[0].segment.rest_detected_sec, 0.4)
        self.assertLessEqual(
            events[0].segment.clip_end_sec,
            events[0].segment.rest_detected_sec,
        )

    def test_non_monotonic_timestamp_is_rejected(self):
        controller = AutoKnee42Controller(
            AutoTriggerConfig(reference_rest_enabled=False),
            analysis_fn=frame_analysis,
        )
        trigger = np.zeros(225, dtype=np.float32)
        controller.add_observation(1.0, trigger, ("one", "one"))

        with self.assertRaisesRegex(ValueError, "monotonic"):
            controller.add_observation(0.9, trigger, ("two", "two"))

    def test_direct_observation_rejects_nonfinite_timestamp(self):
        trigger = np.zeros(225, dtype=np.float32)

        for timestamp in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(timestamp=timestamp):
                controller = AutoKnee42Controller(
                    AutoTriggerConfig(reference_rest_enabled=False),
                    analysis_fn=frame_analysis,
                )
                with self.assertRaisesRegex(ValueError, "finite"):
                    controller.add_observation(
                        timestamp,
                        trigger,
                        ("values", "mask"),
                    )

    def test_video_eof_completes_an_existing_end_confirmation_only(self):
        controller = AutoKnee42Controller(
            AutoTriggerConfig(
                start_hold_sec=0.10,
                pre_roll_sec=0.10,
                end_hold_sec=0.30,
                end_rest_vote_ratio=0.75,
                min_segment_sec=0.10,
                knee_geometry_enabled=False,
                reference_rest_enabled=False,
            ),
            analysis_fn=frame_analysis,
        )
        for timestamp, code in (
            (0.0, 0),
            (0.1, 1),
            (0.2, 1),
            (0.3, 0),
            (0.35, 1),
        ):
            controller.add_observation(
                timestamp,
                np.full(225, code, dtype=np.float32),
                (f"values-{timestamp}", f"mask-{timestamp}"),
            )

        self.assertEqual(controller.state, "END_CONFIRM")
        event = controller.finalize_video_eof()

        self.assertTrue(event.infer)
        self.assertEqual(event.message, "visible_rest_finalize")
        self.assertEqual(
            event.features,
            (
                ("values-0.0", "mask-0.0"),
                ("values-0.1", "mask-0.1"),
                ("values-0.2", "mask-0.2"),
                ("values-0.3", "mask-0.3"),
                ("values-0.35", "mask-0.35"),
            ),
        )

    def test_video_eof_does_not_fabricate_an_end_while_signing_is_active(self):
        controller = AutoKnee42Controller(
            AutoTriggerConfig(
                start_hold_sec=0.10,
                pre_roll_sec=0.10,
                end_hold_sec=0.30,
                min_segment_sec=0.10,
                knee_geometry_enabled=False,
                reference_rest_enabled=False,
            ),
            analysis_fn=frame_analysis,
        )
        for timestamp in (0.0, 0.1, 0.2):
            controller.add_observation(
                timestamp,
                np.ones(225, dtype=np.float32),
                (f"values-{timestamp}", f"mask-{timestamp}"),
            )

        self.assertEqual(controller.state, "SIGNING_ACTIVE")
        event = controller.finalize_video_eof(frame_interval_sec=0.10)

        self.assertFalse(event.infer)
        self.assertEqual(controller.state, "SIGNING_ACTIVE")

    def test_held_trigger_samples_preserve_30hz_timing_with_frame_step_two(self):
        controller = AutoKnee42Controller(
            AutoTriggerConfig(
                start_hold_sec=0.10,
                pre_roll_sec=0.10,
                end_hold_sec=0.30,
                end_rest_vote_ratio=0.90,
                min_segment_sec=0.10,
                knee_geometry_enabled=False,
                reference_rest_enabled=False,
            ),
            analysis_fn=frame_analysis,
        )
        events = []
        for frame_index, code in enumerate((0, 1, 1, 1, 0, 0, 0, 0, 0, 0)):
            timestamp = frame_index * 2.0 / 30.0
            event = controller.add_held_observation(
                timestamp,
                np.full(225, code, dtype=np.float32),
                (f"values-{frame_index}", f"mask-{frame_index}"),
                frame_interval_sec=1.0 / 30.0,
                sample_count=2,
            )
            if event.infer:
                events.append(event)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].message, "visible_rest_finalize")

    def test_held_samples_use_exact_irregular_collected_timestamps(self):
        controller = AutoKnee42Controller(
            AutoTriggerConfig(
                knee_geometry_enabled=False,
                reference_rest_enabled=False,
            ),
            analysis_fn=frame_analysis,
        )

        controller.add_held_observation_at_times(
            [0.033, 0.071],
            np.ones(225, dtype=np.float32),
            ("values", "mask"),
        )

        self.assertEqual(controller.sample_timestamps, (0.033, 0.071))

    def test_held_samples_reject_empty_nonfinite_and_regressing_sequences(self):
        trigger = np.ones(225, dtype=np.float32)
        invalid_sequences = (
            ([], "empty"),
            ([0.033, float("nan")], "finite"),
            ([0.071, 0.033], "monotonic"),
        )

        for timestamps, message in invalid_sequences:
            with self.subTest(timestamps=timestamps):
                controller = AutoKnee42Controller(
                    AutoTriggerConfig(reference_rest_enabled=False),
                    analysis_fn=frame_analysis,
                )
                with self.assertRaisesRegex(ValueError, message):
                    controller.add_held_observation_at_times(
                        timestamps,
                        trigger,
                        ("values", "mask"),
                    )
                self.assertEqual(controller.sample_timestamps, ())

    def test_held_samples_reject_regression_against_previous_observation(self):
        controller = AutoKnee42Controller(
            AutoTriggerConfig(reference_rest_enabled=False),
            analysis_fn=frame_analysis,
        )
        trigger = np.ones(225, dtype=np.float32)
        controller.add_observation(0.071, trigger, ("first", "first"))

        with self.assertRaisesRegex(ValueError, "monotonic"):
            controller.add_held_observation_at_times(
                [0.033],
                trigger,
                ("second", "second"),
            )

    def test_held_samples_reuse_one_real_motion_analysis(self):
        calls = []

        def counted_analysis(previous, current, config):
            calls.append((previous, current.copy()))
            return frame_analysis(previous, current, config)

        controller = AutoKnee42Controller(
            AutoTriggerConfig(
                start_hold_sec=0.10,
                pre_roll_sec=0.10,
                end_hold_sec=0.30,
                min_segment_sec=0.10,
                knee_geometry_enabled=False,
                reference_rest_enabled=False,
            ),
            analysis_fn=counted_analysis,
        )

        controller.add_held_observation_at_times(
            [0.033, 0.071],
            np.ones(225, dtype=np.float32),
            ("values", "mask"),
        )

        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
