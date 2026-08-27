from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import recognition.realtime.auto_trigger as auto_trigger_module

from recognition.realtime.auto_trigger import (
    SEGMENT_STATE_ACTIVE,
    SEGMENT_STATE_COOLDOWN,
    AutoFrameAnalysis,
    AutoTriggerConfig,
    AutoTriggerEngine,
    analyze_frame_vector,
    load_auto_trigger_config,
)


def analysis(
    *,
    rest: bool = False,
    motion: float = 0.0,
    hands_detected: int = 2,
    wrists_visible: bool = True,
) -> AutoFrameAnalysis:
    return AutoFrameAnalysis(
        visible_rest_blank=rest,
        hidden_rest_blank=False,
        torso_motion_score=motion,
        hand_motion_score=motion,
        effective_motion_score=motion,
        hands_on_knees=rest,
        knee_landmarks_valid=True,
        wrists_detected=wrists_visible,
        torso_valid=True,
        explicit_hands_detected=hands_detected,
        wrist_source_left="hand" if wrists_visible else "none",
        wrist_source_right="hand" if wrists_visible else "none",
    )


def frame(value: float) -> np.ndarray:
    return np.full(225, value, dtype=np.float32)


def body_frame(
    *,
    hands_visible: bool = True,
    hands_on_knees: bool = True,
    unmirrored_orientation: bool = False,
    wrist_y: float | None = None,
) -> np.ndarray:
    pose = np.zeros((33, 3), dtype=np.float32)
    pose[11] = [0.60 if unmirrored_orientation else 0.40, 0.30, 0.0]
    pose[12] = [0.40 if unmirrored_orientation else 0.60, 0.30, 0.0]
    pose[13] = [0.36, 0.48, 0.0]
    pose[14] = [0.64, 0.48, 0.0]
    pose[23] = [0.44, 0.70, 0.0]
    pose[24] = [0.56, 0.70, 0.0]
    pose[25] = [0.36, 0.90, 0.0]
    pose[26] = [0.64, 0.90, 0.0]
    left_knee_x = 0.64 if unmirrored_orientation else 0.36
    right_knee_x = 0.36 if unmirrored_orientation else 0.64
    knee_y = 0.90 if wrist_y is None else wrist_y
    pose[15] = [left_knee_x, knee_y, 0.0] if hands_on_knees else [0.53 if unmirrored_orientation else 0.47, 0.42, 0.0]
    pose[16] = [right_knee_x, knee_y, 0.0] if hands_on_knees else [0.47 if unmirrored_orientation else 0.53, 0.42, 0.0]
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)
    if hands_visible:
        left[:] = pose[15]
        right[:] = pose[16]
    return np.concatenate([pose.reshape(-1), left.reshape(-1), right.reshape(-1)])


def activate(engine: AutoTriggerEngine, start_time: float = 0.0, dt: float = 0.1) -> float:
    now = start_time
    engine.update(frame(now), analysis(rest=True), now)
    for _ in range(4):
        now += dt
        engine.update(frame(now), analysis(motion=0.08), now)
    return now


class AutoTriggerEngineTests(unittest.TestCase):
    def test_six_tenths_preroll_preserves_slow_onset_before_threshold_hold(self):
        engine = AutoTriggerEngine(
            AutoTriggerConfig(
                start_motion_threshold=0.04,
                start_hold_sec=0.10,
                pre_roll_sec=0.60,
                end_hold_sec=0.30,
            )
        )

        for timestamp, motion in [
            (0.0, 0.0),
            (0.1, 0.01),
            (0.2, 0.02),
            (0.3, 0.03),
            (0.4, 0.05),
            (0.5, 0.05),
        ]:
            engine.update(
                frame(timestamp),
                analysis(rest=timestamp == 0.0, motion=motion),
                timestamp,
            )

        self.assertEqual(engine.state, SEGMENT_STATE_ACTIVE)
        self.assertAlmostEqual(engine.clip_start_sec, 0.0, places=6)

    def test_confirmed_rest_trims_to_low_motion_anchor_plus_safety_tail(self):
        engine = AutoTriggerEngine(
            AutoTriggerConfig(
                start_hold_sec=0.10,
                pre_roll_sec=0.10,
                end_hold_sec=0.30,
                end_rest_vote_ratio=1.0,
                end_safety_tail_sec=0.15,
            )
        )
        now = activate(engine, dt=0.1)
        low_motion_start = now + 0.1
        for _ in range(7):
            now += 0.1
            self.assertIsNone(
                engine.update(frame(now), analysis(rest=False, motion=0.001), now)
            )
        rest_detected = now + 0.1
        event = None
        while event is None:
            now += 0.1
            event = engine.update(frame(now), analysis(rest=True, motion=0.0), now)

        self.assertAlmostEqual(event.clip_end_sec, low_motion_start + 0.15, places=6)
        self.assertAlmostEqual(event.rest_detected_sec, rest_detected, places=6)
        self.assertEqual(event.boundary_policy, "low_motion_anchor_v1")

    def test_mid_gesture_pause_is_not_used_as_final_anchor(self):
        engine = AutoTriggerEngine(
            AutoTriggerConfig(
                start_hold_sec=0.10,
                pre_roll_sec=0.10,
                end_hold_sec=0.30,
                end_rest_vote_ratio=1.0,
                end_safety_tail_sec=0.15,
            )
        )
        now = activate(engine, dt=0.1)
        for _ in range(3):
            now += 0.1
            engine.update(frame(now), analysis(rest=False, motion=0.001), now)
        for _ in range(2):
            now += 0.1
            engine.update(frame(now), analysis(rest=False, motion=0.08), now)
        final_low_motion_start = now + 0.1
        for _ in range(2):
            now += 0.1
            engine.update(frame(now), analysis(rest=False, motion=0.001), now)
        event = None
        while event is None:
            now += 0.1
            event = engine.update(frame(now), analysis(rest=True, motion=0.0), now)

        self.assertAlmostEqual(
            event.clip_end_sec,
            final_low_motion_start + 0.15,
            places=6,
        )

    def test_start_requires_hold_and_includes_preroll(self):
        config = AutoTriggerConfig(
            start_hold_sec=0.20,
            pre_roll_sec=0.20,
            end_hold_sec=0.50,
        )
        engine = AutoTriggerEngine(config)

        engine.update(frame(0.0), analysis(rest=True), 0.0)
        engine.update(frame(0.1), analysis(motion=0.08), 0.1)
        engine.update(frame(0.2), analysis(motion=0.08), 0.2)
        self.assertNotEqual(engine.state, SEGMENT_STATE_ACTIVE)

        engine.update(frame(0.3), analysis(motion=0.08), 0.3)

        self.assertEqual(engine.state, SEGMENT_STATE_ACTIVE)
        self.assertIsNotNone(engine.clip_start_sec)
        self.assertAlmostEqual(engine.clip_start_sec, 0.0, places=6)
        self.assertEqual([round(sample.timestamp_sec, 2) for sample in engine.segment_samples], [0.0, 0.1, 0.2, 0.3])

    def test_low_motion_inside_sentence_does_not_end_without_rest_pose(self):
        engine = AutoTriggerEngine(AutoTriggerConfig(start_hold_sec=0.20, end_hold_sec=0.50))
        now = activate(engine)

        for _ in range(10):
            now += 0.1
            event = engine.update(frame(now), analysis(rest=False, motion=0.001), now)
            self.assertIsNone(event)

        self.assertEqual(engine.state, SEGMENT_STATE_ACTIVE)

    def test_end_window_tolerates_one_missing_wrist_frame(self):
        engine = AutoTriggerEngine(
            AutoTriggerConfig(
                start_hold_sec=0.20,
                pre_roll_sec=0.20,
                end_hold_sec=0.50,
                end_rest_vote_ratio=0.80,
            )
        )
        now = activate(engine)
        rest_start = now + 0.1
        event = None
        for is_rest, wrists_visible in [
            (True, True),
            (True, True),
            (False, False),
            (True, True),
            (True, True),
            (True, True),
        ]:
            now += 0.1
            event = engine.update(
                frame(now),
                analysis(rest=is_rest, wrists_visible=wrists_visible),
                now,
            )

        self.assertIsNotNone(event)
        assert event is not None
        self.assertAlmostEqual(event.clip_end_sec, rest_start, places=6)
        self.assertAlmostEqual(event.finalize_sec, now, places=6)
        self.assertEqual(event.reason, "visible_rest_finalize")
        self.assertEqual(engine.state, SEGMENT_STATE_COOLDOWN)

    def test_sustained_rest_reports_rest_start_not_finalize_time(self):
        engine = AutoTriggerEngine(
            AutoTriggerConfig(start_hold_sec=0.20, end_hold_sec=0.50, end_rest_vote_ratio=0.80)
        )
        now = activate(engine)
        rest_start = now + 0.1
        event = None
        while event is None:
            now += 0.1
            event = engine.update(frame(now), analysis(rest=True), now)

        self.assertAlmostEqual(event.clip_end_sec, rest_start, places=6)
        self.assertAlmostEqual(event.finalize_sec - event.clip_end_sec, 0.50, places=6)
        self.assertTrue(all(sample.timestamp_sec < event.clip_end_sec for sample in event.samples))

    def test_timeout_finalizes_without_rest_pose(self):
        engine = AutoTriggerEngine(
            AutoTriggerConfig(
                start_hold_sec=0.20,
                min_segment_sec=0.20,
                max_segment_sec=0.60,
                end_hold_sec=0.50,
            )
        )
        now = activate(engine)
        event = None
        while event is None:
            now += 0.1
            event = engine.update(frame(now), analysis(rest=False, motion=0.03), now)

        self.assertEqual(event.reason, "timeout_finalize")
        self.assertAlmostEqual(event.clip_end_sec, event.finalize_sec, places=6)

    def test_engine_returns_to_idle_and_detects_a_second_segment(self):
        engine = AutoTriggerEngine(
            AutoTriggerConfig(
                start_hold_sec=0.10,
                pre_roll_sec=0.10,
                end_hold_sec=0.20,
                end_rest_vote_ratio=0.75,
                cooldown_sec=0.20,
                min_segment_sec=0.10,
            )
        )
        events = []
        timeline = [
            (0.0, analysis(rest=True)),
            (0.1, analysis(motion=0.08)),
            (0.2, analysis(motion=0.08)),
            (0.3, analysis(rest=True)),
            (0.4, analysis(rest=True)),
            (0.5, analysis(rest=True)),
            (0.6, analysis(rest=True)),
            (0.7, analysis(rest=True)),
            (0.8, analysis(motion=0.08)),
            (0.9, analysis(motion=0.08)),
            (1.0, analysis(rest=True)),
            (1.1, analysis(rest=True)),
            (1.2, analysis(rest=True)),
        ]
        for timestamp_sec, frame_analysis in timeline:
            event = engine.update(
                frame(timestamp_sec),
                frame_analysis,
                timestamp_sec,
            )
            if event is not None:
                events.append(event)

        self.assertEqual(len(events), 2)
        self.assertLess(events[0].clip_end_sec, events[1].clip_start_sec)

    def test_time_based_boundaries_are_consistent_across_sampling_rates(self):
        def run(dt: float) -> tuple[float, float]:
            engine = AutoTriggerEngine(
                AutoTriggerConfig(
                    start_hold_sec=0.20,
                    pre_roll_sec=0.20,
                    end_hold_sec=0.50,
                    end_rest_vote_ratio=0.80,
                )
            )
            now = 0.0
            event = None
            while now < 2.0 and event is None:
                rest = now < 0.5 or now >= 1.2
                motion = 0.08 if 0.5 <= now < 1.2 else 0.0
                event = engine.update(frame(now), analysis(rest=rest, motion=motion), now)
                now = round(now + dt, 10)
            assert event is not None
            return event.clip_start_sec, event.clip_end_sec

        start_10, end_10 = run(0.1)
        start_20, end_20 = run(0.05)

        self.assertLessEqual(abs(start_10 - start_20), 0.06)
        self.assertLessEqual(abs(end_10 - end_20), 0.06)

    def test_reference_rest_finalizes_when_pose_knee_geometry_drifts(self):
        config = AutoTriggerConfig(
            start_motion_threshold=0.01,
            start_hold_sec=0.10,
            pre_roll_sec=0.0,
            end_hold_sec=0.20,
            end_rest_vote_ratio=0.75,
            reference_rest_enabled=True,
            reference_seed_sec=0.20,
            reference_rest_distance_threshold=0.01,
        )
        engine = AutoTriggerEngine(config)
        rest_frame = body_frame(hands_visible=True, hands_on_knees=True)
        moving_frame = body_frame(hands_visible=True, hands_on_knees=False)
        moving_frame_2 = moving_frame.copy()
        moving_frame_2[99:] += 0.03
        previous = None
        for timestamp in (0.0, 0.1, 0.2):
            current = analyze_frame_vector(previous, rest_frame, config)
            engine.update(rest_frame, current, timestamp)
            previous = rest_frame
        for timestamp, current_frame in ((0.3, moving_frame), (0.4, moving_frame_2)):
            current = analyze_frame_vector(previous, current_frame, config)
            engine.update(current_frame, current, timestamp)
            previous = current_frame
        self.assertEqual(engine.state, SEGMENT_STATE_ACTIVE)

        event = None
        for timestamp in (0.5, 0.6, 0.7, 0.8):
            observed = analyze_frame_vector(previous, rest_frame, config)
            reference_only = replace(
                observed,
                visible_rest_blank=False,
                hands_on_knees=False,
            )
            event = engine.update(rest_frame, reference_only, timestamp)
            previous = rest_frame

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.reason, "reference_rest_finalize")

    def test_reference_rest_uses_pose_wrists_when_hand_landmarks_disappear(self):
        config = AutoTriggerConfig(
            start_motion_threshold=0.01,
            start_hold_sec=0.10,
            pre_roll_sec=0.0,
            end_hold_sec=0.20,
            end_rest_vote_ratio=0.75,
            knee_geometry_enabled=False,
            hidden_rest_enabled=False,
            reference_rest_enabled=True,
            reference_seed_sec=0.20,
            reference_rest_distance_threshold=0.02,
        )
        engine = AutoTriggerEngine(config)
        rest_frame = body_frame(hands_visible=True, hands_on_knees=True)
        moving_frame = body_frame(hands_visible=True, hands_on_knees=False)
        moving_frame_2 = moving_frame.copy()
        moving_frame_2[99:] += 0.03
        pose_only_rest = body_frame(hands_visible=False, hands_on_knees=True)
        previous = None
        for timestamp in (0.0, 0.1, 0.2):
            observed = analyze_frame_vector(previous, rest_frame, config)
            engine.update(rest_frame, observed, timestamp)
            previous = rest_frame
        for timestamp, current_frame in ((0.3, moving_frame), (0.4, moving_frame_2)):
            observed = analyze_frame_vector(previous, current_frame, config)
            engine.update(current_frame, observed, timestamp)
            previous = current_frame
        self.assertEqual(engine.state, SEGMENT_STATE_ACTIVE)

        event = None
        for timestamp in (0.5, 0.6, 0.7, 0.8, 0.9):
            observed = analyze_frame_vector(previous, pose_only_rest, config)
            event = engine.update(pose_only_rest, observed, timestamp)
            previous = pose_only_rest
            if event is not None:
                break

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.reason, "reference_rest_finalize")


class AutoFrameAnalysisTests(unittest.TestCase):
    def test_pose_wrist_rest_signature_survives_missing_hand_landmarks(self):
        current = body_frame(hands_visible=False, hands_on_knees=True)

        result = analyze_frame_vector(current, current, AutoTriggerConfig())

        self.assertIsNone(result.rest_signature)
        self.assertIsNotNone(result.wrist_rest_signature)

    def test_visible_still_hands_on_knees_are_rest(self):
        current = body_frame(hands_visible=True, hands_on_knees=True)

        result = analyze_frame_vector(current, current, AutoTriggerConfig())

        self.assertTrue(result.visible_rest_blank)
        self.assertTrue(result.wrists_detected)
        self.assertTrue(result.hands_on_knees)
        self.assertTrue(result.knee_landmarks_valid)
        self.assertEqual(result.explicit_hands_detected, 2)

    def test_crossed_handedness_is_still_detected_on_knees(self):
        current = body_frame(
            hands_visible=True,
            hands_on_knees=True,
            unmirrored_orientation=True,
        )

        result = analyze_frame_vector(current, current, AutoTriggerConfig())

        self.assertTrue(result.visible_rest_blank)
        self.assertTrue(result.hands_on_knees)

    def test_missing_hands_are_not_rest_when_hidden_rest_is_disabled(self):
        current = body_frame(hands_visible=False, hands_on_knees=True)

        result = analyze_frame_vector(current, current, AutoTriggerConfig(hidden_rest_enabled=False))

        self.assertFalse(result.visible_rest_blank)
        self.assertFalse(result.hidden_rest_blank)
        self.assertFalse(result.is_blank)

    def test_low_motion_at_chest_is_not_rest(self):
        current = body_frame(hands_visible=True, hands_on_knees=False)

        result = analyze_frame_vector(current, current, AutoTriggerConfig())

        self.assertFalse(result.visible_rest_blank)
        self.assertFalse(result.hands_on_knees)

    def test_low_motion_above_knees_is_not_rest(self):
        current = body_frame(
            hands_visible=True,
            hands_on_knees=True,
            wrist_y=0.50,
        )

        result = analyze_frame_vector(current, current, AutoTriggerConfig())

        self.assertFalse(result.visible_rest_blank)
        self.assertFalse(result.hands_on_knees)

    def test_missing_knee_landmarks_cannot_produce_visible_rest(self):
        current = body_frame(hands_visible=True, hands_on_knees=True)
        current[25 * 3:27 * 3] = 0.0

        result = analyze_frame_vector(current, current, AutoTriggerConfig())

        self.assertFalse(result.knee_landmarks_valid)
        self.assertFalse(result.visible_rest_blank)

    def test_hand_motion_dominates_effective_motion(self):
        previous = body_frame(hands_visible=True, hands_on_knees=True)
        current = previous.copy()
        current[99:162:3] += 0.10

        result = analyze_frame_vector(previous, current, AutoTriggerConfig())

        self.assertGreater(result.hand_motion_score, 0.05)
        self.assertEqual(result.effective_motion_score, result.hand_motion_score)

    def test_pose_arm_motion_survives_missing_hand_landmarks(self):
        previous = body_frame(hands_visible=False, hands_on_knees=True)
        current = body_frame(hands_visible=False, hands_on_knees=False)

        result = analyze_frame_vector(previous, current, AutoTriggerConfig())

        self.assertEqual(result.explicit_hands_detected, 0)
        self.assertGreater(result.hand_motion_score, 0.03)
        self.assertGreater(result.effective_motion_score, 0.03)


class AutoTriggerConfigTests(unittest.TestCase):
    def test_json_values_load_and_explicit_overrides_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "auto.json"
            path.write_text(
                json.dumps(
                    {
                        "end_hold_sec": 0.60,
                        "end_rest_vote_ratio": 0.90,
                        "hidden_rest_enabled": True,
                    }
                ),
                encoding="utf-8",
            )

            config = load_auto_trigger_config(
                path,
                overrides={
                    "end_hold_sec": 0.40,
                    "start_motion_threshold": None,
                    "hidden_rest_enabled": False,
                },
            )

        self.assertEqual(config.end_hold_sec, 0.40)
        self.assertEqual(config.end_rest_vote_ratio, 0.90)
        self.assertFalse(config.hidden_rest_enabled)
        self.assertEqual(config.start_motion_threshold, AutoTriggerConfig().start_motion_threshold)

    def test_formal_loader_requires_exact_schema_and_start_not_above_blank(self):
        loader = getattr(auto_trigger_module, "load_formal_auto_trigger_config", None)
        self.assertTrue(callable(loader), "formal auto-trigger loader is missing")
        valid = AutoTriggerConfig(
            start_motion_threshold=0.015,
            blank_motion_threshold=0.022,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "auto_trigger_knee_ivcam_local.json"
            payload = asdict(valid)
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(loader(root), valid)
            for mutation, expected in (
                ({key: value for key, value in payload.items() if key != "start_hold_sec"}, "start_hold_sec"),
                ({**payload, "unexpected": 1}, "unexpected"),
                ({**payload, "start_motion_threshold": 0.03, "blank_motion_threshold": 0.02}, "start_motion_threshold"),
            ):
                with self.subTest(expected=expected):
                    path.write_text(json.dumps(mutation), encoding="utf-8")
                    with self.assertRaisesRegex((TypeError, ValueError), expected):
                        loader(root)

    def test_numeric_config_rejects_bool_nan_and_infinity(self):
        for field_name, value in (
            ("start_hold_sec", True),
            ("blank_motion_threshold", float("nan")),
            ("max_segment_sec", float("inf")),
            ("pose_visibility_threshold", -float("inf")),
        ):
            with self.subTest(field=field_name, value=value):
                with self.assertRaisesRegex((TypeError, ValueError), field_name):
                    AutoTriggerConfig(**{field_name: value})


class CalibrationTelemetryTests(unittest.TestCase):
    @staticmethod
    def _eligible(*, motion: float = 0.0, hands: int = 2) -> AutoFrameAnalysis:
        return replace(
            analysis(motion=motion, hands_detected=hands),
            rest_signature=(0.0,) * 8,
            wrist_rest_signature=(0.0,) * 4,
        )

    def test_reference_seed_requires_continuous_eligible_samples(self):
        engine = AutoTriggerEngine(
            AutoTriggerConfig(reference_rest_enabled=True, reference_seed_sec=0.5)
        )
        engine.update(frame(0.0), self._eligible(), 0.0)
        engine.update(frame(0.3), self._eligible(), 0.3)
        engine.update(frame(0.4), self._eligible(hands=1), 0.4)
        engine.update(frame(0.5), self._eligible(), 0.5)
        engine.update(frame(0.8), self._eligible(), 0.8)

        telemetry = getattr(engine, "calibration_telemetry", None)
        self.assertIsNotNone(telemetry, "structured calibration telemetry is missing")
        self.assertFalse(telemetry.calibrated)
        self.assertEqual(telemetry.status.value, "collecting_reference")
        self.assertEqual(telemetry.sample_count, 2)
        self.assertAlmostEqual(telemetry.elapsed_sec, 0.3)


if __name__ == "__main__":
    unittest.main()
