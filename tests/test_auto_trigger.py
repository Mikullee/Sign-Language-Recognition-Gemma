from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

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
        hands_at_sides=rest,
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
    hands_at_sides: bool = True,
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
    left_side_x = 0.70 if unmirrored_orientation else 0.30
    right_side_x = 0.30 if unmirrored_orientation else 0.70
    side_y = 0.78 if wrist_y is None else wrist_y
    pose[15] = [left_side_x, side_y, 0.0] if hands_at_sides else [0.53 if unmirrored_orientation else 0.47, 0.42, 0.0]
    pose[16] = [right_side_x, side_y, 0.0] if hands_at_sides else [0.47 if unmirrored_orientation else 0.53, 0.42, 0.0]
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


class AutoFrameAnalysisTests(unittest.TestCase):
    def test_visible_still_hands_at_sides_are_rest(self):
        current = body_frame(hands_visible=True, hands_at_sides=True)

        result = analyze_frame_vector(current, current, AutoTriggerConfig())

        self.assertTrue(result.visible_rest_blank)
        self.assertTrue(result.wrists_detected)
        self.assertTrue(result.hands_at_sides)
        self.assertEqual(result.explicit_hands_detected, 2)

    def test_unmirrored_handedness_is_still_detected_at_image_sides(self):
        current = body_frame(
            hands_visible=True,
            hands_at_sides=True,
            unmirrored_orientation=True,
        )

        result = analyze_frame_vector(current, current, AutoTriggerConfig())

        self.assertTrue(result.visible_rest_blank)
        self.assertTrue(result.hands_at_sides)

    def test_missing_hands_are_not_rest_when_hidden_rest_is_disabled(self):
        current = body_frame(hands_visible=False, hands_at_sides=True)

        result = analyze_frame_vector(current, current, AutoTriggerConfig(hidden_rest_enabled=False))

        self.assertFalse(result.visible_rest_blank)
        self.assertFalse(result.hidden_rest_blank)
        self.assertFalse(result.is_blank)

    def test_low_motion_at_chest_is_not_rest(self):
        current = body_frame(hands_visible=True, hands_at_sides=False)

        result = analyze_frame_vector(current, current, AutoTriggerConfig())

        self.assertFalse(result.visible_rest_blank)
        self.assertFalse(result.hands_at_sides)

    def test_low_motion_at_abdomen_is_not_rest_even_when_wrists_span_sides(self):
        current = body_frame(
            hands_visible=True,
            hands_at_sides=True,
            wrist_y=0.62,
        )

        result = analyze_frame_vector(current, current, AutoTriggerConfig())

        self.assertFalse(result.visible_rest_blank)
        self.assertFalse(result.hands_at_sides)

    def test_hand_motion_dominates_effective_motion(self):
        previous = body_frame(hands_visible=True, hands_at_sides=True)
        current = previous.copy()
        current[99:162:3] += 0.10

        result = analyze_frame_vector(previous, current, AutoTriggerConfig())

        self.assertGreater(result.hand_motion_score, 0.05)
        self.assertEqual(result.effective_motion_score, result.hand_motion_score)

    def test_pose_arm_motion_survives_missing_hand_landmarks(self):
        previous = body_frame(hands_visible=False, hands_at_sides=True)
        current = body_frame(hands_visible=False, hands_at_sides=False)

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


if __name__ == "__main__":
    unittest.main()
