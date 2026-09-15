from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from recognition.realtime.auto_trigger import AutoFrameAnalysis
from recognition.transformer.live_trigger import (
    SEGMENT_STATE_IDLE,
    SEGMENT_STATE_REARMING,
    WebAutoTriggerConfig,
    WebAutoTriggerEngine,
    load_web_trigger_config,
)


ROOT = Path(__file__).resolve().parents[1]


def frame(value: float) -> np.ndarray:
    return np.full(225, value, dtype=np.float32)


def observed(
    *,
    rest: bool,
    motion: float,
    palm_value: float | None,
    wrist_value: float | None,
    hands_detected: int = 2,
) -> AutoFrameAnalysis:
    base = AutoFrameAnalysis(
        visible_rest_blank=rest,
        hidden_rest_blank=False,
        torso_motion_score=motion,
        hand_motion_score=motion,
        effective_motion_score=motion,
        hands_on_knees=rest,
        knee_landmarks_valid=True,
        wrists_detected=wrist_value is not None,
        torso_valid=True,
        explicit_hands_detected=hands_detected,
        wrist_source_left="hand" if hands_detected == 2 else "pose",
        wrist_source_right="hand" if hands_detected == 2 else "pose",
    )
    return replace(
        base,
        rest_signature=(None if palm_value is None else (palm_value,) * 6),
        wrist_rest_signature=(None if wrist_value is None else (wrist_value,) * 6),
    )


class WebAutoTriggerTests(unittest.TestCase):
    def test_web_service_defaults_to_live_trigger_config(self):
        from webservice.server import DEFAULT_TRIGGER_CONFIG

        self.assertEqual(
            DEFAULT_TRIGGER_CONFIG.as_posix(),
            "configs/auto_trigger_knee_web_live.json",
        )

    def test_web_live_config_is_fail_safe_and_rearms(self):
        config = load_web_trigger_config(
            ROOT / "configs" / "auto_trigger_knee_web_live.json"
        )
        self.assertEqual(config.max_segment_sec, 10.0)
        self.assertTrue(config.adaptive_rearm_enabled)
        self.assertEqual(config.adaptive_rearm_hold_sec, 0.5)
        self.assertTrue(config.adaptive_rearm_requires_knee_rest)

    def test_adaptive_rearm_hold_must_be_non_negative(self):
        with self.assertRaisesRegex(ValueError, "Adaptive re-arm hold"):
            WebAutoTriggerConfig(adaptive_rearm_hold_sec=-0.1)

    def test_initial_calibration_requires_consecutive_stable_wrist_samples(self):
        config = WebAutoTriggerConfig(
            reference_rest_enabled=True,
            reference_seed_sec=0.20,
        )
        engine = WebAutoTriggerEngine(config)
        rest = observed(
            rest=True,
            motion=0.0,
            palm_value=None,
            wrist_value=0.1,
            hands_detected=0,
        )
        moving = observed(
            rest=False,
            motion=0.08,
            palm_value=None,
            wrist_value=0.5,
            hands_detected=0,
        )

        engine.update(frame(0.0), rest, 0.0)
        engine.update(frame(0.1), moving, 0.1)
        engine.update(frame(0.2), rest, 0.2)
        engine.update(frame(0.3), rest, 0.3)

        self.assertFalse(engine.calibrated)
        engine.update(frame(0.4), rest, 0.4)
        self.assertTrue(engine.calibrated)

    def test_timeout_waits_for_stable_rest_and_refreshes_reference(self):
        config = WebAutoTriggerConfig(
            start_motion_threshold=0.01,
            blank_motion_threshold=0.02,
            start_hold_sec=0.10,
            pre_roll_sec=0.0,
            max_segment_sec=0.50,
            min_segment_sec=0.10,
            cooldown_sec=0.10,
            reference_rest_enabled=True,
            reference_seed_sec=0.20,
            adaptive_rearm_enabled=True,
            adaptive_rearm_hold_sec=0.20,
        )
        engine = WebAutoTriggerEngine(config)

        for timestamp in (0.0, 0.1, 0.2):
            engine.update(
                frame(timestamp),
                observed(
                    rest=True,
                    motion=0.0,
                    palm_value=0.0,
                    wrist_value=0.0,
                ),
                timestamp,
            )
        self.assertTrue(engine.calibrated)
        self.assertEqual(engine.reference_revision, 1)

        event = None
        for timestamp in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            event = engine.update(
                frame(timestamp),
                observed(
                    rest=False,
                    motion=0.08,
                    palm_value=0.5,
                    wrist_value=0.5,
                ),
                timestamp,
            )
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.reason, "timeout_finalize")

        for timestamp in (0.9, 1.0, 1.1):
            self.assertIsNone(
                engine.update(
                    frame(timestamp),
                    observed(
                        rest=False,
                        motion=0.08,
                        palm_value=0.5,
                        wrist_value=0.5,
                    ),
                    timestamp,
                )
            )
        self.assertEqual(engine.state, SEGMENT_STATE_REARMING)
        self.assertEqual(engine.reference_revision, 1)

        for timestamp in (1.2, 1.3, 1.4):
            engine.update(
                frame(timestamp),
                observed(
                    rest=True,
                    motion=0.0,
                    palm_value=0.2,
                    wrist_value=0.2,
                ),
                timestamp,
            )
        self.assertEqual(engine.state, SEGMENT_STATE_IDLE)
        self.assertEqual(engine.reference_revision, 2)
        np.testing.assert_allclose(engine._rest_wrist_reference_signature, 0.2)

    def test_pose_wrists_can_seed_reference_when_hands_are_missing(self):
        config = WebAutoTriggerConfig(
            start_motion_threshold=0.01,
            reference_rest_enabled=True,
            reference_seed_sec=0.20,
        )
        engine = WebAutoTriggerEngine(config)

        last = None
        for timestamp in (0.0, 0.1, 0.2):
            last = observed(
                rest=True,
                motion=0.0,
                palm_value=None,
                wrist_value=0.1,
                hands_detected=0,
            )
            engine.update(frame(timestamp), last, timestamp)

        self.assertTrue(engine.calibrated)
        self.assertIsNone(engine._rest_reference_signature)
        np.testing.assert_allclose(engine._rest_wrist_reference_signature, 0.1)
        self.assertEqual(engine.rest_signature_status(last), "pose_wrist_fallback")
        self.assertTrue(
            engine._is_start_candidate(
                observed(
                    rest=False,
                    motion=0.08,
                    palm_value=None,
                    wrist_value=0.5,
                    hands_detected=0,
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
