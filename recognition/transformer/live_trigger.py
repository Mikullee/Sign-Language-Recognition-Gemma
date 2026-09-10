"""Transformer/Web auto-trigger extensions without changing the archived v13 source.

The v13 release binds ``recognition.realtime.auto_trigger`` by SHA-256.  The
current Transformer and Web paths therefore extend that state machine here so
new behavior cannot silently rewrite the provenance of the published archive.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np

from recognition.realtime.auto_trigger import (
    SEGMENT_STATE_COOLDOWN,
    SEGMENT_STATE_IDLE,
    AutoFrameAnalysis,
    AutoTriggerConfig,
    AutoTriggerEngine,
    SegmentResult,
    analyze_frame_vector,
)
from recognition.realtime.knee42_controllers import AutoKnee42Controller


SEGMENT_STATE_REARMING = "REARMING"


@dataclass(frozen=True)
class WebAutoTriggerConfig(AutoTriggerConfig):
    """Auto-trigger settings owned by the current Transformer/Web path."""

    adaptive_rearm_enabled: bool = False
    adaptive_rearm_hold_sec: float = 0.50
    adaptive_rearm_requires_knee_rest: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.adaptive_rearm_hold_sec < 0:
            raise ValueError("Adaptive re-arm hold must be non-negative.")


def load_web_trigger_config(path: str | Path) -> WebAutoTriggerConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Web auto-trigger config JSON must contain an object.")
    allowed = {field.name for field in fields(WebAutoTriggerConfig)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unknown web auto-trigger config keys: {', '.join(sorted(unknown))}")
    return WebAutoTriggerConfig(**payload)


class WebAutoTriggerEngine(AutoTriggerEngine):
    """Add rest-gated re-arming and pose-wrist fallback to the v13 engine."""

    config: WebAutoTriggerConfig

    def __init__(self, config: WebAutoTriggerConfig):
        super().__init__(config)
        self._rearm_start_sec: float | None = None
        self._rearm_signatures: list[np.ndarray] = []
        self._rearm_wrist_signatures: list[np.ndarray] = []
        self._rearm_ready = False
        self.reference_revision = 0

    @property
    def calibrated(self) -> bool:
        return bool(
            not self.config.reference_rest_enabled
            or self._rest_reference_signature is not None
            or self._rest_wrist_reference_signature is not None
        )

    def reset(self) -> None:
        super().reset()
        self._clear_rearm_window()
        self.reference_revision = 0

    def update(
        self,
        frame_vector: np.ndarray,
        analysis: AutoFrameAnalysis,
        timestamp_sec: float,
    ) -> SegmentResult | None:
        if not self.config.adaptive_rearm_enabled or self.state not in {
            SEGMENT_STATE_COOLDOWN,
            SEGMENT_STATE_REARMING,
        }:
            result = super().update(frame_vector, analysis, timestamp_sec)
            if result is not None:
                self._clear_rearm_window()
            return result

        timestamp_sec = float(timestamp_sec)
        if self._last_timestamp_sec is not None and timestamp_sec < self._last_timestamp_sec:
            raise ValueError("Frame timestamps must be monotonic.")
        self._last_timestamp_sec = timestamp_sec

        if self.state == SEGMENT_STATE_COOLDOWN:
            self._update_rearm(timestamp_sec, analysis, transition_when_ready=False)
            if timestamp_sec < self._cooldown_until_sec:
                return None
            self.clip_start_sec = None
            self._pre_roll.clear()
            self._low_motion_start_sec = None
            self.state = SEGMENT_STATE_REARMING
            if self._rearm_ready:
                self._complete_rearm()
            return None

        self._update_rearm(timestamp_sec, analysis)
        return None

    def _is_start_candidate(self, analysis: AutoFrameAnalysis) -> bool:
        if self.config.reference_rest_enabled and not self.calibrated:
            return False
        reference_distance = self._reference_distance(analysis)
        departed_reference_pose = bool(
            reference_distance is not None
            and reference_distance >= self.config.reference_departure_distance_threshold
        )
        if self._is_rest_candidate(analysis) and not departed_reference_pose:
            return False
        if (
            analysis.effective_motion_score < self.config.start_motion_threshold
            and not departed_reference_pose
        ):
            return False
        if (
            self.config.temporal_classifier_enabled
            and analysis.temporal_active_probability is not None
            and analysis.temporal_active_probability
            < self.config.temporal_start_probability_threshold
        ):
            return False
        return True

    def _update_rest_reference(
        self,
        timestamp_sec: float,
        analysis: AutoFrameAnalysis,
    ) -> None:
        if not self.config.reference_rest_enabled or self.calibrated:
            return
        if not self._is_stable_reference_sample(analysis):
            self._reference_seed_start_sec = None
            self._reference_signatures = []
            self._reference_wrist_signatures = []
            return
        if self._reference_seed_start_sec is None:
            self._reference_seed_start_sec = timestamp_sec
        if analysis.rest_signature is not None:
            self._reference_signatures.append(
                np.asarray(analysis.rest_signature, dtype=np.float32)
            )
        assert analysis.wrist_rest_signature is not None
        self._reference_wrist_signatures.append(
            np.asarray(analysis.wrist_rest_signature, dtype=np.float32)
        )
        if timestamp_sec - self._reference_seed_start_sec >= self.config.reference_seed_sec:
            self._set_rest_reference(
                self._reference_signatures,
                self._reference_wrist_signatures,
            )

    def _is_stable_reference_sample(self, analysis: AutoFrameAnalysis) -> bool:
        return bool(
            analysis.wrist_rest_signature is not None
            and analysis.wrists_detected
            and analysis.torso_valid
            and analysis.effective_motion_score
            <= self.config.reference_seed_motion_threshold
        )

    def _is_safe_rearm_sample(self, analysis: AutoFrameAnalysis) -> bool:
        if not self._is_stable_reference_sample(analysis):
            return False
        if not self.config.adaptive_rearm_requires_knee_rest:
            return True
        if analysis.hands_on_knees:
            return True
        distance = self._reference_distance(analysis)
        return bool(
            distance is not None
            and distance <= self.config.reference_rest_distance_threshold
        )

    def _update_rearm(
        self,
        timestamp_sec: float,
        analysis: AutoFrameAnalysis,
        *,
        transition_when_ready: bool = True,
    ) -> None:
        if self._rearm_ready:
            if transition_when_ready:
                self._complete_rearm()
            return
        if not self._is_safe_rearm_sample(analysis):
            self._clear_rearm_window()
            return
        if self._rearm_start_sec is None:
            self._rearm_start_sec = timestamp_sec
        if analysis.rest_signature is not None:
            self._rearm_signatures.append(
                np.asarray(analysis.rest_signature, dtype=np.float32)
            )
        assert analysis.wrist_rest_signature is not None
        self._rearm_wrist_signatures.append(
            np.asarray(analysis.wrist_rest_signature, dtype=np.float32)
        )
        if (
            timestamp_sec - self._rearm_start_sec + 1e-9
            < self.config.adaptive_rearm_hold_sec
        ):
            return
        self._rearm_ready = True
        if transition_when_ready:
            self._complete_rearm()

    def _complete_rearm(self) -> None:
        if not self._rearm_ready:
            return
        self._set_rest_reference(
            self._rearm_signatures,
            self._rearm_wrist_signatures,
        )
        self.state = SEGMENT_STATE_IDLE
        self._pre_roll.clear()
        self._active_start_sec = None
        self._clear_rearm_window()

    def _set_rest_reference(
        self,
        signatures: list[np.ndarray],
        wrist_signatures: list[np.ndarray],
    ) -> None:
        if not wrist_signatures:
            raise ValueError("A wrist reference requires at least one sample.")
        self._rest_reference_signature = (
            np.median(np.stack(signatures), axis=0).astype(np.float32)
            if signatures
            else None
        )
        self._rest_wrist_reference_signature = np.median(
            np.stack(wrist_signatures), axis=0
        ).astype(np.float32)
        self.reference_revision += 1

    def _clear_rearm_window(self) -> None:
        self._rearm_start_sec = None
        self._rearm_signatures = []
        self._rearm_wrist_signatures = []
        self._rearm_ready = False

    def rest_signature_status(self, analysis: AutoFrameAnalysis | None) -> str:
        if not self.calibrated:
            return "uncalibrated"
        if analysis is None or not analysis.torso_valid:
            return "missing_pose"
        if analysis.wrist_rest_signature is None or not analysis.wrists_detected:
            return "missing_wrists"
        if analysis.rest_signature is None:
            return "pose_wrist_fallback"
        return "palm_and_wrist"


class WebAutoKnee42Controller(AutoKnee42Controller):
    """Use the Web-specific engine while retaining the shared feature buffer."""

    engine: WebAutoTriggerEngine

    def __init__(
        self,
        config: WebAutoTriggerConfig | AutoTriggerConfig,
        *,
        initial_mode: str = "auto",
        analysis_fn: Callable[
            [np.ndarray | None, np.ndarray, AutoTriggerConfig], AutoFrameAnalysis
        ] = analyze_frame_vector,
    ):
        if not isinstance(config, WebAutoTriggerConfig):
            config = WebAutoTriggerConfig(**config.to_dict())
        super().__init__(config, initial_mode=initial_mode, analysis_fn=analysis_fn)
        self.engine = WebAutoTriggerEngine(config)

    @property
    def calibrated(self) -> bool:
        return self.engine.calibrated
