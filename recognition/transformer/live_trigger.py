"""Transformer/Web auto-trigger extensions without changing the archived v13 source.

The v13 release binds ``recognition.realtime.auto_trigger`` by SHA-256.  The
current Transformer and Web paths therefore extend that state machine here so
new behavior cannot silently rewrite the provenance of the published archive.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from recognition.realtime.auto_trigger import (
    SEGMENT_STATE_COOLDOWN,
    SEGMENT_STATE_ACTIVE,
    SEGMENT_STATE_END_CONFIRM,
    SEGMENT_STATE_IDLE,
    AutoFrameAnalysis,
    AutoTriggerConfig,
    AutoTriggerEngine,
    SegmentResult,
    FrameSample,
    analyze_frame_vector,
)
from recognition.realtime.knee42_controllers import AutoKnee42Controller, ControllerEvent, SegmentEvidence
from recognition.transformer.knee_geometry import analyze_knee_frame, sanitize_trigger


SEGMENT_STATE_REARMING = "REARMING"


@dataclass(frozen=True)
class WebAutoTriggerConfig(AutoTriggerConfig):
    """Auto-trigger settings owned by the current Transformer/Web path."""

    adaptive_rearm_enabled: bool = False
    adaptive_rearm_hold_sec: float = 0.50
    adaptive_rearm_requires_knee_rest: bool = False
    knee_gate_enabled: bool = False
    knee_occlusion_grace_sec: float = 1.0
    observation_gap_sec: float = 0.25
    body_shift_threshold: float = 0.50
    body_scale_ratio_threshold: float = 1.25
    knee_min_thigh_progress_ratio: float = 0.35

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.adaptive_rearm_hold_sec < 0:
            raise ValueError("Adaptive re-arm hold must be non-negative.")
        if any(isinstance(value, (int, float)) and not np.isfinite(value)
               for value in self.to_dict().values()):
            raise ValueError("Trigger settings must be finite.")
        if not 0 < self.observation_gap_sec or not 0 <= self.knee_occlusion_grace_sec <= 1:
            raise ValueError("Invalid observation gap or knee occlusion grace (maximum 1 second).")
        if self.body_shift_threshold <= 0 or self.body_scale_ratio_threshold <= 1:
            raise ValueError("Invalid body relocation thresholds.")
        if not 0 < self.pose_visibility_threshold <= 1:
            raise ValueError("Pose visibility threshold must be in (0, 1].")
        if self.knee_gate_enabled and not (self.reference_rest_enabled and self.adaptive_rearm_enabled
                                          and self.adaptive_rearm_requires_knee_rest):
            raise ValueError("Knee gate requires reference calibration and knee-gated rearming.")
        if self.knee_gate_enabled and not (.25 <= self.knee_min_thigh_progress_ratio
                                          < self.knee_max_thigh_progress_ratio <= 1.5):
            raise ValueError("Strict knee geometry must stay in the lower-thigh/knee region.")


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
        self._body_reference = None
        self._current_body = None
        self._last_visible_knee_sec = None
        self._needs_recalibration = False
        self.last_segment = None
        self.last_analysis = None
        self._end_onset_sec = None
        self._seed_body = self._rearm_body = None

    @property
    def calibrated(self) -> bool:
        return bool(
            not self.config.reference_rest_enabled
            or self._rest_reference_signature is not None
            or self._rest_wrist_reference_signature is not None
        ) and not self._needs_recalibration

    def reset(self) -> None:
        super().reset()
        self._clear_rearm_window()
        self.reference_revision = 0
        self._body_reference = self._current_body = None
        self._last_visible_knee_sec = None
        self._needs_recalibration = False
        self.last_segment = self.last_analysis = None
        self._end_onset_sec = None
        self._seed_body = self._rearm_body = None

    def update(
        self,
        frame_vector: np.ndarray,
        analysis: AutoFrameAnalysis,
        timestamp_sec: float,
    ) -> SegmentResult | None:
        timestamp_sec = float(timestamp_sec)
        if not np.isfinite(timestamp_sec):
            raise ValueError("Frame timestamps must be finite.")
        if self._last_timestamp_sec is not None and timestamp_sec < self._last_timestamp_sec:
            raise ValueError("Frame timestamps must be monotonic.")
        self.last_segment = None
        self.last_analysis = analysis
        if self.config.knee_gate_enabled:
            self._current_body = getattr(analysis, "body_anchor", None)
            if self._body_moved():
                self._needs_recalibration = True
                if self.state == SEGMENT_STATE_IDLE and self._body_reference is not None:
                    self.state = SEGMENT_STATE_REARMING
            if analysis.knee_landmarks_valid and analysis.torso_valid:
                self._last_visible_knee_sec = timestamp_sec
            if (self._last_timestamp_sec is not None
                    and timestamp_sec - self._last_timestamp_sec > self.config.observation_gap_sec):
                self._clear_rearm_window()
                self._clear_seed_window()
                self._pre_roll.clear()
                self._active_start_sec = None
                if self.state in {SEGMENT_STATE_ACTIVE, SEGMENT_STATE_END_CONFIRM}:
                    last_observed = self._last_timestamp_sec
                    self._last_timestamp_sec = timestamp_sec
                    return self._finalize(last_observed, timestamp_sec, "tracking_gap", boundary_policy="tracking_gap")
        if not self.config.adaptive_rearm_enabled or self.state not in {
            SEGMENT_STATE_COOLDOWN,
            SEGMENT_STATE_REARMING,
        }:
            result = self._update_tracking(frame_vector, analysis, timestamp_sec)
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

    def _update_tracking(self, frame_vector, analysis, timestamp_sec):
        """Use observed elapsed time; keep the sample *before* a window edge.

        The archived engine drops that sample and then asks the remaining
        window to span a full hold, which fails on non-divisor frame rates.
        No synthetic observations or future frames are used here.
        """
        timestamp_sec = float(timestamp_sec)
        if not np.isfinite(timestamp_sec):
            raise ValueError("Frame timestamps must be finite.")
        if self._last_timestamp_sec is not None and timestamp_sec < self._last_timestamp_sec:
            raise ValueError("Frame timestamps must be monotonic.")
        self._last_timestamp_sec = timestamp_sec
        sample = FrameSample(timestamp_sec, np.asarray(frame_vector, dtype=np.float32).copy())
        self._update_rest_reference(timestamp_sec, analysis)
        if self.state == SEGMENT_STATE_COOLDOWN:
            if timestamp_sec < self._cooldown_until_sec:
                return None
            self.state = SEGMENT_STATE_IDLE
            self._pre_roll.clear()
        if self.state == SEGMENT_STATE_IDLE:
            return self._update_idle(sample, analysis)
        self.segment_samples.append(sample)
        if self.clip_start_sec is not None and timestamp_sec - self.clip_start_sec >= self.config.max_segment_sec:
            return self._finalize(timestamp_sec, timestamp_sec, "timeout_finalize")
        rest = self._is_rest_candidate(analysis)
        if not rest:
            self._end_onset_sec = None
        elif self.state == SEGMENT_STATE_END_CONFIRM and self._end_onset_sec is None:
            self._end_onset_sec = timestamp_sec
        if self.config.knee_gate_enabled and not rest and self.state == SEGMENT_STATE_END_CONFIRM:
            # No confirmation may bridge a chest pause or a lost wrist.
            self.state = SEGMENT_STATE_ACTIVE
            self._end_votes.clear()
            self._end_onset_sec = None
        if self.state == SEGMENT_STATE_ACTIVE:
            if rest:
                self.state = SEGMENT_STATE_END_CONFIRM
                self._end_votes = deque([(timestamp_sec, True)])
                self._end_onset_sec = timestamp_sec
            return None
        self._end_votes.append((timestamp_sec, rest))
        cutoff = timestamp_sec - self.config.end_hold_sec
        # Retain one predecessor so variable FPS covers the window exactly.
        while len(self._end_votes) > 1 and self._end_votes[1][0] <= cutoff:
            self._end_votes.popleft()
        if not any(vote for _, vote in self._end_votes):
            self.state = SEGMENT_STATE_ACTIVE
            self._end_votes.clear()
            self._end_onset_sec = None
            return None
        elapsed = timestamp_sec - self._end_votes[0][0]
        votes = list(self._end_votes)
        rest_time = sum(
            max(0.0, end - max(start, cutoff))
            for (start, yes), (end, _) in zip(votes, votes[1:]) if yes
        )
        if rest and elapsed + 1e-9 >= self.config.end_hold_sec and rest_time + 1e-9 >= self.config.end_hold_sec * self.config.end_rest_vote_ratio:
            boundary = self._end_onset_sec if self._end_onset_sec is not None else next(t for t, yes in votes if yes)
            reason = "visible_rest_finalize" if analysis.visible_rest_blank else "reference_rest_finalize"
            return self._finalize(boundary, timestamp_sec, reason, rest_detected_sec=boundary)
        return None

    def _is_start_candidate(self, analysis: AutoFrameAnalysis) -> bool:
        if self.config.knee_gate_enabled and not (analysis.torso_valid and analysis.wrists_detected):
            return False
        if self.config.knee_gate_enabled and self._is_rest_candidate(analysis):
            return False
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
        if (not self.config.reference_rest_enabled or self.calibrated
                or self.state != SEGMENT_STATE_IDLE):
            return
        if not self._is_stable_reference_sample(analysis):
            self._clear_seed_window()
            return
        if self.config.knee_gate_enabled and self._body_difference(self._seed_body, self._current_body):
            self._clear_seed_window()
        if self._reference_seed_start_sec is None:
            self._reference_seed_start_sec = timestamp_sec
            self._seed_body = self._current_body
        if analysis.rest_signature is not None:
            self._reference_signatures.append(
                np.asarray(analysis.rest_signature, dtype=np.float32)
            )
        assert analysis.wrist_rest_signature is not None
        self._reference_wrist_signatures.append(
            np.asarray(analysis.wrist_rest_signature, dtype=np.float32)
        )
        if timestamp_sec - self._reference_seed_start_sec + 1e-9 >= self.config.reference_seed_sec:
            self._set_rest_reference(
                self._reference_signatures,
                self._reference_wrist_signatures,
            )

    def _is_stable_reference_sample(self, analysis: AutoFrameAnalysis) -> bool:
        return bool(
            (not self.config.knee_gate_enabled or (analysis.knee_landmarks_valid and analysis.hands_on_knees))
            and
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
        if self.config.knee_gate_enabled:
            return False
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
        if not self._is_safe_rearm_sample(analysis):
            self._clear_rearm_window()
            return
        if self.config.knee_gate_enabled and self._body_difference(self._rearm_body, self._current_body):
            self._clear_rearm_window()
        if self._rearm_ready:
            if transition_when_ready:
                self._complete_rearm()
            return
        if self._rearm_start_sec is None:
            self._rearm_start_sec = timestamp_sec
            self._rearm_body = self._current_body
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
        self._body_reference = self._current_body
        self._needs_recalibration = False
        self._clear_seed_window()

    def _clear_seed_window(self):
        self._reference_seed_start_sec = None
        self._reference_signatures = []
        self._reference_wrist_signatures = []
        self._seed_body = None

    def _body_moved(self):
        return self._body_difference(self._body_reference, self._current_body)

    def _body_difference(self, reference, current):
        if reference is None or current is None:
            return False
        old, new = np.asarray(reference), np.asarray(current)
        shift = max(np.linalg.norm(new[:2] - old[:2]), np.linalg.norm(new[2:4] - old[2:4])) / old[-1]
        ratio = max(new[-1] / old[-1], old[-1] / new[-1])
        return bool(shift > self.config.body_shift_threshold or ratio > self.config.body_scale_ratio_threshold)

    def _reference_distance(self, analysis):
        if not self.config.knee_gate_enabled:
            return super()._reference_distance(analysis)
        if self._rest_wrist_reference_signature is None or analysis.wrist_rest_signature is None:
            return None
        return float(np.sqrt(np.mean((np.asarray(analysis.wrist_rest_signature) - self._rest_wrist_reference_signature) ** 2)))

    def _is_rest_candidate(self, analysis):
        if not self.config.knee_gate_enabled:
            return super()._is_rest_candidate(analysis)
        if not (analysis.torso_valid and analysis.wrists_detected):
            return False
        if analysis.effective_motion_score > self.config.blank_motion_threshold:
            return False
        if analysis.knee_landmarks_valid:
            return bool(analysis.hands_on_knees)
        age = (float("inf") if self._last_visible_knee_sec is None or self._last_timestamp_sec is None
               else self._last_timestamp_sec - self._last_visible_knee_sec)
        distance = self._reference_distance(analysis)
        return bool(self.calibrated and not self._body_moved()
                    and age <= self.config.knee_occlusion_grace_sec + 1e-9
                    and distance is not None and distance <= self.config.reference_rest_distance_threshold)

    def _finalize(self, *args, **kwargs):
        result = super()._finalize(*args, **kwargs)
        self.last_segment = result
        self._clear_rearm_window()
        self._end_onset_sec = None
        return result

    def _clear_rearm_window(self) -> None:
        self._rearm_start_sec = None
        self._rearm_signatures = []
        self._rearm_wrist_signatures = []
        self._rearm_ready = False
        self._rearm_body = None

    def rest_signature_status(self, analysis: AutoFrameAnalysis | None) -> str:
        if self.config.knee_gate_enabled:
            if analysis is None or not analysis.torso_valid:
                return "missing_pose"
            if not analysis.wrists_detected:
                return "missing_wrists"
            if not self.calibrated and not analysis.knee_landmarks_valid:
                return "waiting_visible_knees"
            if not self.calibrated:
                return "waiting_knee_calibration"
            if self._is_rest_candidate(analysis) and not analysis.knee_landmarks_valid:
                return "calibrated_wrist_fallback"
            if not analysis.knee_landmarks_valid:
                return "waiting_visible_knees"
            if analysis.hands_on_knees and analysis.effective_motion_score > self.config.blank_motion_threshold:
                return "knee_pose_moving"
            return "visible_knee_rest" if analysis.hands_on_knees else "waiting_knee_return"
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
        self.last_analysis = None

    @property
    def calibrated(self) -> bool:
        return self.engine.calibrated

    def add_observation(self, timestamp_sec, trigger_values, feature, *, pose_visibility=None):
        timestamp_sec = float(timestamp_sec)
        if not np.isfinite(timestamp_sec):
            raise ValueError("Frame timestamps must be finite.")
        if self._last_timestamp is not None and timestamp_sec < self._last_timestamp:
            raise ValueError("Frame timestamps must be monotonic.")
        if timestamp_sec == self._last_timestamp:
            return ControllerEvent(self.state, message="duplicate_timestamp")
        if self.mode == "manual":
            return super().add_observation(timestamp_sec, trigger_values, feature)
        trigger = np.asarray(trigger_values, dtype=np.float32)
        if trigger.shape != (225,):
            raise ValueError(f"expected 225 trigger values, got {trigger.shape}")
        if self.config.knee_gate_enabled:
            trigger = sanitize_trigger(trigger, pose_visibility, self.config.pose_visibility_threshold)
            interval = 0 if self._last_timestamp is None else timestamp_sec - self._last_timestamp
            self.last_analysis = analyze_knee_frame(self._previous_trigger, trigger, self.config, interval,
                                                   visibility_available=pose_visibility is not None)
        else:
            self.last_analysis = self._analysis_fn(self._previous_trigger, trigger, self.config)
        self._previous_trigger = trigger.copy()
        self._last_timestamp = timestamp_sec
        self._timestamps.append(timestamp_sec)
        self._features[timestamp_sec] = feature
        self._trim(timestamp_sec)
        segment = self.engine.update(trigger, self.last_analysis, timestamp_sec)
        if segment is None:
            return ControllerEvent(self.state)
        evidence = SegmentEvidence(segment.clip_start_sec, segment.clip_end_sec,
                                   segment.finalize_sec, segment.reason, segment.rest_detected_sec,
                                   segment.boundary_policy)
        if segment.reason in {"timeout_finalize", "tracking_gap"}:
            return ControllerEvent(self.state, message=segment.reason, segment=evidence)
        if segment.duration_sec < self.config.min_segment_sec:
            return ControllerEvent(self.state, message="short_segment", segment=evidence)
        # Only accepted segments need model features. Long gaps may already
        # have evicted old observations and must never attempt this lookup.
        features = tuple(self._features[sample.timestamp_sec] for sample in segment.samples)
        return ControllerEvent(self.state, infer=bool(features), features=features,
                               message=segment.reason, segment=evidence)

    def add_held_observation(self, timestamp_sec, trigger_values, feature, *, frame_interval_sec, sample_count):
        # Current Web must never turn one detector output into several votes.
        if sample_count != 1:
            raise ValueError("Web trigger accepts real observations only (sample_count=1).")
        return self.add_observation(timestamp_sec, trigger_values, feature)

    def finalize_video_eof(self, *, frame_interval_sec):
        if frame_interval_sec <= 0:
            raise ValueError("frame_interval_sec must be positive")
        return ControllerEvent(self.state, message="incomplete_eof" if self.state in
                               {SEGMENT_STATE_ACTIVE, SEGMENT_STATE_END_CONFIRM} else "eof")

    def reset(self):
        result = super().reset()
        self.last_analysis = None
        return result
