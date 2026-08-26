"""Temporal interaction controllers for Knee42 manual and sliding inference."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from recognition.realtime.auto_trigger import (
    AutoFrameAnalysis,
    AutoTriggerConfig,
    AutoTriggerEngine,
    analyze_frame_vector,
)
from recognition.realtime.knee42_clock import MAX_PRACTICAL_FPS


MIN_PRACTICAL_TIMESTAMP_INTERVAL_SEC = 1.0 / MAX_PRACTICAL_FPS
MAX_BUFFERED_OBSERVATIONS = 4096
MAX_HELD_OBSERVATION_SAMPLES = 256
MAX_EOF_SYNTHETIC_SAMPLES = 256
_TIMESTAMP_INTERVAL_EPSILON_SEC = 1e-12


@dataclass(frozen=True)
class SegmentEvidence:
    clip_start_sec: float
    clip_end_sec: float
    finalize_sec: float
    reason: str
    rest_detected_sec: float | None = None
    boundary_policy: str = "first_confirmed_rest"


@dataclass(frozen=True)
class ControllerEvent:
    state: str
    infer: bool = False
    features: tuple[Any, ...] = ()
    message: str = ""
    segment: SegmentEvidence | None = None


class ManualController:
    """Space-controlled record/stop state machine."""

    def __init__(self):
        self.state = "idle"
        self._features: list[Any] = []

    @property
    def features(self) -> list[Any]:
        return list(self._features)

    def on_space(self) -> ControllerEvent:
        if self.state in {"idle", "result"}:
            self._features.clear()
            self.state = "recording"
            return ControllerEvent(self.state, message="recording started")
        if self.state == "recording":
            if not self._features:
                self.state = "error"
                return ControllerEvent(self.state, message="no feature frames were recorded")
            self.state = "infer"
            return ControllerEvent(
                self.state,
                infer=True,
                features=tuple(self._features),
                message="recording stopped",
            )
        return ControllerEvent(self.state, message="press R to reset before recording")

    def add_feature(self, feature: Any) -> ControllerEvent:
        if self.state == "recording":
            self._features.append(feature)
        return ControllerEvent(self.state)

    def mark_result(self) -> ControllerEvent:
        if self.state != "infer":
            raise RuntimeError(f"cannot mark result from state {self.state}")
        self.state = "result"
        return ControllerEvent(self.state)

    def mark_error(self, message: str) -> ControllerEvent:
        self.state = "error"
        return ControllerEvent(self.state, message=message)

    def reset(self) -> ControllerEvent:
        self._features.clear()
        self.state = "idle"
        return ControllerEvent(self.state)


class SlidingController:
    """Bounded rolling feature window with deterministic inference throttling."""

    def __init__(self, *, window: int = 64, inference_stride: int = 8):
        if window <= 0 or inference_stride <= 0:
            raise ValueError("window and inference_stride must be positive")
        self.window = window
        self.inference_stride = inference_stride
        self._features: deque[Any] = deque(maxlen=window)
        self._received = 0

    @property
    def features(self) -> list[Any]:
        return list(self._features)

    @property
    def state(self) -> str:
        return "sliding" if len(self._features) == self.window else "warming"

    def add_feature(self, feature: Any) -> ControllerEvent:
        self._features.append(feature)
        self._received += 1
        ready = (
            len(self._features) == self.window
            and (self._received - self.window) % self.inference_stride == 0
        )
        return ControllerEvent(
            self.state,
            infer=ready,
            features=tuple(self._features) if ready else (),
        )

    def reset(self) -> ControllerEvent:
        self._features.clear()
        self._received = 0
        return ControllerEvent(self.state)


class AutoKnee42Controller:
    """Adapt timestamped 225-value trigger decisions to Knee42 model features."""

    def __init__(
        self,
        config: AutoTriggerConfig,
        *,
        initial_mode: str = "auto",
        analysis_fn: Callable[
            [np.ndarray | None, np.ndarray, AutoTriggerConfig], AutoFrameAnalysis
        ] = analyze_frame_vector,
    ):
        if initial_mode not in {"auto", "manual"}:
            raise ValueError("initial_mode must be auto or manual")
        self.config = config
        self.engine = AutoTriggerEngine(config)
        self.mode = initial_mode
        self._manual = ManualController()
        self._analysis_fn = analysis_fn
        self._previous_trigger: np.ndarray | None = None
        self._last_rest_trigger: np.ndarray | None = None
        self._last_rest_feature: Any = None
        self._features: dict[float, Any] = {}
        self._timestamps: deque[float] = deque()
        self._last_timestamp: float | None = None
        self._buffer_horizon_sec = (
            config.max_segment_sec
            + config.pre_roll_sec
            + config.start_hold_sec
            + config.end_hold_sec
            + config.cooldown_sec
            + 1.0
        )

    @property
    def state(self) -> str:
        return self.engine.state if self.mode == "auto" else self._manual.state

    @property
    def buffered_observations(self) -> int:
        return len(self._timestamps)

    @property
    def sample_timestamps(self) -> tuple[float, ...]:
        return tuple(self._timestamps)

    @property
    def calibrated(self) -> bool:
        return bool(
            not self.config.reference_rest_enabled
            or self.engine._rest_reference_signature is not None
        )

    def add_observation(
        self,
        timestamp_sec: float,
        trigger_values: np.ndarray,
        feature: Any,
    ) -> ControllerEvent:
        timestamp_sec = float(timestamp_sec)
        self._validate_timestamp(timestamp_sec, self._last_timestamp)
        trigger = np.asarray(trigger_values, dtype=np.float32)
        if trigger.shape != (225,):
            raise ValueError(f"expected 225 trigger values, got {trigger.shape}")
        if self.mode == "manual":
            self._last_timestamp = timestamp_sec
            return self._manual.add_feature(feature)
        analysis = self._analysis_fn(self._previous_trigger, trigger, self.config)
        self._previous_trigger = trigger.copy()
        return self._add_analyzed_observation(
            timestamp_sec,
            trigger,
            feature,
            analysis,
        )

    def _add_analyzed_observation(
        self,
        timestamp_sec: float,
        trigger: np.ndarray,
        feature: Any,
        analysis: AutoFrameAnalysis,
    ) -> ControllerEvent:
        self._validate_timestamp(timestamp_sec, self._last_timestamp)
        self._last_timestamp = timestamp_sec
        self._timestamps.append(timestamp_sec)
        self._features[timestamp_sec] = feature
        self._trim(timestamp_sec)
        state_before_update = self.engine.state
        rest_candidate = self.engine._is_rest_candidate(analysis)
        segment = self.engine.update(trigger, analysis, timestamp_sec)
        if rest_candidate and state_before_update in {"SIGNING_ACTIVE", "END_CONFIRM"}:
            self._last_rest_trigger = trigger.copy()
            self._last_rest_feature = feature
        if segment is None:
            return ControllerEvent(self.state)
        if segment.duration_sec < self.config.min_segment_sec:
            return ControllerEvent(self.state, message="short_segment")
        missing = [
            sample.timestamp_sec
            for sample in segment.samples
            if sample.timestamp_sec not in self._features
        ]
        if missing:
            raise RuntimeError(f"recognition feature missing for trigger timestamps: {missing}")
        features = tuple(self._features[sample.timestamp_sec] for sample in segment.samples)
        return ControllerEvent(
            self.state,
            infer=bool(features),
            features=features,
            message=segment.reason,
            segment=SegmentEvidence(
                clip_start_sec=segment.clip_start_sec,
                clip_end_sec=segment.clip_end_sec,
                finalize_sec=segment.finalize_sec,
                reason=segment.reason,
                rest_detected_sec=segment.rest_detected_sec,
                boundary_policy=segment.boundary_policy,
            ),
        )

    def add_held_observation(
        self,
        timestamp_sec: float,
        trigger_values: np.ndarray,
        feature: Any,
        *,
        frame_interval_sec: float,
        sample_count: int,
    ) -> ControllerEvent:
        """Compatibility wrapper for callers that still provide a nominal interval."""
        frame_interval_sec = float(frame_interval_sec)
        sample_count = int(sample_count)
        if (
            not np.isfinite(frame_interval_sec)
            or frame_interval_sec < MIN_PRACTICAL_TIMESTAMP_INTERVAL_SEC
        ):
            raise ValueError("frame_interval_sec must have a finite practical cadence")
        if sample_count <= 0 or sample_count > MAX_HELD_OBSERVATION_SAMPLES:
            raise ValueError(
                "sample_count must be positive and at most the held observation bound "
                f"{MAX_HELD_OBSERVATION_SAMPLES}"
            )
        timestamp_sec = float(timestamp_sec)
        return self.add_held_observation_at_times(
            [
                timestamp_sec + sample_index * frame_interval_sec
                for sample_index in range(sample_count)
            ],
            trigger_values,
            feature,
        )

    def add_held_observation_at_times(
        self,
        timestamp_sec_values: Sequence[float],
        trigger_values: np.ndarray,
        feature: Any,
    ) -> ControllerEvent:
        """Hold one detector result at each exact collected source timestamp."""
        timestamps = tuple(float(value) for value in timestamp_sec_values)
        if not timestamps:
            raise ValueError("held observation timestamps cannot be empty")
        if not all(np.isfinite(value) for value in timestamps):
            raise ValueError("held observation timestamps must be finite")
        previous = self._last_timestamp
        for timestamp_sec in timestamps:
            self._validate_timestamp(timestamp_sec, previous)
            previous = timestamp_sec

        trigger = np.asarray(trigger_values, dtype=np.float32)
        if trigger.shape != (225,):
            raise ValueError(f"expected 225 trigger values, got {trigger.shape}")
        if self.mode == "manual":
            event = ControllerEvent(self.state)
            inference_event: ControllerEvent | None = None
            for timestamp_sec in timestamps:
                event = self.add_observation(timestamp_sec, trigger, feature)
                if event.infer and inference_event is None:
                    inference_event = event
            return inference_event or event

        analysis = self._analysis_fn(self._previous_trigger, trigger, self.config)
        self._previous_trigger = trigger.copy()
        event = ControllerEvent(self.state)
        inference_event = None
        for timestamp_sec in timestamps:
            event = self._add_analyzed_observation(
                timestamp_sec,
                trigger,
                feature,
                analysis,
            )
            if event.infer and inference_event is None:
                inference_event = event
        return inference_event or event

    def toggle_mode(self) -> ControllerEvent:
        target = "manual" if self.mode == "auto" else "auto"
        self.reset()
        self.mode = target
        return ControllerEvent(self.state, message=f"mode {target}")

    def on_space(self) -> ControllerEvent:
        if self.mode != "manual":
            return ControllerEvent(self.state, message="space is available in manual mode")
        return self._manual.on_space()

    def finalize_video_eof(
        self,
        *,
        frame_interval_sec: float | None = None,
    ) -> ControllerEvent:
        """Complete an already-started rest confirmation at recorded-video EOF."""
        if frame_interval_sec is not None:
            frame_interval_sec = float(frame_interval_sec)
            if (
                not np.isfinite(frame_interval_sec)
                or frame_interval_sec < MIN_PRACTICAL_TIMESTAMP_INTERVAL_SEC
            ):
                raise ValueError(
                    "frame_interval_sec must have a finite practical cadence"
                )
        if (
            self.mode != "auto"
            or self.state != "END_CONFIRM"
            or self._last_timestamp is None
            or self._last_rest_trigger is None
        ):
            return ControllerEvent(self.state)
        observed_timestamps = tuple(self._timestamps)
        observed_intervals = [
            later - earlier
            for earlier, later in zip(
                observed_timestamps[:-1],
                observed_timestamps[1:],
            )
            if later > earlier
        ]
        if not observed_intervals:
            raise ValueError(
                "cannot finalize video EOF without advancing collected timestamps"
            )
        frame_interval_sec = float(np.median(observed_intervals))
        if frame_interval_sec < MIN_PRACTICAL_TIMESTAMP_INTERVAL_SEC:
            raise ValueError("collected EOF timestamps have an impractical cadence")
        # Allow one transition frame plus endpoint/granularity slack before the
        # end-hold window contains a full interval of repeated rest evidence.
        repeats = min(
            MAX_EOF_SYNTHETIC_SAMPLES,
            int(np.ceil(self.config.end_hold_sec / frame_interval_sec)) + 4,
        )
        event = ControllerEvent(self.state)
        for _ in range(repeats):
            event = self.add_observation(
                self._last_timestamp + frame_interval_sec,
                self._last_rest_trigger.copy(),
                self._last_rest_feature,
            )
            if event.infer:
                return event
        return event

    def mark_result(self) -> ControllerEvent:
        if self.mode == "manual":
            return self._manual.mark_result()
        return ControllerEvent(self.state)

    def reset(self) -> ControllerEvent:
        self.engine.reset()
        self._manual.reset()
        self._previous_trigger = None
        self._last_rest_trigger = None
        self._last_rest_feature = None
        self._features.clear()
        self._timestamps.clear()
        self._last_timestamp = None
        return ControllerEvent(self.state)

    def _trim(self, timestamp_sec: float) -> None:
        cutoff = timestamp_sec - self._buffer_horizon_sec
        while self._timestamps and self._timestamps[0] < cutoff:
            expired = self._timestamps.popleft()
            self._features.pop(expired, None)
        while len(self._timestamps) > MAX_BUFFERED_OBSERVATIONS:
            expired = self._timestamps.popleft()
            self._features.pop(expired, None)

    @staticmethod
    def _validate_timestamp(timestamp_sec: float, previous: float | None) -> None:
        if not np.isfinite(timestamp_sec):
            raise ValueError("Frame timestamps must be finite.")
        if previous is None:
            return
        interval_sec = timestamp_sec - previous
        if interval_sec <= 0.0:
            raise ValueError("Frame timestamps must be strictly monotonic increasing.")
        if (
            interval_sec + _TIMESTAMP_INTERVAL_EPSILON_SEC
            < MIN_PRACTICAL_TIMESTAMP_INTERVAL_SEC
        ):
            raise ValueError(
                "Frame timestamp cadence exceeds the practical 240 FPS limit."
            )
