"""Timestamp contracts for live capture and deterministic video replay."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable


LIVE_CLOCK_MODE = "live_perf_counter"
VIDEO_PENDING_MODE = "video_timestamp_pending"
VIDEO_SOURCE_MODE = "video_source_timestamp"
VIDEO_FALLBACK_MODE = "video_nominal_fps_fallback"


@dataclass(frozen=True)
class FramePacket:
    """One decoded frame with its source-owned timestamp contract."""

    frame: Any
    timestamp_sec: float
    clock_mode: str


class LiveClock:
    """Elapsed monotonic wall clock for successfully decoded live frames."""

    def __init__(self, *, perf_counter: Callable[[], float] = time.perf_counter):
        self._perf_counter = perf_counter
        self._origin: float | None = None
        self._previous_reading: float | None = None

    @property
    def clock_mode(self) -> str:
        return LIVE_CLOCK_MODE

    def next_timestamp(self) -> float:
        reading = float(self._perf_counter())
        if not math.isfinite(reading):
            raise ValueError(f"live perf_counter reading must be finite, got {reading!r}")
        if self._previous_reading is not None and reading < self._previous_reading:
            raise ValueError(
                "live perf_counter reading regressed "
                f"from {self._previous_reading!r} to {reading!r}"
            )
        if self._origin is None:
            self._origin = reading
        self._previous_reading = reading
        return reading - self._origin


class VideoClock:
    """Preserve usable container times or select one sticky nominal-FPS fallback."""

    def __init__(self, *, nominal_fps: float):
        nominal_fps = float(nominal_fps)
        if not math.isfinite(nominal_fps) or nominal_fps <= 0.0:
            raise ValueError(
                f"video fallback FPS must be finite positive, got {nominal_fps!r}"
            )
        self.nominal_fps = nominal_fps
        self._clock_mode = VIDEO_PENDING_MODE
        self._source_origin_sec: float | None = None
        self._previous_source_sec: float | None = None
        self._previous_timestamp_sec: float | None = None
        self._previous_frame_index: int | None = None
        self._fallback_anchor_sec: float | None = None
        self._fallback_anchor_frame_index: int | None = None

    @property
    def clock_mode(self) -> str:
        return self._clock_mode

    def finalize(self) -> None:
        """Resolve an undecidable one-frame/stalled stream when EOF is observed."""
        if self._clock_mode == VIDEO_PENDING_MODE:
            self._clock_mode = VIDEO_FALLBACK_MODE

    def next_timestamp(self, *, pos_msec: float, frame_index: int) -> float:
        frame_index = int(frame_index)
        if frame_index < 1:
            raise ValueError(f"video frame_index must be positive, got {frame_index!r}")
        if (
            self._previous_frame_index is not None
            and frame_index <= self._previous_frame_index
        ):
            raise ValueError(
                "video frame_index must increase monotonically, "
                f"got {frame_index!r} after {self._previous_frame_index!r}"
            )

        if self._clock_mode == VIDEO_FALLBACK_MODE:
            timestamp_sec = self._fallback_timestamp(frame_index)
        else:
            timestamp_sec = self._source_or_fallback_timestamp(pos_msec, frame_index)

        if not math.isfinite(timestamp_sec):
            raise ValueError(f"video timestamp must be finite, got {timestamp_sec!r}")
        if (
            self._previous_timestamp_sec is not None
            and timestamp_sec < self._previous_timestamp_sec
        ):
            raise ValueError(
                "video timestamp regressed "
                f"from {self._previous_timestamp_sec!r} to {timestamp_sec!r}"
            )
        self._previous_timestamp_sec = timestamp_sec
        self._previous_frame_index = frame_index
        return timestamp_sec

    def _source_or_fallback_timestamp(
        self,
        pos_msec: float,
        frame_index: int,
    ) -> float:
        try:
            source_sec = float(pos_msec) / 1000.0
        except (TypeError, ValueError):
            return self._select_fallback(frame_index)
        if not math.isfinite(source_sec) or source_sec < 0.0:
            return self._select_fallback(frame_index)

        if self._source_origin_sec is None:
            self._source_origin_sec = source_sec
            self._previous_source_sec = source_sec
            return 0.0

        assert self._previous_source_sec is not None
        if self._clock_mode == VIDEO_PENDING_MODE:
            if source_sec <= self._previous_source_sec:
                return self._select_fallback(frame_index)
            self._clock_mode = VIDEO_SOURCE_MODE
        elif source_sec <= self._previous_source_sec:
            return self._select_fallback(frame_index)

        self._previous_source_sec = source_sec
        return source_sec - self._source_origin_sec

    def _select_fallback(self, frame_index: int) -> float:
        timestamp_sec = self._fallback_timestamp(frame_index)
        if (
            self._previous_timestamp_sec is not None
            and timestamp_sec <= self._previous_timestamp_sec
        ):
            self._fallback_anchor_sec = (
                self._previous_timestamp_sec + 1.0 / self.nominal_fps
            )
            self._fallback_anchor_frame_index = frame_index
            timestamp_sec = self._fallback_anchor_sec
        self._clock_mode = VIDEO_FALLBACK_MODE
        return timestamp_sec

    def _fallback_timestamp(self, frame_index: int) -> float:
        if self._fallback_anchor_sec is not None:
            assert self._fallback_anchor_frame_index is not None
            return self._fallback_anchor_sec + (
                frame_index - self._fallback_anchor_frame_index
            ) / self.nominal_fps
        return (frame_index - 1) / self.nominal_fps
