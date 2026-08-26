"""RGB-only OpenCV camera and video sources for the Knee42 Windows tester."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from recognition.realtime.knee42_clock import (
    MAX_PRACTICAL_FPS,
    FramePacket,
    LiveClock,
    VideoClock,
)


DEFAULT_CAMERA_FPS = 30.0


def apply_video_transform(
    frame: np.ndarray,
    rotation_degrees: float,
    *,
    horizontal_mirror: bool = False,
) -> np.ndarray:
    """Apply explicit container rotation, then an independently approved mirror."""
    degrees = int(round(float(rotation_degrees))) % 360
    if degrees not in {0, 90, 180, 270}:
        raise ValueError(f"video rotation must be a right angle, got {rotation_degrees!r}")
    if degrees == 0 and not horizontal_mirror:
        return frame
    turns_counterclockwise = {0: 0, 90: 3, 180: 2, 270: 1}[degrees]
    output = np.rot90(np.asarray(frame), k=turns_counterclockwise, axes=(0, 1))
    if horizontal_mirror:
        output = np.flip(output, axis=1)
    return np.ascontiguousarray(output)


class OpenCVFrameSource:
    """Small shared interface over an OpenCV color-frame capture."""

    def __init__(
        self,
        capture: Any,
        status: str,
        cv2_module: Any,
        *,
        clock: LiveClock | VideoClock,
        rotation_degrees: float = 0.0,
        horizontal_mirror: bool = False,
    ):
        self._capture = capture
        self._status = status
        self._cv2 = cv2_module
        self._clock = clock
        self._frame_index = 0
        self._rotation_degrees = rotation_degrees
        self._horizontal_mirror = horizontal_mirror

    @property
    def status(self) -> str:
        return self._status

    @property
    def rotation_degrees(self) -> float:
        return float(self._rotation_degrees)

    @property
    def horizontal_mirror(self) -> bool:
        return bool(self._horizontal_mirror)

    @property
    def fps(self) -> float:
        try:
            value = float(self._capture.get(self._cv2.CAP_PROP_FPS))
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value) and 0.0 < value <= MAX_PRACTICAL_FPS:
            return value
        if isinstance(self._clock, LiveClock):
            return DEFAULT_CAMERA_FPS
        return self._clock.nominal_fps

    @property
    def clock_mode(self) -> str:
        return self._clock.clock_mode

    def read_packet(self) -> FramePacket | None:
        ok, frame = self._capture.read()
        if not ok:
            if isinstance(self._clock, VideoClock):
                self._clock.finalize()
            return None
        self._frame_index += 1
        if isinstance(self._clock, VideoClock):
            pos_msec_property = getattr(self._cv2, "CAP_PROP_POS_MSEC", None)
            pos_msec = (
                float("nan")
                if pos_msec_property is None
                else self._capture.get(pos_msec_property)
            )
            timestamp_sec = self._clock.next_timestamp(
                pos_msec=pos_msec,
                frame_index=self._frame_index,
            )
        else:
            timestamp_sec = self._clock.next_timestamp()
        transformed = apply_video_transform(
            frame,
            self._rotation_degrees,
            horizontal_mirror=self._horizontal_mirror,
        )
        return FramePacket(
            frame=transformed,
            timestamp_sec=timestamp_sec,
            clock_mode=self._clock.clock_mode,
        )

    def read(self):
        """Return the legacy ``(ok, frame)`` pair while advancing the source clock."""
        packet = self.read_packet()
        return (False, None) if packet is None else (True, packet.frame)

    def release(self) -> None:
        self._capture.release()


def _load_cv2(cv2_module: Any | None):
    if cv2_module is not None:
        return cv2_module
    import cv2

    return cv2


def _camera_capture(cv2_module: Any, index: int, platform_name: str):
    if platform_name.startswith("win"):
        return cv2_module.VideoCapture(index, cv2_module.CAP_DSHOW)
    return cv2_module.VideoCapture(index)


def open_camera(
    camera_index: int | None,
    *,
    max_index: int = 9,
    cv2_module: Any | None = None,
    platform_name: str | None = None,
    perf_counter: Callable[[], float] | None = None,
) -> OpenCVFrameSource:
    """Open an explicit camera or probe indices 0..max_index without leaking handles."""
    cv2_module = _load_cv2(cv2_module)
    platform_name = platform_name or sys.platform
    if max_index < 0:
        raise ValueError("max_index must be non-negative")
    if camera_index is not None and camera_index < 0:
        raise ValueError("camera_index must be non-negative")
    indices = [camera_index] if camera_index is not None else list(range(max_index + 1))
    for index in indices:
        capture = _camera_capture(cv2_module, int(index), platform_name)
        if capture.isOpened():
            clock = LiveClock() if perf_counter is None else LiveClock(perf_counter=perf_counter)
            return OpenCVFrameSource(
                capture,
                f"camera:{index}",
                cv2_module,
                clock=clock,
            )
        capture.release()
    if camera_index is not None:
        raise RuntimeError(f"camera index {camera_index} is unavailable")
    raise RuntimeError(f"no RGB camera found at camera indices 0..{max_index}")


def open_video(
    path: Path,
    *,
    cv2_module: Any | None = None,
) -> OpenCVFrameSource:
    """Open a color video through the same read/release/fps interface as a camera."""
    cv2_module = _load_cv2(cv2_module)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2_module.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open color video: {path}")
    try:
        clock = VideoClock(nominal_fps=float(capture.get(cv2_module.CAP_PROP_FPS)))
    except (TypeError, ValueError):
        capture.release()
        raise
    orientation_meta = getattr(cv2_module, "CAP_PROP_ORIENTATION_META", None)
    rotation_degrees = float(capture.get(orientation_meta)) if orientation_meta is not None else 0.0
    orientation_auto = getattr(cv2_module, "CAP_PROP_ORIENTATION_AUTO", None)
    if orientation_auto is not None and hasattr(capture, "set"):
        capture.set(orientation_auto, 0)
    return OpenCVFrameSource(
        capture,
        f"video:{path}",
        cv2_module,
        clock=clock,
        rotation_degrees=rotation_degrees,
    )
