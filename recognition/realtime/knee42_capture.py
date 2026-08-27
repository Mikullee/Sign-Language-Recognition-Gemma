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
from recognition.realtime.knee42_orientation import (
    InputOrientation,
    RotationSetting,
    resolve_rotation,
)


DEFAULT_CAMERA_FPS = 30.0
_MIRROR_UNSET = object()


def _right_angle_degrees(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"video rotation must be a right angle, got {value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"video rotation must be a right angle, got {value!r}") from exc
    if not math.isfinite(numeric) or numeric not in {0.0, 90.0, 180.0, 270.0}:
        raise ValueError(f"video rotation must be a right angle, got {value!r}")
    return int(numeric)


def _resolve_mirror_aliases(
    *,
    horizontal_mirror: object = _MIRROR_UNSET,
    input_mirror: object = _MIRROR_UNSET,
) -> bool:
    for name, value in (
        ("horizontal_mirror", horizontal_mirror),
        ("input_mirror", input_mirror),
    ):
        if value is not _MIRROR_UNSET and type(value) is not bool:
            raise TypeError(f"{name} must be bool, got {value!r}")
    if horizontal_mirror is not _MIRROR_UNSET and input_mirror is not _MIRROR_UNSET:
        if horizontal_mirror != input_mirror:
            raise ValueError("conflicting horizontal_mirror and input_mirror values")
        return bool(input_mirror)
    if input_mirror is not _MIRROR_UNSET:
        return bool(input_mirror)
    if horizontal_mirror is not _MIRROR_UNSET:
        return bool(horizontal_mirror)
    return False


def apply_video_transform(
    frame: np.ndarray,
    rotation_degrees: float,
    *,
    horizontal_mirror: object = _MIRROR_UNSET,
    input_mirror: object = _MIRROR_UNSET,
) -> np.ndarray:
    """Apply resolved rotation, then an independently declared input mirror.

    ``horizontal_mirror`` remains as the legacy spelling. New source code uses
    ``input_mirror`` so this model-affecting transform cannot be confused with
    display-only mirroring.
    """
    mirror_enabled = _resolve_mirror_aliases(
        horizontal_mirror=horizontal_mirror,
        input_mirror=input_mirror,
    )
    degrees = _right_angle_degrees(rotation_degrees)
    if degrees == 0 and not mirror_enabled:
        return frame
    turns_counterclockwise = {0: 0, 90: 3, 180: 2, 270: 1}[degrees]
    output = np.rot90(np.asarray(frame), k=turns_counterclockwise, axes=(0, 1))
    if mirror_enabled:
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
        horizontal_mirror: object = _MIRROR_UNSET,
        input_mirror: object = _MIRROR_UNSET,
    ):
        self._capture = capture
        self._status = status
        self._cv2 = cv2_module
        self._clock = clock
        self._frame_index = 0
        self._rotation_degrees = _right_angle_degrees(rotation_degrees)
        self._input_mirror = _resolve_mirror_aliases(
            horizontal_mirror=horizontal_mirror,
            input_mirror=input_mirror,
        )

    @property
    def status(self) -> str:
        return self._status

    @property
    def rotation_degrees(self) -> float:
        return float(self._rotation_degrees)

    @property
    def resolved_rotation(self) -> int:
        return int(self._rotation_degrees)

    @property
    def input_mirror(self) -> bool:
        return bool(self._input_mirror)

    @property
    def horizontal_mirror(self) -> bool:
        """Legacy alias for the declared model-input mirror."""
        return self.input_mirror

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
            input_mirror=self._input_mirror,
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


def _disable_and_verify_auto_orientation(capture: Any, cv2_module: Any) -> None:
    property_id = getattr(cv2_module, "CAP_PROP_ORIENTATION_AUTO", None)
    if property_id is None:
        raise RuntimeError(
            "OpenCV automatic video orientation control is unavailable; "
            "refusing ambiguous video orientation"
        )
    setter = getattr(capture, "set", None)
    if not callable(setter):
        raise RuntimeError(
            "OpenCV capture cannot disable automatic video orientation; "
            "refusing ambiguous video orientation"
        )
    try:
        accepted = setter(property_id, 0)
    except Exception as exc:
        raise RuntimeError("failed to disable OpenCV automatic video orientation") from exc
    if not bool(accepted):
        raise RuntimeError("OpenCV refused to disable automatic video orientation")
    try:
        reported = float(capture.get(property_id))
    except Exception as exc:
        raise RuntimeError(
            "cannot verify OpenCV automatic video orientation is disabled"
        ) from exc
    if not math.isfinite(reported) or reported != 0.0:
        raise RuntimeError(
            "OpenCV automatic video orientation remains enabled or unverifiable "
            f"after disable request: {reported!r}"
        )


def open_camera(
    camera_index: int | None,
    *,
    max_index: int = 9,
    cv2_module: Any | None = None,
    platform_name: str | None = None,
    perf_counter: Callable[[], float] | None = None,
    rotation: RotationSetting = "auto",
    input_mirror: bool = False,
) -> OpenCVFrameSource:
    """Open an explicit camera or probe indices 0..max_index without leaking handles."""
    orientation = InputOrientation(
        rotation=rotation,
        input_mirror=input_mirror,
        display_mirror=False,
    )
    resolved_rotation = resolve_rotation(
        orientation.rotation,
        source_kind="camera",
    )
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
                rotation_degrees=resolved_rotation,
                input_mirror=orientation.input_mirror,
            )
        capture.release()
    if camera_index is not None:
        raise RuntimeError(f"camera index {camera_index} is unavailable")
    raise RuntimeError(f"no RGB camera found at camera indices 0..{max_index}")


def open_video(
    path: Path,
    *,
    cv2_module: Any | None = None,
    rotation: RotationSetting = "auto",
    input_mirror: bool = False,
) -> OpenCVFrameSource:
    """Open a color video through the same read/release/fps interface as a camera."""
    orientation = InputOrientation(
        rotation=rotation,
        input_mirror=input_mirror,
        display_mirror=False,
    )
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
        _disable_and_verify_auto_orientation(capture, cv2_module)
        orientation_meta = getattr(cv2_module, "CAP_PROP_ORIENTATION_META", None)
        metadata_rotation = (
            capture.get(orientation_meta)
            if orientation.rotation == "auto" and orientation_meta is not None
            else 0
        )
        rotation_degrees = resolve_rotation(
            orientation.rotation,
            source_kind="video",
            metadata_rotation=metadata_rotation,
        )
        return OpenCVFrameSource(
            capture,
            f"video:{path}",
            cv2_module,
            clock=clock,
            rotation_degrees=rotation_degrees,
            input_mirror=orientation.input_mirror,
        )
    except BaseException:
        capture.release()
        raise
