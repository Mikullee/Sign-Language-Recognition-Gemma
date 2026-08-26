"""Declared image orientation and anatomical handedness contracts for Knee42."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Literal


RotationSetting = Literal["auto", 0, 90, 180, 270]
ResolvedRotation = Literal[0, 90, 180, 270]
AnatomicalHandSlot = Literal["left", "right"]
_RIGHT_ANGLE_ROTATIONS = frozenset({0, 90, 180, 270})


class MirrorMode(str, Enum):
    """Strict command/config spelling for a declared mirror switch."""

    OFF = "off"
    ON = "on"

    @classmethod
    def parse(cls, value: object) -> "MirrorMode":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError(f"mirror mode must be 'off' or 'on', got {value!r}")
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"mirror mode must be 'off' or 'on', got {value!r}") from exc

    @property
    def enabled(self) -> bool:
        return self is MirrorMode.ON


def parse_rotation(value: object) -> RotationSetting:
    """Parse only the public ``auto|0|90|180|270`` rotation spellings."""
    if value == "auto" and isinstance(value, str):
        return "auto"
    if isinstance(value, str) and value in {"0", "90", "180", "270"}:
        return int(value)  # type: ignore[return-value]
    if type(value) is int and value in _RIGHT_ANGLE_ROTATIONS:
        return value  # type: ignore[return-value]
    raise ValueError(
        "rotation must be 'auto' or one of 0, 90, 180, 270; "
        f"got {value!r}"
    )


def _validate_declared_rotation(value: object) -> RotationSetting:
    if value == "auto" and isinstance(value, str):
        return "auto"
    if type(value) is int and value in _RIGHT_ANGLE_ROTATIONS:
        return value  # type: ignore[return-value]
    raise ValueError(
        "rotation must be 'auto' or an integer 0, 90, 180, or 270; "
        f"got {value!r}"
    )


def _metadata_rotation(value: object) -> ResolvedRotation:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"video rotation metadata must be a right angle, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric not in _RIGHT_ANGLE_ROTATIONS:
        raise ValueError(f"video rotation metadata must be a right angle, got {value!r}")
    return int(numeric)  # type: ignore[return-value]


@dataclass(frozen=True)
class InputOrientation:
    """Immutable declared rotation plus independent model/display mirror switches."""

    rotation: RotationSetting = "auto"
    input_mirror: bool = False
    display_mirror: bool = False

    def __post_init__(self) -> None:
        _validate_declared_rotation(self.rotation)
        if type(self.input_mirror) is not bool:
            raise TypeError(f"input_mirror must be bool, got {self.input_mirror!r}")
        if type(self.display_mirror) is not bool:
            raise TypeError(f"display_mirror must be bool, got {self.display_mirror!r}")

    @property
    def description(self) -> dict[str, str | int | bool]:
        return {
            "rotation": self.rotation,
            "input_mirror": self.input_mirror,
            "display_mirror": self.display_mirror,
        }

    def describe(self) -> dict[str, str | int | bool]:
        """Return a JSON-ready copy of the declared orientation."""
        return self.description


def resolve_rotation(
    rotation: RotationSetting,
    *,
    source_kind: Literal["camera", "video"],
    metadata_rotation: object = 0,
) -> ResolvedRotation:
    """Resolve camera/video auto policy without silently normalizing bad values."""
    declared = _validate_declared_rotation(rotation)
    if source_kind not in {"camera", "video"}:
        raise ValueError(f"source_kind must be 'camera' or 'video', got {source_kind!r}")
    if declared != "auto":
        return declared
    if source_kind == "camera":
        return 0
    return _metadata_rotation(metadata_rotation)


def anatomical_hand_slot(
    mediapipe_label: object,
    *,
    pixels_mirrored: bool,
) -> AnatomicalHandSlot:
    """Map MediaPipe's mirror-relative label to one anatomical model slot."""
    if type(pixels_mirrored) is not bool:
        raise TypeError(f"pixels_mirrored must be bool, got {pixels_mirrored!r}")
    if mediapipe_label not in {"Left", "Right"} or not isinstance(mediapipe_label, str):
        raise ValueError(
            "MediaPipe handedness must be exactly 'Left' or 'Right'; "
            f"got {mediapipe_label!r}"
        )
    if pixels_mirrored:
        return mediapipe_label.lower()  # type: ignore[return-value]
    return "right" if mediapipe_label == "Left" else "left"


# This is an input interpretation rule, not data augmentation. If horizontal
# mirror augmentation is ever introduced, it must also swap anatomical hand
# slots and every MediaPipe pose left/right pair. No augmentation is enabled here.
