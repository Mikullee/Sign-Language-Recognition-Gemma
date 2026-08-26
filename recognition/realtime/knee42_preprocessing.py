"""Exact realtime form of the frozen Knee42 feature preprocessing contract."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from recognition.realtime.knee42_orientation import anatomical_hand_slot


POSE_KEEP = tuple(index for index in range(33) if index not in (25, 26))
HAND_LANDMARKS = 21
LANDMARK_DIM = len(POSE_KEEP) * 3 + 2 * HAND_LANDMARKS * 3
MODEL_INPUT_DIM = LANDMARK_DIM * 2


@dataclass(frozen=True)
class FrameObservation:
    """Synchronized trigger and frozen-recognizer views of one MediaPipe result."""

    trigger_values: np.ndarray
    recognition_values: np.ndarray
    recognition_mask: np.ndarray
    display_pose: np.ndarray | None = None
    display_left_hand: np.ndarray | None = None
    display_right_hand: np.ndarray | None = None


def _landmark_array(
    landmarks: Sequence[Any] | None,
    expected_count: int,
    name: str,
) -> np.ndarray:
    if landmarks is None:
        return np.full((expected_count, 3), np.nan, dtype=np.float32)
    if len(landmarks) != expected_count:
        raise ValueError(f"{name} requires {expected_count} landmarks, found {len(landmarks)}")
    return np.asarray(
        [[item.x, item.y, item.z] for item in landmarks],
        dtype=np.float32,
    )


def flatten_landmarks(
    pose_landmarks: Sequence[Any] | None,
    left_hand_landmarks: Sequence[Any] | None,
    right_hand_landmarks: Sequence[Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten pose-without-knees, left hand, then right hand into 219 values."""
    full_pose = _landmark_array(pose_landmarks, 33, "pose")
    pose = full_pose[np.asarray(POSE_KEEP, dtype=np.int64)]
    left = _landmark_array(left_hand_landmarks, HAND_LANDMARKS, "left hand")
    right = _landmark_array(right_hand_landmarks, HAND_LANDMARKS, "right hand")
    values = np.concatenate((pose.reshape(-1), left.reshape(-1), right.reshape(-1))).astype(
        np.float32
    )
    if values.shape != (LANDMARK_DIM,):
        raise AssertionError(f"unexpected Knee42 landmark shape: {values.shape}")
    return values, np.isfinite(values)


def _result_landmark_groups(
    hand_result: Any,
    pose_result: Any,
    *,
    pixels_mirrored: bool,
) -> tuple[Sequence[Any] | None, Sequence[Any] | None, Sequence[Any] | None]:
    if type(pixels_mirrored) is not bool:
        raise TypeError(f"pixels_mirrored must be bool, got {pixels_mirrored!r}")
    pose_groups = getattr(pose_result, "pose_landmarks", []) if pose_result is not None else []
    pose_landmarks = pose_groups[0] if pose_groups else None
    hands: dict[str, Sequence[Any] | None] = {"left": None, "right": None}
    if hand_result is not None:
        handedness_groups = getattr(hand_result, "handedness", [])
        landmark_groups = getattr(hand_result, "hand_landmarks", [])
        if len(handedness_groups) != len(landmark_groups):
            raise ValueError(
                "MediaPipe handedness and hand landmark group counts do not match"
            )
        for handedness, landmarks in zip(handedness_groups, landmark_groups):
            if not handedness:
                raise ValueError("MediaPipe handedness group is empty")
            if len(landmarks) != HAND_LANDMARKS:
                continue
            label = getattr(handedness[0], "category_name", None)
            slot = anatomical_hand_slot(label, pixels_mirrored=pixels_mirrored)
            if hands[slot] is not None:
                raise ValueError(f"ambiguous duplicate anatomical {slot} handedness")
            hands[slot] = landmarks
    return pose_landmarks, hands["left"], hands["right"]


def observation_from_results(
    hand_result: Any,
    pose_result: Any,
    *,
    pixels_mirrored: bool,
) -> FrameObservation:
    """Create trigger/model views using the required pixel-handedness policy."""
    pose, left, right = _result_landmark_groups(
        hand_result,
        pose_result,
        pixels_mirrored=pixels_mirrored,
    )
    full_pose = _landmark_array(pose, 33, "pose")
    full_left = _landmark_array(left, HAND_LANDMARKS, "left hand")
    full_right = _landmark_array(right, HAND_LANDMARKS, "right hand")
    trigger = np.nan_to_num(
        np.concatenate((full_pose.reshape(-1), full_left.reshape(-1), full_right.reshape(-1))),
        nan=0.0,
    ).astype(np.float32)
    values, mask = flatten_landmarks(pose, left, right)
    return FrameObservation(
        trigger_values=trigger,
        recognition_values=values,
        recognition_mask=mask,
        display_pose=full_pose,
        display_left_hand=full_left,
        display_right_hand=full_right,
    )


def landmarks_from_results(
    hand_result: Any,
    pose_result: Any,
    *,
    pixels_mirrored: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Map MediaPipe results to anatomical slots in the frozen feature contract."""
    observation = observation_from_results(
        hand_result,
        pose_result,
        pixels_mirrored=pixels_mirrored,
    )
    return observation.recognition_values, observation.recognition_mask


def normalize_frame(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply the frozen shoulder-relative normalization without filling missing data."""
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.bool_)
    if values.shape != (LANDMARK_DIM,) or mask.shape != values.shape:
        raise ValueError(f"expected ({LANDMARK_DIM},) values/mask, got {values.shape}/{mask.shape}")
    result = values.copy()
    points = result.reshape(-1, 3)
    point_mask = mask.reshape(-1, 3).all(axis=1)
    pose_index = {source: target for target, source in enumerate(POSE_KEEP)}
    left_shoulder = pose_index[11]
    right_shoulder = pose_index[12]
    if point_mask[left_shoulder] and point_mask[right_shoulder]:
        center = (points[left_shoulder] + points[right_shoulder]) / 2.0
        scale = float(
            np.linalg.norm(points[left_shoulder, :2] - points[right_shoulder, :2])
        )
    elif np.any(point_mask):
        valid = points[point_mask]
        center = valid.mean(axis=0)
        scale = float(np.linalg.norm(np.ptp(valid[:, :2], axis=0)))
    else:
        return result
    scale = max(scale, 1e-3)
    points[point_mask] = (points[point_mask] - center) / scale
    return points.reshape(-1).astype(np.float32)


def materialize_sequence(
    values: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    sequence_length: int = 64,
) -> np.ndarray:
    """Sample, train-standardize, neutral-fill, and concatenate the observed mask."""
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.bool_)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != LANDMARK_DIM or values.shape[0] == 0:
        raise ValueError(f"expected non-empty [frames,{LANDMARK_DIM}] values, got {values.shape}")
    if mask.shape != values.shape:
        raise ValueError(f"mask shape {mask.shape} does not match values {values.shape}")
    if mean.shape != (LANDMARK_DIM,) or std.shape != mean.shape or np.any(std <= 0):
        raise ValueError("invalid 219-dimensional train-only standardizer")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if np.any(np.isfinite(values) & ~mask):
        raise ValueError("mask/value mismatch")
    indices = np.rint(np.linspace(0, len(values) - 1, sequence_length)).astype(np.int64)
    sampled_values = values[indices]
    sampled_mask = mask[indices]
    standardized = (sampled_values - mean) / std
    standardized = np.where(sampled_mask, standardized, 0.0).astype(np.float32)
    return np.concatenate((standardized, sampled_mask.astype(np.float32)), axis=1)
