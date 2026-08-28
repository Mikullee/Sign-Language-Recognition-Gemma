"""Offline video -> 219-value sequences for the Transformer path.

The realtime path already turns MediaPipe results into the 219-value contract
via ``recognition.realtime.knee42_preprocessing``.  This module covers the
offline case, where frames come from a video file rather than a camera loop, and
adds the two things a file needs but a live stream does not: reading frames with
timestamps, and deciding whether the footage is mirrored.

MediaPipe Tasks is driven directly here, in IMAGE mode, with frames fed
**un-flipped** and the reported handedness taken as-is.  That is exactly how the
training feature cache was built (``horizontal_mirror: False``; see
``scripts/prepare_knee42_features_final.py``), and matching it matters more than
matching any other inference stack.

Note on mirror detection: the pose landmarker assigns anatomical left/right from
body appearance, so it re-labels the sides when it sees a mirrored person.  The
normalized left-shoulder x therefore reads about +0.5 whether or not the footage
is mirrored, and **cannot** be used to detect mirroring here.  It remains useful
as a sanity check on the normalization itself.  Footage that genuinely needs
un-mirroring must say so via the extractor's ``selfie_flip`` flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from recognition.realtime.knee42_preprocessing import (
    HAND_LANDMARKS,
    LANDMARK_DIM,
    POSE_KEEP,
    normalize_frame,
)


POSE_LANDMARKS = 33
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

#: Column of the signer's left shoulder x inside a normalized 219-value frame.
LEFT_SHOULDER_X = POSE_KEEP.index(LEFT_SHOULDER) * 3


@dataclass
class TrackedFrame:
    """One video frame: capture time plus whatever MediaPipe found in it."""

    index: int
    timestamp: float
    pose: np.ndarray | None = None
    hands: dict[str, np.ndarray] = field(default_factory=dict)

    def wrist(self, side: str) -> np.ndarray | None:
        landmarks = self.hands.get(side)
        return None if landmarks is None else landmarks[0]

    @property
    def has_hands(self) -> bool:
        return bool(self.hands)


def _landmarks_to_array(landmarks: Iterable[Any], expected: int) -> np.ndarray:
    points = np.asarray([[point.x, point.y, point.z] for point in landmarks], dtype=np.float32)
    if points.shape != (expected, 3):
        raise ValueError(f"expected {expected} landmarks, got {points.shape}")
    return points


class MediaPipeLandmarkExtractor:
    """Run the MediaPipe hand and pose landmarkers over a video file.

    ``hand_model`` and ``pose_model`` should point at the same ``.task`` files the
    feature cache was extracted with -- notably ``pose_landmarker_lite``, not
    ``full``.  Their SHA-256 values are recorded in every cached ``.npz``.

    ``selfie_flip`` un-mirrors the frame *before* detection.  Correcting the
    coordinates afterwards is not equivalent: the landmarker would still have
    seen a mirrored person, which is not the orientation it was calibrated on
    here.  Use ``probe_selfie_flip`` to decide the flag from the footage.
    """

    def __init__(
        self,
        hand_model: Path | str,
        pose_model: Path | str,
        *,
        max_hands: int = 2,
        resize_width: int = 0,
        selfie_flip: bool = False,
    ) -> None:
        from mediapipe import Image, ImageFormat
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )

        self._Image = Image
        self._ImageFormat = ImageFormat
        self.resize_width = int(resize_width)
        self.selfie_flip = bool(selfie_flip)

        for name, path in (("hand", hand_model), ("pose", pose_model)):
            if not Path(path).is_file():
                raise FileNotFoundError(f"{name} landmarker model not found: {path}")

        # IMAGE mode, not VIDEO: the feature cache, the data validator and the
        # realtime path all detect frame-by-frame with no temporal tracking, so
        # anything else would put inference on a different distribution than the
        # one the model was trained on. See scripts/prepare_knee42_features_final.py.
        self._hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(hand_model)),
                running_mode=VisionTaskRunningMode.IMAGE,
                num_hands=max_hands,
            )
        )
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(pose_model)),
                running_mode=VisionTaskRunningMode.IMAGE,
            )
        )

    def close(self) -> None:
        try:
            self._hand.close()
        finally:
            self._pose.close()

    def __enter__(self) -> "MediaPipeLandmarkExtractor":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def run_video(
        self,
        video_path: Path | str,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[TrackedFrame]:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps:
            fps = 30.0
        total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        report_every = max(1, round(fps))

        frames: list[TrackedFrame] = []
        try:
            while True:
                ok, raw = capture.read()
                if not ok:
                    break
                image = cv2.flip(raw, 1) if self.selfie_flip else raw
                if self.resize_width and image.shape[1] > self.resize_width:
                    height = round(image.shape[0] * self.resize_width / image.shape[1])
                    image = cv2.resize(
                        image, (self.resize_width, height), interpolation=cv2.INTER_AREA
                    )
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frames.append(self._track(rgb, len(frames), len(frames) / fps))
                if progress and len(frames) % report_every == 0:
                    progress(len(frames), total)
        finally:
            capture.release()
        return frames

    def _track(self, rgb: np.ndarray, index: int, timestamp: float) -> TrackedFrame:
        image = self._Image(image_format=self._ImageFormat.SRGB, data=rgb)
        hand_result = self._hand.detect(image)
        pose_result = self._pose.detect(image)

        pose = None
        if getattr(pose_result, "pose_landmarks", None):
            pose = _landmarks_to_array(pose_result.pose_landmarks[0], POSE_LANDMARKS)

        hands: dict[str, np.ndarray] = {}
        handedness_groups = getattr(hand_result, "handedness", []) or []
        landmark_groups = getattr(hand_result, "hand_landmarks", []) or []
        for handedness, landmarks in zip(handedness_groups, landmark_groups):
            if not handedness or len(landmarks) != HAND_LANDMARKS:
                continue
            side = str(handedness[0].category_name).strip().capitalize()
            if side in ("Left", "Right"):
                hands[side] = _landmarks_to_array(landmarks, HAND_LANDMARKS)

        return TrackedFrame(index=index, timestamp=timestamp, pose=pose, hands=hands)


def frame_to_219(frame: TrackedFrame, *, mirrored: bool) -> np.ndarray | None:
    """One tracked frame -> ``(219,)`` shoulder-normalized values, or ``None``.

    ``None`` means both shoulders were not visible, so the frame cannot be
    normalized and must be dropped rather than guessed at.  Coordinates that were
    simply not detected stay NaN; the sequence stage interpolates them.
    """
    pose = np.full((POSE_LANDMARKS, 3), np.nan, dtype=np.float32)
    if frame.pose is not None:
        pose = frame.pose.astype(np.float32).copy()
        if mirrored:
            pose[:, 0] = 1.0 - pose[:, 0]

    if not (np.isfinite(pose[LEFT_SHOULDER]).all() and np.isfinite(pose[RIGHT_SHOULDER]).all()):
        return None

    slots = {
        side: np.full((HAND_LANDMARKS, 3), np.nan, dtype=np.float32)
        for side in ("Left", "Right")
    }
    for side, landmarks in frame.hands.items():
        hand = landmarks.astype(np.float32).copy()
        if mirrored:
            hand[:, 0] = 1.0 - hand[:, 0]
        slots[side] = hand

    values = np.concatenate(
        (
            pose[np.asarray(POSE_KEEP, dtype=np.int64)].reshape(-1),
            slots["Left"].reshape(-1),
            slots["Right"].reshape(-1),
        )
    ).astype(np.float32)
    return normalize_frame(values, np.isfinite(values))


def observation_from_frame(frame: TrackedFrame) -> "FrameObservation":
    """Build the shared trigger/recognition views from already-extracted arrays.

    ``knee42_preprocessing.observation_from_results`` does this from live
    MediaPipe result objects. Landmarks that arrive as plain arrays -- from a
    browser running MediaPipe itself, or from a replayed capture -- need the same
    two views, derived identically: 225 trigger values with missing points zeroed,
    and 219 shoulder-normalized recognition values that keep NaN where a landmark
    was never seen.
    """
    from recognition.realtime.knee42_preprocessing import FrameObservation

    pose = (
        frame.pose.astype(np.float32)
        if frame.pose is not None
        else np.full((POSE_LANDMARKS, 3), np.nan, dtype=np.float32)
    )
    hands = {
        side: frame.hands.get(side, np.full((HAND_LANDMARKS, 3), np.nan, dtype=np.float32))
        for side in ("Left", "Right")
    }

    trigger = np.nan_to_num(
        np.concatenate(
            (pose.reshape(-1), hands["Left"].reshape(-1), hands["Right"].reshape(-1))
        ),
        nan=0.0,
    ).astype(np.float32)

    values = np.concatenate(
        (
            pose[np.asarray(POSE_KEEP, dtype=np.int64)].reshape(-1),
            hands["Left"].reshape(-1),
            hands["Right"].reshape(-1),
        )
    ).astype(np.float32)
    mask = np.isfinite(values)

    return FrameObservation(
        trigger_values=trigger,
        recognition_values=normalize_frame(values, mask),
        recognition_mask=mask,
        display_pose=pose,
        display_left_hand=hands["Left"],
        display_right_hand=hands["Right"],
    )


def frames_to_sequence(
    frames: Iterable[TrackedFrame],
    *,
    mirrored: bool,
    keep_gaps: bool = False,
) -> list[np.ndarray | None]:
    """Convert tracked frames to 219-value vectors, optionally keeping ``None`` gaps."""
    sequence: list[np.ndarray | None] = []
    for frame in frames:
        vector = frame_to_219(frame, mirrored=mirrored)
        if vector is None and not keep_gaps:
            continue
        sequence.append(vector)
    return sequence


def left_shoulder_x_mean(matrix: np.ndarray) -> float:
    """Mean normalized x of the signer's left shoulder; the training value is +0.5."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None]
    if matrix.shape[1] != LANDMARK_DIM:
        raise ValueError(f"expected [frames,{LANDMARK_DIM}] values, got {matrix.shape}")
    return float(np.nanmean(matrix[:, LEFT_SHOULDER_X]))


def convention_check(matrix: np.ndarray) -> tuple[float, bool]:
    """Sanity-check the normalization: the signer's left shoulder should sit at +x.

    Returns ``(measured, ok)``.  A negative value means the 219-value assembly or
    the shoulder normalization is wrong -- it does **not** mean the footage is
    mirrored, which this measurement cannot detect (see the module docstring).
    """
    measured = left_shoulder_x_mean(matrix)
    return measured, bool(measured > 0)
