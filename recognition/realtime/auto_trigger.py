from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np


SEGMENT_STATE_IDLE = "IDLE_BLANK"
SEGMENT_STATE_ACTIVE = "SIGNING_ACTIVE"
SEGMENT_STATE_END_CONFIRM = "END_CONFIRM"
SEGMENT_STATE_COOLDOWN = "FORCED_FINALIZE_COOLDOWN"

POSE_SIZE = 33 * 3
HAND_SIZE = 21 * 3
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24


@dataclass(frozen=True)
class AutoTriggerConfig:
    start_motion_threshold: float = 0.040
    blank_motion_threshold: float = 0.018
    start_hold_sec: float = 0.20
    end_hold_sec: float = 0.50
    end_rest_vote_ratio: float = 0.80
    pre_roll_sec: float = 0.20
    max_segment_sec: float = 9.60
    min_segment_sec: float = 0.80
    cooldown_sec: float = 0.67
    torso_motion_weight: float = 0.35
    hidden_rest_enabled: bool = False
    side_margin_ratio: float = 0.18
    chest_drop_ratio: float = 0.28
    pose_visibility_threshold: float = 0.35

    def __post_init__(self) -> None:
        if self.start_motion_threshold < 0 or self.blank_motion_threshold < 0:
            raise ValueError("Motion thresholds must be non-negative.")
        if self.start_hold_sec < 0 or self.pre_roll_sec < 0:
            raise ValueError("Start hold and pre-roll must be non-negative.")
        if self.end_hold_sec <= 0:
            raise ValueError("End hold must be positive.")
        if not 0 < self.end_rest_vote_ratio <= 1:
            raise ValueError("End rest vote ratio must be in (0, 1].")
        if self.max_segment_sec <= 0 or self.min_segment_sec < 0:
            raise ValueError("Segment durations are invalid.")
        if self.max_segment_sec < self.min_segment_sec:
            raise ValueError("Maximum segment duration must be at least the minimum duration.")
        if self.cooldown_sec < 0:
            raise ValueError("Cooldown must be non-negative.")

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class AutoFrameAnalysis:
    visible_rest_blank: bool
    hidden_rest_blank: bool
    torso_motion_score: float
    hand_motion_score: float
    effective_motion_score: float
    hands_at_sides: bool
    wrists_detected: bool
    torso_valid: bool
    explicit_hands_detected: int
    wrist_source_left: str
    wrist_source_right: str

    @property
    def is_blank(self) -> bool:
        return bool(self.visible_rest_blank or self.hidden_rest_blank)


@dataclass(frozen=True)
class FrameSample:
    timestamp_sec: float
    frame_vector: np.ndarray


@dataclass(frozen=True)
class SegmentResult:
    samples: list[FrameSample]
    clip_start_sec: float
    clip_end_sec: float
    finalize_sec: float
    reason: str

    @property
    def frame_vectors(self) -> list[np.ndarray]:
        return [sample.frame_vector.copy() for sample in self.samples]

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.clip_end_sec - self.clip_start_sec)


def load_auto_trigger_config(
    path: str | Path | None = None,
    overrides: dict[str, object | None] | None = None,
) -> AutoTriggerConfig:
    values: dict[str, object] = {}
    allowed = {field.name for field in fields(AutoTriggerConfig)}
    if path:
        config_path = Path(path)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Auto-trigger config JSON must contain an object.")
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown auto-trigger config keys: {', '.join(sorted(unknown))}")
        values.update(payload)
    for key, value in (overrides or {}).items():
        if key not in allowed:
            raise ValueError(f"Unknown auto-trigger override: {key}")
        if value is not None:
            values[key] = value
    return AutoTriggerConfig(**values)


def decompose_frame_vector(frame_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vector = np.asarray(frame_vector, dtype=np.float32)
    expected_size = POSE_SIZE + HAND_SIZE * 2
    if vector.size != expected_size:
        raise ValueError(f"Expected frame vector with {expected_size} values, got {vector.size}.")
    pose = vector[:POSE_SIZE].reshape(33, 3)
    left = vector[POSE_SIZE:POSE_SIZE + HAND_SIZE].reshape(21, 3)
    right = vector[POSE_SIZE + HAND_SIZE:].reshape(21, 3)
    return pose, left, right


def _valid_point(points: np.ndarray, index: int) -> np.ndarray | None:
    point = points[index]
    if np.allclose(point, 0.0):
        return None
    return point


def _masked_motion(previous_points: np.ndarray, current_points: np.ndarray) -> float:
    valid_mask = np.any(previous_points != 0, axis=1) & np.any(current_points != 0, axis=1)
    if not np.any(valid_mask):
        return 0.0
    diff = current_points[valid_mask] - previous_points[valid_mask]
    return float(np.sqrt(np.mean(np.square(diff), dtype=np.float32)))


def analyze_frame_vector(
    previous_frame_vector: np.ndarray | None,
    current_frame_vector: np.ndarray,
    config: AutoTriggerConfig,
) -> AutoFrameAnalysis:
    current_pose, current_left, current_right = decompose_frame_vector(current_frame_vector)
    left_hand_visible = bool(np.any(current_left != 0))
    right_hand_visible = bool(np.any(current_right != 0))
    explicit_hands_detected = int(left_hand_visible) + int(right_hand_visible)

    left_wrist = _valid_point(current_left, 0) if left_hand_visible else None
    right_wrist = _valid_point(current_right, 0) if right_hand_visible else None
    left_source = "hand" if left_wrist is not None else "none"
    right_source = "hand" if right_wrist is not None else "none"
    if left_wrist is None:
        left_wrist = _valid_point(current_pose, LEFT_WRIST)
        if left_wrist is not None:
            left_source = "pose"
    if right_wrist is None:
        right_wrist = _valid_point(current_pose, RIGHT_WRIST)
        if right_wrist is not None:
            right_source = "pose"

    left_shoulder = _valid_point(current_pose, LEFT_SHOULDER)
    right_shoulder = _valid_point(current_pose, RIGHT_SHOULDER)
    left_hip = _valid_point(current_pose, LEFT_HIP)
    right_hip = _valid_point(current_pose, RIGHT_HIP)
    torso_valid = all(point is not None for point in (left_shoulder, right_shoulder, left_hip, right_hip))
    wrists_detected = left_wrist is not None and right_wrist is not None

    torso_motion = 0.0
    hand_motion = 0.0
    if previous_frame_vector is not None:
        previous_pose, previous_left, previous_right = decompose_frame_vector(previous_frame_vector)
        torso_indices = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_HIP, RIGHT_HIP]
        torso_motion = _masked_motion(previous_pose[torso_indices], current_pose[torso_indices])
        left_motion = _masked_motion(previous_left, current_left)
        right_motion = _masked_motion(previous_right, current_right)
        pose_arm_indices = [LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]
        pose_arm_motion = _masked_motion(
            previous_pose[pose_arm_indices],
            current_pose[pose_arm_indices],
        )
        hand_motion = max(left_motion, right_motion, pose_arm_motion)
    effective_motion = max(hand_motion, config.torso_motion_weight * torso_motion)

    hands_at_sides = False
    if torso_valid and wrists_detected:
        assert left_shoulder is not None and right_shoulder is not None
        assert left_hip is not None and right_hip is not None
        assert left_wrist is not None and right_wrist is not None
        torso_center_x = float((left_shoulder[0] + right_shoulder[0]) / 2.0)
        shoulder_width = max(float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])), 1e-3)
        shoulder_y = float((left_shoulder[1] + right_shoulder[1]) / 2.0)
        hip_y = float((left_hip[1] + right_hip[1]) / 2.0)
        torso_height = max(hip_y - shoulder_y, 1e-3)
        chest_y = shoulder_y + config.chest_drop_ratio * torso_height
        side_margin = config.side_margin_ratio * shoulder_width
        image_left_x = min(float(left_wrist[0]), float(right_wrist[0]))
        image_right_x = max(float(left_wrist[0]), float(right_wrist[0]))
        wrist_floor_y = max(chest_y, hip_y)
        both_wrists_low = left_wrist[1] >= wrist_floor_y and right_wrist[1] >= wrist_floor_y
        spans_both_sides = (
            image_left_x <= torso_center_x - side_margin
            and image_right_x >= torso_center_x + side_margin
        )
        hands_at_sides = bool(both_wrists_low and spans_both_sides)

    visible_rest_blank = bool(
        explicit_hands_detected == 2
        and hands_at_sides
        and effective_motion <= config.blank_motion_threshold
    )
    hidden_rest_blank = bool(
        config.hidden_rest_enabled
        and torso_valid
        and explicit_hands_detected == 0
        and torso_motion <= config.blank_motion_threshold
    )
    return AutoFrameAnalysis(
        visible_rest_blank=visible_rest_blank,
        hidden_rest_blank=hidden_rest_blank,
        torso_motion_score=torso_motion,
        hand_motion_score=hand_motion,
        effective_motion_score=effective_motion,
        hands_at_sides=hands_at_sides,
        wrists_detected=wrists_detected,
        torso_valid=torso_valid,
        explicit_hands_detected=explicit_hands_detected,
        wrist_source_left=left_source,
        wrist_source_right=right_source,
    )


class AutoTriggerEngine:
    def __init__(self, config: AutoTriggerConfig):
        self.config = config
        self.state = SEGMENT_STATE_IDLE
        self.clip_start_sec: float | None = None
        self.segment_samples: list[FrameSample] = []
        self._pre_roll: deque[FrameSample] = deque()
        self._active_start_sec: float | None = None
        self._end_votes: deque[tuple[float, bool]] = deque()
        self._cooldown_until_sec = 0.0
        self._last_timestamp_sec: float | None = None

    def reset(self) -> None:
        self.state = SEGMENT_STATE_IDLE
        self.clip_start_sec = None
        self.segment_samples = []
        self._pre_roll.clear()
        self._active_start_sec = None
        self._end_votes.clear()
        self._cooldown_until_sec = 0.0
        self._last_timestamp_sec = None

    def update(
        self,
        frame_vector: np.ndarray,
        analysis: AutoFrameAnalysis,
        timestamp_sec: float,
    ) -> SegmentResult | None:
        timestamp_sec = float(timestamp_sec)
        if self._last_timestamp_sec is not None and timestamp_sec < self._last_timestamp_sec:
            raise ValueError("Frame timestamps must be monotonic.")
        self._last_timestamp_sec = timestamp_sec
        sample = FrameSample(timestamp_sec, np.asarray(frame_vector, dtype=np.float32).copy())

        if self.state == SEGMENT_STATE_COOLDOWN:
            if timestamp_sec < self._cooldown_until_sec:
                return None
            self.state = SEGMENT_STATE_IDLE
            self.clip_start_sec = None
            self._pre_roll.clear()

        if self.state == SEGMENT_STATE_IDLE:
            return self._update_idle(sample, analysis)

        self.segment_samples.append(sample)
        if self.clip_start_sec is not None and timestamp_sec - self.clip_start_sec >= self.config.max_segment_sec:
            return self._finalize(timestamp_sec, timestamp_sec, "timeout_finalize")

        rest_candidate = self._is_rest_candidate(analysis)
        if self.state == SEGMENT_STATE_ACTIVE:
            if rest_candidate:
                self.state = SEGMENT_STATE_END_CONFIRM
                self._end_votes = deque([(timestamp_sec, True)])
            return None

        self._end_votes.append((timestamp_sec, rest_candidate))
        cutoff = timestamp_sec - self.config.end_hold_sec
        while self._end_votes and self._end_votes[0][0] < cutoff - 1e-9:
            self._end_votes.popleft()

        if not any(vote for _, vote in self._end_votes):
            self.state = SEGMENT_STATE_ACTIVE
            self._end_votes.clear()
            return None

        window_elapsed = timestamp_sec - self._end_votes[0][0]
        vote_ratio = sum(1 for _, vote in self._end_votes if vote) / len(self._end_votes)
        if window_elapsed + 1e-9 >= self.config.end_hold_sec and vote_ratio >= self.config.end_rest_vote_ratio:
            clip_end_sec = next(time_sec for time_sec, vote in self._end_votes if vote)
            reason = "hidden_rest_finalize" if analysis.hidden_rest_blank and not analysis.visible_rest_blank else "visible_rest_finalize"
            return self._finalize(clip_end_sec, timestamp_sec, reason)
        return None

    def _update_idle(self, sample: FrameSample, analysis: AutoFrameAnalysis) -> None:
        self._pre_roll.append(sample)
        history_sec = self.config.pre_roll_sec + self.config.start_hold_sec
        cutoff = sample.timestamp_sec - history_sec
        while self._pre_roll and self._pre_roll[0].timestamp_sec < cutoff - 1e-9:
            self._pre_roll.popleft()

        if self._is_start_candidate(analysis):
            if self._active_start_sec is None:
                self._active_start_sec = sample.timestamp_sec
        else:
            self._active_start_sec = None

        if (
            self._active_start_sec is not None
            and sample.timestamp_sec - self._active_start_sec + 1e-9 >= self.config.start_hold_sec
        ):
            requested_start = self._active_start_sec - self.config.pre_roll_sec
            selected = [item for item in self._pre_roll if item.timestamp_sec >= requested_start - 1e-9]
            if not selected:
                selected = [sample]
            self.segment_samples = [
                FrameSample(item.timestamp_sec, item.frame_vector.copy())
                for item in selected
            ]
            self.clip_start_sec = self.segment_samples[0].timestamp_sec
            self.state = SEGMENT_STATE_ACTIVE
            self._active_start_sec = None
            self._end_votes.clear()
        return None

    def _is_start_candidate(self, analysis: AutoFrameAnalysis) -> bool:
        if self._is_rest_candidate(analysis):
            return False
        if analysis.effective_motion_score < self.config.start_motion_threshold:
            return False
        return True

    def _is_rest_candidate(self, analysis: AutoFrameAnalysis) -> bool:
        if analysis.visible_rest_blank:
            return True
        return bool(self.config.hidden_rest_enabled and analysis.hidden_rest_blank)

    def _finalize(
        self,
        clip_end_sec: float,
        finalize_sec: float,
        reason: str,
    ) -> SegmentResult:
        assert self.clip_start_sec is not None
        selected = [
            FrameSample(sample.timestamp_sec, sample.frame_vector.copy())
            for sample in self.segment_samples
            if sample.timestamp_sec < clip_end_sec - 1e-9
        ]
        if not selected and self.segment_samples:
            selected = [
                FrameSample(
                    self.segment_samples[0].timestamp_sec,
                    self.segment_samples[0].frame_vector.copy(),
                )
            ]
        result = SegmentResult(
            samples=selected,
            clip_start_sec=self.clip_start_sec,
            clip_end_sec=float(clip_end_sec),
            finalize_sec=float(finalize_sec),
            reason=reason,
        )
        self.state = SEGMENT_STATE_COOLDOWN
        self._cooldown_until_sec = finalize_sec + self.config.cooldown_sec
        self.clip_start_sec = None
        self.segment_samples = []
        self._pre_roll.clear()
        self._active_start_sec = None
        self._end_votes.clear()
        return result
