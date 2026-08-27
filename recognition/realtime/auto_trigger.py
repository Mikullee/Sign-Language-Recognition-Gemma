from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, fields
from enum import Enum
from numbers import Real
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
LEFT_KNEE = 25
RIGHT_KNEE = 26
FORMAL_AUTO_TRIGGER_CONFIG_NAME = "auto_trigger_knee_ivcam_local.json"


class CalibrationState(str, Enum):
    DISABLED = "disabled"
    MISSING_REST_SIGNATURE = "missing_rest_signature"
    REQUIRES_TWO_HANDS = "requires_two_hands"
    MOTION_ABOVE_SEED_THRESHOLD = "motion_above_seed_threshold"
    COLLECTING_REFERENCE = "collecting_reference"
    CALIBRATED = "calibrated"


@dataclass(frozen=True)
class CalibrationTelemetry:
    status: CalibrationState
    blocker: CalibrationState | None
    elapsed_sec: float
    sample_count: int

    @property
    def calibrated(self) -> bool:
        return self.status in {CalibrationState.DISABLED, CalibrationState.CALIBRATED}

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "blocker": None if self.blocker is None else self.blocker.value,
            "elapsed_sec": self.elapsed_sec,
            "sample_count": self.sample_count,
            "calibrated": self.calibrated,
        }


@dataclass(frozen=True)
class AutoTriggerConfig:
    start_motion_threshold: float = 0.040
    blank_motion_threshold: float = 0.018
    start_hold_sec: float = 0.20
    end_hold_sec: float = 0.50
    end_rest_vote_ratio: float = 0.80
    end_safety_tail_sec: float = 0.0
    pre_roll_sec: float = 0.20
    max_segment_sec: float = 9.60
    min_segment_sec: float = 0.80
    cooldown_sec: float = 0.67
    torso_motion_weight: float = 0.35
    # A camera that ends around the knees can use the learned shoulder/hand
    # rest signature only, avoiding unstable hip/knee pose estimates.
    knee_geometry_enabled: bool = True
    hidden_rest_enabled: bool = False
    knee_lateral_thigh_margin_ratio: float = 0.55
    # Seated cameras commonly see the resting palm on the upper thigh while
    # PoseLandmarker places the knee below it.  Negative progress is therefore
    # intentionally permitted, but chest-level hands remain outside the band.
    knee_min_thigh_progress_ratio: float = -0.85
    knee_max_thigh_progress_ratio: float = 1.25
    reference_rest_enabled: bool = False
    reference_seed_sec: float = 1.00
    reference_seed_motion_threshold: float = 0.035
    reference_rest_distance_threshold: float = 0.28
    reference_departure_distance_threshold: float = 0.10
    temporal_classifier_enabled: bool = False
    temporal_start_probability_threshold: float = 0.55
    temporal_rest_probability_threshold: float = 0.55
    pose_visibility_threshold: float = 0.35

    def __post_init__(self) -> None:
        numeric_fields = (
            "start_motion_threshold",
            "blank_motion_threshold",
            "start_hold_sec",
            "end_hold_sec",
            "end_rest_vote_ratio",
            "end_safety_tail_sec",
            "pre_roll_sec",
            "max_segment_sec",
            "min_segment_sec",
            "cooldown_sec",
            "torso_motion_weight",
            "knee_lateral_thigh_margin_ratio",
            "knee_min_thigh_progress_ratio",
            "knee_max_thigh_progress_ratio",
            "reference_seed_sec",
            "reference_seed_motion_threshold",
            "reference_rest_distance_threshold",
            "reference_departure_distance_threshold",
            "temporal_start_probability_threshold",
            "temporal_rest_probability_threshold",
            "pose_visibility_threshold",
        )
        for name in numeric_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a finite real number, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        for name in (
            "knee_geometry_enabled",
            "hidden_rest_enabled",
            "reference_rest_enabled",
            "temporal_classifier_enabled",
        ):
            value = getattr(self, name)
            if type(value) is not bool:
                raise TypeError(f"{name} must be bool, got {value!r}")
        if self.start_motion_threshold < 0 or self.blank_motion_threshold < 0:
            raise ValueError("start_motion_threshold and blank_motion_threshold must be non-negative")
        if self.start_hold_sec < 0 or self.pre_roll_sec < 0:
            raise ValueError("start_hold_sec and pre_roll_sec must be non-negative")
        if self.end_hold_sec <= 0:
            raise ValueError("end_hold_sec must be positive")
        if not 0 <= self.end_safety_tail_sec <= self.end_hold_sec:
            raise ValueError("end_safety_tail_sec must be in [0, end_hold_sec]")
        if not 0 < self.end_rest_vote_ratio <= 1:
            raise ValueError("end_rest_vote_ratio must be in (0, 1]")
        if self.max_segment_sec <= 0 or self.min_segment_sec < 0:
            raise ValueError("min_segment_sec must be non-negative and max_segment_sec positive")
        if self.max_segment_sec < self.min_segment_sec:
            raise ValueError("max_segment_sec must be at least min_segment_sec")
        if self.cooldown_sec < 0:
            raise ValueError("cooldown_sec must be non-negative")
        if not 0 <= self.torso_motion_weight <= 1:
            raise ValueError("torso_motion_weight must be in [0, 1]")
        if self.knee_lateral_thigh_margin_ratio <= 0:
            raise ValueError("knee_lateral_thigh_margin_ratio must be positive")
        if self.knee_min_thigh_progress_ratio >= self.knee_max_thigh_progress_ratio:
            raise ValueError(
                "knee_min_thigh_progress_ratio must be below knee_max_thigh_progress_ratio"
            )
        if self.reference_seed_sec < 0 or self.reference_seed_motion_threshold < 0:
            raise ValueError(
                "reference_seed_sec and reference_seed_motion_threshold must be non-negative"
            )
        if (
            self.reference_rest_distance_threshold <= 0
            or self.reference_departure_distance_threshold <= 0
        ):
            raise ValueError(
                "reference_rest_distance_threshold and "
                "reference_departure_distance_threshold must be positive"
            )
        if not 0 < self.temporal_start_probability_threshold < 1:
            raise ValueError("temporal_start_probability_threshold must be in (0, 1)")
        if not 0 < self.temporal_rest_probability_threshold < 1:
            raise ValueError("temporal_rest_probability_threshold must be in (0, 1)")
        if not 0 <= self.pose_visibility_threshold <= 1:
            raise ValueError("pose_visibility_threshold must be in [0, 1]")

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class AutoFrameAnalysis:
    visible_rest_blank: bool
    hidden_rest_blank: bool
    torso_motion_score: float
    hand_motion_score: float
    effective_motion_score: float
    hands_on_knees: bool
    knee_landmarks_valid: bool
    wrists_detected: bool
    torso_valid: bool
    explicit_hands_detected: int
    wrist_source_left: str
    wrist_source_right: str
    rest_signature: tuple[float, ...] | None = None
    wrist_rest_signature: tuple[float, ...] | None = None
    temporal_active_probability: float | None = None

    @property
    def is_blank(self) -> bool:
        return bool(self.visible_rest_blank or self.hidden_rest_blank)

    @property
    def hands_at_sides(self) -> bool:
        """Backward-compatible name for callers predating seated knee triggering."""
        return self.hands_on_knees


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
    rest_detected_sec: float | None = None
    boundary_policy: str = "first_confirmed_rest"

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


def load_formal_auto_trigger_config(release_root: str | Path) -> AutoTriggerConfig:
    """Load the one release-root trigger config with an exact, complete schema."""
    config_path = Path(release_root) / FORMAL_AUTO_TRIGGER_CONFIG_NAME
    if not config_path.is_file():
        raise FileNotFoundError(f"formal auto-trigger config missing: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid formal auto-trigger config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"formal auto-trigger config {config_path} must contain an object")
    required = {field.name for field in fields(AutoTriggerConfig)}
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required)
    if missing:
        raise ValueError(
            f"formal auto-trigger config {config_path} missing fields: {', '.join(missing)}"
        )
    if unknown:
        raise ValueError(
            f"formal auto-trigger config {config_path} has unknown fields: {', '.join(unknown)}"
        )
    config = AutoTriggerConfig(**payload)
    if config.start_motion_threshold > config.blank_motion_threshold:
        raise ValueError(
            "formal auto-trigger config start_motion_threshold must be <= "
            "blank_motion_threshold"
        )
    return config


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


def _hand_center(hand: np.ndarray) -> np.ndarray | None:
    """Return a robust palm centre while rejecting an all-zero hand landmark set."""
    valid = hand[np.any(hand != 0, axis=1)]
    if not len(valid):
        return None
    # Palm anchors are stabler than fingertips when a relaxed hand is partially cropped.
    anchors = valid[: min(len(valid), 18)]
    return np.median(anchors, axis=0)


def _assignment_is_on_knees(
    wrists: tuple[np.ndarray, np.ndarray],
    hand_centres: tuple[np.ndarray | None, np.ndarray | None],
    hips: tuple[np.ndarray, np.ndarray],
    knees: tuple[np.ndarray, np.ndarray],
    scale: float,
    config: AutoTriggerConfig,
    assignment: tuple[int, int],
) -> bool:
    """Check one left/right (or crossed) hand-to-knee association."""
    for hand_index, knee_index in enumerate(assignment):
        wrist = wrists[hand_index]
        hand_centre = hand_centres[hand_index]
        hip = hips[knee_index]
        knee = knees[knee_index]
        thigh = knee[:2] - hip[:2]
        thigh_sq = float(np.dot(thigh, thigh))
        if hand_centre is None or thigh_sq <= 1e-8:
            return False
        wrist_progress = float(np.dot(wrist[:2] - hip[:2], thigh) / thigh_sq)
        palm_progress = float(np.dot(hand_centre[:2] - hip[:2], thigh) / thigh_sq)
        wrist_projection = hip[:2] + wrist_progress * thigh
        palm_projection = hip[:2] + palm_progress * thigh
        wrist_lateral = float(np.linalg.norm(wrist[:2] - wrist_projection))
        palm_lateral = float(np.linalg.norm(hand_centre[:2] - palm_projection))
        if not (config.knee_min_thigh_progress_ratio <= wrist_progress <= config.knee_max_thigh_progress_ratio):
            return False
        if not (config.knee_min_thigh_progress_ratio <= palm_progress <= config.knee_max_thigh_progress_ratio):
            return False
        if wrist_lateral > config.knee_lateral_thigh_margin_ratio * scale:
            return False
        if palm_lateral > config.knee_lateral_thigh_margin_ratio * scale:
            return False
    return True


def _rest_signature(
    left_wrist: np.ndarray | None,
    right_wrist: np.ndarray | None,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    left_shoulder: np.ndarray | None,
    right_shoulder: np.ndarray | None,
) -> tuple[float, ...] | None:
    """Encode wrist/palm positions in torso coordinates for seated rest matching."""
    if any(point is None for point in (left_wrist, right_wrist, left_shoulder, right_shoulder)):
        return None
    left_palm = _hand_center(left_hand)
    right_palm = _hand_center(right_hand)
    if left_palm is None or right_palm is None:
        return None
    assert left_wrist is not None and right_wrist is not None
    assert left_shoulder is not None and right_shoulder is not None
    centre = (left_shoulder[:2] + right_shoulder[:2]) / 2.0
    scale = max(float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])), 1e-3)
    pairs = [(left_wrist[:2], left_palm[:2]), (right_wrist[:2], right_palm[:2])]
    pairs.sort(key=lambda pair: float(pair[0][0]))
    values: list[float] = []
    for wrist, palm in pairs:
        values.extend(((wrist - centre) / scale).tolist())
        values.extend(((palm - centre) / scale).tolist())
    return tuple(float(value) for value in values)


def _wrist_rest_signature(
    left_wrist: np.ndarray | None,
    right_wrist: np.ndarray | None,
    left_shoulder: np.ndarray | None,
    right_shoulder: np.ndarray | None,
) -> tuple[float, ...] | None:
    """Encode pose/hand wrists even when explicit palm landmarks disappear."""
    if any(point is None for point in (left_wrist, right_wrist, left_shoulder, right_shoulder)):
        return None
    assert left_wrist is not None and right_wrist is not None
    assert left_shoulder is not None and right_shoulder is not None
    centre = (left_shoulder[:2] + right_shoulder[:2]) / 2.0
    scale = max(float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])), 1e-3)
    wrists = sorted((left_wrist[:2], right_wrist[:2]), key=lambda point: float(point[0]))
    values: list[float] = []
    for wrist in wrists:
        values.extend(((wrist - centre) / scale).tolist())
    return tuple(float(value) for value in values)


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
    left_knee = _valid_point(current_pose, LEFT_KNEE)
    right_knee = _valid_point(current_pose, RIGHT_KNEE)
    torso_valid = all(point is not None for point in (left_shoulder, right_shoulder, left_hip, right_hip))
    knee_landmarks_valid = all(point is not None for point in (left_knee, right_knee))
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

    hands_on_knees = False
    if (
        config.knee_geometry_enabled
        and torso_valid
        and knee_landmarks_valid
        and wrists_detected
        and explicit_hands_detected == 2
    ):
        assert left_shoulder is not None and right_shoulder is not None
        assert left_hip is not None and right_hip is not None
        assert left_knee is not None and right_knee is not None
        assert left_wrist is not None and right_wrist is not None
        shoulder_width = max(float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])), 1e-3)
        thigh_lengths = [
            float(np.linalg.norm(left_knee[:2] - left_hip[:2])),
            float(np.linalg.norm(right_knee[:2] - right_hip[:2])),
        ]
        body_scale = max(shoulder_width, float(np.median(thigh_lengths)), 1e-3)
        hand_centres = (_hand_center(current_left), _hand_center(current_right))
        wrists = (left_wrist, right_wrist)
        hips = (left_hip, right_hip)
        knees = (left_knee, right_knee)
        hands_on_knees = (
            _assignment_is_on_knees(wrists, hand_centres, hips, knees, body_scale, config, (0, 1))
            or _assignment_is_on_knees(wrists, hand_centres, hips, knees, body_scale, config, (1, 0))
        )

    visible_rest_blank = bool(
        explicit_hands_detected == 2
        and hands_on_knees
        and effective_motion <= config.blank_motion_threshold
    )
    hidden_rest_blank = bool(
        config.hidden_rest_enabled
        and torso_valid
        and explicit_hands_detected == 0
        and torso_motion <= config.blank_motion_threshold
    )
    rest_signature = _rest_signature(
        left_wrist,
        right_wrist,
        current_left,
        current_right,
        left_shoulder,
        right_shoulder,
    )
    wrist_rest_signature = _wrist_rest_signature(
        left_wrist,
        right_wrist,
        left_shoulder,
        right_shoulder,
    )
    return AutoFrameAnalysis(
        visible_rest_blank=visible_rest_blank,
        hidden_rest_blank=hidden_rest_blank,
        torso_motion_score=torso_motion,
        hand_motion_score=hand_motion,
        effective_motion_score=effective_motion,
        hands_on_knees=hands_on_knees,
        knee_landmarks_valid=knee_landmarks_valid,
        wrists_detected=wrists_detected,
        torso_valid=torso_valid,
        explicit_hands_detected=explicit_hands_detected,
        wrist_source_left=left_source,
        wrist_source_right=right_source,
        rest_signature=rest_signature,
        wrist_rest_signature=wrist_rest_signature,
    )


class AutoTriggerEngine:
    def __init__(self, config: AutoTriggerConfig):
        self.config = config
        self.state = SEGMENT_STATE_IDLE
        self.clip_start_sec: float | None = None
        self.segment_samples: list[FrameSample] = []
        self._pre_roll: deque[FrameSample] = deque()
        self._active_start_sec: float | None = None
        self._low_motion_start_sec: float | None = None
        self._end_votes: deque[tuple[float, bool]] = deque()
        self._cooldown_until_sec = 0.0
        self._last_timestamp_sec: float | None = None
        self._reference_seed_start_sec: float | None = None
        self._reference_signatures: list[np.ndarray] = []
        self._reference_wrist_signatures: list[np.ndarray] = []
        self._rest_reference_signature: np.ndarray | None = None
        self._rest_wrist_reference_signature: np.ndarray | None = None
        self._calibration_telemetry = self._initial_calibration_telemetry()

    def _initial_calibration_telemetry(self) -> CalibrationTelemetry:
        if not self.config.reference_rest_enabled:
            return CalibrationTelemetry(CalibrationState.DISABLED, None, 0.0, 0)
        return CalibrationTelemetry(
            CalibrationState.MISSING_REST_SIGNATURE,
            CalibrationState.MISSING_REST_SIGNATURE,
            0.0,
            0,
        )

    @property
    def calibration_telemetry(self) -> CalibrationTelemetry:
        return self._calibration_telemetry

    def reset(self) -> None:
        self.state = SEGMENT_STATE_IDLE
        self.clip_start_sec = None
        self.segment_samples = []
        self._pre_roll.clear()
        self._active_start_sec = None
        self._low_motion_start_sec = None
        self._end_votes.clear()
        self._cooldown_until_sec = 0.0
        self._last_timestamp_sec = None
        self._reference_seed_start_sec = None
        self._reference_signatures = []
        self._reference_wrist_signatures = []
        self._rest_reference_signature = None
        self._rest_wrist_reference_signature = None
        self._calibration_telemetry = self._initial_calibration_telemetry()

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
        # Learn the per-video seated waiting pose before allowing an action.
        # This must run regardless of the state so a transient early movement
        # cannot prevent the initial reference from ever being completed.
        self._update_rest_reference(timestamp_sec, analysis)

        if self.state == SEGMENT_STATE_COOLDOWN:
            if timestamp_sec < self._cooldown_until_sec:
                return None
            self.state = SEGMENT_STATE_IDLE
            self.clip_start_sec = None
            self._pre_roll.clear()
            self._low_motion_start_sec = None

        if self.state == SEGMENT_STATE_IDLE:
            return self._update_idle(sample, analysis)

        self.segment_samples.append(sample)
        if analysis.effective_motion_score <= self.config.blank_motion_threshold:
            if self._low_motion_start_sec is None:
                self._low_motion_start_sec = timestamp_sec
        else:
            self._low_motion_start_sec = None
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
            rest_detected_sec = next(time_sec for time_sec, vote in self._end_votes if vote)
            clip_end_sec = rest_detected_sec
            boundary_policy = "first_confirmed_rest"
            if (
                self.config.end_safety_tail_sec > 0
                and self._low_motion_start_sec is not None
            ):
                clip_end_sec = min(
                    rest_detected_sec,
                    self._low_motion_start_sec + self.config.end_safety_tail_sec,
                )
                boundary_policy = "low_motion_anchor_v1"
            if analysis.hidden_rest_blank and not analysis.visible_rest_blank:
                reason = "hidden_rest_finalize"
            elif analysis.visible_rest_blank:
                reason = "visible_rest_finalize"
            else:
                reason = "reference_rest_finalize"
            return self._finalize(
                clip_end_sec,
                timestamp_sec,
                reason,
                rest_detected_sec=rest_detected_sec,
                boundary_policy=boundary_policy,
            )
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
            self._low_motion_start_sec = None
            self._end_votes.clear()
        return None

    def _is_start_candidate(self, analysis: AutoFrameAnalysis) -> bool:
        if self.config.reference_rest_enabled and self._rest_reference_signature is None:
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
            and analysis.temporal_active_probability < self.config.temporal_start_probability_threshold
        ):
            return False
        return True

    def _is_rest_candidate(self, analysis: AutoFrameAnalysis) -> bool:
        rest = False
        if analysis.visible_rest_blank:
            rest = True
        elif (
            self.config.reference_rest_enabled
            and (distance := self._reference_distance(analysis)) is not None
            and analysis.effective_motion_score <= self.config.blank_motion_threshold
        ):
            if distance <= self.config.reference_rest_distance_threshold:
                rest = True
        elif self.config.hidden_rest_enabled and analysis.hidden_rest_blank:
            rest = True
        if not rest:
            return False
        if (
            self.config.temporal_classifier_enabled
            and analysis.temporal_active_probability is not None
            and 1.0 - analysis.temporal_active_probability < self.config.temporal_rest_probability_threshold
        ):
            return False
        return True

    def _reference_distance(self, analysis: AutoFrameAnalysis) -> float | None:
        if self._rest_reference_signature is not None and analysis.rest_signature is not None:
            signature = np.asarray(analysis.rest_signature, dtype=np.float32)
            return float(np.sqrt(np.mean(np.square(signature - self._rest_reference_signature))))
        if (
            self._rest_wrist_reference_signature is not None
            and analysis.wrist_rest_signature is not None
        ):
            signature = np.asarray(analysis.wrist_rest_signature, dtype=np.float32)
            return float(
                np.sqrt(
                    np.mean(np.square(signature - self._rest_wrist_reference_signature))
                )
            )
        return None

    def _update_rest_reference(self, timestamp_sec: float, analysis: AutoFrameAnalysis) -> None:
        if not self.config.reference_rest_enabled:
            self._calibration_telemetry = CalibrationTelemetry(
                CalibrationState.DISABLED, None, 0.0, 0
            )
            return
        if self._rest_reference_signature is not None:
            self._calibration_telemetry = CalibrationTelemetry(
                CalibrationState.CALIBRATED,
                None,
                self.config.reference_seed_sec,
                len(self._reference_signatures),
            )
            return
        blocker: CalibrationState | None = None
        if analysis.explicit_hands_detected != 2:
            blocker = CalibrationState.REQUIRES_TWO_HANDS
        elif analysis.rest_signature is None or analysis.wrist_rest_signature is None:
            blocker = CalibrationState.MISSING_REST_SIGNATURE
        elif analysis.effective_motion_score > self.config.reference_seed_motion_threshold:
            blocker = CalibrationState.MOTION_ABOVE_SEED_THRESHOLD
        # Do not start the countdown from camera-open: pose/hand landmarks can
        # be absent for the first frames while iVCam autofocus settles.
        if blocker is not None:
            self._reference_seed_start_sec = None
            self._reference_signatures.clear()
            self._reference_wrist_signatures.clear()
            self._calibration_telemetry = CalibrationTelemetry(blocker, blocker, 0.0, 0)
            return
        if self._reference_seed_start_sec is None:
            self._reference_seed_start_sec = timestamp_sec
        assert analysis.rest_signature is not None
        assert analysis.wrist_rest_signature is not None
        self._reference_signatures.append(np.asarray(analysis.rest_signature, dtype=np.float32))
        self._reference_wrist_signatures.append(
            np.asarray(analysis.wrist_rest_signature, dtype=np.float32)
        )
        elapsed_sec = max(0.0, timestamp_sec - self._reference_seed_start_sec)
        self._calibration_telemetry = CalibrationTelemetry(
            CalibrationState.COLLECTING_REFERENCE,
            None,
            elapsed_sec,
            len(self._reference_signatures),
        )
        if elapsed_sec >= self.config.reference_seed_sec:
            self._rest_reference_signature = np.median(
                np.stack(self._reference_signatures), axis=0
            ).astype(np.float32)
            self._rest_wrist_reference_signature = np.median(
                np.stack(self._reference_wrist_signatures), axis=0
            ).astype(np.float32)
            self._calibration_telemetry = CalibrationTelemetry(
                CalibrationState.CALIBRATED,
                None,
                elapsed_sec,
                len(self._reference_signatures),
            )

    def _finalize(
        self,
        clip_end_sec: float,
        finalize_sec: float,
        reason: str,
        *,
        rest_detected_sec: float | None = None,
        boundary_policy: str | None = None,
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
            rest_detected_sec=(
                None if rest_detected_sec is None else float(rest_detected_sec)
            ),
            boundary_policy=(
                boundary_policy
                or ("timeout" if reason == "timeout_finalize" else "first_confirmed_rest")
            ),
        )
        self.state = SEGMENT_STATE_COOLDOWN
        self._cooldown_until_sec = finalize_sec + self.config.cooldown_sec
        self.clip_start_sec = None
        self.segment_samples = []
        self._pre_roll.clear()
        self._active_start_sec = None
        self._low_motion_start_sec = None
        self._end_votes.clear()
        return result
