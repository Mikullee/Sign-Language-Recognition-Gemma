"""Visibility-aware trigger geometry. Never modifies recognition features."""
from dataclasses import dataclass

import numpy as np

from recognition.realtime.auto_trigger import AutoFrameAnalysis, analyze_frame_vector

TORSO = [11, 12, 23, 24]


@dataclass(frozen=True)
class KneeFrameAnalysis(AutoFrameAnalysis):
    body_anchor: tuple[float, ...] | None = None
    visibility_available: bool = False


def sanitize_trigger(vector, visibility, threshold):
    """Unknown pose visibility is not positive evidence of a visible knee."""
    points = np.asarray(vector, dtype=np.float32).reshape(75, 3).copy()
    valid = (np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)
             & (points[:, :2] >= 0).all(axis=1) & (points[:, :2] <= 1).all(axis=1))
    if visibility is None:
        valid[:33] = False
    else:
        visibility = np.asarray(visibility, dtype=np.float32)
        if visibility.shape != (33,):
            raise ValueError("pose_visibility must have 33 values")
        valid[:33] &= np.isfinite(visibility) & (visibility >= threshold)
    points[~valid] = 0
    return points.reshape(225)


def _anchor(points):
    if not np.any(points[TORSO] != 0, axis=1).all():
        return None
    shoulder = points[[11, 12], :2].mean(axis=0)
    hip = points[[23, 24], :2].mean(axis=0)
    scale = float(np.linalg.norm(points[11, :2] - points[12, :2]))
    if scale < .02:
        return None
    return (*shoulder.tolist(), *hip.tolist(), scale)


def _normalized(points, anchor):
    return (points[:, :2] - np.asarray(anchor[:2])) / anchor[-1]


def analyze_knee_frame(previous, current, config, interval_sec, *, visibility_available):
    """Coordinates in body widths, motion in body widths/second (causal)."""
    base = analyze_frame_vector(previous, current, config)
    points = current.reshape(75, 3)
    anchor = _anchor(points)
    wrists = []
    for hand, pose in ((33, 15), (54, 16)):
        wrists.append(points[hand] if np.any(points[hand]) else points[pose])
    wrists_valid = all(np.any(w) for w in wrists)
    knees_valid = bool(np.any(points[[25, 26]] != 0, axis=1).all())
    on_knees = False
    if anchor is not None and knees_valid and wrists_valid:
        # Permit either handedness assignment, but never a chest-level wrist.
        def assignment(order):
            for wrist, side in zip(wrists, order):
                hip, knee = points[23 + side, :2], points[25 + side, :2]
                thigh = knee - hip
                length = float(np.linalg.norm(thigh))
                if length < .1 * anchor[-1]:
                    return False
                progress = float(np.dot(wrist[:2] - hip, thigh) / (length * length))
                lateral = float(np.linalg.norm(wrist[:2] - hip - progress * thigh))
                if not config.knee_min_thigh_progress_ratio <= progress <= config.knee_max_thigh_progress_ratio:
                    return False
                if lateral > config.knee_lateral_thigh_margin_ratio * anchor[-1]:
                    return False
            return True
        on_knees = assignment((0, 1)) or assignment((1, 0))
    speed = 0.0
    torso_speed = 0.0
    if previous is not None and interval_sec > 0 and anchor is not None:
        old = previous.reshape(75, 3)
        old_anchor = _anchor(old)
        if old_anchor is not None:
            delta = (_normalized(points, anchor) - _normalized(old, old_anchor)) / interval_sec
            valid = np.any(points != 0, axis=1) & np.any(old != 0, axis=1)
            def rms(indices):
                indices = np.asarray(indices)
                selected = indices[valid[indices]]
                return float(np.sqrt(np.mean(delta[selected] ** 2))) if len(selected) else 0.0
            speed = max(rms(range(33, 54)), rms(range(54, 75)), rms([13, 14, 15, 16]))
            torso_speed = rms(TORSO)
            # Relative coordinates must not disguise a moving calibration
            # candidate as still. Translation invariance applies to *where*
            # the person sits, not to movement during a required stable hold.
            body_speed = float(np.linalg.norm(np.asarray(anchor[:2]) - old_anchor[:2])) / old_anchor[-1] / interval_sec
            torso_speed = max(torso_speed, body_speed)
    effective = max(speed, config.torso_motion_weight * torso_speed)
    return KneeFrameAnalysis(**{
        **base.__dict__,
        "visible_rest_blank": bool(on_knees and effective <= config.blank_motion_threshold),
        "hidden_rest_blank": False,
        "hands_on_knees": bool(on_knees),
        "knee_landmarks_valid": knees_valid,
        "torso_valid": anchor is not None,
        "wrists_detected": wrists_valid,
        "effective_motion_score": effective,
        "hand_motion_score": speed,
        "torso_motion_score": torso_speed,
        "body_anchor": anchor,
        "visibility_available": visibility_available,
    })
