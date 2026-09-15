"""Visibility-aware trigger geometry. Never modifies recognition features."""
from dataclasses import dataclass

import numpy as np

from recognition.realtime.auto_trigger import AutoFrameAnalysis, analyze_frame_vector

TORSO = [11, 12, 23, 24]


@dataclass(frozen=True)
class KneeFrameAnalysis(AutoFrameAnalysis):
    body_anchor: tuple[float, ...] | None = None
    visibility_available: bool = False
    rest_motion_score: float | None = None
    rest_motion_ready: bool = True
    raw_hands_on_knees: bool | None = None
    knee_region_margin: float | None = None


def align_trigger_hands(vector):
    """Associate trigger hands with pose wrists; never alter model features.

    Tasks handedness may change while a hand stays still. A mislabeled hand
    must also not be counted once as a hand and again as a pose fallback.
    """
    points = np.asarray(vector).reshape(75, 3).copy()
    hands = [points[s:s+21].copy() for s in (33, 54) if np.any(points[s])]
    pose_sides = [s for s in (0, 1) if np.any(points[15+s])]
    points[33:] = 0
    if not hands:
        return points.ravel()
    if len(pose_sides) == 2 and len(hands) == 2:
        direct = sum(np.linalg.norm(hands[s][0,:2]-points[15+s,:2]) for s in (0,1))
        crossed = sum(np.linalg.norm(hands[1-s][0,:2]-points[15+s,:2]) for s in (0,1))
        ordered = hands if direct <= crossed else hands[::-1]
        for side, hand in enumerate(ordered):
            points[33+21*side:54+21*side] = hand
    elif pose_sides:
        # Greedy matching is unambiguous with only one hand or one pose wrist.
        distances = [(float(np.linalg.norm(hand[0,:2]-points[15+s,:2])),i,s)
                     for i,hand in enumerate(hands) for s in pose_sides]
        distance,i,side = min(distances)
        anchor = _anchor(points)
        # A hand far from the only visible pose wrist belongs to the other side.
        if len(pose_sides)==1 and len(hands)==1 and anchor and distance > .5*anchor[-1]:
            side = 1-side
        points[33+21*side:54+21*side] = hands[i]
        if len(hands)==2:
            points[33+21*(1-side):54+21*(1-side)] = hands[1-i]
    else:
        # Pose wrists absent: retain two distinct hands in spatial order.
        for side,hand in enumerate(sorted(hands,key=lambda h:float(h[0,0]))):
            points[33+21*side:54+21*side] = hand
    return points.ravel()


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
    region_margin = None
    if anchor is not None and knees_valid and wrists_valid:
        rest_points = []
        for wrist,start in zip(wrists,(33,54)):
            candidates = [wrist[:2]]
            knuckles = points[np.asarray([5,9,13,17])+start]
            valid = np.any(knuckles != 0,axis=1)
            if np.any(points[start]) and valid.sum() >= 3:
                candidates.append(np.median(knuckles[valid,:2],axis=0))
            rest_points.append(candidates)
        torso_height = abs(anchor[3]-anchor[1])

        def knee_margin(point, side):
            hip, knee = points[23+side,:2], points[25+side,:2]
            # A seated thigh can be nearly edge-on. Its projected length is
            # not a reliable denominator for a palm resting on the knee.
            height_margin = (point[1]-min(hip[1],knee[1])+.15*torso_height)/anchor[-1]
            circle_margin = config.knee_lateral_thigh_margin_ratio - np.linalg.norm(point-knee)/anchor[-1]
            thigh = knee-hip
            length = float(np.linalg.norm(thigh))
            if length < .1*anchor[-1]:
                return min(height_margin,circle_margin)
            progress = float(np.dot(point-hip,thigh)/(length*length))
            lateral = float(np.linalg.norm(point-hip-progress*thigh))
            thigh_margin = min((progress-config.knee_min_thigh_progress_ratio)*length/anchor[-1],
                               (config.knee_max_thigh_progress_ratio-progress)*length/anchor[-1],
                               config.knee_lateral_thigh_margin_ratio-lateral/anchor[-1])
            return min(height_margin,max(circle_margin,thigh_margin))

        # Each distinct hand must match a different visible knee.
        def assignment(order):
            return min(max(knee_margin(point,side) for point in candidates)
                       for candidates,side in zip(rest_points,order))
        region_margin = float(max(assignment((0, 1)),assignment((1, 0))))
        on_knees = region_margin >= 0
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
        "raw_hands_on_knees": bool(on_knees),
        "knee_region_margin": region_margin,
        "knee_landmarks_valid": knees_valid,
        "torso_valid": anchor is not None,
        "wrists_detected": wrists_valid,
        "effective_motion_score": effective,
        "hand_motion_score": speed,
        "torso_motion_score": torso_speed,
        "body_anchor": anchor,
        "visibility_available": visibility_available,
    })
