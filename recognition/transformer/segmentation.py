"""Offline motion segmentation and whole-video analysis for the Transformer path.

This is the *offline* counterpart to the realtime auto-trigger state machine in
``recognition.realtime.auto_trigger``.  A recorded video has no live feedback
loop, so segmentation here is a single backward-looking pass over wrist motion
energy rather than a hold-and-confirm state machine.  The two are deliberately
separate: the realtime thresholds are calibrated against a live rest reference,
which a file does not have.

The result dictionary is JSON-serializable and is what the web service returns.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from recognition.transformer.landmarks import (
    MediaPipeLandmarkExtractor,
    TrackedFrame,
    convention_check,
    frame_to_219,
)


DEFAULT_SEGMENTATION = {
    "motion_threshold": 0.10,
    "min_duration": 0.3,
    "max_duration": 4.0,
    "pause": 0.4,
}

MIN_SEGMENT_FRAMES = 4


def motion_energy(frames: Sequence[TrackedFrame]) -> list[float]:
    """Per-frame wrist speed in normalized units per second; no hands means zero.

    A wrist that appears or disappears between frames is scored as full motion:
    hands entering or leaving the picture is movement, and treating it as
    stillness would let a segment end while the signer is still signing.
    """
    energy = [0.0]
    for index in range(1, len(frames)):
        delta = max(frames[index].timestamp - frames[index - 1].timestamp, 1e-3)
        best = 0.0
        for side in ("Left", "Right"):
            previous = frames[index - 1].wrist(side)
            current = frames[index].wrist(side)
            if previous is not None and current is not None:
                step = current[:2] - previous[:2]
                best = max(best, float(np.linalg.norm(step)) / delta)
            elif previous is not None or current is not None:
                best = max(best, 1.0)
        energy.append(best)
    return energy


def segment_frames(
    frames: Sequence[TrackedFrame],
    *,
    motion_threshold: float = DEFAULT_SEGMENTATION["motion_threshold"],
    min_duration: float = DEFAULT_SEGMENTATION["min_duration"],
    max_duration: float = DEFAULT_SEGMENTATION["max_duration"],
    pause: float = DEFAULT_SEGMENTATION["pause"],
) -> list[tuple[float, float]]:
    """Split a tracked video into ``(start, end)`` spans of sustained motion."""
    if not frames:
        return []
    energy = motion_energy(frames)
    segments: list[tuple[float, float]] = []
    start: float | None = None
    last_active: float | None = None

    for frame, value in zip(frames, energy):
        moment = frame.timestamp
        active = value >= motion_threshold and frame.has_hands
        if active:
            if start is None:
                start = moment
            last_active = moment
            if moment - start >= max_duration:
                segments.append((start, moment))
                start = last_active = None
        elif start is not None and last_active is not None and moment - last_active >= pause:
            if last_active - start >= min_duration:
                segments.append((start, last_active))
            start = last_active = None

    if start is not None and last_active is not None and last_active - start >= min_duration:
        segments.append((start, last_active))
    return segments


def analyze_frames(
    frames: Sequence[TrackedFrame],
    recognizer: Any,
    *,
    mirrored: bool = False,
    topk: int = 3,
    min_segment_frames: int = MIN_SEGMENT_FRAMES,
    **segmentation: float,
) -> dict:
    """Motion segmentation and top-k per segment, plus a whole-clip result.

    The whole-clip result matters: single-word clips often contain no pause at
    all, so segmentation legitimately finds nothing and it is the only answer.

    ``mirrored`` defaults to False, the convention the training cache was built
    under. Footage that needs un-mirroring should be flipped at the extractor
    (``selfie_flip``) rather than here, so the landmarker sees the corrected
    image rather than corrected coordinates.
    """
    if not frames:
        raise ValueError("no frames to analyze")

    mirror_flag = bool(mirrored)
    vectors = [frame_to_219(frame, mirrored=mirror_flag) for frame in frames]
    usable = [vector for vector in vectors if vector is not None]

    result: dict[str, Any] = {
        "mirrored": mirror_flag,
        "n_frames": len(frames),
        "n_valid": len(usable),
        "segments": [],
        "whole": None,
        "left_shoulder_x": None,
        "convention_ok": None,
    }
    if len(usable) < min_segment_frames:
        result["message"] = (
            f"only {len(usable)} frames had both shoulders visible "
            f"(at least {min_segment_frames} are required)"
        )
        return result

    everything = np.stack(usable)
    measured, convention_ok = convention_check(everything)
    result["left_shoulder_x"] = round(measured, 4)
    result["convention_ok"] = convention_ok

    def ranked(matrix: np.ndarray) -> list[dict]:
        return [
            {"label": label, "text": text, "prob": round(float(probability), 6)}
            for label, text, probability in recognizer.predict(matrix, topk=topk)
        ]

    result["whole"] = {"n_frames": len(usable), "top": ranked(everything)}

    options = {**DEFAULT_SEGMENTATION, **segmentation}
    timestamps = np.asarray([frame.timestamp for frame in frames])
    for index, (start, end) in enumerate(segment_frames(frames, **options), 1):
        low = int(np.searchsorted(timestamps, start))
        high = int(np.searchsorted(timestamps, end, side="right"))
        span = [vector for vector in vectors[low:high] if vector is not None]
        item = {
            "index": index,
            "start": round(float(start), 2),
            "end": round(float(end), 2),
            "duration": round(float(end - start), 2),
            "n_frames": len(span),
            "top": [],
            "skipped": len(span) < min_segment_frames,
        }
        if not item["skipped"]:
            item["top"] = ranked(np.stack(span))
        result["segments"].append(item)
    return result


def analyze_video(
    video_path: Path | str,
    recognizer: Any,
    *,
    hand_model: Path | str,
    pose_model: Path | str,
    resize_width: int = 960,
    selfie_flip: bool = False,
    max_seconds: float = 0.0,
    progress: Callable[[int, int], None] | None = None,
    **analysis: Any,
) -> dict:
    """Track a whole video, then analyze it. ``progress(done, total)`` is optional."""
    started = time.perf_counter()
    with MediaPipeLandmarkExtractor(
        hand_model, pose_model, resize_width=resize_width, selfie_flip=selfie_flip
    ) as extractor:
        frames = extractor.run_video(video_path, progress=progress)
    tracking_seconds = time.perf_counter() - started

    if max_seconds > 0:
        frames = [frame for frame in frames if frame.timestamp <= max_seconds]
    if not frames:
        raise ValueError("no frames could be read from the video")

    analyzing = time.perf_counter()
    result = analyze_frames(frames, recognizer, **analysis)
    result["track_seconds"] = round(tracking_seconds, 2)
    result["analyze_seconds"] = round(time.perf_counter() - analyzing, 2)
    result["n_tracked"] = len(frames)
    result["duration"] = round(float(frames[-1].timestamp), 2)
    result["track_fps"] = round(len(frames) / max(tracking_seconds, 1e-6), 1)
    result["selfie_flip"] = bool(selfie_flip)
    return result
