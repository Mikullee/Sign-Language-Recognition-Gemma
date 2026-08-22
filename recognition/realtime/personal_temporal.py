from __future__ import annotations

"""Small per-person temporal idle/signing classifier built from landmark caches."""

from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np

from recognition.realtime.auto_trigger import (
    AutoFrameAnalysis,
    AutoTriggerConfig,
    analyze_frame_vector,
)


FEATURE_VERSION = 1


def _signature(analysis: AutoFrameAnalysis) -> np.ndarray | None:
    if analysis.rest_signature is None:
        return None
    return np.asarray(analysis.rest_signature, dtype=np.float32)


def _mean_window(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values.copy()
    result = np.empty_like(values, dtype=np.float32)
    cumulative = np.concatenate((np.zeros((1, values.shape[1]), dtype=np.float32), np.cumsum(values, axis=0)))
    for index in range(len(values)):
        start = max(0, index - width + 1)
        result[index] = (cumulative[index + 1] - cumulative[start]) / (index - start + 1)
    return result


def build_temporal_features(
    timestamps_sec: np.ndarray,
    frame_vectors: np.ndarray,
    config: AutoTriggerConfig,
) -> np.ndarray:
    """Create frame-local and short-window landmark features without sentence labels."""
    timestamps = np.asarray(timestamps_sec, dtype=np.float32)
    vectors = np.asarray(frame_vectors, dtype=np.float32)
    if not len(vectors):
        return np.empty((0, 32), dtype=np.float32)
    analyses: list[AutoFrameAnalysis] = []
    previous = None
    signatures: list[np.ndarray | None] = []
    for vector in vectors:
        analysis = analyze_frame_vector(previous, vector, config)
        analyses.append(analysis)
        signatures.append(_signature(analysis))
        previous = vector
    valid = [signature for signature, time_sec in zip(signatures, timestamps) if signature is not None and time_sec <= config.reference_seed_sec]
    if not valid:
        valid = [signature for signature in signatures if signature is not None]
    reference = np.median(np.stack(valid), axis=0).astype(np.float32) if valid else np.zeros(8, dtype=np.float32)
    resolved = np.stack([signature if signature is not None else reference for signature in signatures]).astype(np.float32)
    delta = resolved - reference
    dt = np.maximum(np.diff(timestamps, prepend=timestamps[0]), 1.0 / 30.0)
    velocity = np.vstack((np.zeros((1, 8), dtype=np.float32), np.diff(resolved, axis=0) / dt[1:, None])).astype(np.float32)
    motion = np.asarray([item.effective_motion_score for item in analyses], dtype=np.float32)
    acceleration = np.abs(np.diff(motion, prepend=motion[0])) / dt
    hand_distance = np.linalg.norm(resolved[:, :2] - resolved[:, 4:6], axis=1)
    delta_norm = np.linalg.norm(delta, axis=1)
    scalar = np.column_stack((
        motion,
        acceleration,
        hand_distance,
        delta_norm,
        np.asarray([item.hands_on_knees for item in analyses], dtype=np.float32),
        np.asarray([item.explicit_hands_detected / 2.0 for item in analyses], dtype=np.float32),
        np.asarray([item.knee_landmarks_valid for item in analyses], dtype=np.float32),
        np.asarray([signature is not None for signature in signatures], dtype=np.float32),
    )).astype(np.float32)
    fps = max(1.0 / float(np.median(dt[1:])) if len(dt) > 1 else 30.0, 1.0)
    short = _mean_window(scalar[:, :4], max(1, round(fps * 0.20)))
    long = _mean_window(scalar[:, :4], max(1, round(fps * 0.50)))
    return np.concatenate((delta, velocity, scalar, short, long), axis=1).astype(np.float32)


def build_frame_labels(timestamps_sec: np.ndarray, start_sec: float, end_sec: float) -> np.ndarray:
    timestamps = np.asarray(timestamps_sec, dtype=np.float32)
    return ((timestamps >= start_sec) & (timestamps <= end_sec)).astype(np.float32)


class PersonalTemporalModel:
    """Serializable weighted logistic-regression classifier using only NumPy."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray, weights: np.ndarray, bias: float):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.maximum(np.asarray(scale, dtype=np.float32), 1e-6)
        self.weights = np.asarray(weights, dtype=np.float32)
        self.bias = float(bias)

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, iterations: int = 1200, learning_rate: float = 0.12, l2: float = 1e-3) -> "PersonalTemporalModel":
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32)
        if x.ndim != 2 or len(x) != len(y) or len(x) == 0 or len(np.unique(y)) != 2:
            raise ValueError("Temporal training requires aligned features containing both idle and signing frames.")
        mean = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), 1e-5)
        normalized = (x - mean) / scale
        weights = np.zeros(normalized.shape[1], dtype=np.float32)
        bias = 0.0
        positive = max(float(y.sum()), 1.0)
        negative = max(float(len(y) - y.sum()), 1.0)
        sample_weight = np.where(y > 0.5, len(y) / (2.0 * positive), len(y) / (2.0 * negative)).astype(np.float32)
        for _ in range(iterations):
            logits = np.clip(normalized @ weights + bias, -30.0, 30.0)
            probability = 1.0 / (1.0 + np.exp(-logits))
            residual = (probability - y) * sample_weight
            weights -= learning_rate * ((normalized.T @ residual) / len(y) + l2 * weights)
            bias -= learning_rate * float(residual.mean())
        return cls(mean, scale, weights, bias)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        logits = np.clip(((x - self.mean) / self.scale) @ self.weights + self.bias, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, version=np.array(FEATURE_VERSION), mean=self.mean, scale=self.scale, weights=self.weights, bias=np.array(self.bias, dtype=np.float32))
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "PersonalTemporalModel":
        with np.load(Path(path), allow_pickle=False) as payload:
            if int(payload["version"]) != FEATURE_VERSION:
                raise ValueError("Unsupported personal temporal model version.")
            return cls(payload["mean"], payload["scale"], payload["weights"], float(payload["bias"]))


class PersonalTemporalPredictor:
    """Online version of feature extraction for realtime state-machine integration."""

    def __init__(self, model: PersonalTemporalModel, config: AutoTriggerConfig):
        self.model = model
        self.config = config
        self._seed: list[np.ndarray] = []
        self._seed_start: float | None = None
        self._reference: np.ndarray | None = None
        self._previous_signature: np.ndarray | None = None
        self._previous_motion = 0.0
        self._history: deque[tuple[float, np.ndarray]] = deque()

    def update(self, timestamp_sec: float, analysis: AutoFrameAnalysis) -> float | None:
        signature = _signature(analysis)
        if self._seed_start is None:
            self._seed_start = timestamp_sec
        if self._reference is None:
            if signature is not None and analysis.effective_motion_score <= self.config.reference_seed_motion_threshold:
                self._seed.append(signature)
            if timestamp_sec - self._seed_start >= self.config.reference_seed_sec:
                if not self._seed:
                    return None
                self._reference = np.median(np.stack(self._seed), axis=0).astype(np.float32)
            else:
                return None
        assert self._reference is not None
        resolved = signature if signature is not None else self._reference
        delta = resolved - self._reference
        if self._previous_signature is None:
            velocity = np.zeros(8, dtype=np.float32)
            acceleration = 0.0
        else:
            elapsed = max(timestamp_sec - self._history[-1][0], 1.0 / 30.0)
            velocity = (resolved - self._previous_signature) / elapsed
            acceleration = abs(analysis.effective_motion_score - self._previous_motion) / elapsed
        scalar = np.array([
            analysis.effective_motion_score, acceleration,
            float(np.linalg.norm(resolved[:2] - resolved[4:6])), float(np.linalg.norm(delta)),
            float(analysis.hands_on_knees), analysis.explicit_hands_detected / 2.0,
            float(analysis.knee_landmarks_valid), float(signature is not None),
        ], dtype=np.float32)
        self._history.append((timestamp_sec, scalar))
        while self._history and timestamp_sec - self._history[0][0] > 0.50:
            self._history.popleft()
        history = list(self._history)
        short = np.mean([value[:4] for time_sec, value in history if timestamp_sec - time_sec <= 0.20], axis=0)
        long = np.mean([value[:4] for _, value in history], axis=0)
        feature = np.concatenate((delta, velocity, scalar, short, long))[None, :]
        self._previous_signature = resolved.copy()
        self._previous_motion = analysis.effective_motion_score
        return float(self.model.predict_proba(feature)[0])


def with_temporal_probability(analysis: AutoFrameAnalysis, probability: float | None) -> AutoFrameAnalysis:
    return replace(analysis, temporal_active_probability=probability)
