"""Sequence feature contract for the Knee42 Transformer recognizer.

The frame-level contract (MediaPipe result -> 219 shoulder-normalized values plus
an observation mask) is shared with the legacy BiGRU path and lives in
``recognition.realtime.knee42_preprocessing``.  This module only implements the
stage where the two paths diverge: turning a variable-length ``[frames, 219]``
sequence into the fixed ``[64, 657]`` tensor the Transformer was trained on.

Unlike the BiGRU contract, the Transformer does not consume the observation mask
and does not apply a train-only standardizer.  Missing coordinates are linearly
interpolated along the time axis instead, and the model sees position, velocity
and acceleration channels of the shoulder-normalized coordinates directly.
"""
from __future__ import annotations

import numpy as np


SEQUENCE_LENGTH = 64
LANDMARK_DIM = 219
MODEL_INPUT_DIM = LANDMARK_DIM * 3


def interp_missing(values: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN gaps per dimension along the time axis.

    Edges are held at the nearest observed value.  A dimension that is missing
    for the whole sequence collapses to zero, which is the neutral value in the
    shoulder-normalized coordinate space.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != LANDMARK_DIM:
        raise ValueError(f"expected [frames,{LANDMARK_DIM}] values, got {values.shape}")
    if len(values) == 0:
        raise ValueError("cannot interpolate an empty sequence")
    frames = np.arange(len(values))
    filled = values.copy()
    for dimension in range(values.shape[1]):
        column = values[:, dimension]
        observed = np.isfinite(column)
        if observed.all():
            continue
        if not observed.any():
            filled[:, dimension] = 0.0
            continue
        filled[:, dimension] = np.interp(frames, frames[observed], column[observed])
    return filled


def resample(values: np.ndarray, sequence_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Linearly resample the time axis to a fixed number of frames."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"expected a non-empty 2-D sequence, got {values.shape}")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if len(values) == 1:
        return np.repeat(values, sequence_length, axis=0).astype(np.float32)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, sequence_length)
    resampled = np.empty((sequence_length, values.shape[1]), dtype=np.float32)
    for dimension in range(values.shape[1]):
        resampled[:, dimension] = np.interp(target, source, values[:, dimension])
    return resampled


def featurize(
    values: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """Turn ``[frames, 219]`` into the ``[64, 657]`` position/velocity/acceleration tensor.

    Velocity and acceleration are first-order differences taken *after* the
    resampling step, so they are expressed per resampled frame rather than per
    captured frame.  Both channels repeat their first value at index 0 so the
    output keeps the full ``sequence_length``.
    """
    positions = resample(values, sequence_length)
    velocity = np.diff(positions, axis=0, prepend=positions[:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
    features = np.concatenate([positions, velocity, acceleration], axis=1)
    if features.shape != (sequence_length, MODEL_INPUT_DIM):
        raise AssertionError(f"unexpected Transformer feature shape: {features.shape}")
    return features.astype(np.float32)


def materialize_sequence(
    values: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """Full sequence stage: interpolate missing coordinates, then featurize."""
    return featurize(interp_missing(values), sequence_length)
