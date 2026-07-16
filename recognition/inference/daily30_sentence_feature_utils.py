from __future__ import annotations

import numpy as np


def append_delta_features(frames: np.ndarray) -> np.ndarray:
    if len(frames) == 0:
        return frames.astype(np.float32)
    delta = np.zeros_like(frames, dtype=np.float32)
    if len(frames) > 1:
        delta[1:] = frames[1:] - frames[:-1]
    return np.concatenate([frames, delta], axis=-1).astype(np.float32)


def zscore_sequence_features(frames: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    if len(frames) == 0:
        return frames.astype(np.float32)
    mean = frames.mean(axis=0, keepdims=True)
    std = frames.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return ((frames - mean) / std).astype(np.float32)


def build_feature_sequence(
    frames: np.ndarray,
    append_delta: bool = True,
    zscore_features: bool = True,
) -> np.ndarray:
    out = frames.astype(np.float32, copy=True)
    if append_delta:
        out = append_delta_features(out)
    if zscore_features:
        out = zscore_sequence_features(out)
    return out.astype(np.float32)


def resample_sequence(frames: np.ndarray, target_length: int) -> np.ndarray:
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if len(frames) == 0:
        return np.zeros((target_length, 0), dtype=np.float32)
    if len(frames) == target_length:
        return frames.astype(np.float32, copy=True)
    positions = np.linspace(0, len(frames) - 1, num=target_length)
    indices = np.clip(np.round(positions).astype(int), 0, len(frames) - 1)
    return frames[indices].astype(np.float32, copy=True)


def partial_observation_sequence(frames: np.ndarray, fraction: float) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    total = len(frames)
    keep = max(1, int(np.ceil(total * fraction)))
    out = np.zeros_like(frames, dtype=np.float32)
    out[:keep] = frames[:keep]
    return out


def boundary_shift_sequence(frames: np.ndarray, shift_ratio: float) -> np.ndarray:
    total = len(frames)
    shift = int(round(total * shift_ratio))
    out = np.zeros_like(frames, dtype=np.float32)
    if shift >= 0:
        keep = max(0, total - shift)
        if keep > 0:
            out[:keep] = frames[shift : shift + keep]
    else:
        gap = min(total, -shift)
        keep = max(0, total - gap)
        if keep > 0:
            out[gap : gap + keep] = frames[:keep]
    return out


def blank_context_sequence(frames: np.ndarray, pre_ratio: float, post_ratio: float) -> np.ndarray:
    if pre_ratio < 0 or post_ratio < 0:
        raise ValueError("blank ratios must be non-negative")
    total = len(frames)
    pre = int(round(total * pre_ratio))
    post = int(round(total * post_ratio))
    padded = np.concatenate(
        [
            np.zeros((pre, frames.shape[1]), dtype=np.float32),
            frames.astype(np.float32, copy=False),
            np.zeros((post, frames.shape[1]), dtype=np.float32),
        ],
        axis=0,
    )
    return resample_sequence(padded, total)


def sliding_window_sequences(frames: np.ndarray, context_ratio: float = 0.25, stride_ratio: float = 0.1) -> list[np.ndarray]:
    if context_ratio < 0:
        raise ValueError("context_ratio must be non-negative")
    if not 0.0 < stride_ratio <= 1.0:
        raise ValueError("stride_ratio must be in (0, 1]")
    total = len(frames)
    pad = int(round(total * context_ratio))
    stride = max(1, int(round(total * stride_ratio)))
    padded = np.concatenate(
        [
            np.zeros((pad, frames.shape[1]), dtype=np.float32),
            frames.astype(np.float32, copy=False),
            np.zeros((pad, frames.shape[1]), dtype=np.float32),
        ],
        axis=0,
    )
    windows: list[np.ndarray] = []
    max_start = max(0, len(padded) - total)
    for start in range(0, max_start + 1, stride):
        windows.append(padded[start : start + total].astype(np.float32, copy=True))
    if not windows:
        windows.append(resample_sequence(padded, total))
    elif max_start % stride != 0:
        windows.append(padded[max_start : max_start + total].astype(np.float32, copy=True))
    return windows
