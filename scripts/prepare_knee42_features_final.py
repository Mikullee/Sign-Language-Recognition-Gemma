#!/usr/bin/env python3
"""Create the single Knee42 landmark contract used for every split and experiment.

The cache deliberately contains no knee coordinates or knee-specific masks.  Raw
missing landmarks remain NaN; the trainer standardizes observed values and only
then replaces missing values with the neutral value (zero), while retaining the
matching non-knee coordinate mask as model input.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.realtime.knee42_capture import open_video  # noqa: E402


POSE_KEEP = tuple(index for index in range(33) if index not in (25, 26))
POSE_COORDS = 3
HAND_LANDMARKS = 21
LANDMARK_DIM = len(POSE_KEEP) * POSE_COORDS + 2 * HAND_LANDMARKS * POSE_COORDS
CACHE_VERSION = "knee42_features_upright_v2"

FEATURE_CONFIG = {
    "cache_version": CACHE_VERSION,
    "extractor": "mediapipe_tasks_image_mode",
    "frame_step": 2,
    "pose_landmarks_kept": list(POSE_KEEP),
    "pose_landmarks_removed": [25, 26],
    "pose_fields": ["x", "y", "z"],
    "hand_fields": ["x", "y", "z"],
    "landmark_value_dim": LANDMARK_DIM,
    "model_input": "standardized_values_then_neutral_fill_concatenated_with_non_knee_coordinate_mask",
    "forbidden_inputs": ["knee_coordinates", "knee_visibility", "knee_presence", "knee_masks", "knee_quality_status", "candidate_flag", "signer_missingness_pattern"],
    "video_orientation": "container_rotation_metadata_applied_explicitly",
    "horizontal_mirror": False,
}
SCHEMA_SHA256 = hashlib.sha256(json.dumps(FEATURE_CONFIG, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def cache_path(out_dir: Path, sample_id: str) -> Path:
    if not sample_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in sample_id):
        raise ValueError(f"unsafe sample_id: {sample_id!r}")
    return out_dir / f"{sample_id}.npz"


def cache_matches(
    path: Path,
    source_sha256: str,
    extractor_commit: str,
    hand_model_sha256: str,
    pose_model_sha256: str,
) -> bool:
    try:
        with np.load(path, allow_pickle=False) as payload:
            return (
                str(payload["cache_version"].item()) == CACHE_VERSION
                and str(payload["source_sha256"].item()) == source_sha256
                and str(payload["schema_sha256"].item()) == SCHEMA_SHA256
                and payload["values"].ndim == 2
                and payload["values"].shape[1] == LANDMARK_DIM
                and payload["mask"].shape == payload["values"].shape
                and int(payload["rotation_metadata_degrees"].item()) in {0, 90, 180, 270}
                and not bool(payload["horizontal_mirror"].item())
                and str(payload["extractor_commit"].item()) == extractor_commit
                and str(payload["hand_model_sha256"].item()) == hand_model_sha256
                and str(payload["pose_model_sha256"].item()) == pose_model_sha256
            )
    except (OSError, KeyError, ValueError):
        return False


def _landmark_xyz(landmarks, index: int) -> np.ndarray:
    landmark = landmarks[index]
    return np.asarray([landmark.x, landmark.y, landmark.z], dtype=np.float32)


def extract_frame(hand_result, pose_result) -> tuple[np.ndarray, np.ndarray]:
    pose = np.full((len(POSE_KEEP), 3), np.nan, dtype=np.float32)
    pose_landmarks = getattr(pose_result, "pose_landmarks", [])
    full_pose = pose_landmarks[0] if pose_landmarks else None
    if full_pose:
        for out_index, source_index in enumerate(POSE_KEEP):
            pose[out_index] = _landmark_xyz(full_pose, source_index)

    hands = {"left": np.full((HAND_LANDMARKS, 3), np.nan, dtype=np.float32), "right": np.full((HAND_LANDMARKS, 3), np.nan, dtype=np.float32)}
    for handedness, landmarks in zip(getattr(hand_result, "handedness", []), getattr(hand_result, "hand_landmarks", [])):
        if not handedness or len(landmarks) != HAND_LANDMARKS:
            continue
        label = handedness[0].category_name.lower()
        if label in hands:
            hands[label] = np.stack([_landmark_xyz(landmarks, index) for index in range(HAND_LANDMARKS)])

    values = np.concatenate([pose.reshape(-1), hands["left"].reshape(-1), hands["right"].reshape(-1)]).astype(np.float32)
    return values, np.isfinite(values)


def normalize_frame(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Shoulder-relative normalization without inventing values for missing points."""
    result = values.copy()
    points = result.reshape(-1, 3)
    point_mask = mask.reshape(-1, 3).all(axis=1)
    pose_index = {source: target for target, source in enumerate(POSE_KEEP)}
    left_shoulder = pose_index[11]
    right_shoulder = pose_index[12]
    if point_mask[left_shoulder] and point_mask[right_shoulder]:
        center = (points[left_shoulder] + points[right_shoulder]) / 2.0
        scale = float(np.linalg.norm(points[left_shoulder, :2] - points[right_shoulder, :2]))
    elif np.any(point_mask):
        valid = points[point_mask]
        center = valid.mean(axis=0)
        scale = float(np.linalg.norm(np.ptp(valid[:, :2], axis=0)))
    else:
        return result
    scale = max(scale, 1e-3)
    points[point_mask] = (points[point_mask] - center) / scale
    return points.reshape(-1).astype(np.float32)


def extract_video(video_path: Path, hands: HandLandmarker, pose: PoseLandmarker, frame_step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    capture = open_video(video_path, cv2_module=cv2)
    rotation_degrees = int(round(capture.rotation_degrees)) % 360
    all_values: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                values, mask = extract_frame(hands.detect(image), pose.detect(image))
                all_values.append(normalize_frame(values, mask))
                all_masks.append(mask)
            frame_index += 1
    finally:
        capture.release()
    if not all_values:
        raise RuntimeError(f"no readable frames: {video_path}")
    values = np.stack(all_values).astype(np.float32)
    masks = np.stack(all_masks).astype(np.bool_)
    valid_frames = np.any(masks, axis=1)
    return values, masks, valid_frames, rotation_degrees


def atomic_save(
    path: Path,
    values: np.ndarray,
    mask: np.ndarray,
    frame_mask: np.ndarray,
    source_sha256: str,
    rotation_degrees: int,
    extractor_commit: str,
    hand_model_sha256: str,
    pose_model_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.stem + ".", suffix=".npz", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary,
            values=values,
            mask=mask,
            valid_frame_mask=frame_mask,
            source_sha256=np.asarray(source_sha256),
            cache_version=np.asarray(CACHE_VERSION),
            schema_sha256=np.asarray(SCHEMA_SHA256),
            rotation_metadata_degrees=np.asarray(rotation_degrees, dtype=np.int16),
            horizontal_mirror=np.asarray(False),
            extractor_commit=np.asarray(extractor_commit),
            hand_model_sha256=np.asarray(hand_model_sha256),
            pose_model_sha256=np.asarray(pose_model_sha256),
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one knee-free final feature cache for Knee42 manifests.")
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--hand-model", required=True, type=Path)
    parser.add_argument("--pose-model", required=True, type=Path)
    parser.add_argument("--progress-json", required=True, type=Path)
    parser.add_argument("--feature-config-json", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.frame_step != FEATURE_CONFIG["frame_step"]:
        raise ValueError(f"features_final is fixed at frame_step={FEATURE_CONFIG['frame_step']}")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if len(args.code_commit) != 40 or any(character not in "0123456789abcdef" for character in args.code_commit.lower()):
        raise ValueError("code-commit must be a full 40-character Git SHA")

    by_id: dict[str, dict[str, str]] = {}
    for manifest in args.manifest:
        for row in read_manifest(manifest):
            existing = by_id.setdefault(row["sample_id"], row)
            if existing["sha256"] != row["sha256"] or existing["original_file_path"] != row["original_file_path"]:
                raise ValueError(f"inconsistent manifest row for {row['sample_id']}")
    all_samples = [by_id[key] for key in sorted(by_id)]
    samples = all_samples[args.shard_index :: args.num_shards]
    hand_model_sha256 = sha256_file(args.hand_model)
    pose_model_sha256 = sha256_file(args.pose_model)
    write_json(
        args.feature_config_json,
        {
            **FEATURE_CONFIG,
            "schema_sha256": SCHEMA_SHA256,
            "expected_samples": len(all_samples),
            "parallel_shards": args.num_shards,
            "extractor_commit": args.code_commit.lower(),
            "hand_model_sha256": hand_model_sha256,
            "pose_model_sha256": pose_model_sha256,
        },
    )

    completed = cache_hits = 0
    failures: list[dict[str, str]] = []
    hand_options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(args.hand_model.resolve())), running_mode=VisionTaskRunningMode.IMAGE, num_hands=2)
    pose_options = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(args.pose_model.resolve())), running_mode=VisionTaskRunningMode.IMAGE)
    with HandLandmarker.create_from_options(hand_options) as hands, PoseLandmarker.create_from_options(pose_options) as pose:
        for index, row in enumerate(samples, start=1):
            try:
                target = cache_path(args.out_dir, row["sample_id"])
                if cache_matches(
                    target,
                    row["sha256"],
                    args.code_commit.lower(),
                    hand_model_sha256,
                    pose_model_sha256,
                ):
                    cache_hits += 1
                else:
                    values, mask, frame_mask, rotation_degrees = extract_video(
                        Path(row["original_file_path"]), hands, pose, args.frame_step
                    )
                    if sha256_file(Path(row["original_file_path"])) != row["sha256"]:
                        raise RuntimeError("source SHA-256 differs from manifest")
                    atomic_save(
                        target,
                        values,
                        mask,
                        frame_mask,
                        row["sha256"],
                        rotation_degrees,
                        args.code_commit.lower(),
                        hand_model_sha256,
                        pose_model_sha256,
                    )
                completed += 1
            except Exception as exc:  # keep exact failure inventory for the gate
                failures.append({"sample_id": row["sample_id"], "error": str(exc)})
            if index == 1 or index % 10 == 0 or failures:
                write_json(args.progress_json, {"expected_total": len(all_samples), "expected_shard": len(samples), "processed": index, "completed": completed, "cache_hits": cache_hits, "failed": len(failures), "failures": failures[-100:], "schema_sha256": SCHEMA_SHA256, "shard_index": args.shard_index, "num_shards": args.num_shards})
                print(json.dumps({"shard_index": args.shard_index, "processed": index, "expected_shard": len(samples), "completed": completed, "cache_hits": cache_hits, "failed": len(failures)}, ensure_ascii=False), flush=True)
    write_json(args.progress_json, {"expected_total": len(all_samples), "expected_shard": len(samples), "processed": len(samples), "completed": completed, "cache_hits": cache_hits, "failed": len(failures), "failures": failures, "schema_sha256": SCHEMA_SHA256, "shard_index": args.shard_index, "num_shards": args.num_shards})
    if failures or completed != len(samples):
        raise SystemExit(f"features_final incomplete: {len(failures)} failures")


if __name__ == "__main__":
    main()
