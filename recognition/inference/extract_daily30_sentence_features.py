from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions

from recognition.inference.daily30_sentence_feature_utils import build_feature_sequence


POSE_SIZE = 33 * 3
HAND_SIZE = 21 * 3
FEATURE_DIM = POSE_SIZE + HAND_SIZE * 2
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract daily30 sentence landmarks to fixed-length NPZ features.")
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hand-model", required=True)
    ap.add_argument("--pose-model", required=True)
    ap.add_argument("--sequence-length", type=int, default=96)
    ap.add_argument("--frame-step", type=int, default=2)
    ap.add_argument("--append-delta", dest="append_delta", action="store_true")
    ap.add_argument("--no-append-delta", dest="append_delta", action="store_false")
    ap.add_argument("--zscore-features", dest="zscore_features", action="store_true")
    ap.add_argument("--no-zscore-features", dest="zscore_features", action="store_false")
    ap.set_defaults(append_delta=True, zscore_features=True)
    return ap.parse_args()


def extract_frame_vector(hand_result, pose_result) -> np.ndarray:
    values: list[float] = []

    pose_landmarks = getattr(pose_result, "pose_landmarks", [])
    if pose_landmarks:
        for lm in pose_landmarks[0]:
            values.extend([lm.x, lm.y, lm.z])
    else:
        values.extend([0.0] * POSE_SIZE)

    left = [0.0] * HAND_SIZE
    right = [0.0] * HAND_SIZE
    for handedness, landmarks in zip(getattr(hand_result, "handedness", []), getattr(hand_result, "hand_landmarks", [])):
        label = handedness[0].category_name.lower()
        flat: list[float] = []
        for lm in landmarks:
            flat.extend([lm.x, lm.y, lm.z])
        if label == "left":
            left = flat
        else:
            right = flat

    values.extend(left)
    values.extend(right)
    return np.array(values, dtype=np.float32)


def normalize_relative_frames(frames: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(frames, dtype=np.float32)
    for i, frame in enumerate(frames):
        pose = frame[:POSE_SIZE].reshape(33, 3).copy()
        left = frame[POSE_SIZE:POSE_SIZE + HAND_SIZE].reshape(21, 3).copy()
        right = frame[POSE_SIZE + HAND_SIZE:].reshape(21, 3).copy()

        left_shoulder = pose[LEFT_SHOULDER]
        right_shoulder = pose[RIGHT_SHOULDER]
        if np.any(left_shoulder) and np.any(right_shoulder):
            center = (left_shoulder + right_shoulder) / 2.0
            scale = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
        else:
            valid_pose = pose[np.any(pose != 0, axis=1)]
            if len(valid_pose):
                center = valid_pose.mean(axis=0)
                scale = float(np.linalg.norm(np.ptp(valid_pose[:, :2], axis=0)))
            else:
                center = np.array([0.5, 0.5, 0.0], dtype=np.float32)
                scale = 1.0

        scale = max(scale, 1e-3)
        for arr in (pose, left, right):
            valid_mask = np.any(arr != 0, axis=1)
            arr[valid_mask] = (arr[valid_mask] - center) / scale
        normalized[i] = np.concatenate([pose.reshape(-1), left.reshape(-1), right.reshape(-1)]).astype(np.float32)
    return normalized


def resize_seq(seq: np.ndarray, target_len: int) -> np.ndarray:
    if len(seq) == 0:
        return np.zeros((target_len, seq.shape[-1]), dtype=np.float32)
    if len(seq) == target_len:
        return seq.astype(np.float32)
    idx = np.linspace(0, len(seq) - 1, num=target_len)
    idx0 = np.floor(idx).astype(int)
    idx1 = np.ceil(idx).astype(int)
    alpha = idx - idx0
    out = (1.0 - alpha)[:, None] * seq[idx0] + alpha[:, None] * seq[idx1]
    return out.astype(np.float32)


def main() -> None:
    args = parse_args()
    split_rows = list(csv.DictReader(Path(args.split_csv).open("r", encoding="utf-8-sig", newline="")))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(Path(args.hand_model).resolve())),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_hands=2,
    )
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(Path(args.pose_model).resolve())),
        running_mode=VisionTaskRunningMode.IMAGE,
    )

    with (
        HandLandmarker.create_from_options(hand_options) as hand_landmarker,
        PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
    ):
        for row in tqdm(split_rows, desc="extract_sentence_features"):
            video_path = Path(row["file_path"])
            if not video_path.exists():
                continue

            cap = cv2.VideoCapture(str(video_path))
            frame_vectors: list[np.ndarray] = []
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % args.frame_step != 0:
                    frame_idx += 1
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                frame_vectors.append(
                    extract_frame_vector(
                        hand_landmarker.detect(mp_image),
                        pose_landmarker.detect(mp_image),
                    )
                )
                frame_idx += 1
            cap.release()

            if not frame_vectors:
                continue

            raw_frames = np.stack(frame_vectors, axis=0)
            rel_frames = normalize_relative_frames(raw_frames)
            feat_arr = resize_seq(rel_frames, args.sequence_length)
            feat_arr = build_feature_sequence(
                feat_arr,
                append_delta=bool(args.append_delta),
                zscore_features=bool(args.zscore_features),
            )

            np.savez_compressed(
                out_dir / f"{Path(row['video_name']).stem}.npz",
                feature=feat_arr,
                num_frames_raw=int(raw_frames.shape[0]),
                video_name=row["video_name"],
                template_id=row["template_id"],
                signer_id=row["signer_id"],
                split=row["split"],
            )


if __name__ == "__main__":
    main()
