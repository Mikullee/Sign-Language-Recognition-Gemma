#!/usr/bin/env python3
"""Inventory and MediaPipe quality validation for the legacy42 and knee42 corpora.

This program deliberately contains no model training, evaluation, tuning, or checkpoint
creation.  It only reads source videos and writes CSV/JSON reports plus re-usable private
landmark caches.  Source videos are never renamed, moved, or altered.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
POSE_N = 33
HAND_N = 21
POSE_DIM = POSE_N * 4  # x,y,z,visibility; hands have x,y,z only
HAND_DIM = HAND_N * 3
VISIBILITY_MIN = 0.50

LABELS = [
    "你好", "早安", "晚安", "謝謝", "對不起", "再見", "請再說一次", "請慢一點", "我聽不懂", "我知道",
    "我不知道", "可以", "不可以", "我要喝水", "我要上廁所", "我肚子餓", "我累了", "我不舒服", "請幫我", "我要看醫生",
    "現在幾點", "今天星期幾", "你叫什麼名字", "你住哪裡", "多少錢", "太貴了", "我不要", "我要這個", "台北", "新北",
    "桃園", "台中", "台南", "高雄", "新竹", "宜蘭", "花蓮", "我住在台北", "我在新竹上班", "你住在宜蘭嗎", "我明天要去花蓮", "我是桃園人",
]
K42 = {f"K42_{number:02d}": text for number, text in enumerate(LABELS, start=1)}
LEGACY_TO_K42 = {
    **{f"DAILY_{number:02d}": f"K42_{number:02d}" for number in range(1, 24)},
    "DAILY_25": "K42_24", "DAILY_27": "K42_25", "DAILY_28": "K42_26", "DAILY_29": "K42_27", "DAILY_30": "K42_28",
    "PLACE_08": "K42_29", "PLACE_09": "K42_30", "PLACE_10": "K42_31", "PLACE_13": "K42_32", "PLACE_18": "K42_33",
    "PLACE_19": "K42_34", "PLACE_11": "K42_35", "PLACE_21": "K42_36", "PLACE_22": "K42_37",
    "GEOSEQ_31": "K42_38", "GEOSEQ_32": "K42_39", "GEOSEQ_33": "K42_40", "GEOSEQ_34": "K42_41", "GEOSEQ_35": "K42_42",
}
LEGACY_LOOKUP = {(kind, int(index)): target for key, target in LEGACY_TO_K42.items() for kind, index in [key.split("_")]}
KNEE_NAME_RE = re.compile(r"(?:^|[_\-\s])(?:K42[_\-]?(?P<label>0?[1-9]|[1-3]\d|4[0-2])|(?P<label2>0?[1-9]|[1-3]\d|4[0-2]))(?:[_\-\s]|$)", re.I)
LEGACY_RE = re.compile(r"(?:DAILY|PLACE|GEOSEQ)[_\- ]?(\d{1,2})|(?:^|[_\- ])([SP])(\d{2})(?:[_\- ]|$)", re.I)
SIGNER_RE = re.compile(r"(?:^|[_\-])(?:SLR[_\-])?(?:[A-Z]+[_\-])?(?P<signer>[HJLPX])(?:[_\-]|$)", re.I)
TRIAL_RE = re.compile(r"(?:^|[_\-])(?:T|TRIAL|TAKE)[_\-]?(?P<trial>\d{1,3})(?:[_\-]|\.|$)", re.I)


@dataclass
class VideoRow:
    sample_id: str
    source: str
    label_id: str
    display_text: str
    signer_id: str
    trial_id: str
    original_file_path: str
    relative_path: str
    video_name: str
    file_size_bytes: int
    sha256: str
    fps: float | str
    width: int | str
    height: int | str
    duration_sec: float | str
    decode_status: str
    reported_frame_count: int | str
    decoded_frame_count: int | str
    parse_issues: str
    source_label: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(source: str, relative_path: str) -> str:
    return hashlib.sha256(f"{source}:{relative_path.replace(os.sep, '/') }".encode()).hexdigest()[:24]


def split_issues(items: Iterable[str]) -> str:
    return ";".join(sorted(set(item for item in items if item)))


def read_manifest(path: Path | None) -> dict[str, dict[str, str]]:
    """Index optional legacy/new manifest rows by normalized relative path and basename."""
    if path is None:
        return {}
    records: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            normalized = (row.get("relative_path") or row.get("file_path") or row.get("video_name") or "").replace("\\", "/").lstrip("/")
            if normalized:
                records[normalized] = row
                records[Path(normalized).name] = row
    return records


def manifest_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def infer_legacy_label(path: Path, manifest_row: dict[str, str]) -> tuple[str, str, list[str]]:
    issues: list[str] = []
    explicit = manifest_value(manifest_row, "legacy_label_id", "source_label", "unified_label_id", "label_id")
    haystack = " ".join([explicit, *path.parts, path.stem]).upper()
    direct = re.search(r"\b(DAILY|PLACE|GEOSEQ)[_\- ]?(\d{1,2})\b", haystack)
    kind, raw = (direct.group(1), int(direct.group(2))) if direct else ("", -1)
    if not direct:
        # S01/S08 normally denotes DAILY/PLACE, while P01–P05 denotes GEOSEQ 31–35.
        compact = LEGACY_RE.search(haystack)
        if compact and compact.group(2):
            prefix, raw = compact.group(2).upper(), int(compact.group(3))
            if prefix == "P":
                kind, raw = "GEOSEQ", raw + 30
            else:
                # Sxx is ambiguous without the parent source directory: never guess.
                parent = " ".join(path.parts).upper()
                if "DAILY" in parent:
                    kind = "DAILY"
                elif "PLACE" in parent:
                    kind = "PLACE"
                else:
                    issues.append("ambiguous_legacy_S_index")
    target = LEGACY_LOOKUP.get((kind, raw))
    if not target:
        issues.append("legacy_label_not_in_explicit_k42_mapping")
        return "", "", issues
    return target, f"{kind}_{raw:02d}", issues


def infer_knee_label(path: Path, manifest_row: dict[str, str]) -> tuple[str, list[str]]:
    issues: list[str] = []
    supplied = manifest_value(manifest_row, "label_id", "class_id", "label")
    candidates = [supplied, *path.parts, path.stem]
    label = ""
    for value in candidates:
        match = re.search(r"K42[_\- ]?(\d{1,2})\b", value, re.I)
        if match:
            label = f"K42_{int(match.group(1)):02d}"
            break
    if not label:
        # The supplied Drive uses folders such as ``02.早安``.  A bare number is
        # deliberately insufficient: it must be paired with the official text.
        for part in path.parts:
            match = re.match(r"^\s*(\d{1,2})[._\-\s]+(.+?)\s*$", part)
            if match and 1 <= int(match.group(1)) <= 42:
                candidate = int(match.group(1))
                expected = K42[f"K42_{candidate:02d}"]
                if match.group(2) == expected:
                    label = f"K42_{candidate:02d}"
                    break
    if not label:
        issues.append("knee42_label_unparseable")
    file_label = re.search(r"(?:^|[_\-])L(\d{1,2})(?:[_\-]|$)", path.stem, re.I)
    if file_label:
        filename_label = f"K42_{int(file_label.group(1)):02d}"
        if filename_label not in K42:
            issues.append("filename_label_out_of_range")
        elif label and filename_label != label:
            issues.append(f"filename_folder_label_mismatch:{filename_label}->{label}")
    return label, issues


def infer_signer_trial(path: Path, manifest_row: dict[str, str]) -> tuple[str, str, list[str]]:
    issues: list[str] = []
    signer = manifest_value(manifest_row, "signer_id", "signer", "person_id").upper()
    trial = manifest_value(manifest_row, "trial_id", "take_id", "trial", "take")
    if not signer:
        match = SIGNER_RE.search(path.stem)
        signer = match.group("signer").upper() if match else ""
    if not trial:
        match = TRIAL_RE.search(path.stem)
        trial = str(int(match.group("trial"))).zfill(2) if match else ""
    if not signer:
        issues.append("signer_unparseable")
    if not trial:
        issues.append("trial_unparseable")
    return signer, trial, issues


def inspect_video(path: Path) -> tuple[dict[str, Any], str]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"fps": "", "width": "", "height": "", "duration_sec": "", "reported_frame_count": "", "decoded_frame_count": ""}, "OPEN_FAILED"
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    decoded = 0
    try:
        while True:
            ok, _frame = cap.read()
            if not ok:
                break
            decoded += 1
    except cv2.error:
        cap.release()
        return {"fps": fps, "width": width, "height": height, "duration_sec": decoded / fps if fps else "", "reported_frame_count": reported, "decoded_frame_count": decoded}, "DECODE_EXCEPTION"
    finally:
        cap.release()
    status = "OK" if decoded else "EMPTY_OR_UNREADABLE"
    # Some phone encoders write an inaccurate container frame count.  We have
    # nevertheless decoded the stream until clean EOF, so report this as a
    # metadata anomaly rather than claiming the source is corrupt.
    if status == "OK" and reported > 0 and decoded < reported * 0.95:
        status = "OK_FRAME_COUNT_METADATA_MISMATCH"
    return {"fps": fps, "width": width, "height": height, "duration_sec": decoded / fps if fps else "", "reported_frame_count": reported, "decoded_frame_count": decoded}, status


def inventory(source: str, root: Path, manifest: dict[str, dict[str, str]]) -> list[VideoRow]:
    rows: list[VideoRow] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS):
        relative = path.relative_to(root).as_posix()
        manifest_row = manifest.get(relative, manifest.get(path.name, {}))
        if source == "legacy42":
            label, source_label, label_issues = infer_legacy_label(path, manifest_row)
        else:
            label, label_issues = infer_knee_label(path, manifest_row)
            source_label = manifest_value(manifest_row, "label_id", "class_id", "label")
            if not source_label:
                match = re.search(r"(?:^|[_\-])L(\d{1,2})(?:[_\-]|$)", path.stem, re.I)
                source_label = f"L{int(match.group(1)):02d}" if match else ""
        signer, trial, id_issues = infer_signer_trial(path, manifest_row)
        if source == "knee42" and signer:
            folder_signers = {
                match.group(1).upper()
                for part in path.parts
                for match in [re.match(r"^([HJLPX])(?:\s|_|-|$)", part, re.I)]
                if match
            }
            if folder_signers and signer not in folder_signers:
                id_issues.append("filename_folder_signer_mismatch")
        metadata, decode = inspect_video(path)
        issues = [*label_issues, *id_issues]
        if label and K42[label] not in " ".join(path.parts):
            issues.append("canonical_display_text_not_in_path")
        if source == "knee42" and signer and signer not in {"H", "J", "L", "P", "X"}:
            issues.append("unexpected_knee42_signer")
        rows.append(VideoRow(
            sample_id=stable_id(source, relative), source=source, label_id=label, display_text=K42.get(label, ""), signer_id=signer, trial_id=trial,
            original_file_path=str(path.resolve()), relative_path=relative, video_name=path.name, file_size_bytes=path.stat().st_size, sha256=sha256_file(path),
            decode_status=decode, parse_issues=split_issues(issues), source_label=source_label, **metadata,
        ))
    return rows


def save_csv(path: Path, rows: list[dict[str, Any]] | list[VideoRow], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(row) if isinstance(row, VideoRow) else row for row in rows]
    if fields is None:
        # Failed landmark rows intentionally have fewer metrics; retain every column.
        fields = list(dict.fromkeys(key for row in data for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


def empty_raw() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.zeros((POSE_N, 4), np.float32), np.zeros((HAND_N, 3), np.float32), np.zeros((HAND_N, 3), np.float32)


def pose_array(result: Any) -> np.ndarray:
    values, _, _ = empty_raw()
    items = getattr(result, "pose_landmarks", [])
    if items:
        for index, lm in enumerate(items[0][:POSE_N]):
            values[index] = [lm.x, lm.y, lm.z, getattr(lm, "visibility", 0.0)]
    return values


def hand_arrays(result: Any) -> tuple[np.ndarray, np.ndarray]:
    _, left, right = empty_raw()
    for handedness, landmarks in zip(getattr(result, "handedness", []), getattr(result, "hand_landmarks", [])):
        target = left if handedness and handedness[0].category_name.lower() == "left" else right
        for index, lm in enumerate(landmarks[:HAND_N]):
            target[index] = [lm.x, lm.y, lm.z]
    return left, right


class Extractor:
    def __init__(self, hand_model: Path, pose_model: Path, frame_step: int):
        self.hand_model, self.pose_model, self.frame_step = hand_model, pose_model, frame_step
        self.hands: HandLandmarker | None = None
        self.pose: PoseLandmarker | None = None

    def __enter__(self) -> "Extractor":
        self.hands = HandLandmarker.create_from_options(HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(self.hand_model)), running_mode=VisionTaskRunningMode.IMAGE, num_hands=2))
        self.pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(self.pose_model)), running_mode=VisionTaskRunningMode.IMAGE))
        return self

    def __exit__(self, *_: Any) -> None:
        if self.hands: self.hands.close()
        if self.pose: self.pose.close()

    def run(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        assert self.hands and self.pose
        cap = cv2.VideoCapture(str(path)); poses: list[np.ndarray] = []; lefts: list[np.ndarray] = []; rights: list[np.ndarray] = []; total = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok: break
                total += 1
                if (total - 1) % self.frame_step: continue
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                poses.append(pose_array(self.pose.detect(image)))
                left, right = hand_arrays(self.hands.detect(image)); lefts.append(left); rights.append(right)
        finally:
            cap.release()
        if not poses: raise RuntimeError("no readable frames for landmark extraction")
        return np.stack(poses), np.stack(lefts), np.stack(rights), total


def longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0; best = max(best, current)
    return best


def safe_rate(mask: np.ndarray) -> float:
    return round(float(np.mean(mask)) if len(mask) else 0.0, 6)


def movement_noise(poses: np.ndarray, visible: np.ndarray) -> tuple[float, float]:
    centers = (poses[:, 11, :2] + poses[:, 12, :2]) / 2
    both = visible[:, 11] & visible[:, 12]
    moves = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    valid_moves = both[1:] & both[:-1]
    if not np.any(valid_moves): return 0.0, 0.0
    values = moves[valid_moves]; base = float(np.median(values) + 1e-6)
    edge = max(1, len(moves) // 10)
    edge_value = float(np.median(np.concatenate([moves[:edge], moves[-edge:]])))
    return round(float(np.percentile(values, 95)), 6), round(edge_value / base, 3)


def cache_path(feature_root: Path, row: VideoRow) -> Path:
    return feature_root / row.source / "features" / f"{row.sample_id}.npz"


def write_cache(path: Path, row: VideoRow, pose: np.ndarray, left: np.ndarray, right: np.ndarray, original_frames: int, frame_step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle: temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, cache_version="knee42_landmarks_v1", source_sha256=row.sha256, pose=pose, left_hand=left, right_hand=right, original_frame_count=original_frames, frame_step=frame_step)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def cached(path: Path, sha: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    try:
        with np.load(path, allow_pickle=False) as values:
            if str(values["cache_version"].item()) != "knee42_landmarks_v1" or str(values["source_sha256"].item()) != sha: return None
            return values["pose"], values["left_hand"], values["right_hand"], int(values["original_frame_count"].item())
    except (OSError, KeyError, ValueError):
        return None


def quality_from_landmarks(row: VideoRow, pose: np.ndarray, left: np.ndarray, right: np.ndarray, total_frames: int, frame_step: int) -> dict[str, Any]:
    visible = pose[:, :, 3] >= VISIBILITY_MIN
    has_pose, has_left, has_right = np.any(visible, axis=1), np.any(left, axis=(1, 2)), np.any(right, axis=(1, 2))
    both_hands, all_zero = has_left & has_right, ~(has_pose | has_left | has_right)
    shoulder = visible[:, 11] & visible[:, 12]; wrist = visible[:, 15] & visible[:, 16]; hip = visible[:, 23] & visible[:, 24]; knee = visible[:, 25] & visible[:, 26]
    effective_fps = float(row.fps) / frame_step if isinstance(row.fps, float) and row.fps > 0 else 0.0
    p95_motion, edge_noise_ratio = movement_noise(pose, visible)
    issues: list[str] = [item for item in row.parse_issues.split(";") if item]
    if row.decode_status != "OK": issues.append(f"decode_{row.decode_status.lower()}")
    if row.file_size_bytes == 0: issues.append("empty_file")
    duration = float(row.duration_sec) if isinstance(row.duration_sec, float) else 0.0
    if duration < 0.5: issues.append("extremely_short_video")
    if safe_rate(all_zero) > 0.30: issues.append("high_all_zero_feature_ratio")
    if safe_rate(has_pose) < 0.55: issues.append("person_often_out_of_frame")
    if p95_motion > 0.08: issues.append("possible_camera_shake")
    if edge_noise_ratio > 5.0: issues.append("boundary_motion_noise")
    missing_seconds = longest_run(~(has_left | has_right)) / effective_fps if effective_fps else 0.0
    if row.source == "knee42":
        if safe_rate(shoulder) < 0.65: issues.append("low_shoulder_visibility")
        if safe_rate(has_left | has_right) < 0.65: issues.append("low_hand_visibility")
        if safe_rate(wrist) < 0.55: issues.append("low_wrist_visibility")
        if safe_rate(hip) < 0.65: issues.append("low_hip_visibility")
        if safe_rate(knee) < 0.65: issues.append("low_knee_visibility")
        edge = max(1, len(knee) // 10)
        if safe_rate(knee[:edge]) < 0.60: issues.append("start_not_knee_visible")
        if safe_rate(knee[-edge:]) < 0.60: issues.append("end_not_knee_visible")
    else:
        elbow = visible[:, 13] & visible[:, 14]
        if safe_rate(shoulder) < 0.55: issues.append("low_shoulder_visibility")
        if safe_rate(elbow) < 0.45: issues.append("low_elbow_visibility")
        if safe_rate(wrist) < 0.45: issues.append("low_wrist_visibility")
        if missing_seconds > 1.5: issues.append("long_hand_missing")
    reject_terms = {"empty_file", "extremely_short_video", "high_all_zero_feature_ratio", "person_often_out_of_frame", "decode_open_failed", "decode_empty_or_unreadable", "decode_decode_exception", "low_hip_visibility", "low_knee_visibility"}
    review_terms = {"low_shoulder_visibility", "low_hand_visibility", "low_wrist_visibility", "start_not_knee_visible", "end_not_knee_visible", "low_elbow_visibility", "long_hand_missing", "boundary_motion_noise", "possible_camera_shake"}
    state = "REJECT_CANDIDATE" if any(item in reject_terms or item.startswith("legacy_label_not") or item.endswith("unparseable") for item in issues) else "REVIEW" if any(item in review_terms for item in issues) else "PASS"
    return {
        "sample_id": row.sample_id, "source": row.source, "label_id": row.label_id, "display_text": row.display_text, "signer_id": row.signer_id, "trial_id": row.trial_id,
        "total_frame_count": total_frames, "feature_frame_count": len(pose), "successful_feature_frame_count": int(np.sum(~all_zero)),
        "pose_success_rate": safe_rate(has_pose), "left_hand_success_rate": safe_rate(has_left), "right_hand_success_rate": safe_rate(has_right), "both_hands_visible_rate": safe_rate(both_hands),
        "shoulder_visible_rate": safe_rate(shoulder), "wrist_visible_rate": safe_rate(wrist), "hip_visible_rate": safe_rate(hip), "knee_visible_rate": safe_rate(knee),
        "all_zero_feature_frame_rate": safe_rate(all_zero), "longest_hand_missing_sec": round(missing_seconds, 4), "feature_sequence_length": len(pose),
        "p95_shoulder_motion": p95_motion, "boundary_motion_noise_ratio": edge_noise_ratio, "quality_status": state, "quality_issues": split_issues(issues), "feature_cache_path": str(cache_path(Path("."), row)),
    }


def extract_quality(rows: list[VideoRow], feature_root: Path, extractor: Extractor, frame_step: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        target = cache_path(feature_root, row)
        try:
            content = cached(target, row.sha256)
            if content is None:
                content = extractor.run(Path(row.original_file_path)); write_cache(target, row, *content, frame_step)
            quality = quality_from_landmarks(row, *content, frame_step)
            quality["feature_cache_path"] = str(target)
        except Exception as exc:
            quality = {"sample_id": row.sample_id, "source": row.source, "label_id": row.label_id, "display_text": row.display_text, "signer_id": row.signer_id, "trial_id": row.trial_id, "quality_status": "REJECT_CANDIDATE", "quality_issues": f"feature_extraction_failed:{type(exc).__name__}:{exc}", "feature_cache_path": str(target)}
        output.append(quality)
        if index % 25 == 0 or index == len(rows): print(json.dumps({"source": row.source, "feature_progress": index, "total": len(rows)}, ensure_ascii=False), flush=True)
    return output


def apply_duplicates(inventory_rows: list[VideoRow], quality_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sha: dict[str, list[VideoRow]] = defaultdict(list)
    for row in inventory_rows:
        if row.sha256: by_sha[row.sha256].append(row)
    quality_by_id = {row["sample_id"]: row for row in quality_rows}
    report: list[dict[str, Any]] = []
    for sha, members in sorted(by_sha.items()):
        if len(members) < 2: continue
        canonical = sorted(members, key=lambda item: (item.source, item.relative_path))[0]
        for member in members:
            report.append({"sha256": sha, "sample_id": member.sample_id, "source": member.source, "relative_path": member.relative_path, "canonical_sample_id": canonical.sample_id, "is_canonical": member is canonical, "duplicate_group_size": len(members)})
            if member is not canonical:
                item = quality_by_id[member.sample_id]
                item["quality_issues"] = split_issues([item.get("quality_issues", ""), f"duplicate_content_of:{canonical.sample_id}"])
                item["quality_status"] = "REJECT_CANDIDATE"
    return report


def apply_identity_anomalies(inventory_rows: list[VideoRow], quality_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag duplicate label/signer/trial identities without altering source files."""
    by_identity: dict[tuple[str, str, str, str], list[VideoRow]] = defaultdict(list)
    by_name: dict[tuple[str, str], list[VideoRow]] = defaultdict(list)
    for row in inventory_rows:
        by_name[(row.source, row.video_name.lower())].append(row)
        if row.label_id and row.signer_id and row.trial_id:
            by_identity[(row.source, row.label_id, row.signer_id, row.trial_id)].append(row)
    qmap = {row["sample_id"]: row for row in quality_rows}
    report: list[dict[str, Any]] = []
    for key, members in sorted(by_identity.items()):
        if len(members) > 1:
            for member in members:
                item = qmap[member.sample_id]
                item["quality_issues"] = split_issues([item.get("quality_issues", ""), "duplicate_label_signer_trial"])
                if item.get("quality_status") == "PASS": item["quality_status"] = "REVIEW"
                report.append({"issue": "duplicate_label_signer_trial", "sample_id": member.sample_id, "source": key[0], "label_id": key[1], "signer_id": key[2], "trial_id": key[3], "relative_path": member.relative_path, "group_size": len(members)})
    for (source, name), members in sorted(by_name.items()):
        if len(members) > 1:
            for member in members:
                report.append({"issue": "duplicate_video_name", "sample_id": member.sample_id, "source": source, "label_id": member.label_id, "signer_id": member.signer_id, "trial_id": member.trial_id, "relative_path": member.relative_path, "group_size": len(members)})
    for row in inventory_rows:
        if row.source != "knee42" or not row.trial_id:
            continue
        try: valid_trial = 1 <= int(row.trial_id) <= 15
        except ValueError: valid_trial = False
        if not valid_trial:
            item = qmap[row.sample_id]
            item["quality_issues"] = split_issues([item.get("quality_issues", ""), "trial_outside_01_15"])
            if item.get("quality_status") == "PASS": item["quality_status"] = "REVIEW"
            report.append({"issue": "trial_outside_01_15", "sample_id": row.sample_id, "source": row.source, "label_id": row.label_id, "signer_id": row.signer_id, "trial_id": row.trial_id, "relative_path": row.relative_path, "group_size": 1})
    return report


def quality_report(rows: list[VideoRow], quality: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    qmap = {row["sample_id"]: row for row in quality}
    count_by_key: Counter[tuple[str, str]] = Counter((row.label_id, row.signer_id) for row in rows)
    report: list[dict[str, Any]] = []
    for label_id, text in K42.items():
        members = [row for row in rows if row.label_id == label_id]
        statuses = Counter(qmap[row.sample_id].get("quality_status", "REJECT_CANDIDATE") for row in members)
        record: dict[str, Any] = {"source": source, "row_type": "label", "label_id": label_id, "display_text": text, "discovered_videos": len(members), "PASS": statuses["PASS"], "REVIEW": statuses["REVIEW"], "REJECT_CANDIDATE": statuses["REJECT_CANDIDATE"], "expected_per_signer": 15 if source == "knee42" else ""}
        report.append(record)
        for signer in (["H", "J", "L", "P", "X"] if source == "knee42" else sorted({row.signer_id for row in members if row.signer_id})):
            individual = [row for row in members if row.signer_id == signer]; individual_statuses = Counter(qmap[row.sample_id].get("quality_status", "REJECT_CANDIDATE") for row in individual)
            report.append({**record, "row_type": "label_signer", "signer_id": signer, "discovered_videos": len(individual), "PASS": individual_statuses["PASS"], "REVIEW": individual_statuses["REVIEW"], "REJECT_CANDIDATE": individual_statuses["REJECT_CANDIDATE"], "count_issue": "" if source != "knee42" or len(individual) == 15 else f"expected_15_got_{len(individual)}"})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only knee42/legacy42 validation and landmark cache builder (no ML training).")
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--knee-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--feature-root", required=True, type=Path, help="Private Git-ignored dataset_staging directory")
    parser.add_argument("--hand-model", required=True, type=Path)
    parser.add_argument("--pose-model", required=True, type=Path)
    parser.add_argument("--legacy-manifest", type=Path)
    parser.add_argument("--knee-manifest", type=Path)
    parser.add_argument("--frame-step", type=int, default=1)
    args = parser.parse_args()
    if args.frame_step < 1: parser.error("--frame-step must be >= 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    legacy = inventory("legacy42", args.legacy_root.resolve(), read_manifest(args.legacy_manifest))
    knee = inventory("knee42", args.knee_root.resolve(), read_manifest(args.knee_manifest))
    save_csv(args.out_dir / "legacy42_inventory.csv", legacy)
    save_csv(args.out_dir / "knee42_inventory.csv", knee)
    with Extractor(args.hand_model.resolve(), args.pose_model.resolve(), args.frame_step) as extractor:
        legacy_quality = extract_quality(legacy, args.feature_root.resolve(), extractor, args.frame_step)
        knee_quality = extract_quality(knee, args.feature_root.resolve(), extractor, args.frame_step)
    identity_anomalies = apply_identity_anomalies([*legacy, *knee], [*legacy_quality, *knee_quality])
    duplicates = [*identity_anomalies, *apply_duplicates([*legacy, *knee], [*legacy_quality, *knee_quality])]
    save_csv(args.out_dir / "legacy42_feature_quality.csv", legacy_quality)
    save_csv(args.out_dir / "knee42_feature_quality.csv", knee_quality)
    legacy_report, knee_report = quality_report(legacy, legacy_quality, "legacy42"), quality_report(knee, knee_quality, "knee42")
    save_csv(args.out_dir / "legacy42_quality_report.csv", legacy_report)
    save_csv(args.out_dir / "knee42_quality_report.csv", knee_report)
    combined = [row for row in [*legacy, *knee] if next(item for item in [*legacy_quality, *knee_quality] if item["sample_id"] == row.sample_id).get("quality_status") == "PASS"]
    save_csv(args.out_dir / "combined42_inventory.csv", combined)
    save_csv(args.out_dir / "combined42_duplicate_report.csv", duplicates)
    manual = [item for item in [*legacy_quality, *knee_quality] if item.get("quality_status") == "REVIEW"]
    save_csv(args.out_dir / "combined42_manual_review.csv", manual)
    knee_counts = Counter((row.label_id, row.signer_id) for row in knee)
    summary = {"script": "validate_knee42_data.py", "training_or_checkpoints_created": False, "quality_rules_version": "knee42_v1", "legacy42": {"videos": len(legacy), "status_counts": dict(Counter(item.get("quality_status") for item in legacy_quality)), "unmapped": sum(not row.label_id for row in legacy)}, "knee42": {"videos": len(knee), "expected_videos": 3150, "is_complete_3150": len(knee) == 3150 and all(knee_counts[(label, signer)] == 15 for label in K42 for signer in "H J L P X".split()), "status_counts": dict(Counter(item.get("quality_status") for item in knee_quality)), "unmapped": sum(not row.label_id for row in knee), "count_anomalies": [{"label_id": label, "signer_id": signer, "found": knee_counts[(label, signer)]} for label in K42 for signer in "H J L P X".split() if knee_counts[(label, signer)] != 15]}, "combined42": {"pass_videos": len(combined), "manual_review_videos": len(manual), "duplicate_or_identity_issue_rows": len(duplicates)}}
    (args.out_dir / "combined42_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
