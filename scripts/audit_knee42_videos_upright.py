#!/usr/bin/env python3
"""Create a resumable, Test/J-free per-video Knee42 upright audit inventory."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.realtime.knee42_capture import apply_video_transform  # noqa: E402

EXPECTED_ROWS = 2252
EXPECTED_SPLITS = {"train": 1634, "dev": 618}
SAMPLE_FRACTIONS = (0.05, 0.25, 0.50, 0.75, 0.95)
TILE_SIZE = (240, 180)
FINAL_STATUSES = {"PASS", "FAIL", "REVIEW"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_research_scope(rows: list[dict[str, str]]) -> None:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} Train/Dev rows, got {len(rows)}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate sample_id in research manifest")
    split_counts = Counter(row["split"].strip().lower() for row in rows)
    if dict(split_counts) != EXPECTED_SPLITS:
        raise ValueError(f"unexpected Train/Dev counts: {dict(split_counts)}")
    forbidden = [
        row["sample_id"]
        for row in rows
        if row["split"].strip().lower() == "test" or row["signer_id"].strip().upper() == "J"
    ]
    if forbidden:
        raise ValueError(f"REFUSED Test/J rows: {forbidden[:5]}")
    allowed = {"train": {"L", "P", "X"}, "dev": {"H"}}
    wrong = [
        row["sample_id"]
        for row in rows
        if row["signer_id"].strip().upper() not in allowed[row["split"].strip().lower()]
    ]
    if wrong:
        raise ValueError(f"split/signer policy violation: {wrong[:5]}")


def validate_final_audit(rows: list[dict[str, str]]) -> None:
    validate_research_scope(rows)
    incomplete = [row["sample_id"] for row in rows if row.get("final_status") not in FINAL_STATUSES]
    if incomplete:
        raise ValueError(f"audit has {len(incomplete)} rows without a final visual conclusion")
    missing_evidence = [
        row["sample_id"]
        for row in rows
        if not row.get("visual_reviewer") or not row.get("visual_evidence_path") or not row.get("final_reason")
    ]
    if missing_evidence:
        raise ValueError(f"audit has {len(missing_evidence)} rows without visual evidence")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized = cv2.resize(frame, (max(1, round(source_width * scale)), max(1, round(source_height * scale))))
    output = np.full((height, width, 3), 24, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    output[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return output


def contact_sheet(path: Path, row: dict[str, str], destination: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return {"decode_opened": False, "sampled_frames": 0, "rotation_metadata_degrees": ""}
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    stored_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    stored_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    rotation = int(round(float(capture.get(cv2.CAP_PROP_ORIENTATION_META) or 0.0))) % 360
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    tiles: list[np.ndarray] = []
    sampled_indices: list[int] = []
    for fraction in SAMPLE_FRACTIONS:
        ok, frame, target = False, None, 0
        for retreat in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20):
            candidate_fraction = max(0.0, fraction - retreat)
            target = max(
                0,
                min(max(frame_count - 1, 0), round(max(frame_count - 1, 0) * candidate_fraction)),
            )
            capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = capture.read()
            if ok:
                break
        if not ok:
            continue
        corrected = apply_video_transform(frame, rotation, horizontal_mirror=False)
        tile = letterbox(corrected, *TILE_SIZE)
        cv2.putText(tile, f"{round(fraction * 100):02d}%", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, f"{round(fraction * 100):02d}%", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
        sampled_indices.append(target)
    capture.release()
    if tiles:
        while len(tiles) < len(SAMPLE_FRACTIONS):
            tiles.append(np.full_like(tiles[0], 24))
        body = np.hstack(tiles)
        header = np.zeros((34, body.shape[1], 3), np.uint8)
        title = (
            f"{row['sample_id']} {row['label_id']} signer={row['signer_id']} "
            f"trial={row['trial_id']} split={row['split']} rotation={rotation}"
        )
        cv2.putText(header, title, (7, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), np.vstack([header, body]), [cv2.IMWRITE_JPEG_QUALITY, 82]):
            raise RuntimeError(f"cannot write contact sheet: {destination}")
    upright_width, upright_height = (stored_height, stored_width) if rotation in {90, 270} else (stored_width, stored_height)
    return {
        "decode_opened": True,
        "frame_count": frame_count,
        "fps": round(fps, 6),
        "stored_width": stored_width,
        "stored_height": stored_height,
        "rotation_metadata_degrees": rotation,
        "upright_width": upright_width,
        "upright_height": upright_height,
        "sampled_frames": len(sampled_indices),
        "sampled_frame_indices": ";".join(map(str, sampled_indices)),
    }


def build_atlases(contact_paths: list[Path], destination: Path, rows_per_atlas: int = 8) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for start in range(0, len(contact_paths), rows_per_atlas):
        images = [cv2.imread(str(path)) for path in contact_paths[start : start + rows_per_atlas]]
        images = [image for image in images if image is not None]
        if not images:
            continue
        width = max(image.shape[1] for image in images)
        padded = []
        for image in images:
            if image.shape[1] < width:
                image = cv2.copyMakeBorder(image, 0, 0, 0, width - image.shape[1], cv2.BORDER_CONSTANT, value=(16, 16, 16))
            padded.append(image)
        atlas = np.vstack(padded)
        target = destination / f"atlas_{count + 1:04d}.jpg"
        if not cv2.imwrite(str(target), atlas, [cv2.IMWRITE_JPEG_QUALITY, 86]):
            raise RuntimeError(f"cannot write atlas: {target}")
        count += 1
    return count


def visual_order_key(row: dict[str, Any]) -> tuple[int, str, int, str]:
    try:
        label_number = int(str(row["label_id"]).rsplit("_", 1)[-1])
    except ValueError:
        label_number = 999
    try:
        trial_number = int(row["trial_id"])
    except ValueError:
        trial_number = 999
    return label_number, str(row["signer_id"]), trial_number, str(row["sample_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite audit directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    validate_research_scope(manifest_rows)
    sha_counts = Counter(row["sha256"] for row in manifest_rows)
    audit_path = args.out_dir / "preliminary_video_audit.csv"
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and audit_path.is_file():
        with audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
            completed = {row["sample_id"]: row for row in csv.DictReader(handle)}
    results: list[dict[str, Any]] = []
    for index, row in enumerate(manifest_rows, 1):
        if row["sample_id"] in completed:
            results.append(completed[row["sample_id"]])
            continue
        path = Path(row["original_file_path"])
        sheet_rel = Path("contact_sheets") / f"{row['sample_id']}.jpg"
        exists = path.is_file()
        actual_sha = sha256_file(path) if exists else ""
        media = contact_sheet(path, row, args.out_dir / sheet_rel) if exists else {"decode_opened": False, "sampled_frames": 0, "rotation_metadata_degrees": ""}
        reasons = []
        if not exists:
            reasons.append("missing_file")
        elif actual_sha != row["sha256"]:
            reasons.append("sha256_mismatch")
        if not media.get("decode_opened"):
            reasons.append("decode_failed")
        if int(media.get("sampled_frames", 0)) != len(SAMPLE_FRACTIONS):
            reasons.append("incomplete_contact_sheet")
        if row["display_text"] not in path.parent.name:
            reasons.append("label_text_not_in_parent_directory")
        if sha_counts[row["sha256"]] > 1:
            reasons.append("duplicate_content_sha256")
        automatic_status = "FAIL" if any(item in reasons for item in ("missing_file", "sha256_mismatch", "decode_failed")) else "REVIEW" if reasons else "PASS_AUTOMATIC"
        results.append({
            **{key: row[key] for key in ("sample_id", "label_id", "display_text", "signer_id", "trial_id", "split", "original_file_path", "relative_path", "sha256")},
            "actual_sha256": actual_sha,
            **media,
            "horizontal_mirror_applied": False,
            "content_sha256_occurrences": sha_counts[row["sha256"]],
            "automatic_status": automatic_status,
            "automatic_reasons": ";".join(reasons),
            "visual_status": "PENDING",
            "dominant_hand_visual": "PENDING",
            "gesture_matches_label_visual": "PENDING",
            "full_body_hands_face_visible_visual": "PENDING",
            "sequence_complete_visual": "PENDING",
            "horizontal_mirror_visual": "PENDING",
            "visual_reviewer": "",
            "visual_evidence_path": sheet_rel.as_posix(),
            "final_status": "PENDING",
            "final_reason": "",
        })
        if index % 25 == 0:
            atomic_csv(audit_path, results)
            atomic_json(args.out_dir / "progress.json", {"completed": len(results), "expected": EXPECTED_ROWS})
            print(f"progress {len(results)}/{EXPECTED_ROWS}", flush=True)
    atomic_csv(audit_path, results)
    visual_rows = sorted(results, key=visual_order_key)
    contact_paths = [
        args.out_dir / row["visual_evidence_path"]
        for row in visual_rows
        if (args.out_dir / row["visual_evidence_path"]).is_file()
    ]
    rows_per_atlas = 12
    atlas_count = build_atlases(contact_paths, args.out_dir / "atlases_by_label", rows_per_atlas)
    review_template = []
    for visual_index, row in enumerate(visual_rows):
        review_template.append({
            "atlas_path": f"atlases_by_label/atlas_{visual_index // rows_per_atlas + 1:04d}.jpg",
            "atlas_row": visual_index % rows_per_atlas + 1,
            "sample_id": row["sample_id"],
            "label_id": row["label_id"],
            "display_text": row["display_text"],
            "signer_id": row["signer_id"],
            "trial_id": row["trial_id"],
            "visual_status": "PENDING",
            "dominant_hand_visual": "PENDING",
            "gesture_matches_label_visual": "PENDING",
            "full_body_hands_face_visible_visual": "PENDING",
            "sequence_complete_visual": "PENDING",
            "horizontal_mirror_visual": "PENDING",
            "visual_reviewer": "",
            "visual_reason": "",
        })
    atomic_csv(args.out_dir / "atlas_review_template.csv", review_template)
    summary = {
        "status": "PRELIMINARY_VISUAL_REVIEW_REQUIRED",
        "scope": "Train=L/P/X and Dev=H only; Test/J=0",
        "rows": len(results),
        "split_counts": dict(Counter(row["split"] for row in results)),
        "automatic_status_counts": dict(Counter(row["automatic_status"] for row in results)),
        "rotation_counts": dict(Counter(str(row["rotation_metadata_degrees"]) for row in results)),
        "contact_sheets": len(contact_paths),
        "atlases": atlas_count,
        "manifest_sha256": sha256_file(args.manifest),
    }
    atomic_json(args.out_dir / "preliminary_summary.json", summary)
    atomic_json(args.out_dir / "progress.json", {"completed": len(results), "expected": EXPECTED_ROWS, "contact_sheets": len(contact_paths), "atlases": atlas_count})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
