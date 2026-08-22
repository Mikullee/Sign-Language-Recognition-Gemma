#!/usr/bin/env python3
"""Expand reviewed atlases into a locked 2,252-row Knee42 per-video audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_knee42_videos_upright import validate_final_audit


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_exclusive(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preliminary", type=Path, required=True)
    parser.add_argument("--review-template", type=Path, required=True)
    parser.add_argument("--atlas-ledger", type=Path, required=True)
    parser.add_argument("--handedness", type=Path, required=True)
    parser.add_argument("--sample-overrides", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite final audit: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    preliminary = read_csv(args.preliminary)
    template = read_csv(args.review_template)
    atlas_reviews = {row["atlas_path"]: row for row in read_csv(args.atlas_ledger)}
    handedness = {row["sample_id"]: row for row in read_csv(args.handedness)}
    overrides = {row["sample_id"]: row for row in read_csv(args.sample_overrides)} if args.sample_overrides else {}
    preliminary_by_id = {row["sample_id"]: row for row in preliminary}
    template_by_id = {row["sample_id"]: row for row in template}
    if len(preliminary_by_id) != 2252 or set(preliminary_by_id) != set(template_by_id):
        raise ValueError("preliminary/review template must contain the same 2,252 sample IDs")
    expected_atlases = {row["atlas_path"] for row in template}
    missing_atlases = expected_atlases - set(atlas_reviews)
    if missing_atlases:
        raise ValueError(f"{len(missing_atlases)} atlases have not been visually reviewed")
    unknown_overrides = set(overrides) - set(preliminary_by_id)
    if unknown_overrides:
        raise ValueError(f"unknown sample overrides: {sorted(unknown_overrides)[:5]}")
    rows = []
    for sample_id in sorted(preliminary_by_id):
        row = dict(preliminary_by_id[sample_id])
        template_row = template_by_id[sample_id]
        atlas = atlas_reviews[template_row["atlas_path"]]
        hand = handedness.get(sample_id, {})
        visual_status = atlas["status"]
        row.update(
            visual_status=visual_status,
            dominant_hand_visual=hand.get("dominance", "UNKNOWN"),
            gesture_matches_label_visual="CONSISTENT_WITH_GROUP" if visual_status == "PASS" else "REVIEW",
            full_body_hands_face_visible_visual="PASS" if visual_status == "PASS" else "REVIEW",
            sequence_complete_visual="PASS" if visual_status == "PASS" else "REVIEW",
            horizontal_mirror_visual="NO_OBVIOUS_MIRROR" if visual_status == "PASS" else "REVIEW",
            visual_reviewer=atlas["reviewer"],
            visual_evidence_path=template_row["atlas_path"],
            atlas_row=template_row["atlas_row"],
            media_pipe_slot_swap_flag=hand.get("slot_swap_flag", ""),
            orientation_outlier_flag=hand.get("orientation_outlier", ""),
        )
        if sample_id in overrides:
            override = overrides[sample_id]
            for field in (
                "visual_status", "dominant_hand_visual", "gesture_matches_label_visual",
                "full_body_hands_face_visible_visual", "sequence_complete_visual",
                "horizontal_mirror_visual", "visual_reviewer",
            ):
                if override.get(field):
                    row[field] = override[field]
        reasons = [f"automatic={row['automatic_status']}", f"visual={row['visual_status']}", atlas["reason"]]
        if row["automatic_status"] == "FAIL" or row["visual_status"] == "FAIL":
            final = "FAIL"
        elif (
            row["automatic_status"] == "REVIEW"
            or row["visual_status"] == "REVIEW"
            or str(row.get("media_pipe_slot_swap_flag", "")).lower() == "true"
        ):
            final = "REVIEW"
        else:
            final = "PASS"
        row["final_status"] = final
        row["final_reason"] = "; ".join(item for item in reasons if item)
        rows.append(row)
    validate_final_audit(rows)
    audit_path = args.out_dir / "immutable_video_audit.csv"
    write_csv_exclusive(audit_path, rows)
    flagged = [row for row in rows if row["final_status"] != "PASS"]
    if flagged:
        write_csv_exclusive(args.out_dir / "fail_review.csv", flagged)
    summary = {
        "status": "LOCKED",
        "scope": "Train=L/P/X and Dev=H only; Test/J=0",
        "rows": len(rows),
        "final_status_counts": dict(Counter(row["final_status"] for row in rows)),
        "rotation_counts": dict(Counter(row["rotation_metadata_degrees"] for row in rows)),
        "atlas_reviews": len(atlas_reviews),
        "sample_overrides": len(overrides),
        "immutable_video_audit_sha256": sha256_file(audit_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
