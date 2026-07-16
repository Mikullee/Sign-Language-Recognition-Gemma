from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
SENTENCE_RE = re.compile(r"^(?P<sentence_id>\d{2})_(?P<sentence_text>.+)$")
FILE_RE = re.compile(
    r"^SLR_(?P<signer>[A-Za-z])_S(?P<sentence_id>\d{2})(?:-W(?P<word_id>\d+))?_T(?P<trial>\d+)\.(?P<ext>mp4|mov|avi|mkv|m4v|webm)$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build daily30 fixed-sentence manifest, labels, and split.")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-templates-csv", required=True)
    ap.add_argument("--out-labels-csv", required=True)
    ap.add_argument("--out-split-csv", required=True)
    ap.add_argument("--out-report-json", required=True)
    ap.add_argument("--exclude-sentences", default="24,26")
    ap.add_argument("--split-mode", default="per_video_fallback", choices=["per_video_fallback", "global_signer_holdout"])
    ap.add_argument("--holdout-signer", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", default="0.7,0.15,0.15")
    return ap.parse_args()


def parse_excluded(text: str) -> set[str]:
    excluded = set()
    for token in text.split(","):
        token = token.strip()
        if token:
            excluded.add(token.zfill(2))
    return excluded


def parse_ratios(text: str) -> tuple[float, float, float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) != 3:
        raise ValueError("ratios must contain exactly 3 numbers, e.g. 0.7,0.15,0.15")
    total = sum(values)
    if total <= 0:
        raise ValueError("ratios sum must be positive")
    return tuple(v / total for v in values)


def read_sentence_dirs(data_root: Path, excluded: set[str]) -> list[tuple[str, str, Path]]:
    results: list[tuple[str, str, Path]] = []
    for sentence_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        match = SENTENCE_RE.match(sentence_dir.name)
        if not match:
            continue
        sentence_id = match.group("sentence_id")
        sentence_text = match.group("sentence_text")
        if sentence_id in excluded:
            continue
        results.append((sentence_id, sentence_text, sentence_dir))
    return results


def build_rows(data_root: Path, excluded: set[str]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    templates: list[dict[str, str]] = []
    labels: list[dict[str, str]] = []
    mismatches: list[dict[str, str]] = []

    for sentence_id, sentence_text, sentence_dir in read_sentence_dirs(data_root, excluded):
        template_id = f"T{sentence_id}"
        sequence_id = f"SEQ{sentence_id}"
        templates.append(
            {
                "template_id": template_id,
                "sequence_id": sequence_id,
                "sentence_id": sentence_id,
                "sentence_text": sentence_text,
                "version": "daily30_v1",
                "status": "active",
                "notes": "",
            }
        )

        for video_path in sorted(p for p in sentence_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES):
            signer_id = ""
            trial_id = ""
            naming_ok = True
            match = FILE_RE.match(video_path.name)
            if match:
                signer_id = match.group("signer").upper()
                trial_id = match.group("trial").zfill(2)
                if match.group("sentence_id") != sentence_id:
                    naming_ok = False
                if match.group("word_id"):
                    naming_ok = False
            else:
                naming_ok = False

            row = {
                "video_name": video_path.name,
                "template_id": template_id,
                "sequence_id": sequence_id,
                "sentence_id": sentence_id,
                "sentence_text": sentence_text,
                "file_path": str(video_path.resolve()),
                "relative_path": str(video_path.relative_to(data_root)),
                "signer_id": signer_id,
                "trial_id": trial_id,
                "group_key": f"{sequence_id}|{signer_id}" if signer_id else sequence_id,
                "naming_ok": "1" if naming_ok else "0",
            }
            labels.append(row)
            if not naming_ok:
                mismatches.append(row)

    return templates, labels, mismatches


def split_template_rows(
    rows: list[dict[str, str]],
    ratios: tuple[float, float, float],
    rng: random.Random,
) -> tuple[list[dict[str, str]], bool]:
    n_total = len(rows)
    if n_total == 0:
        return [], False

    train_ratio, dev_ratio, _ = ratios
    n_train = int(round(n_total * train_ratio))
    n_dev = int(round(n_total * dev_ratio))
    if n_total >= 3:
        n_dev = max(1, n_dev)
        n_test = max(1, n_total - n_train - n_dev)
        n_train = max(1, n_total - n_dev - n_test)
    else:
        n_test = max(0, n_total - n_train - n_dev)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["group_key"]].append(row)

    if len(groups) < 3:
        items = list(rows)
        rng.shuffle(items)
        out: list[dict[str, str]] = []
        for i, row in enumerate(items):
            if i < n_train:
                split = "train"
            elif i < n_train + n_dev:
                split = "dev"
            else:
                split = "test"
            new_row = dict(row)
            new_row["split"] = split
            out.append(new_row)
        return out, True

    keys = list(groups.keys())
    rng.shuffle(keys)
    keys.sort(key=lambda k: len(groups[k]), reverse=True)

    count_train = 0
    count_dev = 0
    out = []
    for key in keys:
        block = groups[key]
        if count_train < n_train:
            split = "train"
            count_train += len(block)
        elif count_dev < n_dev:
            split = "dev"
            count_dev += len(block)
        else:
            split = "test"
        for row in block:
            new_row = dict(row)
            new_row["split"] = split
            out.append(new_row)
    return out, False


def choose_holdout_signer(labels: list[dict[str, str]], requested: str, rng: random.Random) -> str:
    signers = sorted({row["signer_id"] for row in labels if row["signer_id"]})
    if requested:
        signer = requested.strip().upper()
        if signer not in signers:
            raise ValueError(f"holdout signer {signer!r} not found in dataset signers {signers}")
        return signer
    if len(signers) < 2:
        raise ValueError("global_signer_holdout requires at least 2 distinct signer_id values")
    candidates = list(signers)
    rng.shuffle(candidates)
    return candidates[0]


def split_holdout_rows(
    rows: list[dict[str, str]],
    rng: random.Random,
) -> list[dict[str, str]]:
    items = list(rows)
    items.sort(key=lambda row: (row["trial_id"], row["video_name"]))
    if len(items) > 1:
        rng.shuffle(items)

    n_dev = max(1, len(items) // 2)
    if n_dev >= len(items):
        n_dev = len(items) - 1
    n_test = len(items) - n_dev
    if n_test <= 0:
        raise ValueError("holdout signer split requires at least 2 videos per template")

    out: list[dict[str, str]] = []
    for i, row in enumerate(items):
        new_row = dict(row)
        new_row["split"] = "dev" if i < n_dev else "test"
        out.append(new_row)
    return out


def split_global_signer_holdout(
    labels: list[dict[str, str]],
    holdout_signer: str,
    rng: random.Random,
) -> tuple[list[dict[str, str]], dict[str, dict[str, int]], dict[str, str]]:
    by_template: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in labels:
        by_template[row["template_id"]].append(row)

    split_rows: list[dict[str, str]] = []
    template_counts: dict[str, dict[str, int]] = {}
    skipped_templates: dict[str, str] = {}
    for template_id in sorted(by_template):
        rows = by_template[template_id]
        train_rows = [row for row in rows if row["signer_id"] and row["signer_id"] != holdout_signer]
        holdout_rows = [row for row in rows if row["signer_id"] == holdout_signer]
        if not train_rows:
            skipped_templates[template_id] = "no_train_rows"
            continue
        if len(holdout_rows) < 2:
            skipped_templates[template_id] = "insufficient_holdout_rows"
            continue

        for row in train_rows:
            new_row = dict(row)
            new_row["split"] = "train"
            split_rows.append(new_row)
        split_rows.extend(split_holdout_rows(holdout_rows, rng))

        counts = Counter(row["split"] for row in split_rows if row["template_id"] == template_id)
        template_counts[template_id] = {
            "total": len(rows),
            "train": counts.get("train", 0),
            "dev": counts.get("dev", 0),
            "test": counts.get("test", 0),
        }

    return split_rows, template_counts, skipped_templates


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(data_root)

    excluded = parse_excluded(args.exclude_sentences)
    ratios = parse_ratios(args.ratios)
    rng = random.Random(args.seed)

    templates, labels, mismatches = build_rows(data_root, excluded)
    if not labels:
        raise RuntimeError(f"No sentence videos found under {data_root}")

    by_template: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in labels:
        by_template[row["template_id"]].append(row)

    split_rows: list[dict[str, str]] = []
    fallback_templates: list[str] = []
    holdout_signer = ""
    skipped_templates: dict[str, str] = {}
    if args.split_mode == "global_signer_holdout":
        holdout_signer = choose_holdout_signer(labels, args.holdout_signer, rng)
        split_rows, template_counts, skipped_templates = split_global_signer_holdout(labels, holdout_signer, rng)
        if not split_rows:
            raise RuntimeError("global_signer_holdout produced no usable rows")
    else:
        for template_id in sorted(by_template):
            rows, used_fallback = split_template_rows(by_template[template_id], ratios, rng)
            split_rows.extend(rows)
            if used_fallback:
                fallback_templates.append(template_id)

    split_rows.sort(key=lambda row: row["video_name"])
    labels_sorted = sorted(labels, key=lambda row: row["video_name"])
    templates_sorted = sorted(templates, key=lambda row: row["template_id"])

    write_csv(
        Path(args.out_templates_csv),
        templates_sorted,
        ["template_id", "sequence_id", "sentence_id", "sentence_text", "version", "status", "notes"],
    )
    write_csv(
        Path(args.out_labels_csv),
        labels_sorted,
        [
            "video_name",
            "template_id",
            "sequence_id",
            "sentence_id",
            "sentence_text",
            "file_path",
            "relative_path",
            "signer_id",
            "trial_id",
            "group_key",
            "naming_ok",
        ],
    )
    write_csv(
        Path(args.out_split_csv),
        split_rows,
        [
            "video_name",
            "template_id",
            "sequence_id",
            "sentence_id",
            "sentence_text",
            "file_path",
            "relative_path",
            "signer_id",
            "trial_id",
            "group_key",
            "naming_ok",
            "split",
        ],
    )

    split_counts = Counter(row["split"] for row in split_rows)
    if args.split_mode != "global_signer_holdout":
        template_counts = {}
        for template_id in sorted(by_template):
            counts = Counter(row["split"] for row in split_rows if row["template_id"] == template_id)
            template_counts[template_id] = {
                "total": len(by_template[template_id]),
                "train": counts.get("train", 0),
                "dev": counts.get("dev", 0),
                "test": counts.get("test", 0),
            }

    report = {
        "data_root": str(data_root),
        "excluded_sentences": sorted(excluded),
        "split_mode": args.split_mode,
        "holdout_signer": holdout_signer,
        "num_templates": len(templates_sorted),
        "num_videos": len(labels_sorted),
        "num_split_videos": len(split_rows),
        "by_split": dict(split_counts),
        "num_naming_mismatches": len(mismatches),
        "fallback_templates": fallback_templates,
        "skipped_templates": skipped_templates,
        "templates": template_counts,
    }
    Path(args.out_report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
