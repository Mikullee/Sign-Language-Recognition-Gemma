"""Recognize a recorded video with the Knee42 Transformer bundle.

Tracks the video with MediaPipe, splits it on wrist motion, and prints the top-k
for each segment plus a whole-clip result.  Use it to sanity-check a bundle, or
to score footage without setting up a camera.

Usage
-----
    python scripts/analyze_knee42_video.py clip.mp4
    python scripts/analyze_knee42_video.py clip.mp4 --json result.json

The MediaPipe ``.task`` models must sit in ``models/`` (see models/README.md);
they are not redistributed with this repository.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recognition.config import preview_paths
from recognition.transformer.recognizer import Knee42TransformerRecognizer
from recognition.transformer.segmentation import DEFAULT_SEGMENTATION, analyze_video


def format_result(result: dict) -> str:
    lines = [
        f"frames tracked : {result.get('n_tracked')} in {result.get('track_seconds')}s "
        f"({result.get('track_fps')} fps)",
        f"usable frames  : {result['n_valid']}/{result['n_frames']}",
        f"selfie flip    : {result.get('selfie_flip')}",
        f"normalization  : left shoulder x {result['left_shoulder_x']} "
        f"({'OK' if result['convention_ok'] else 'WRONG -- expected positive'})",
        f"segments       : {len(result['segments'])}",
        "",
    ]
    for segment in result["segments"]:
        head = (
            f"  #{segment['index']:<3} {segment['start']:>6.2f} - {segment['end']:>6.2f}"
            f"  ({segment['duration']:.2f}s, {segment['n_frames']} frames)"
        )
        if segment["skipped"]:
            lines.append(head + "  [too short, skipped]")
            continue
        top = "  ".join(
            f"{item['label']}/{item['text']} {item['prob']:.2f}" for item in segment["top"]
        )
        lines.append(head + "  " + top)
    if result.get("whole"):
        whole = "  ".join(
            f"{item['label']}/{item['text']} {item['prob']:.2f}"
            for item in result["whole"]["top"]
        )
        lines.append(f"\n  whole clip ({result['whole']['n_frames']} frames)  {whole}")
    if result.get("message"):
        lines.append(f"\n  {result['message']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    paths = preview_paths()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--bundle", type=Path, default=paths.runtime_bundle_dir)
    parser.add_argument("--hand-model", type=Path, default=paths.hand_model)
    parser.add_argument("--pose-model", type=Path, default=paths.pose_model)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resize-width", type=int, default=960)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument(
        "--selfie-flip",
        action="store_true",
        help="un-mirror each frame before detection, for selfie-recorded footage",
    )
    parser.add_argument("--motion-threshold", type=float, default=DEFAULT_SEGMENTATION["motion_threshold"])
    parser.add_argument("--min-duration", type=float, default=DEFAULT_SEGMENTATION["min_duration"])
    parser.add_argument("--max-duration", type=float, default=DEFAULT_SEGMENTATION["max_duration"])
    parser.add_argument("--pause", type=float, default=DEFAULT_SEGMENTATION["pause"])
    parser.add_argument("--json", type=Path, default=None, help="also write the raw result here")
    args = parser.parse_args(argv)

    recognizer = Knee42TransformerRecognizer(args.bundle, device=args.device)
    result = analyze_video(
        args.video,
        recognizer,
        hand_model=args.hand_model,
        pose_model=args.pose_model,
        resize_width=args.resize_width,
        max_seconds=args.max_seconds,
        selfie_flip=args.selfie_flip,
        topk=args.topk,
        motion_threshold=args.motion_threshold,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        pause=args.pause,
    )
    print(format_result(result))
    if args.json:
        args.json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nraw result written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
