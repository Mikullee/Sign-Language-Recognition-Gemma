"""Audit recordings for AUTO-triggered Knee42 recognition segments."""
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from recognition.realtime.knee42_controllers import SegmentEvidence


CSV_FIELDS = (
    "segment_index",
    "clip_start_sec",
    "clip_end_sec",
    "finalize_sec",
    "reason",
    "rest_detected_sec",
    "boundary_policy",
    "top1_label",
    "top1_text",
    "top1_confidence",
    "top3_json",
    "exact_clip",
    "context_clip",
    "metadata_json",
)


@dataclass(frozen=True)
class RecordingSummary:
    session_dir: Path
    source_path: Path
    segment_count: int
    frame_count: int


def frame_bounds(
    source_origin_sec: float,
    clip_start_sec: float,
    clip_end_sec: float,
    fps: float,
    total_frames: int,
) -> tuple[int, int]:
    """Return clamped [start, end) frame bounds, including the end timestamp."""
    if fps <= 0 or total_frames < 0:
        raise ValueError("fps must be positive and total_frames cannot be negative")
    start = math.floor((float(clip_start_sec) - float(source_origin_sec)) * fps + 1e-7)
    end = math.floor((float(clip_end_sec) - float(source_origin_sec)) * fps + 1e-7) + 1
    return max(0, min(total_frames, start)), max(0, min(total_frames, end))


def _safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return cleaned or "unknown"


class SegmentSessionRecorder:
    """Stream a session source, then materialize exact/context AUTO clips."""

    def __init__(
        self,
        output_root: Path,
        *,
        fps: float,
        frame_size: tuple[int, int],
        source_origin_sec: float,
        context_sec: float = 1.0,
        now: Callable[[], str] | None = None,
        cv2_module: Any | None = None,
    ):
        if fps <= 0 or frame_size[0] <= 0 or frame_size[1] <= 0 or context_sec < 0:
            raise ValueError("invalid recording geometry or timing")
        if cv2_module is None:
            import cv2 as cv2_module
        self.cv2 = cv2_module
        self.fps = float(fps)
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.source_origin_sec = float(source_origin_sec)
        self.context_sec = float(context_sec)
        stamp = (now or (lambda: datetime.now().strftime("%Y%m%d_%H%M%S")))()
        self.session_dir = self._create_unique_session(Path(output_root), stamp)
        for name in ("segments", "context", "metadata"):
            (self.session_dir / name).mkdir()
        self.source_path, self._writer, self._codec = self._open_writer(
            self.session_dir / "session_source"
        )
        self._csv_handle = (self.session_dir / "segments.csv").open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=CSV_FIELDS)
        self._csv_writer.writeheader()
        self._jsonl_handle = (self.session_dir / "segments.jsonl").open("w", encoding="utf-8")
        self._segments: list[dict[str, Any]] = []
        self._frame_count = 0
        self._summary: RecordingSummary | None = None

    @staticmethod
    def _create_unique_session(root: Path, stamp: str) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        for suffix in range(1000):
            name = f"Knee42-session-{stamp}" if suffix == 0 else f"Knee42-session-{stamp}_{suffix:02d}"
            candidate = root / name
            try:
                candidate.mkdir()
                return candidate
            except FileExistsError:
                continue
        raise RuntimeError("cannot allocate a unique recording session directory")

    def _open_writer(self, base: Path):
        for extension, fourcc_name in ((".mp4", "mp4v"), (".avi", "MJPG")):
            path = base.with_suffix(extension)
            writer = self.cv2.VideoWriter(
                str(path),
                self.cv2.VideoWriter_fourcc(*fourcc_name),
                self.fps,
                self.frame_size,
            )
            if writer.isOpened():
                return path, writer, fourcc_name
            writer.release()
        raise RuntimeError(f"cannot open a video writer under {self.session_dir}")

    @property
    def segment_count(self) -> int:
        return len(self._segments)

    def add_frame(self, frame_bgr: np.ndarray) -> None:
        if self._summary is not None:
            raise RuntimeError("recording session is already stopped")
        frame = np.asarray(frame_bgr)
        expected = (self.frame_size[1], self.frame_size[0], 3)
        if frame.shape != expected:
            raise ValueError(f"expected BGR frame shape {expected}, got {frame.shape}")
        self._writer.write(frame)
        self._frame_count += 1

    def record_segment(self, evidence: SegmentEvidence, result: Any) -> None:
        if self._summary is not None:
            raise RuntimeError("recording session is already stopped")
        index = len(self._segments) + 1
        label = _safe_label(result.top1.label_id)
        source_extension = self.source_path.suffix
        exact_rel = Path("segments") / f"segment_{index:04d}_{label}{source_extension}"
        context_rel = Path("context") / f"segment_{index:04d}_{label}_context{source_extension}"
        metadata_rel = Path("metadata") / f"segment_{index:04d}.json"
        top3 = [
            {
                "label_id": str(item.label_id),
                "display_text": str(item.display_text),
                "confidence": float(item.confidence),
            }
            for item in result.top3
        ]
        payload = {
            "segment_index": index,
            **asdict(evidence),
            "top1": top3[0],
            "top3": top3,
            "source_origin_sec": self.source_origin_sec,
            "source_fps": self.fps,
            "source_video": self.source_path.name,
            "exact_clip": exact_rel.as_posix(),
            "context_clip": context_rel.as_posix(),
        }
        row = {
            "segment_index": index,
            "clip_start_sec": f"{evidence.clip_start_sec:.6f}",
            "clip_end_sec": f"{evidence.clip_end_sec:.6f}",
            "finalize_sec": f"{evidence.finalize_sec:.6f}",
            "reason": evidence.reason,
            "rest_detected_sec": (
                ""
                if evidence.rest_detected_sec is None
                else f"{evidence.rest_detected_sec:.6f}"
            ),
            "boundary_policy": evidence.boundary_policy,
            "top1_label": top3[0]["label_id"],
            "top1_text": top3[0]["display_text"],
            "top1_confidence": f"{top3[0]['confidence']:.9f}",
            "top3_json": json.dumps(top3, ensure_ascii=False, separators=(",", ":")),
            "exact_clip": exact_rel.as_posix(),
            "context_clip": context_rel.as_posix(),
            "metadata_json": metadata_rel.as_posix(),
        }
        (self.session_dir / metadata_rel).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        self._csv_writer.writerow(row)
        self._csv_handle.flush()
        self._jsonl_handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._jsonl_handle.flush()
        self._segments.append(payload)

    def stop(self) -> RecordingSummary:
        if self._summary is not None:
            return self._summary
        self._writer.release()
        self._csv_handle.close()
        self._jsonl_handle.close()
        for segment in self._segments:
            self._materialize(segment, context=False)
            self._materialize(segment, context=True)
        self._summary = RecordingSummary(
            session_dir=self.session_dir,
            source_path=self.source_path,
            segment_count=len(self._segments),
            frame_count=self._frame_count,
        )
        (self.session_dir / "session_summary.json").write_text(
            json.dumps(
                {
                    "session_dir": self.session_dir.name,
                    "source_video": self.source_path.name,
                    "source_origin_sec": self.source_origin_sec,
                    "source_fps": self.fps,
                    "frame_count": self._frame_count,
                    "segment_count": len(self._segments),
                    "status": "FINALIZED",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return self._summary

    def _materialize(self, segment: dict[str, Any], *, context: bool) -> None:
        padding = self.context_sec if context else 0.0
        start, end = frame_bounds(
            self.source_origin_sec,
            float(segment["clip_start_sec"]) - padding,
            float(segment["clip_end_sec"]) + padding,
            self.fps,
            self._frame_count,
        )
        destination = self.session_dir / (
            segment["context_clip"] if context else segment["exact_clip"]
        )
        capture = self.cv2.VideoCapture(str(self.source_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot reopen session source {self.source_path}")
        writer = self.cv2.VideoWriter(
            str(destination),
            self.cv2.VideoWriter_fourcc(*self._codec),
            self.fps,
            self.frame_size,
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"cannot create evidence clip {destination}")
        try:
            capture.set(self.cv2.CAP_PROP_POS_FRAMES, start)
            for _index in range(start, end):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"session source ended while creating {destination}")
                writer.write(frame)
        finally:
            writer.release()
            capture.release()
