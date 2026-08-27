"""Audit recordings for AUTO-triggered Knee42 recognition segments."""
from __future__ import annotations

import csv
import json
import math
import re
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from recognition.realtime.knee42_clock import MAX_PRACTICAL_FPS
from recognition.realtime.probability_reporting import (
    probability_policy_record,
    validate_raw_probability,
)


CSV_FIELDS = (
    "segment_index",
    "clip_start_sec",
    "clip_end_sec",
    "finalize_sec",
    "reason",
    "rest_detected_sec",
    "boundary_policy",
    "decision_reason",
    "boundary_decision_json",
    "runtime_context_json",
    "top1_label",
    "top1_text",
    "top1_raw_probability",
    "top3_json",
    "exact_clip",
    "context_clip",
    "metadata_json",
)


class SegmentEvidenceLike(Protocol):
    clip_start_sec: float
    clip_end_sec: float
    finalize_sec: float
    reason: str
    rest_detected_sec: float | None
    boundary_policy: str


@runtime_checkable
class BoundaryDecisionTelemetryLike(Protocol):
    clip_start_sec: float
    clip_end_sec: float
    finalize_sec: float
    finalize_reason: str
    rest_detected_sec: float | None
    boundary_policy: str

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class _RecordingSegmentEvidence:
    clip_start_sec: float
    clip_end_sec: float
    finalize_sec: float
    reason: str
    rest_detected_sec: float | None
    boundary_policy: str


@dataclass(frozen=True)
class RecordingSummary:
    session_dir: Path
    source_path: Path
    segment_count: int
    frame_count: int


@dataclass(frozen=True)
class RecordingRuntimeContext:
    """Immutable capture identity; later release stages bind optional root assets."""

    clock_mode: str
    resolved_rotation: int
    input_mirror: bool
    display_mirror: bool
    trigger_config_sha256: str
    trigger_provenance_sha256: str
    release_root_manifest_sha256: str | None = None
    model_component_manifest_sha256: str | None = None
    hand_task_sha256: str | None = None
    pose_task_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.clock_mode, str) or not self.clock_mode.strip():
            raise ValueError("clock_mode must be a non-empty string")
        if type(self.resolved_rotation) is not int or self.resolved_rotation not in {
            0,
            90,
            180,
            270,
        }:
            raise ValueError("resolved_rotation must be one of 0, 90, 180, 270")
        for name in ("input_mirror", "display_mirror"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        for name in (
            "trigger_config_sha256",
            "trigger_provenance_sha256",
            "release_root_manifest_sha256",
            "model_component_manifest_sha256",
            "hand_task_sha256",
            "pose_task_sha256",
        ):
            value = getattr(self, name)
            if value is None and name not in {
                "trigger_config_sha256",
                "trigger_provenance_sha256",
            }:
                continue
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def timestamp_frame_bounds(
    frame_timestamps_sec: Sequence[float],
    clip_start_sec: float,
    clip_end_sec: float,
) -> tuple[int, int]:
    """Return [left, right) bounds using exact captured-frame timestamps."""
    return (
        bisect_left(frame_timestamps_sec, float(clip_start_sec)),
        bisect_right(frame_timestamps_sec, float(clip_end_sec)),
    )


def _safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return cleaned or "unknown"


def _strict_json_dumps(
    payload: object,
    *,
    indent: int | None = None,
    compact: bool = False,
) -> str:
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "allow_nan": False,
    }
    if indent is not None:
        options["indent"] = indent
    if compact:
        options["separators"] = (",", ":")
    try:
        return json.dumps(payload, **options)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "recording JSON must be serializable and contain only finite numbers"
        ) from error


def _atomic_write_text(path: Path, payload: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _segment_evidence_payload(evidence: SegmentEvidenceLike) -> dict[str, Any]:
    return {
        "clip_start_sec": evidence.clip_start_sec,
        "clip_end_sec": evidence.clip_end_sec,
        "finalize_sec": evidence.finalize_sec,
        "reason": evidence.reason,
        "rest_detected_sec": evidence.rest_detected_sec,
        "boundary_policy": evidence.boundary_policy,
    }


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
        runtime_context: RecordingRuntimeContext | None = None,
        now: Callable[[], str] | None = None,
        cv2_module: Any | None = None,
    ):
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0.0 or fps > MAX_PRACTICAL_FPS:
            raise ValueError(
                "recording source FPS must be finite and in the practical range "
                f"(0, {MAX_PRACTICAL_FPS:g}]"
            )
        if frame_size[0] <= 0 or frame_size[1] <= 0 or context_sec < 0:
            raise ValueError("invalid recording geometry or timing")
        if cv2_module is None:
            import cv2 as cv2_module
        self.cv2 = cv2_module
        self.fps = fps
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.source_origin_sec = float(source_origin_sec)
        self.context_sec = float(context_sec)
        if runtime_context is not None and not isinstance(
            runtime_context, RecordingRuntimeContext
        ):
            raise TypeError("runtime_context must be RecordingRuntimeContext or None")
        self.runtime_context = runtime_context
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
        self._frame_timestamps_sec: list[float] = []
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

    def add_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        timestamp_sec: float | None = None,
    ) -> None:
        if self._summary is not None:
            raise RuntimeError("recording session is already stopped")
        frame = np.asarray(frame_bgr)
        expected = (self.frame_size[1], self.frame_size[0], 3)
        if frame.shape != expected:
            raise ValueError(f"expected BGR frame shape {expected}, got {frame.shape}")
        if timestamp_sec is None:
            timestamp_sec = self.source_origin_sec + self._frame_count / self.fps
        timestamp_sec = float(timestamp_sec)
        if not math.isfinite(timestamp_sec):
            raise ValueError("recording frame timestamp must be finite")
        if (
            self._frame_timestamps_sec
            and timestamp_sec < self._frame_timestamps_sec[-1]
        ):
            raise ValueError("recording frame timestamps must be nondecreasing")
        self._writer.write(frame)
        self._frame_timestamps_sec.append(timestamp_sec)
        self._frame_count += 1

    def record_segment(self, evidence: SegmentEvidenceLike, result: Any) -> None:
        self._record_boundary(evidence, result, boundary_decision=None)

    def record_boundary_decision(
        self,
        decision: BoundaryDecisionTelemetryLike,
        result: Any | None = None,
    ) -> None:
        if not isinstance(decision, BoundaryDecisionTelemetryLike):
            raise TypeError("decision must satisfy the boundary telemetry contract")
        self._record_boundary(
            _RecordingSegmentEvidence(
                clip_start_sec=decision.clip_start_sec,
                clip_end_sec=decision.clip_end_sec,
                finalize_sec=decision.finalize_sec,
                reason=decision.finalize_reason,
                rest_detected_sec=decision.rest_detected_sec,
                boundary_policy=decision.boundary_policy,
            ),
            result,
            boundary_decision=decision,
        )

    def _record_boundary(
        self,
        evidence: SegmentEvidenceLike,
        result: Any | None,
        *,
        boundary_decision: BoundaryDecisionTelemetryLike | None,
    ) -> None:
        if self._summary is not None:
            raise RuntimeError("recording session is already stopped")
        index = len(self._segments) + 1
        top3 = []
        if result is not None:
            top3 = [
                {
                    "label_id": str(item.label_id),
                    "display_text": str(item.display_text),
                    "raw_probability": validate_raw_probability(item.raw_probability),
                }
                for item in result.top3
            ]
            if not top3:
                raise ValueError("prediction result top3 cannot be empty")
        label = _safe_label(top3[0]["label_id"] if top3 else "dropped")
        source_extension = self.source_path.suffix
        exact_rel = Path("segments") / f"segment_{index:04d}_{label}{source_extension}"
        context_rel = Path("context") / f"segment_{index:04d}_{label}_context{source_extension}"
        metadata_rel = Path("metadata") / f"segment_{index:04d}.json"
        decision_payload = (
            None if boundary_decision is None else boundary_decision.to_dict()
        )
        runtime_context_payload = (
            None if self.runtime_context is None else self.runtime_context.to_dict()
        )
        payload = {
            "segment_index": index,
            **_segment_evidence_payload(evidence),
            "top1": top3[0] if top3 else None,
            "top3": top3,
            "boundary_decision": decision_payload,
            "probability_policy": probability_policy_record(),
            "source_origin_sec": self.source_origin_sec,
            "source_fps": self.fps,
            "source_video": self.source_path.name,
            "exact_clip": exact_rel.as_posix(),
            "context_clip": context_rel.as_posix(),
            "metadata_json": metadata_rel.as_posix(),
            "runtime_context": runtime_context_payload,
        }
        row = {
            "segment_index": index,
            "clip_start_sec": float(evidence.clip_start_sec),
            "clip_end_sec": float(evidence.clip_end_sec),
            "finalize_sec": float(evidence.finalize_sec),
            "reason": evidence.reason,
            "rest_detected_sec": (
                ""
                if evidence.rest_detected_sec is None
                else float(evidence.rest_detected_sec)
            ),
            "boundary_policy": evidence.boundary_policy,
            "decision_reason": (
                "" if boundary_decision is None else boundary_decision.decision_reason
            ),
            "boundary_decision_json": _strict_json_dumps(
                decision_payload, compact=True
            ),
            "runtime_context_json": _strict_json_dumps(
                runtime_context_payload, compact=True
            ),
            "top1_label": top3[0]["label_id"] if top3 else "",
            "top1_text": top3[0]["display_text"] if top3 else "",
            "top1_raw_probability": top3[0]["raw_probability"] if top3 else "",
            "top3_json": _strict_json_dumps(top3, compact=True),
            "exact_clip": exact_rel.as_posix(),
            "context_clip": context_rel.as_posix(),
            "metadata_json": metadata_rel.as_posix(),
        }
        metadata_payload = _strict_json_dumps(payload, indent=2) + "\n"
        jsonl_payload = _strict_json_dumps(payload, compact=True) + "\n"
        _atomic_write_text(self.session_dir / metadata_rel, metadata_payload)
        self._csv_writer.writerow(row)
        self._csv_handle.flush()
        self._jsonl_handle.write(jsonl_payload)
        self._jsonl_handle.flush()
        self._segments.append(payload)

    def stop(self) -> RecordingSummary:
        if self._summary is not None:
            return self._summary
        finalized_segments: list[dict[str, Any]] = []
        metadata_documents: list[tuple[Path, str]] = []
        for segment in self._segments:
            finalized = dict(segment)
            finalized["clock_timestamps"] = self._clock_timestamps(finalized)
            finalized_segments.append(finalized)
            metadata_documents.append(
                (
                    self.session_dir / str(finalized["metadata_json"]),
                    _strict_json_dumps(finalized, indent=2) + "\n",
                )
            )
        segments_document = "".join(
            _strict_json_dumps(segment, compact=True) + "\n"
            for segment in finalized_segments
        )
        summary = RecordingSummary(
            session_dir=self.session_dir,
            source_path=self.source_path,
            segment_count=len(finalized_segments),
            frame_count=self._frame_count,
        )
        summary_document = _strict_json_dumps(
            {
                "session_dir": self.session_dir.name,
                "source_video": self.source_path.name,
                "source_origin_sec": self.source_origin_sec,
                "source_fps": self.fps,
                "frame_count": self._frame_count,
                "frame_timestamps_sec": self._frame_timestamps_sec,
                "segment_count": len(finalized_segments),
                "probability_policy": probability_policy_record(),
                "runtime_context": (
                    None
                    if self.runtime_context is None
                    else self.runtime_context.to_dict()
                ),
                "status": "FINALIZED",
            },
            indent=2,
        ) + "\n"
        self._writer.release()
        self._csv_handle.close()
        self._jsonl_handle.close()
        for metadata_path, metadata_document in metadata_documents:
            _atomic_write_text(metadata_path, metadata_document)
        for segment in finalized_segments:
            self._materialize(segment, context=False)
            self._materialize(segment, context=True)
        _atomic_write_text(
            self.session_dir / "segments.jsonl",
            segments_document,
        )
        _atomic_write_text(
            self.session_dir / "session_summary.json",
            summary_document,
        )
        self._segments = finalized_segments
        self._summary = summary
        return summary

    def _clock_timestamps(self, segment: dict[str, Any]) -> dict[str, Any]:
        exact_start, exact_end = timestamp_frame_bounds(
            self._frame_timestamps_sec,
            float(segment["clip_start_sec"]),
            float(segment["clip_end_sec"]),
        )
        context_start, context_end = timestamp_frame_bounds(
            self._frame_timestamps_sec,
            float(segment["clip_start_sec"]) - self.context_sec,
            float(segment["clip_end_sec"]) + self.context_sec,
        )
        return {
            "selection_basis": "captured_frame_timestamps_sec",
            "session_frame_timestamps_sec": list(self._frame_timestamps_sec),
            "exact_frame_bounds": [exact_start, exact_end],
            "exact_frame_timestamps_sec": self._frame_timestamps_sec[
                exact_start:exact_end
            ],
            "context_frame_bounds": [context_start, context_end],
            "context_frame_timestamps_sec": self._frame_timestamps_sec[
                context_start:context_end
            ],
        }

    def _materialize(self, segment: dict[str, Any], *, context: bool) -> None:
        padding = self.context_sec if context else 0.0
        start, end = timestamp_frame_bounds(
            self._frame_timestamps_sec,
            float(segment["clip_start_sec"]) - padding,
            float(segment["clip_end_sec"]) + padding,
        )
        destination = self.session_dir / (
            segment["context_clip"] if context else segment["exact_clip"]
        )
        temporary = destination.with_name(
            f"{destination.stem}.tmp{destination.suffix}"
        )
        capture = self.cv2.VideoCapture(str(self.source_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot reopen session source {self.source_path}")
        writer = None
        try:
            writer = self.cv2.VideoWriter(
                str(temporary),
                self.cv2.VideoWriter_fourcc(*self._codec),
                self.fps,
                self.frame_size,
            )
            if not writer.isOpened():
                raise RuntimeError(f"cannot create evidence clip {destination}")
            capture.set(self.cv2.CAP_PROP_POS_FRAMES, start)
            for _index in range(start, end):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"session source ended while creating {destination}")
                writer.write(frame)
            writer.release()
            writer = None
            capture.release()
            temporary.replace(destination)
        except BaseException:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        finally:
            if writer is not None:
                writer.release()
            capture.release()
