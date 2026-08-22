from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions

from recognition.config import preview_paths
from recognition.inference.extract_daily30_sentence_features import extract_frame_vector
from recognition.realtime.auto_trigger import (
    AutoTriggerConfig,
    AutoTriggerEngine,
    SegmentResult,
    analyze_frame_vector,
    load_auto_trigger_config,
)
from recognition.realtime.personal_temporal import PersonalTemporalModel, PersonalTemporalPredictor, with_temporal_probability


PATHS = preview_paths()
DEFAULT_CACHE_DIR = Path("artifacts") / "realtime" / "best_current"
DEFAULT_END_SEARCH_GRID = {
    "end_hold_sec": [0.25, 0.35, 0.50],
    "end_rest_vote_ratio": [0.75, 0.80, 0.90],
    "blank_motion_threshold": [0.018, 0.024, 0.030],
    "knee_lateral_thigh_margin_ratio": [0.55, 0.80, 1.05],
    "knee_min_thigh_progress_ratio": [-1.25, -1.00, -0.75],
    "reference_rest_distance_threshold": [0.18, 0.28, 0.38],
}
DEFAULT_START_SEARCH_GRID = {
    "start_motion_threshold": [0.015, 0.024, 0.032],
    "start_hold_sec": [0.07, 0.13, 0.20],
    "pre_roll_sec": [0.10, 0.18, 0.25],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and calibrate fixed-sentence auto-trigger boundaries."
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument(
        "--landmark-cache-dir",
        default=str(PATHS.results_dir / "auto_trigger_landmark_cache"),
    )
    parser.add_argument("--base-auto-config", default="")
    parser.add_argument("--boundary-tolerance-sec", type=float, default=0.30)
    parser.add_argument("--detector-frame-skip", type=int, default=2)
    parser.add_argument("--inference-max-width", type=int, default=960)
    parser.add_argument("--model-cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument("--skip-debug-videos", action="store_true")
    return parser.parse_args(argv)


@dataclass(frozen=True)
class VideoAnnotation:
    video_path: Path
    expected_label: str
    start_sec: float
    end_sec: float
    notes: str


@dataclass(frozen=True)
class VideoLandmarkCache:
    video_path: Path
    fps: float
    frame_width: int
    frame_height: int
    timestamps_sec: np.ndarray
    frame_vectors: np.ndarray


@dataclass(frozen=True)
class VideoBoundaryMetrics:
    video_path: Path
    expected_label: str
    gt_start_sec: float
    gt_end_sec: float
    predicted_start_sec: float | None
    predicted_end_sec: float | None
    finalize_sec: float | None
    start_error_sec: float | None
    end_error_sec: float | None
    finalize_delay_sec: float | None
    segment_count: int
    extra_segment_count: int
    false_start_before_gt: bool
    passed: bool
    finalize_reason: str
    notes: str

    @classmethod
    def missing(cls, annotation: VideoAnnotation) -> "VideoBoundaryMetrics":
        return cls(
            video_path=annotation.video_path,
            expected_label=annotation.expected_label,
            gt_start_sec=annotation.start_sec,
            gt_end_sec=annotation.end_sec,
            predicted_start_sec=None,
            predicted_end_sec=None,
            finalize_sec=None,
            start_error_sec=None,
            end_error_sec=None,
            finalize_delay_sec=None,
            segment_count=0,
            extra_segment_count=0,
            false_start_before_gt=False,
            passed=False,
            finalize_reason="missing_segment",
            notes=annotation.notes,
        )

    @classmethod
    def from_prediction(
        cls,
        annotation: VideoAnnotation,
        prediction: SegmentResult,
        tolerance_sec: float,
        segment_count: int = 1,
    ) -> "VideoBoundaryMetrics":
        start_error = prediction.clip_start_sec - annotation.start_sec
        end_error = prediction.clip_end_sec - annotation.end_sec
        false_start = prediction.clip_start_sec < annotation.start_sec - tolerance_sec
        return cls(
            video_path=annotation.video_path,
            expected_label=annotation.expected_label,
            gt_start_sec=annotation.start_sec,
            gt_end_sec=annotation.end_sec,
            predicted_start_sec=prediction.clip_start_sec,
            predicted_end_sec=prediction.clip_end_sec,
            finalize_sec=prediction.finalize_sec,
            start_error_sec=start_error,
            end_error_sec=end_error,
            finalize_delay_sec=prediction.finalize_sec - prediction.clip_end_sec,
            segment_count=segment_count,
            extra_segment_count=max(0, segment_count - 1),
            false_start_before_gt=false_start,
            passed=bool(
                segment_count == 1
                and not false_start
                and abs(start_error) <= tolerance_sec
                and abs(end_error) <= tolerance_sec
            ),
            finalize_reason=prediction.reason,
            notes=annotation.notes,
        )

    def to_row(self) -> dict[str, object]:
        return {
            # Reports may leave the private processing host.  A filename is
            # sufficient to identify these fixed benchmark clips and avoids
            # exporting an account-specific absolute path.
            "video_path": self.video_path.name,
            "expected_label": self.expected_label,
            "gt_start_sec": self.gt_start_sec,
            "gt_end_sec": self.gt_end_sec,
            "predicted_start_sec": self.predicted_start_sec,
            "predicted_end_sec": self.predicted_end_sec,
            "finalize_sec": self.finalize_sec,
            "start_error_sec": self.start_error_sec,
            "end_error_sec": self.end_error_sec,
            "finalize_delay_sec": self.finalize_delay_sec,
            "segment_count": self.segment_count,
            "extra_segment_count": self.extra_segment_count,
            "false_start_before_gt": self.false_start_before_gt,
            "passed": self.passed,
            "finalize_reason": self.finalize_reason,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class CandidateEvaluation:
    config: AutoTriggerConfig
    per_video: list[VideoBoundaryMetrics]

    def score(self) -> tuple[float, ...]:
        structural_failures = sum(
            metric.segment_count != 1 or metric.false_start_before_gt
            for metric in self.per_video
        )
        end_errors = [
            abs(metric.end_error_sec)
            for metric in self.per_video
            if metric.end_error_sec is not None
        ]
        start_errors = [
            abs(metric.start_error_sec)
            for metric in self.per_video
            if metric.start_error_sec is not None
        ]
        delays = [
            metric.finalize_delay_sec
            for metric in self.per_video
            if metric.finalize_delay_sec is not None
        ]
        default = AutoTriggerConfig()
        config_distance = sum(
            abs(float(getattr(self.config, key)) - float(getattr(default, key)))
            for key in DEFAULT_END_SEARCH_GRID | DEFAULT_START_SEARCH_GRID
        )
        return (
            float(structural_failures),
            max(end_errors, default=math.inf),
            float(np.mean(end_errors)) if end_errors else math.inf,
            float(np.mean(start_errors)) if start_errors else math.inf,
            float(np.mean(delays)) if delays else math.inf,
            config_distance,
        )


@dataclass(frozen=True)
class ClassificationComparison:
    video_path: Path
    expected_label: str
    manual_label: str
    manual_confidence: float
    auto_label: str
    auto_confidence: float
    manual_correct: bool
    auto_correct: bool
    segmentation_regression: bool
    model_issue: bool

    def to_row(self) -> dict[str, object]:
        payload = asdict(self)
        payload["video_path"] = self.video_path.name
        return payload


def load_annotations(csv_path: str | Path) -> list[VideoAnnotation]:
    path = Path(csv_path).resolve()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"video_path", "expected_label", "start_sec", "end_sec", "notes"}
    if not rows:
        raise ValueError("Annotation CSV is empty.")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Annotation CSV is missing columns: {', '.join(sorted(missing))}")
    annotations = []
    for row_index, row in enumerate(rows, start=2):
        video_path = Path(row["video_path"])
        if not video_path.is_absolute():
            video_path = path.parent / video_path
        video_path = video_path.resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Annotation row {row_index} video does not exist: {video_path}")
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        if start_sec < 0 or end_sec <= start_sec:
            raise ValueError(f"Annotation row {row_index} has invalid start/end times.")
        annotations.append(
            VideoAnnotation(
                video_path=video_path,
                expected_label=row["expected_label"].strip(),
                start_sec=start_sec,
                end_sec=end_sec,
                notes=row.get("notes", "").strip(),
            )
        )
    return annotations


def detect_cached_segments(
    cache: VideoLandmarkCache,
    config: AutoTriggerConfig,
    temporal_model: PersonalTemporalModel | None = None,
) -> list[SegmentResult]:
    engine = AutoTriggerEngine(config)
    temporal_predictor = PersonalTemporalPredictor(temporal_model, config) if temporal_model is not None else None
    segments: list[SegmentResult] = []
    previous: np.ndarray | None = None
    for timestamp_sec, frame_vector in zip(cache.timestamps_sec, cache.frame_vectors):
        analysis = analyze_frame_vector(previous, frame_vector, config)
        if temporal_predictor is not None:
            analysis = with_temporal_probability(
                analysis,
                temporal_predictor.update(float(timestamp_sec), analysis),
            )
        event = engine.update(frame_vector, analysis, float(timestamp_sec))
        if event is not None:
            segments.append(event)
        previous = frame_vector
    return segments


def evaluate_video_boundaries(
    annotation: VideoAnnotation,
    segments: list[SegmentResult],
    tolerance_sec: float,
) -> VideoBoundaryMetrics:
    if not segments:
        return VideoBoundaryMetrics.missing(annotation)
    return VideoBoundaryMetrics.from_prediction(
        annotation,
        segments[0],
        tolerance_sec=tolerance_sec,
        segment_count=len(segments),
    )


def evaluate_candidate(
    annotations: list[VideoAnnotation],
    caches: dict[Path, VideoLandmarkCache],
    config: AutoTriggerConfig,
    tolerance_sec: float,
    temporal_model: PersonalTemporalModel | None = None,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        config=config,
        per_video=[
            evaluate_video_boundaries(
                annotation,
                detect_cached_segments(caches[annotation.video_path], config, temporal_model),
                tolerance_sec=tolerance_sec,
            )
            for annotation in annotations
        ],
    )


def choose_best_candidate(candidates: Iterable[CandidateEvaluation]) -> CandidateEvaluation:
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("At least one candidate evaluation is required.")
    return min(candidate_list, key=lambda candidate: candidate.score())


def _grid_configs(
    base_config: AutoTriggerConfig,
    grid: dict[str, list[float]],
) -> Iterable[AutoTriggerConfig]:
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        yield replace(base_config, **dict(zip(keys, values)))


def calibrate_auto_trigger(
    annotations: list[VideoAnnotation],
    caches: dict[Path, VideoLandmarkCache],
    base_config: AutoTriggerConfig | None = None,
    tolerance_sec: float = 0.30,
    end_grid: dict[str, list[float]] | None = None,
    start_grid: dict[str, list[float]] | None = None,
) -> tuple[CandidateEvaluation, dict[str, object]]:
    base = base_config or AutoTriggerConfig()
    resolved_end_grid = end_grid or DEFAULT_END_SEARCH_GRID
    resolved_start_grid = start_grid or DEFAULT_START_SEARCH_GRID
    end_candidates = [
        evaluate_candidate(annotations, caches, config, tolerance_sec)
        for config in _grid_configs(base, resolved_end_grid)
    ]
    best_end = choose_best_candidate(end_candidates)
    start_search_required = any(
        metric.start_error_sec is None
        or abs(metric.start_error_sec) > tolerance_sec
        for metric in best_end.per_video
    )
    start_candidates: list[CandidateEvaluation] = []
    if start_search_required:
        start_candidates = [
            evaluate_candidate(annotations, caches, config, tolerance_sec)
            for config in _grid_configs(best_end.config, resolved_start_grid)
        ]
    best = choose_best_candidate([best_end, *start_candidates])
    metadata = {
        "end_candidates": len(end_candidates),
        "start_search_required": start_search_required,
        "start_candidates": len(start_candidates),
        "selection_score": list(best.score()),
    }
    return best, metadata


def _cache_path(
    video_path: Path,
    cache_dir: Path,
    detector_frame_skip: int,
    inference_max_width: int,
) -> Path:
    cache_key = (
        f"{video_path.resolve()}|{detector_frame_skip}|{inference_max_width}"
    ).encode("utf-8")
    digest = hashlib.sha256(cache_key).hexdigest()[:16]
    return cache_dir / f"{video_path.stem}_{digest}.npz"


def extract_video_landmarks(
    video_path: Path,
    detector_frame_skip: int,
    inference_max_width: int,
) -> VideoLandmarkCache:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_vectors: list[np.ndarray] = []
    timestamps_sec: list[float] = []
    frame_index = 0

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(PATHS.hand_model)),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_hands=2,
    )
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(PATHS.pose_model)),
        running_mode=VisionTaskRunningMode.IMAGE,
    )
    try:
        with (
            HandLandmarker.create_from_options(hand_options) as hand_landmarker,
            PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
        ):
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % max(1, detector_frame_skip) == 0:
                    detection_frame = frame
                    if inference_max_width > 0 and frame.shape[1] > inference_max_width:
                        scale = inference_max_width / float(frame.shape[1])
                        detection_frame = cv2.resize(
                            frame,
                            (
                                max(1, int(round(frame.shape[1] * scale))),
                                max(1, int(round(frame.shape[0] * scale))),
                            ),
                            interpolation=cv2.INTER_AREA,
                        )
                    rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    frame_vectors.append(
                        extract_frame_vector(
                            hand_landmarker.detect(mp_image),
                            pose_landmarker.detect(mp_image),
                        )
                    )
                    timestamps_sec.append(frame_index / fps)
                frame_index += 1
    finally:
        capture.release()
    if not frame_vectors:
        raise RuntimeError(f"No landmark frames extracted from video: {video_path}")
    return VideoLandmarkCache(
        video_path=video_path.resolve(),
        fps=fps,
        frame_width=width,
        frame_height=height,
        timestamps_sec=np.asarray(timestamps_sec, dtype=np.float64),
        frame_vectors=np.stack(frame_vectors).astype(np.float32),
    )


def load_or_extract_video_cache(
    video_path: str | Path,
    cache_dir: str | Path,
    detector_frame_skip: int,
    inference_max_width: int,
    extractor: Callable[[Path, int, int], VideoLandmarkCache] | None = None,
) -> VideoLandmarkCache:
    source = Path(video_path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    output_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(
        source,
        output_dir,
        detector_frame_skip,
        inference_max_width,
    )
    source_stat = source.stat()
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as payload:
            if (
                str(payload["source_path"].item()) == str(source)
                and int(payload["source_mtime_ns"].item()) == source_stat.st_mtime_ns
                and int(payload["source_size"].item()) == source_stat.st_size
                and int(payload["detector_frame_skip"].item()) == detector_frame_skip
                and int(payload["inference_max_width"].item()) == inference_max_width
            ):
                return VideoLandmarkCache(
                    video_path=source,
                    fps=float(payload["fps"].item()),
                    frame_width=int(payload["frame_width"].item()),
                    frame_height=int(payload["frame_height"].item()),
                    timestamps_sec=payload["timestamps_sec"].astype(np.float64),
                    frame_vectors=payload["frame_vectors"].astype(np.float32),
                )
    extract = extractor or extract_video_landmarks
    cache = extract(source, detector_frame_skip, inference_max_width)
    np.savez_compressed(
        cache_path,
        source_path=np.array(str(source)),
        source_mtime_ns=np.array(source_stat.st_mtime_ns, dtype=np.int64),
        source_size=np.array(source_stat.st_size, dtype=np.int64),
        detector_frame_skip=np.array(detector_frame_skip, dtype=np.int64),
        inference_max_width=np.array(inference_max_width, dtype=np.int64),
        fps=np.array(cache.fps, dtype=np.float64),
        frame_width=np.array(cache.frame_width, dtype=np.int64),
        frame_height=np.array(cache.frame_height, dtype=np.int64),
        timestamps_sec=cache.timestamps_sec.astype(np.float64),
        frame_vectors=cache.frame_vectors.astype(np.float32),
    )
    return cache


def classify_manual_and_auto_crops(
    cache: VideoLandmarkCache,
    annotation: VideoAnnotation,
    auto_segment: SegmentResult | None,
    predictor: Callable[[list[np.ndarray]], tuple[str, float]],
) -> ClassificationComparison:
    manual_mask = (
        (cache.timestamps_sec >= annotation.start_sec)
        & (cache.timestamps_sec <= annotation.end_sec)
    )
    manual_frames = [
        vector.copy()
        for vector in cache.frame_vectors[manual_mask]
    ]
    auto_frames = auto_segment.frame_vectors if auto_segment is not None else []
    manual_label, manual_confidence = predictor(manual_frames)
    auto_label, auto_confidence = predictor(auto_frames)
    manual_correct = manual_label == annotation.expected_label
    auto_correct = auto_label == annotation.expected_label
    return ClassificationComparison(
        video_path=annotation.video_path,
        expected_label=annotation.expected_label,
        manual_label=manual_label,
        manual_confidence=float(manual_confidence),
        auto_label=auto_label,
        auto_confidence=float(auto_confidence),
        manual_correct=manual_correct,
        auto_correct=auto_correct,
        segmentation_regression=bool(manual_correct and not auto_correct),
        model_issue=bool(not manual_correct and not auto_correct),
    )


def build_sentence_predictor(
    model_cache_dir: str | Path,
) -> Callable[[list[np.ndarray]], tuple[str, float]]:
    import torch

    from recognition.inference.daily30_sentence_feature_utils import build_feature_sequence
    from recognition.inference.daily30_sentence_model_utils import BiGRUSentenceClassifier
    from recognition.inference.daily30_sentence_realtime_utils import (
        ensure_artifacts_cached,
        load_runtime_bundle,
    )
    from recognition.inference.extract_daily30_sentence_features import normalize_relative_frames, resize_seq

    cache_dir = ensure_artifacts_cached(Path(model_cache_dir))
    bundle = load_runtime_bundle(cache_dir)
    preferred_device = str(bundle.get("device", "auto")).lower()
    if preferred_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif preferred_device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = 225 * (2 if bool(bundle["append_delta"]) else 1)
    model = BiGRUSentenceClassifier(
        input_dim=feature_dim,
        hidden_size=int(bundle["hidden_size"]),
        num_layers=int(bundle["num_layers"]),
        dropout=float(bundle["dropout"]),
        num_classes=len(bundle["labels"]),
        pooling=str(bundle["pooling"]),
    )
    model.load_state_dict(torch.load(bundle["paths"]["best_model"], map_location=device))
    model.to(device)
    model.eval()

    def predict(frame_vectors: list[np.ndarray]) -> tuple[str, float]:
        if not frame_vectors:
            return "片段過短", 0.0
        raw_frames = np.stack(frame_vectors, axis=0)
        relative = normalize_relative_frames(raw_frames)
        resized = resize_seq(relative, int(bundle["sequence_length"]))
        features = build_feature_sequence(
            resized,
            append_delta=bool(bundle["append_delta"]),
            zscore_features=bool(bundle["zscore_features"]),
        )
        tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
        predicted_index = int(np.argmax(probabilities))
        label_id = str(bundle["labels"][predicted_index])
        localized = str(bundle["label_display"].get(label_id, label_id))
        return localized, float(probabilities[predicted_index])

    return predict


def write_debug_video(
    annotation: VideoAnnotation,
    metrics: VideoBoundaryMetrics,
    output_path: str | Path,
) -> Path:
    capture = cv2.VideoCapture(str(annotation.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open debug video source: {annotation.video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = frame_count / fps if frame_count > 0 else max(annotation.end_sec, 1.0)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    codec = "MJPG" if destination.suffix.lower() == ".avi" else "mp4v"
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create debug video: {destination}")
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current_sec = frame_index / fps
            canvas = frame.copy()
            cv2.rectangle(canvas, (0, 0), (width, 42), (0, 0, 0), -1)
            status = (
                f"GT {annotation.start_sec:.2f}-{annotation.end_sec:.2f}s | "
                f"PRED {metrics.predicted_start_sec if metrics.predicted_start_sec is not None else -1:.2f}-"
                f"{metrics.predicted_end_sec if metrics.predicted_end_sec is not None else -1:.2f}s | "
                f"NOW {current_sec:.2f}s"
            )
            cv2.putText(
                canvas,
                status,
                (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            timeline_y = max(0, height - 24)
            cv2.rectangle(canvas, (8, timeline_y), (max(8, width - 8), height - 8), (35, 35, 35), -1)

            def timeline_x(time_sec: float) -> int:
                ratio = min(max(time_sec / max(duration_sec, 1e-6), 0.0), 1.0)
                return 8 + int(round(ratio * max(1, width - 16)))

            cv2.line(
                canvas,
                (timeline_x(annotation.start_sec), timeline_y + 4),
                (timeline_x(annotation.end_sec), timeline_y + 4),
                (0, 220, 0),
                5,
            )
            if metrics.predicted_start_sec is not None and metrics.predicted_end_sec is not None:
                cv2.line(
                    canvas,
                    (timeline_x(metrics.predicted_start_sec), timeline_y + 12),
                    (timeline_x(metrics.predicted_end_sec), timeline_y + 12),
                    (0, 80, 255),
                    5,
                )
            current_x = timeline_x(current_sec)
            cv2.line(canvas, (current_x, timeline_y), (current_x, height - 6), (255, 255, 255), 2)
            writer.write(canvas)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    return destination


def write_evaluation_outputs(
    out_dir: str | Path,
    best: CandidateEvaluation,
    comparisons: list[ClassificationComparison],
    search_metadata: dict[str, object],
    tolerance_sec: float,
) -> dict[str, Path]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_by_path = {comparison.video_path: comparison for comparison in comparisons}
    rows = []
    for metric in best.per_video:
        row = metric.to_row()
        comparison = comparison_by_path.get(metric.video_path)
        if comparison is not None:
            row.update(comparison.to_row())
        rows.append(row)

    metrics_path = output_dir / "per_video_metrics.csv"
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = list(rows[0].keys()) if rows else list(VideoBoundaryMetrics.__dataclass_fields__)
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    start_errors = [
        abs(metric.start_error_sec)
        for metric in best.per_video
        if metric.start_error_sec is not None
    ]
    end_errors = [
        abs(metric.end_error_sec)
        for metric in best.per_video
        if metric.end_error_sec is not None
    ]
    classification_summary = {
        "evaluated_count": len(comparisons),
        "manual_correct_count": sum(item.manual_correct for item in comparisons),
        "auto_correct_count": sum(item.auto_correct for item in comparisons),
        "segmentation_regression_count": sum(item.segmentation_regression for item in comparisons),
        "model_issue_count": sum(item.model_issue for item in comparisons),
    }
    summary = {
        "video_count": len(best.per_video),
        "passed_count": sum(metric.passed for metric in best.per_video),
        "all_videos_passed": bool(best.per_video and all(metric.passed for metric in best.per_video)),
        "boundary_tolerance_sec": tolerance_sec,
        "start_mae_sec": float(np.mean(start_errors)) if start_errors else None,
        "end_mae_sec": float(np.mean(end_errors)) if end_errors else None,
        "max_start_error_sec": max(start_errors, default=None),
        "max_end_error_sec": max(end_errors, default=None),
        "extra_segment_count": sum(metric.extra_segment_count for metric in best.per_video),
        "search": search_metadata,
        "classification": classification_summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    best_config_path = output_dir / "best_auto_trigger.json"
    best_config_path.write_text(
        json.dumps(best.config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "metrics": metrics_path,
        "summary": summary_path,
        "best_config": best_config_path,
    }


def install_best_config_if_passed(
    best: CandidateEvaluation,
    destination: str | Path,
    comparisons: list[ClassificationComparison],
) -> Path | None:
    if not best.per_video or not all(metric.passed for metric in best.per_video):
        return None
    if len(comparisons) != len(best.per_video):
        return None
    if any(comparison.segmentation_regression for comparison in comparisons):
        return None
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        json.dumps(best.config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    annotations = load_annotations(args.annotations)
    if args.boundary_tolerance_sec <= 0:
        raise ValueError("Boundary tolerance must be positive.")
    if args.detector_frame_skip <= 0:
        raise ValueError("Detector frame skip must be positive.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else (PATHS.results_dir / f"auto_trigger_eval_{stamp}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    landmark_cache_dir = Path(args.landmark_cache_dir).resolve()
    caches = {
        annotation.video_path: load_or_extract_video_cache(
            annotation.video_path,
            landmark_cache_dir,
            detector_frame_skip=int(args.detector_frame_skip),
            inference_max_width=int(args.inference_max_width),
        )
        for annotation in annotations
    }
    base_config = load_auto_trigger_config(args.base_auto_config or None)
    best, search_metadata = calibrate_auto_trigger(
        annotations,
        caches,
        base_config=base_config,
        tolerance_sec=float(args.boundary_tolerance_sec),
    )
    search_metadata.update(
        {
            "detector_frame_skip": int(args.detector_frame_skip),
            "inference_max_width": int(args.inference_max_width),
            "landmark_cache_dir": str(landmark_cache_dir),
        }
    )

    comparisons: list[ClassificationComparison] = []
    segments_by_path = {
        annotation.video_path: detect_cached_segments(
            caches[annotation.video_path],
            best.config,
        )
        for annotation in annotations
    }
    if not args.skip_classification:
        predictor = build_sentence_predictor(args.model_cache_dir)
        for annotation in annotations:
            segments = segments_by_path[annotation.video_path]
            comparisons.append(
                classify_manual_and_auto_crops(
                    caches[annotation.video_path],
                    annotation,
                    segments[0] if segments else None,
                    predictor,
                )
            )

    output_paths = write_evaluation_outputs(
        output_dir,
        best,
        comparisons=comparisons,
        search_metadata=search_metadata,
        tolerance_sec=float(args.boundary_tolerance_sec),
    )
    installed_config = install_best_config_if_passed(
        best,
        Path(args.model_cache_dir) / "best_auto_trigger.json",
        comparisons=comparisons,
    )
    debug_paths: list[Path] = []
    if not args.skip_debug_videos:
        debug_dir = output_dir / "debug_videos"
        metric_by_path = {metric.video_path: metric for metric in best.per_video}
        for annotation in annotations:
            debug_path = debug_dir / f"{annotation.video_path.stem}_boundary_debug.mp4"
            debug_paths.append(
                write_debug_video(
                    annotation,
                    metric_by_path[annotation.video_path],
                    debug_path,
                )
            )

    print(f"Metrics: {output_paths['metrics']}")
    print(f"Summary: {output_paths['summary']}")
    print(f"Best config: {output_paths['best_config']}")
    if installed_config is not None:
        print(f"Installed realtime config: {installed_config}")
    if debug_paths:
        print(f"Debug videos: {len(debug_paths)} in {debug_paths[0].parent}")


if __name__ == "__main__":
    main()
