from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from recognition.evaluation.eval_auto_trigger_boundaries import (
    CandidateEvaluation,
    VideoAnnotation,
    VideoBoundaryMetrics,
    VideoLandmarkCache,
    calibrate_auto_trigger,
    choose_best_candidate,
    detect_cached_segments,
    evaluate_video_boundaries,
    install_best_config_if_passed,
    load_or_extract_video_cache,
    load_annotations,
    parse_args,
    write_debug_video,
    write_evaluation_outputs,
)
from recognition.realtime.auto_trigger import AutoTriggerConfig, FrameSample, SegmentResult


def body_vector(left_xy: tuple[float, float], right_xy: tuple[float, float]) -> np.ndarray:
    pose = np.zeros((33, 3), dtype=np.float32)
    pose[11] = [0.40, 0.30, 0.0]
    pose[12] = [0.60, 0.30, 0.0]
    pose[13] = [0.36, 0.48, 0.0]
    pose[14] = [0.64, 0.48, 0.0]
    pose[15] = [left_xy[0], left_xy[1], 0.0]
    pose[16] = [right_xy[0], right_xy[1], 0.0]
    pose[23] = [0.44, 0.70, 0.0]
    pose[24] = [0.56, 0.70, 0.0]
    pose[25] = [0.36, 0.90, 0.0]
    pose[26] = [0.64, 0.90, 0.0]
    left = np.tile(np.array([left_xy[0], left_xy[1], 0.0], dtype=np.float32), (21, 1))
    right = np.tile(np.array([right_xy[0], right_xy[1], 0.0], dtype=np.float32), (21, 1))
    return np.concatenate([pose.reshape(-1), left.reshape(-1), right.reshape(-1)])


def synthetic_cache() -> VideoLandmarkCache:
    timestamps = np.round(np.arange(0.0, 2.01, 0.1), 6)
    vectors = []
    for time_sec in timestamps:
        if time_sec < 0.5 or time_sec >= 1.2:
            vectors.append(body_vector((0.36, 0.90), (0.64, 0.90)))
        elif time_sec < 0.8:
            progress = (time_sec - 0.5) / 0.3
            vectors.append(
                body_vector(
                    (0.34 + 0.13 * progress, 0.55 - 0.13 * progress),
                    (0.66 - 0.13 * progress, 0.55 - 0.13 * progress),
                )
            )
        else:
            vectors.append(body_vector((0.47, 0.42), (0.53, 0.42)))
    return VideoLandmarkCache(
        video_path=Path("demo.mp4"),
        fps=10.0,
        frame_width=640,
        frame_height=480,
        timestamps_sec=timestamps,
        frame_vectors=np.stack(vectors),
    )


def segment(start: float, end: float, finalize: float | None = None) -> SegmentResult:
    return SegmentResult(
        samples=[FrameSample(start, np.zeros(225, dtype=np.float32))],
        clip_start_sec=start,
        clip_end_sec=end,
        finalize_sec=finalize if finalize is not None else end + 0.5,
        reason="visible_rest_finalize",
    )


class AnnotationTests(unittest.TestCase):
    def test_load_annotations_resolves_relative_video_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "videos" / "a.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video")
            csv_path = root / "annotations.csv"
            csv_path.write_text(
                "video_path,expected_label,start_sec,end_sec,notes\n"
                "videos/a.mp4,可以,2.0,3.5,first\n",
                encoding="utf-8-sig",
            )

            rows = load_annotations(csv_path)

        self.assertEqual(rows[0].video_path, video.resolve())
        self.assertEqual(rows[0].expected_label, "可以")
        self.assertEqual(rows[0].start_sec, 2.0)
        self.assertEqual(rows[0].end_sec, 3.5)

    def test_cli_supports_annotation_output_and_optional_artifacts(self):
        args = parse_args(
            [
                "--annotations",
                "three_videos.csv",
                "--out-dir",
                "results",
                "--skip-debug-videos",
            ]
        )

        self.assertEqual(args.annotations, "three_videos.csv")
        self.assertEqual(args.out_dir, "results")
        self.assertTrue(args.skip_debug_videos)


class BoundaryEvaluationTests(unittest.TestCase):
    def test_cached_landmarks_produce_one_time_based_segment(self):
        cache = synthetic_cache()
        config = AutoTriggerConfig(
            start_motion_threshold=0.015,
            start_hold_sec=0.10,
            pre_roll_sec=0.10,
            end_hold_sec=0.30,
            end_rest_vote_ratio=0.75,
            min_segment_sec=0.10,
        )

        segments = detect_cached_segments(cache, config)

        self.assertEqual(len(segments), 1)
        self.assertLessEqual(abs(segments[0].clip_start_sec - 0.5), 0.2)
        self.assertAlmostEqual(segments[0].clip_end_sec, 1.3, places=6)
        self.assertAlmostEqual(segments[0].finalize_sec, 1.6, places=6)

    def test_boundary_metrics_mark_extra_segments_invalid(self):
        annotation = VideoAnnotation(Path("demo.mp4"), "可以", 0.5, 1.2, "")

        metrics = evaluate_video_boundaries(
            annotation,
            [segment(0.5, 1.2), segment(1.5, 1.8)],
            tolerance_sec=0.30,
        )

        self.assertFalse(metrics.passed)
        self.assertEqual(metrics.segment_count, 2)
        self.assertEqual(metrics.extra_segment_count, 1)

    def test_candidate_ranking_prioritizes_valid_single_segments_then_end_error(self):
        config = AutoTriggerConfig()
        invalid = CandidateEvaluation(
            config=config,
            per_video=[
                VideoBoundaryMetrics.missing(VideoAnnotation(Path("a.mp4"), "", 1.0, 2.0, ""))
            ],
        )
        valid_worse_end = CandidateEvaluation(
            config=config,
            per_video=[
                VideoBoundaryMetrics.from_prediction(
                    VideoAnnotation(Path("a.mp4"), "", 1.0, 2.0, ""),
                    segment(1.0, 2.25, 2.75),
                    tolerance_sec=0.30,
                )
            ],
        )
        valid_better_end = CandidateEvaluation(
            config=AutoTriggerConfig(end_hold_sec=0.60),
            per_video=[
                VideoBoundaryMetrics.from_prediction(
                    VideoAnnotation(Path("a.mp4"), "", 1.0, 2.0, ""),
                    segment(1.1, 2.05, 2.55),
                    tolerance_sec=0.30,
                )
            ],
        )

        best = choose_best_candidate([invalid, valid_worse_end, valid_better_end])

        self.assertIs(best, valid_better_end)

    def test_calibration_runs_start_search_only_when_stage_one_misses_start(self):
        cache = synthetic_cache()
        annotation = VideoAnnotation(Path("demo.mp4"), "可以", 0.5, 1.3, "")
        end_grid = {
            "end_hold_sec": [0.30],
            "end_rest_vote_ratio": [0.75],
            "blank_motion_threshold": [0.018],
            "knee_lateral_thigh_margin_ratio": [0.55],
            "knee_min_thigh_progress_ratio": [-0.85],
        }
        start_grid = {
            "start_motion_threshold": [0.015],
            "start_hold_sec": [0.10],
            "pre_roll_sec": [0.10],
        }

        best, metadata = calibrate_auto_trigger(
            [annotation],
            {annotation.video_path: cache},
            base_config=AutoTriggerConfig(start_motion_threshold=0.50, min_segment_sec=0.10),
            tolerance_sec=0.30,
            end_grid=end_grid,
            start_grid=start_grid,
        )

        self.assertEqual(metadata["end_candidates"], 1)
        self.assertEqual(metadata["start_candidates"], 1)
        self.assertEqual(best.per_video[0].segment_count, 1)
        self.assertEqual(best.config.start_motion_threshold, 0.015)


class LandmarkCacheTests(unittest.TestCase):
    def test_cached_video_is_extracted_only_once_for_same_settings(self):
        calls = 0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "demo.mp4"
            video_path.write_bytes(b"video")
            cache_dir = root / "cache"

            def extractor(path: Path, detector_frame_skip: int, inference_max_width: int) -> VideoLandmarkCache:
                nonlocal calls
                calls += 1
                cache = synthetic_cache()
                return VideoLandmarkCache(
                    video_path=path,
                    fps=cache.fps,
                    frame_width=cache.frame_width,
                    frame_height=cache.frame_height,
                    timestamps_sec=cache.timestamps_sec,
                    frame_vectors=cache.frame_vectors,
                )

            first = load_or_extract_video_cache(
                video_path,
                cache_dir,
                detector_frame_skip=2,
                inference_max_width=960,
                extractor=extractor,
            )
            second = load_or_extract_video_cache(
                video_path,
                cache_dir,
                detector_frame_skip=2,
                inference_max_width=960,
                extractor=extractor,
            )

        self.assertEqual(calls, 1)
        np.testing.assert_array_equal(first.frame_vectors, second.frame_vectors)
        np.testing.assert_array_equal(first.timestamps_sec, second.timestamps_sec)


class OutputTests(unittest.TestCase):
    def test_write_outputs_creates_metrics_summary_and_best_config(self):
        annotation = VideoAnnotation(Path("C:/private-host-account/demo.mp4"), "可以", 0.5, 1.2, "")
        metrics = VideoBoundaryMetrics.from_prediction(
            annotation,
            segment(0.45, 1.25, 1.75),
            tolerance_sec=0.30,
        )
        candidate = CandidateEvaluation(AutoTriggerConfig(end_hold_sec=0.60), [metrics])
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            paths = write_evaluation_outputs(
                out_dir,
                candidate,
                search_metadata={"end_candidates": 243, "start_candidates": 0},
                tolerance_sec=0.30,
            )
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            config = json.loads(paths["best_config"].read_text(encoding="utf-8"))
            with paths["metrics"].open("r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["video_path"], "demo.mp4")
        self.assertTrue(summary["all_videos_passed"])
        self.assertNotIn("classification", summary)
        self.assertEqual(config["end_hold_sec"], 0.60)

    def test_debug_video_contains_all_source_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.avi"
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (160, 120),
            )
            self.assertTrue(writer.isOpened())
            for index in range(20):
                writer.write(np.full((120, 160, 3), index * 8, dtype=np.uint8))
            writer.release()
            annotation = VideoAnnotation(source, "可以", 0.5, 1.2, "")
            metrics = VideoBoundaryMetrics.from_prediction(
                annotation,
                segment(0.4, 1.3, 1.8),
                tolerance_sec=0.30,
            )
            output = root / "debug.avi"

            write_debug_video(annotation, metrics, output)

            capture = cv2.VideoCapture(str(output))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()

        self.assertEqual(frame_count, 20)

    def test_best_config_installs_only_when_every_video_passes(self):
        """The gate is boundary-only now; the classifier regression check is gone."""
        annotation = VideoAnnotation(Path("demo.mp4"), "可以", 0.5, 1.2, "")
        passing = CandidateEvaluation(
            AutoTriggerConfig(end_hold_sec=0.60),
            [
                VideoBoundaryMetrics.from_prediction(
                    annotation,
                    segment(0.5, 1.2),
                    tolerance_sec=0.30,
                )
            ],
        )
        failing = CandidateEvaluation(
            AutoTriggerConfig(end_hold_sec=0.40),
            [VideoBoundaryMetrics.missing(annotation)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "best_auto_trigger.json"
            self.assertIsNone(install_best_config_if_passed(failing, destination))
            self.assertFalse(destination.exists())

            installed = install_best_config_if_passed(passing, destination)
            self.assertEqual(installed, destination)
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8"))["end_hold_sec"], 0.60
            )

    def test_an_empty_candidate_never_installs(self):
        empty = CandidateEvaluation(AutoTriggerConfig(end_hold_sec=0.60), [])
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "best_auto_trigger.json"
            self.assertIsNone(install_best_config_if_passed(empty, destination))


if __name__ == "__main__":
    unittest.main()
