from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import recognition.realtime.knee42_session_recording as session_recording

from recognition.realtime.auto_trigger import CalibrationState, CalibrationTelemetry
from recognition.realtime.knee42_controllers import (
    BoundaryDecisionTelemetry,
    SegmentEvidence,
)
from recognition.realtime.knee42_session_recording import SegmentSessionRecorder, frame_bounds


def count_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise AssertionError(f"cannot open {path}")
        count = 0
        while True:
            ok, _frame = capture.read()
            if not ok:
                return count
            count += 1
    finally:
        capture.release()


class FrameBoundsTests(unittest.TestCase):
    def test_bounds_are_inclusive_and_clamped_to_session(self):
        self.assertEqual(frame_bounds(10.0, 10.2, 10.5, 10.0, 12), (2, 6))
        self.assertEqual(frame_bounds(10.0, 9.0, 12.0, 10.0, 12), (0, 12))

    def test_timestamp_bounds_use_left_and_right_bisect_for_vfr_frames(self):
        bounds = getattr(session_recording, "timestamp_frame_bounds", None)

        self.assertTrue(callable(bounds), "timestamp_frame_bounds API is missing")
        self.assertEqual(bounds([0.0, 0.1, 0.4], 0.4, 0.4), (2, 3))
        self.assertEqual(bounds([0.0, 0.1, 0.4], -1.0, 1.0), (0, 3))


class SegmentSessionRecorderTests(unittest.TestCase):
    def test_stop_summary_write_failure_is_retryable_and_never_publishes_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = SegmentSessionRecorder(
                Path(temp_dir),
                fps=30.0,
                frame_size=(32, 24),
                source_origin_sec=0.0,
                now=lambda: "retry-summary",
            )
            real_write_text = Path.write_text
            failed = False

            def fail_summary_once(path, data, *args, **kwargs):
                nonlocal failed
                if not failed and path.name in {
                    "session_summary.json",
                    "session_summary.json.tmp",
                }:
                    failed = True
                    raise OSError("summary write failed")
                return real_write_text(path, data, *args, **kwargs)

            with mock.patch.object(
                Path,
                "write_text",
                autospec=True,
                side_effect=fail_summary_once,
            ):
                with self.assertRaisesRegex(OSError, "summary write failed"):
                    recorder.stop()

            self.assertIsNone(recorder._summary)
            self.assertFalse((recorder.session_dir / "session_summary.json").exists())
            self.assertEqual(list(recorder.session_dir.glob("*.tmp")), [])

            summary = recorder.stop()

            self.assertIs(recorder.stop(), summary)
            self.assertTrue((recorder.session_dir / "session_summary.json").is_file())

    def test_boundary_nonfinite_json_fails_before_any_boundary_is_persisted(self):
        decision = BoundaryDecisionTelemetry(
            state_before="END_CONFIRM",
            state_after="FORCED_FINALIZE_COOLDOWN",
            clip_start_sec=0.0,
            clip_end_sec=0.0,
            finalize_sec=0.1,
            finalize_reason="visible_rest_finalize",
            decision_reason="short_segment",
            rest_detected_sec=0.0,
            boundary_policy="first_confirmed_rest",
            threshold_snapshot={"min_segment_sec": float("nan")},
            calibration=CalibrationTelemetry(
                CalibrationState.DISABLED,
                None,
                0.0,
                0,
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = SegmentSessionRecorder(
                Path(temp_dir),
                fps=30.0,
                frame_size=(32, 24),
                source_origin_sec=0.0,
                context_sec=0.0,
                now=lambda: "strict-boundary-json",
            )
            try:
                recorder.add_frame(
                    np.zeros((24, 32, 3), dtype=np.uint8), timestamp_sec=0.0
                )
                with self.assertRaisesRegex(ValueError, "JSON|compliant|finite"):
                    recorder.record_boundary_decision(decision, result=None)

                self.assertEqual(recorder.segment_count, 0)
                self.assertEqual(list((recorder.session_dir / "metadata").iterdir()), [])
                self.assertEqual(
                    (recorder.session_dir / "segments.jsonl").read_text(
                        encoding="utf-8"
                    ),
                    "",
                )
            finally:
                recorder.stop()

    def test_short_boundary_decision_is_persisted_without_prediction(self):
        context = session_recording.RecordingRuntimeContext(
            clock_mode="video_source_timestamp",
            resolved_rotation=0,
            input_mirror=False,
            display_mirror=False,
            trigger_config_sha256="1" * 64,
            trigger_provenance_sha256="2" * 64,
        )
        decision = BoundaryDecisionTelemetry(
            state_before="END_CONFIRM",
            state_after="FORCED_FINALIZE_COOLDOWN",
            clip_start_sec=0.0,
            clip_end_sec=0.0,
            finalize_sec=0.2,
            finalize_reason="visible_rest_finalize",
            decision_reason="short_segment",
            rest_detected_sec=0.0,
            boundary_policy="first_confirmed_rest",
            threshold_snapshot={"min_segment_sec": 0.8},
            calibration=CalibrationTelemetry(
                CalibrationState.DISABLED,
                None,
                0.0,
                0,
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = SegmentSessionRecorder(
                Path(temp_dir),
                fps=30.0,
                frame_size=(32, 24),
                source_origin_sec=0.0,
                runtime_context=context,
                context_sec=0.0,
                now=lambda: "short-boundary",
            )
            try:
                recorder.add_frame(
                    np.zeros((24, 32, 3), dtype=np.uint8), timestamp_sec=0.0
                )
                recorder.record_boundary_decision(decision, result=None)
                summary = recorder.stop()
            finally:
                recorder.stop()
            payload = json.loads(
                (summary.session_dir / "metadata" / "segment_0001.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertIsNone(payload["top1"])
        self.assertEqual(payload["top3"], [])
        self.assertEqual(payload["boundary_decision"]["decision_reason"], "short_segment")
        self.assertEqual(payload["runtime_context"], context.to_dict())

    def test_runtime_context_is_immutable_and_written_to_session_summary(self):
        context_type = getattr(session_recording, "RecordingRuntimeContext", None)
        self.assertIsNotNone(context_type, "recording runtime context API is missing")
        context = context_type(
            clock_mode="video_source_timestamp",
            resolved_rotation=90,
            input_mirror=True,
            display_mirror=False,
            trigger_config_sha256="1" * 64,
            trigger_provenance_sha256="2" * 64,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = SegmentSessionRecorder(
                Path(temp_dir),
                fps=30.0,
                frame_size=(32, 24),
                source_origin_sec=0.0,
                runtime_context=context,
                now=lambda: "runtime-context",
            )
            summary = recorder.stop()
            payload = json.loads(
                (summary.session_dir / "session_summary.json").read_text(encoding="utf-8")
            )
        self.assertEqual(payload["runtime_context"]["clock_mode"], "video_source_timestamp")
        self.assertEqual(payload["runtime_context"]["trigger_config_sha256"], "1" * 64)

    def test_recorder_rejects_nonfinite_or_impractical_source_fps_before_writer(self):
        fake_cv2 = mock.Mock()
        fake_cv2.VideoWriter.side_effect = AssertionError("writer must not be opened")

        with tempfile.TemporaryDirectory() as temp_dir:
            for fps in (float("nan"), float("inf"), 0.0, -1.0, 240.0001):
                with self.subTest(fps=fps):
                    with self.assertRaisesRegex(ValueError, "finite|practical|240"):
                        SegmentSessionRecorder(
                            Path(temp_dir),
                            fps=fps,
                            frame_size=(32, 24),
                            source_origin_sec=0.0,
                            now=lambda: f"invalid-{fps}",
                            cv2_module=fake_cv2,
                        )

        fake_cv2.VideoWriter.assert_not_called()

    def test_vfr_timestamps_select_the_actual_end_frame_and_are_audited(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = SegmentSessionRecorder(
                Path(temp_dir),
                fps=10.0,
                frame_size=(32, 24),
                source_origin_sec=0.0,
                context_sec=0.0,
                now=lambda: "vfr",
            )
            try:
                for index, timestamp_sec in enumerate((0.0, 0.1, 0.4)):
                    recorder.add_frame(
                        np.full((24, 32, 3), index * 80, dtype=np.uint8),
                        timestamp_sec=timestamp_sec,
                    )
                prediction = SimpleNamespace(
                    label_id="K42_01",
                    display_text="你好",
                    raw_probability=1.0,
                )
                recorder.record_segment(
                    SegmentEvidence(0.4, 0.4, 0.4, "visible_rest_finalize"),
                    SimpleNamespace(top1=prediction, top3=(prediction,)),
                )

                summary = recorder.stop()
            finally:
                recorder.stop()

            metadata = json.loads(
                (summary.session_dir / "metadata" / "segment_0001.json").read_text(
                    encoding="utf-8"
                )
            )
            exact = summary.session_dir / metadata["exact_clip"]
            self.assertEqual(count_frames(exact), 1)
            self.assertEqual(metadata["clock_timestamps"]["exact_frame_bounds"], [2, 3])
            self.assertEqual(
                metadata["clock_timestamps"]["exact_frame_timestamps_sec"],
                [0.4],
            )
            self.assertEqual(
                metadata["clock_timestamps"]["session_frame_timestamps_sec"],
                [0.0, 0.1, 0.4],
            )

    def test_add_frame_rejects_nonfinite_and_regressing_timestamps_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = SegmentSessionRecorder(
                Path(temp_dir),
                fps=30.0,
                frame_size=(32, 24),
                source_origin_sec=0.0,
                now=lambda: "timestamps",
            )
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            try:
                recorder.add_frame(frame, timestamp_sec=0.1)

                for timestamp_sec, message in (
                    (float("nan"), "finite"),
                    (0.09, "nondecreasing"),
                ):
                    with self.subTest(timestamp_sec=timestamp_sec):
                        with self.assertRaisesRegex(ValueError, message):
                            recorder.add_frame(frame, timestamp_sec=timestamp_sec)

                self.assertEqual(recorder.stop().frame_count, 1)
            finally:
                recorder.stop()

    def test_stop_materializes_exact_and_context_clips_with_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_probability = 0.12345678901234568
            recorder = SegmentSessionRecorder(
                root,
                fps=10.0,
                frame_size=(32, 24),
                source_origin_sec=10.0,
                now=lambda: "20260818_220000",
            )
            for index in range(12):
                recorder.add_frame(np.full((24, 32, 3), index * 10, dtype=np.uint8))
            result = SimpleNamespace(
                top1=SimpleNamespace(label_id="K42_09", display_text="請再說一次", raw_probability=raw_probability),
                top3=(
                    SimpleNamespace(label_id="K42_09", display_text="請再說一次", raw_probability=raw_probability),
                    SimpleNamespace(label_id="K42_10", display_text="沒有", raw_probability=0.2),
                    SimpleNamespace(label_id="K42_11", display_text="有", raw_probability=0.1),
                ),
            )
            recorder.record_segment(
                SegmentEvidence(
                    10.2,
                    10.5,
                    10.7,
                    "visible_rest_finalize",
                    rest_detected_sec=10.6,
                    boundary_policy="low_motion_anchor_v1",
                ),
                result,
            )

            summary = recorder.stop()

            self.assertEqual(summary.segment_count, 1)
            self.assertTrue(summary.source_path.is_file())
            with (summary.session_dir / "segments.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["rest_detected_sec"]), 10.6)
            self.assertEqual(rows[0]["boundary_policy"], "low_motion_anchor_v1")
            self.assertEqual(rows[0]["top1_label"], "K42_09")
            self.assertEqual(rows[0]["top1_text"], "請再說一次")
            self.assertIn("top1_raw_probability", rows[0])
            self.assertEqual(float(rows[0]["top1_raw_probability"]), raw_probability)
            self.assertEqual(
                json.loads(rows[0]["top3_json"])[0]["raw_probability"],
                raw_probability,
            )
            self.assertNotIn("top1_confidence", rows[0])
            exact = summary.session_dir / rows[0]["exact_clip"]
            context = summary.session_dir / rows[0]["context_clip"]
            self.assertEqual(count_frames(exact), 4)
            self.assertEqual(count_frames(context), 12)
            metadata = json.loads((summary.session_dir / "metadata" / "segment_0001.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["reason"], "visible_rest_finalize")
            self.assertEqual(metadata["rest_detected_sec"], 10.6)
            self.assertEqual(metadata["boundary_policy"], "low_motion_anchor_v1")
            self.assertEqual([item["label_id"] for item in metadata["top3"]], ["K42_09", "K42_10", "K42_11"])
            self.assertEqual(metadata["top1"]["raw_probability"], raw_probability)
            self.assertNotIn("confidence", metadata["top1"])
            self.assertEqual(
                metadata["probability_policy"],
                {
                    "kind": "uncalibrated_softmax",
                    "acceptance_policy": "disabled_no_risk_coverage_evidence",
                    "calibration_artifact": None,
                },
            )
            jsonl = json.loads((summary.session_dir / "segments.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(jsonl["rest_detected_sec"], 10.6)
            self.assertEqual(jsonl["boundary_policy"], "low_motion_anchor_v1")
            summary_payload = json.loads(
                (summary.session_dir / "session_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary_payload["probability_policy"]["acceptance_policy"],
                "disabled_no_risk_coverage_evidence",
            )

    def test_unique_session_and_empty_stop_do_not_fabricate_segments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = SegmentSessionRecorder(root, fps=10.0, frame_size=(32, 24), source_origin_sec=0.0, now=lambda: "same")
            second = SegmentSessionRecorder(root, fps=10.0, frame_size=(32, 24), source_origin_sec=0.0, now=lambda: "same")
            first.add_frame(np.zeros((24, 32, 3), dtype=np.uint8))
            second.add_frame(np.zeros((24, 32, 3), dtype=np.uint8))

            first_summary = first.stop()
            second_summary = second.stop()

            self.assertNotEqual(first_summary.session_dir, second_summary.session_dir)
            self.assertEqual(first_summary.segment_count, 0)
            self.assertFalse(any((first_summary.session_dir / "segments").iterdir()))
            self.assertEqual(first.stop(), first_summary)


if __name__ == "__main__":
    unittest.main()
