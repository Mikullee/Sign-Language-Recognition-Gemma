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

from recognition.realtime.knee42_controllers import SegmentEvidence
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
                top1=SimpleNamespace(label_id="K42_09", display_text="請再說一次", raw_probability=0.7),
                top3=(
                    SimpleNamespace(label_id="K42_09", display_text="請再說一次", raw_probability=0.7),
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
            self.assertEqual(rows[0]["rest_detected_sec"], "10.600000")
            self.assertEqual(rows[0]["boundary_policy"], "low_motion_anchor_v1")
            self.assertEqual(rows[0]["top1_label"], "K42_09")
            self.assertEqual(rows[0]["top1_text"], "請再說一次")
            self.assertIn("top1_raw_probability", rows[0])
            self.assertEqual(rows[0]["top1_raw_probability"], "0.700000000")
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
            self.assertEqual(metadata["top1"]["raw_probability"], 0.7)
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
