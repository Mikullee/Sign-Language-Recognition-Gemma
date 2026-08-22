from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

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


class SegmentSessionRecorderTests(unittest.TestCase):
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
                top1=SimpleNamespace(label_id="K42_09", display_text="請再說一次", confidence=0.7),
                top3=(
                    SimpleNamespace(label_id="K42_09", display_text="請再說一次", confidence=0.7),
                    SimpleNamespace(label_id="K42_10", display_text="沒有", confidence=0.2),
                    SimpleNamespace(label_id="K42_11", display_text="有", confidence=0.1),
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
            exact = summary.session_dir / rows[0]["exact_clip"]
            context = summary.session_dir / rows[0]["context_clip"]
            self.assertEqual(count_frames(exact), 4)
            self.assertEqual(count_frames(context), 12)
            metadata = json.loads((summary.session_dir / "metadata" / "segment_0001.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["reason"], "visible_rest_finalize")
            self.assertEqual(metadata["rest_detected_sec"], 10.6)
            self.assertEqual(metadata["boundary_policy"], "low_motion_anchor_v1")
            self.assertEqual([item["label_id"] for item in metadata["top3"]], ["K42_09", "K42_10", "K42_11"])
            jsonl = json.loads((summary.session_dir / "segments.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(jsonl["rest_detected_sec"], 10.6)
            self.assertEqual(jsonl["boundary_policy"], "low_motion_anchor_v1")

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
