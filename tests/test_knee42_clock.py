from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError

from recognition.realtime.knee42_clock import FramePacket, LiveClock, VideoClock


class FramePacketTests(unittest.TestCase):
    def test_packet_is_immutable(self):
        packet = FramePacket(frame="frame", timestamp_sec=0.0, clock_mode="test")

        with self.assertRaises(FrozenInstanceError):
            packet.timestamp_sec = 1.0


class LiveClockTests(unittest.TestCase):
    def test_variable_and_stalled_cadence_uses_perf_counter_elapsed_time(self):
        readings = iter((10.0, 10.04, 10.04, 10.19))
        clock = LiveClock(perf_counter=readings.__next__)

        timestamps = [clock.next_timestamp() for _ in range(4)]

        self.assertEqual(timestamps[0], 0.0)
        self.assertAlmostEqual(timestamps[1], 0.04)
        self.assertAlmostEqual(timestamps[2], 0.04)
        self.assertAlmostEqual(timestamps[3], 0.19)
        self.assertEqual(clock.clock_mode, "live_perf_counter")

    def test_nonfinite_perf_counter_reading_is_rejected(self):
        for reading in (math.nan, math.inf, -math.inf):
            with self.subTest(reading=reading):
                clock = LiveClock(perf_counter=iter((reading,)).__next__)
                with self.assertRaisesRegex(ValueError, "finite"):
                    clock.next_timestamp()

    def test_regressing_perf_counter_reading_is_rejected(self):
        clock = LiveClock(perf_counter=iter((10.0, 10.1, 10.09)).__next__)
        clock.next_timestamp()
        clock.next_timestamp()

        with self.assertRaisesRegex(ValueError, "regress|monotonic"):
            clock.next_timestamp()


class VideoClockTests(unittest.TestCase):
    def test_vfr_source_timestamps_are_preserved(self):
        clock = VideoClock(nominal_fps=30.0)
        timestamps = []
        modes = []

        for frame_index, pos_msec in enumerate((0.0, 41.0, 127.0), start=1):
            timestamps.append(
                clock.next_timestamp(pos_msec=pos_msec, frame_index=frame_index)
            )
            modes.append(clock.clock_mode)

        self.assertEqual(timestamps[0], 0.0)
        self.assertAlmostEqual(timestamps[1], 0.041)
        self.assertAlmostEqual(timestamps[2], 0.127)
        self.assertEqual(
            modes,
            [
                "video_timestamp_pending",
                "video_source_timestamp",
                "video_source_timestamp",
            ],
        )

    def test_repeated_zero_selects_sticky_nominal_fps_fallback(self):
        clock = VideoClock(nominal_fps=25.0)

        first = clock.next_timestamp(pos_msec=0.0, frame_index=1)
        second = clock.next_timestamp(pos_msec=0.0, frame_index=2)
        third = clock.next_timestamp(pos_msec=127.0, frame_index=3)

        self.assertEqual(first, 0.0)
        self.assertAlmostEqual(second, 1.0 / 25.0)
        self.assertAlmostEqual(third, 2.0 / 25.0)
        self.assertEqual(clock.clock_mode, "video_nominal_fps_fallback")

    def test_invalid_source_timestamp_selects_sticky_fallback(self):
        clock = VideoClock(nominal_fps=20.0)

        first = clock.next_timestamp(pos_msec=math.nan, frame_index=1)
        second = clock.next_timestamp(pos_msec=50.0, frame_index=2)

        self.assertEqual(first, 0.0)
        self.assertAlmostEqual(second, 1.0 / 20.0)
        self.assertEqual(clock.clock_mode, "video_nominal_fps_fallback")

    def test_pending_mode_resolves_to_fallback_at_video_eof(self):
        clock = VideoClock(nominal_fps=30.0)
        self.assertEqual(
            clock.next_timestamp(pos_msec=0.0, frame_index=1),
            0.0,
        )
        self.assertEqual(clock.clock_mode, "video_timestamp_pending")

        clock.finalize()

        self.assertEqual(clock.clock_mode, "video_nominal_fps_fallback")

    def test_regressing_source_timestamp_selects_sticky_monotonic_fallback(self):
        clock = VideoClock(nominal_fps=30.0)
        clock.next_timestamp(pos_msec=0.0, frame_index=1)
        clock.next_timestamp(pos_msec=41.0, frame_index=2)

        third = clock.next_timestamp(pos_msec=20.0, frame_index=3)
        fourth = clock.next_timestamp(pos_msec=200.0, frame_index=4)

        self.assertAlmostEqual(third, 2.0 / 30.0)
        self.assertAlmostEqual(fourth, 3.0 / 30.0)
        self.assertEqual(clock.clock_mode, "video_nominal_fps_fallback")

    def test_stalled_source_timestamp_selects_sticky_nominal_fallback(self):
        clock = VideoClock(nominal_fps=25.0)

        timestamps = [
            clock.next_timestamp(pos_msec=0.0, frame_index=1),
            clock.next_timestamp(pos_msec=41.0, frame_index=2),
            clock.next_timestamp(pos_msec=41.0, frame_index=3),
        ]
        later = clock.next_timestamp(pos_msec=500.0, frame_index=4)

        self.assertEqual(timestamps[0], 0.0)
        self.assertAlmostEqual(timestamps[1], 0.041)
        self.assertAlmostEqual(timestamps[2], 0.08)
        self.assertAlmostEqual(later, 0.12)
        self.assertEqual(clock.clock_mode, "video_nominal_fps_fallback")

    def test_fallback_anchors_after_source_gets_ahead_of_nominal_cadence(self):
        clock = VideoClock(nominal_fps=25.0)
        clock.next_timestamp(pos_msec=0.0, frame_index=1)
        second = clock.next_timestamp(pos_msec=200.0, frame_index=2)

        third = clock.next_timestamp(pos_msec=200.0, frame_index=3)
        fourth = clock.next_timestamp(pos_msec=500.0, frame_index=4)

        self.assertAlmostEqual(second, 0.2)
        self.assertAlmostEqual(third, 0.24)
        self.assertAlmostEqual(fourth, 0.28)
        self.assertEqual(clock.clock_mode, "video_nominal_fps_fallback")

    def test_invalid_nominal_fps_is_rejected(self):
        for fps in (0.0, -1.0, math.nan, math.inf, -math.inf):
            with self.subTest(fps=fps):
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    VideoClock(nominal_fps=fps)


if __name__ == "__main__":
    unittest.main()
