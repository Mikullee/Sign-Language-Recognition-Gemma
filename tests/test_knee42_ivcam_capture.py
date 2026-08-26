from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from recognition.realtime.knee42_capture import open_camera, open_video
from recognition.realtime.knee42_controllers import ManualController, SlidingController


class FakeCapture:
    def __init__(
        self,
        opened: bool,
        frames=None,
        fps: float = 30.0,
        pos_msec=None,
    ):
        self.opened = opened
        self.frames = list(frames or [])
        self.fps_value = fps
        self.pos_msec = list(pos_msec or [])
        self.last_pos_msec = float("nan")
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.frames:
            return False, None
        if self.pos_msec:
            self.last_pos_msec = self.pos_msec.pop(0)
        return True, self.frames.pop(0)

    def release(self):
        self.released = True

    def get(self, property_id):
        if property_id == FakeCv2.CAP_PROP_POS_MSEC:
            return self.last_pos_msec
        return self.fps_value


class FakeCv2:
    CAP_DSHOW = 700
    CAP_PROP_FPS = 5
    CAP_PROP_POS_MSEC = 0

    def __init__(
        self,
        opened_indices=(),
        camera_frames=None,
        video_frames=None,
        video_pos_msec=None,
    ):
        self.opened_indices = set(opened_indices)
        self.camera_frames = list(camera_frames or [])
        self.video_frames = list(video_frames or [])
        self.video_pos_msec = list(video_pos_msec or [])
        self.calls = []
        self.captures = []

    def VideoCapture(self, source, *backend):
        self.calls.append((source, *backend))
        if isinstance(source, int):
            frames = self.camera_frames or [f"camera-{source}"]
            capture = FakeCapture(source in self.opened_indices, frames=frames)
        else:
            capture = FakeCapture(
                bool(self.video_frames),
                frames=self.video_frames,
                fps=25.0,
                pos_msec=self.video_pos_msec,
            )
        self.captures.append(capture)
        return capture


class CaptureTests(unittest.TestCase):
    def test_explicit_windows_camera_uses_directshow_and_color_read(self):
        fake_cv2 = FakeCv2(opened_indices={2})

        source = open_camera(2, cv2_module=fake_cv2, platform_name="win32")

        self.assertEqual(fake_cv2.calls, [(2, fake_cv2.CAP_DSHOW)])
        self.assertEqual(source.read(), (True, "camera-2"))
        self.assertEqual(source.status, "camera:2")
        source.release()
        self.assertTrue(fake_cv2.captures[0].released)

    def test_camera_discovery_is_bounded_and_releases_failed_probes(self):
        fake_cv2 = FakeCv2(opened_indices={2})

        source = open_camera(None, max_index=3, cv2_module=fake_cv2, platform_name="win32")

        self.assertEqual(
            fake_cv2.calls,
            [(0, fake_cv2.CAP_DSHOW), (1, fake_cv2.CAP_DSHOW), (2, fake_cv2.CAP_DSHOW)],
        )
        self.assertTrue(fake_cv2.captures[0].released)
        self.assertTrue(fake_cv2.captures[1].released)
        self.assertFalse(fake_cv2.captures[2].released)
        source.release()

    def test_camera_packets_use_injected_live_clock_not_reported_fps(self):
        fake_cv2 = FakeCv2(
            opened_indices={2},
            camera_frames=["camera-a", "camera-b", "camera-c"],
        )
        perf_counter = iter((100.0, 100.055, 100.310)).__next__
        source = open_camera(
            2,
            cv2_module=fake_cv2,
            platform_name="win32",
            perf_counter=perf_counter,
        )

        packets = [source.read_packet() for _ in range(3)]

        self.assertEqual([packet.frame for packet in packets], ["camera-a", "camera-b", "camera-c"])
        self.assertEqual(packets[0].timestamp_sec, 0.0)
        self.assertAlmostEqual(packets[1].timestamp_sec, 0.055)
        self.assertAlmostEqual(packets[2].timestamp_sec, 0.310)
        self.assertTrue(all(packet.clock_mode == "live_perf_counter" for packet in packets))
        self.assertEqual(source.clock_mode, "live_perf_counter")

    def test_no_camera_releases_every_probe_and_fails(self):
        fake_cv2 = FakeCv2()

        with self.assertRaisesRegex(RuntimeError, "camera indices 0..2"):
            open_camera(None, max_index=2, cv2_module=fake_cv2, platform_name="win32")

        self.assertEqual(len(fake_cv2.captures), 3)
        self.assertTrue(all(item.released for item in fake_cv2.captures))

    def test_video_fallback_has_same_read_release_fps_interface(self):
        fake_cv2 = FakeCv2(video_frames=["frame-a", "frame-b"])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "color.avi"
            path.write_bytes(b"fixture")

            source = open_video(path, cv2_module=fake_cv2)

            self.assertEqual(fake_cv2.calls, [(str(path),)])
            self.assertEqual(source.fps, 25.0)
            self.assertEqual(source.status, f"video:{path}")
            self.assertEqual(source.read(), (True, "frame-a"))
            source.release()
            self.assertTrue(fake_cv2.captures[0].released)

    def test_video_packets_preserve_vfr_pos_msec_timestamps(self):
        fake_cv2 = FakeCv2(
            video_frames=["frame-a", "frame-b", "frame-c"],
            video_pos_msec=[0.0, 41.0, 127.0],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "color.avi"
            path.write_bytes(b"fixture")
            source = open_video(path, cv2_module=fake_cv2)

            packets = [source.read_packet() for _ in range(3)]

        self.assertEqual([packet.frame for packet in packets], ["frame-a", "frame-b", "frame-c"])
        self.assertEqual(packets[0].timestamp_sec, 0.0)
        self.assertAlmostEqual(packets[1].timestamp_sec, 0.041)
        self.assertAlmostEqual(packets[2].timestamp_sec, 0.127)
        self.assertEqual(
            [packet.clock_mode for packet in packets],
            [
                "video_timestamp_pending",
                "video_source_timestamp",
                "video_source_timestamp",
            ],
        )
        self.assertEqual(source.clock_mode, "video_source_timestamp")

    def test_video_packet_fallback_is_sticky_after_pos_msec_stalls(self):
        fake_cv2 = FakeCv2(
            video_frames=["frame-a", "frame-b", "frame-c"],
            video_pos_msec=[0.0, 0.0, 127.0],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "color.avi"
            path.write_bytes(b"fixture")
            source = open_video(path, cv2_module=fake_cv2)

            packets = [source.read_packet() for _ in range(3)]

        self.assertEqual(packets[0].timestamp_sec, 0.0)
        self.assertAlmostEqual(packets[1].timestamp_sec, 1.0 / 25.0)
        self.assertAlmostEqual(packets[2].timestamp_sec, 2.0 / 25.0)
        self.assertEqual(
            [packet.clock_mode for packet in packets],
            [
                "video_timestamp_pending",
                "video_nominal_fps_fallback",
                "video_nominal_fps_fallback",
            ],
        )

    def test_one_frame_video_resolves_pending_mode_when_eof_is_observed(self):
        fake_cv2 = FakeCv2(
            video_frames=["frame-a"],
            video_pos_msec=[0.0],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "color.avi"
            path.write_bytes(b"fixture")
            source = open_video(path, cv2_module=fake_cv2)

            packet = source.read_packet()
            eof = source.read_packet()

        self.assertEqual(packet.timestamp_sec, 0.0)
        self.assertEqual(packet.clock_mode, "video_timestamp_pending")
        self.assertIsNone(eof)
        self.assertEqual(source.clock_mode, "video_nominal_fps_fallback")


class ControllerTests(unittest.TestCase):
    def test_manual_space_starts_and_stops_recording_for_inference(self):
        controller = ManualController()

        self.assertEqual(controller.on_space().state, "recording")
        controller.add_feature("one")
        controller.add_feature("two")
        event = controller.on_space()

        self.assertEqual(event.state, "infer")
        self.assertEqual(event.features, ("one", "two"))
        self.assertEqual(controller.state, "infer")
        controller.mark_result()
        self.assertEqual(controller.state, "result")
        self.assertEqual(controller.on_space().state, "recording")

    def test_manual_empty_recording_enters_error_and_reset_recovers(self):
        controller = ManualController()
        controller.on_space()

        event = controller.on_space()

        self.assertEqual(event.state, "error")
        self.assertIn("no feature", event.message)
        self.assertEqual(controller.reset().state, "idle")

    def test_sliding_window_keeps_latest_64_and_throttles_inference(self):
        controller = SlidingController(window=64, inference_stride=4)
        ready = []

        for index in range(70):
            event = controller.add_feature(index)
            if event.infer:
                ready.append((index, event.features))

        self.assertEqual(controller.features, list(range(6, 70)))
        self.assertEqual([index for index, _ in ready], [63, 67])
        self.assertTrue(all(len(features) == 64 for _, features in ready))


if __name__ == "__main__":
    unittest.main()
