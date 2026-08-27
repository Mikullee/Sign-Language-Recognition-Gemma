from __future__ import annotations

import ctypes
import dataclasses
import inspect
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import recognition.realtime.knee42_display as knee42_display

from recognition.realtime.knee42_display import (
    ResizableDisplay,
    _load_panel_font,
    application_layout,
    draw_information_panel,
    draw_landmark_graph,
    initial_window_size,
    letterbox_rect,
    render_application_canvas,
    render_application_view,
    render_letterboxed,
    resize_interpolation,
    windows_primary_screen_size,
    wrap_panel_lines,
)


class FakeCv2Window:
    WINDOW_NORMAL = 1
    WINDOW_KEEPRATIO = 2
    WND_PROP_FULLSCREEN = 3
    WINDOW_FULLSCREEN = 4

    def __init__(self):
        self.calls = []
        self.rect = (100, 50, 2000, 1100)

    def namedWindow(self, title, flags):
        self.calls.append(("named", title, flags))

    def resizeWindow(self, title, width, height):
        self.calls.append(("resize", title, width, height))

    def moveWindow(self, title, x, y):
        self.calls.append(("move", title, x, y))

    def getWindowImageRect(self, title):
        self.calls.append(("rect", title))
        return self.rect

    def setWindowProperty(self, title, prop, value):
        self.calls.append(("property", title, prop, value))


class FakeCv2Drawing:
    LINE_AA = 16

    def __init__(self):
        self.lines = []
        self.circles = []

    def line(self, _image, start, end, color, thickness, line_type):
        self.lines.append((start, end, color, thickness, line_type))

    def circle(self, _image, center, radius, color, thickness, line_type):
        self.circles.append((center, radius, color, thickness, line_type))

    @staticmethod
    def addWeighted(overlay, _alpha, _base, _beta, _gamma):
        return overlay


class RecordingColorCv2:
    COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
    COLOR_RGB2BGR = cv2.COLOR_RGB2BGR

    def __init__(self):
        self.converted_shapes = []

    def cvtColor(self, image, code):
        self.converted_shapes.append(tuple(image.shape))
        return cv2.cvtColor(image, code)


class RecordingFontLoader:
    def __init__(self):
        self.sizes = []

    def __call__(self, size):
        self.sizes.append(int(size))
        return ImageFont.load_default()


class FakeUser32:
    def __init__(self):
        self.dpi_contexts = []

    def SetProcessDpiAwarenessContext(self, context):
        self.dpi_contexts.append(context.value)
        return 1

    def GetSystemMetrics(self, index):
        return {0: 2560, 1: 1440}[index]


class WindowGeometryTests(unittest.TestCase):
    def test_2k_window_starts_at_85_percent_and_exceeds_two_thirds(self):
        width, height = initial_window_size((2560, 1440), coverage=0.85)

        self.assertEqual((width, height), (2176, 1224))
        self.assertGreaterEqual(width, 2560 * 2 / 3)
        self.assertGreaterEqual(height, 1440 * 2 / 3)

    def test_initial_size_is_clamped_to_screen_and_a_practical_minimum(self):
        self.assertEqual(initial_window_size((800, 600), coverage=0.85), (800, 600))
        self.assertEqual(initial_window_size((3840, 2160), coverage=1.5), (3840, 2160))

    def test_letterbox_preserves_ratio_and_centers_horizontal_video(self):
        rect = letterbox_rect((1280, 720), (1800, 1200))

        self.assertEqual(rect, (0, 94, 1800, 1012))
        self.assertAlmostEqual(rect[2] / rect[3], 16 / 9, places=2)

    def test_letterbox_adds_side_bars_for_portrait_source(self):
        x, y, width, height = letterbox_rect((720, 1280), (1800, 1200))

        self.assertEqual((y, height), (0, 1200))
        self.assertGreater(x, 0)
        self.assertLess(width, 1800)


class HighQualityRenderTests(unittest.TestCase):
    def test_display_prediction_and_e2_panel_use_raw_probability_semantics(self):
        field_names = {
            field.name for field in dataclasses.fields(knee42_display.DisplayPrediction)
        }
        source = inspect.getsource(knee42_display.draw_e2_information_panel)

        self.assertIn("raw_probability", field_names)
        self.assertNotIn("confidence", field_names)
        self.assertIn("RAW PROBABILITY", source)
        self.assertNotIn("confidence", source.lower())

        prediction = knee42_display.DisplayPrediction("K42_01", "你好", 0.25)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(prediction.confidence, prediction.raw_probability)

    def test_state_lamp_palette_changes_with_runtime_state(self):
        mapper = getattr(knee42_display, "state_lamp_bgr", None)

        self.assertTrue(callable(mapper))
        self.assertEqual(mapper("WAITING"), (235, 206, 135))
        self.assertEqual(mapper("SIGNING"), (99, 91, 255))
        self.assertEqual(mapper("END_CONFIRM"), (89, 189, 255))
        self.assertEqual(mapper("RESULT"), (139, 217, 85))
        self.assertEqual(mapper("ERROR"), (95, 54, 255))

    def test_e2_renderer_uses_large_top1_and_crop_only_colour_conversion(self):
        renderer = getattr(knee42_display, "draw_e2_information_panel", None)
        self.assertTrue(callable(renderer))
        source = np.full((108, 192, 3), 80, dtype=np.uint8)
        canvas, layout = render_application_canvas(
            source,
            (2560, 1440),
            cv2_module=cv2,
        )
        before = canvas.copy()
        data = knee42_display.DisplayPanelData(
            top1=knee42_display.DisplayPrediction("K42_12", "可以", 0.268),
            top3=(
                knee42_display.DisplayPrediction("K42_12", "可以", 0.268),
                knee42_display.DisplayPrediction("K42_09", "我聽不懂", 0.061),
                knee42_display.DisplayPrediction("K42_03", "晚安", 0.060),
            ),
            fps=12.6,
            source="camera:0",
            mode="auto",
            state="WAITING",
            recording=False,
            recorded_segments=0,
            model_version="v11",
        )
        cv2_module = RecordingColorCv2()
        font_loader = RecordingFontLoader()

        rendered = renderer(
            canvas,
            layout,
            data,
            cv2_module=cv2_module,
            font_loader=font_loader,
        )

        self.assertEqual(rendered.shape, canvas.shape)
        self.assertGreaterEqual(max(font_loader.sizes), 48)
        self.assertTrue(cv2_module.converted_shapes)
        self.assertNotIn(canvas.shape, cv2_module.converted_shapes)
        x, y, width, height = layout.video_rect
        np.testing.assert_array_equal(
            rendered[y : y + height // 2, x : x + width // 2],
            before[y : y + height // 2, x : x + width // 2],
        )

    def test_default_panel_font_loader_returns_the_first_windows_cjk_font(self):
        expected = object()
        _load_panel_font.cache_clear()

        with mock.patch.object(Path, "is_file", return_value=True), mock.patch(
            "PIL.ImageFont.truetype",
            return_value=expected,
        ) as truetype:
            actual = _load_panel_font(22)

        self.assertIs(actual, expected)
        self.assertEqual(truetype.call_count, 1)

    def test_panel_lines_wrap_by_rendered_pixel_width_without_losing_text(self):
        draw = ImageDraw.Draw(Image.new("RGB", (200, 100)))
        font = ImageFont.load_default()
        source = "Top-1: K42_03 raw probability 14.6%"

        rows = wrap_panel_lines(draw, [source], font, max_width=45)

        self.assertGreater(len(rows), 1)
        self.assertEqual("".join(rows), source)
        self.assertTrue(
            all(draw.textbbox((0, 0), row, font=font)[2] <= 45 for row in rows)
        )

    def test_application_layout_reserves_non_overlapping_panel_and_video_viewport(self):
        layout = application_layout((1920, 1080), (2176, 1224))

        self.assertEqual(layout.canvas_size, (2176, 1224))
        panel_x, panel_y, panel_width, panel_height = layout.panel_rect
        viewport_x, viewport_y, viewport_width, viewport_height = layout.video_viewport_rect
        video_x, video_y, video_width, video_height = layout.video_rect
        self.assertEqual((panel_y, viewport_x, viewport_y), (0, 0, 0))
        self.assertEqual(panel_x, viewport_width)
        self.assertEqual(panel_height, viewport_height)
        self.assertEqual(panel_width + viewport_width, 2176)
        self.assertGreaterEqual(video_x, viewport_x)
        self.assertGreaterEqual(video_y, viewport_y)
        self.assertLessEqual(video_x + video_width, viewport_x + viewport_width)
        self.assertLessEqual(video_y + video_height, viewport_y + viewport_height)
        self.assertAlmostEqual(video_width / video_height, 16 / 9, places=2)

    def test_e2_layout_places_video_left_panel_right_and_footer_below(self):
        layout = application_layout((1920, 1080), (2560, 1440))

        footer_rect = getattr(layout, "footer_rect", None)
        self.assertIsNotNone(footer_rect)
        assert footer_rect is not None
        panel_x, panel_y, panel_width, panel_height = layout.panel_rect
        viewport_x, viewport_y, viewport_width, viewport_height = layout.video_viewport_rect
        footer_x, footer_y, footer_width, footer_height = footer_rect
        self.assertEqual((viewport_x, viewport_y), (0, 0))
        self.assertEqual(panel_x, viewport_width)
        self.assertEqual(panel_y, 0)
        self.assertEqual(panel_height, viewport_height)
        self.assertEqual(panel_x + panel_width, 2560)
        self.assertEqual((footer_x, footer_width), (0, 2560))
        self.assertEqual(footer_y, viewport_height)
        self.assertEqual(footer_y + footer_height, 1440)

    def test_application_canvas_keeps_panel_black_and_does_not_mutate_source(self):
        source = np.full((90, 160, 3), 200, dtype=np.uint8)
        original = source.copy()

        canvas, layout = render_application_canvas(
            source,
            (800, 450),
            cv2_module=cv2,
        )

        panel_x, panel_y, panel_width, panel_height = layout.panel_rect
        video_x, video_y, video_width, video_height = layout.video_rect
        self.assertTrue(np.array_equal(source, original))
        self.assertTrue(
            np.all(canvas[panel_y : panel_y + panel_height, panel_x : panel_x + panel_width] == 0)
        )
        self.assertTrue(
            np.all(canvas[video_y : video_y + video_height, video_x : video_x + video_width] == 200)
        )

    def test_landmark_graph_draws_sky_blue_connections_on_a_copy(self):
        source = np.zeros((100, 100, 3), dtype=np.uint8)
        pose = np.full((33, 3), np.nan, dtype=np.float32)
        pose[11] = (0.20, 0.20, 0.0)
        pose[13] = (0.50, 0.50, 0.0)
        left = np.full((21, 3), np.nan, dtype=np.float32)
        left[0] = (0.10, 0.90, 0.0)
        left[1] = (0.20, 0.80, 0.0)

        rendered = draw_landmark_graph(
            source,
            pose,
            left,
            None,
            cv2_module=cv2,
        )

        self.assertFalse(np.array_equal(rendered, source))
        self.assertTrue(np.all(source == 0))
        coloured = rendered[np.any(rendered > 0, axis=2)]
        self.assertGreater(len(coloured), 0)
        self.assertGreater(float(coloured[:, 0].mean()), float(coloured[:, 2].mean()))

    def test_landmark_graph_ignores_missing_and_out_of_frame_points(self):
        source = np.full((40, 60, 3), 17, dtype=np.uint8)
        pose = np.full((33, 3), np.nan, dtype=np.float32)
        pose[11] = (-0.2, 0.5, 0.0)
        pose[13] = (1.2, 0.5, 0.0)

        rendered = draw_landmark_graph(
            source,
            pose,
            None,
            None,
            cv2_module=cv2,
        )

        self.assertTrue(np.array_equal(rendered, source))

    def test_landmark_graph_draws_knees_but_not_ankles_or_feet(self):
        source = np.zeros((101, 101, 3), dtype=np.uint8)
        pose = np.full((33, 3), np.nan, dtype=np.float32)
        pose[23] = (0.20, 0.20, 0.0)
        pose[24] = (0.80, 0.20, 0.0)
        pose[25] = (0.20, 0.50, 0.0)
        pose[26] = (0.80, 0.50, 0.0)
        pose[27] = (0.20, 0.80, 0.0)
        pose[28] = (0.80, 0.80, 0.0)
        cv2_module = FakeCv2Drawing()

        draw_landmark_graph(
            source,
            pose,
            None,
            None,
            cv2_module=cv2_module,
            alpha=1.0,
        )

        centers = {item[0] for item in cv2_module.circles}
        self.assertIn((20, 50), centers)
        self.assertIn((80, 50), centers)
        self.assertNotIn((20, 80), centers)
        self.assertNotIn((80, 80), centers)
        self.assertTrue(
            all(start[1] <= 50 and end[1] <= 50 for start, end, *_ in cv2_module.lines)
        )

    def test_information_text_changes_only_the_black_panel(self):
        source = np.full((90, 160, 3), 80, dtype=np.uint8)
        canvas, layout = render_application_canvas(source, (800, 450), cv2_module=cv2)
        before = canvas.copy()

        rendered = draw_information_panel(
            canvas,
            layout,
            ["Top-1: K42_03", "raw probability 14.6%", "FPS 21.0"],
            cv2_module=cv2,
            font_loader=lambda _size: ImageFont.load_default(),
        )

        panel_x, panel_y, panel_width, panel_height = layout.panel_rect
        viewport_x, viewport_y, viewport_width, viewport_height = layout.video_viewport_rect
        self.assertTrue(
            np.any(rendered[panel_y : panel_y + panel_height, panel_x : panel_x + panel_width] != 0)
        )
        np.testing.assert_array_equal(
            rendered[viewport_y : viewport_y + viewport_height, viewport_x : viewport_x + viewport_width],
            before[viewport_y : viewport_y + viewport_height, viewport_x : viewport_x + viewport_width],
        )

    def test_full_application_view_keeps_raw_frame_out_of_ui_composition(self):
        source = np.full((100, 160, 3), 25, dtype=np.uint8)
        original = source.copy()
        pose = np.full((33, 3), np.nan, dtype=np.float32)
        pose[11] = (0.25, 0.25, 0.0)
        pose[13] = (0.50, 0.50, 0.0)

        rendered, layout = render_application_view(
            source,
            (900, 540),
            ["Top-1: K42_03", "晚安"],
            pose=pose,
            left_hand=None,
            right_hand=None,
            cv2_module=cv2,
            font_loader=lambda _size: ImageFont.load_default(),
        )

        self.assertTrue(np.array_equal(source, original))
        self.assertEqual(rendered.shape, (540, 900, 3))
        self.assertLess(layout.video_rect[0], layout.panel_rect[0])

    def test_interpolation_uses_area_down_and_cubic_up(self):
        self.assertEqual(resize_interpolation((1920, 1080), (1280, 720), cv2), cv2.INTER_AREA)
        self.assertEqual(resize_interpolation((1280, 720), (1920, 1080), cv2), cv2.INTER_CUBIC)

    def test_render_adds_black_bars_without_changing_source(self):
        source = np.full((100, 100, 3), 255, dtype=np.uint8)
        original = source.copy()

        canvas = render_letterboxed(source, (200, 100), cv2_module=cv2)

        self.assertEqual(canvas.shape, (100, 200, 3))
        self.assertTrue(np.array_equal(source, original))
        self.assertTrue(np.all(canvas[:, :50] == 0))
        self.assertTrue(np.all(canvas[:, 50:150] == 255))
        self.assertTrue(np.all(canvas[:, 150:] == 0))


class ResizableDisplayTests(unittest.TestCase):
    def test_windows_screen_size_enables_per_monitor_dpi_before_reading_2k_metrics(self):
        user32 = FakeUser32()

        size = windows_primary_screen_size(user32=user32)

        self.assertEqual(size, (2560, 1440))
        self.assertEqual(user32.dpi_contexts, [ctypes.c_void_p(-4).value])

    def test_create_sizes_and_centers_a_large_2k_window(self):
        cv2_module = FakeCv2Window()
        display = ResizableDisplay("Knee42 IVCAM RGB", (2560, 1440), coverage=0.85)

        display.create(cv2_module)

        self.assertIn(("resize", "Knee42 IVCAM RGB", 2176, 1224), cv2_module.calls)
        self.assertIn(("move", "Knee42 IVCAM RGB", 192, 108), cv2_module.calls)
        self.assertEqual(display.content_size(cv2_module), (2000, 1100))

    def test_fullscreen_toggle_returns_to_the_large_window(self):
        cv2_module = FakeCv2Window()
        display = ResizableDisplay("Knee42 IVCAM RGB", (2560, 1440), coverage=0.85)
        display.create(cv2_module)

        self.assertTrue(display.toggle_fullscreen(cv2_module))
        self.assertIn(
            ("property", "Knee42 IVCAM RGB", cv2_module.WND_PROP_FULLSCREEN, cv2_module.WINDOW_FULLSCREEN),
            cv2_module.calls,
        )
        self.assertFalse(display.toggle_fullscreen(cv2_module))
        self.assertEqual(cv2_module.calls[-2:], [
            ("resize", "Knee42 IVCAM RGB", 2176, 1224),
            ("move", "Knee42 IVCAM RGB", 192, 108),
        ])


if __name__ == "__main__":
    unittest.main()
