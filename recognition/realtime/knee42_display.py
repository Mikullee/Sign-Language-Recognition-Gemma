"""DPI-aware, aspect-preserving display helpers for Knee42 IVCAM."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


SKY_BLUE_BGR = (235, 206, 135)
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26),
)
POSE_DISPLAY_INDICES = frozenset(range(27))
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


@dataclass(frozen=True)
class ApplicationLayout:
    """Non-overlapping operator panel and aspect-preserving video regions."""

    canvas_size: tuple[int, int]
    panel_rect: tuple[int, int, int, int]
    video_viewport_rect: tuple[int, int, int, int]
    video_rect: tuple[int, int, int, int]
    footer_rect: tuple[int, int, int, int]


@dataclass(frozen=True)
class DisplayPrediction:
    label_id: str
    display_text: str
    confidence: float


@dataclass(frozen=True)
class DisplayPanelData:
    top1: DisplayPrediction | None
    top3: tuple[DisplayPrediction, ...]
    fps: float
    source: str
    mode: str
    state: str
    recording: bool
    recorded_segments: int
    model_version: str


def state_lamp_bgr(state: str) -> tuple[int, int, int]:
    """Return the operator-state lamp colour in OpenCV BGR order."""
    normalized = str(state).strip().upper()
    if normalized in {"SIGNING", "RECORDING"}:
        return (99, 91, 255)
    if normalized in {"END_CONFIRM", "PROCESSING"}:
        return (89, 189, 255)
    if normalized in {"RESULT", "RESULT READY", "COOLDOWN"}:
        return (139, 217, 85)
    if normalized == "ERROR":
        return (95, 54, 255)
    return SKY_BLUE_BGR


def _display_points(
    landmarks: np.ndarray | None,
    expected_count: int,
    frame_size: tuple[int, int],
) -> tuple[list[tuple[int, int] | None], np.ndarray]:
    if landmarks is None:
        values = np.full((expected_count, 3), np.nan, dtype=np.float32)
    else:
        values = np.asarray(landmarks, dtype=np.float32)
        if values.shape != (expected_count, 3):
            raise ValueError(f"expected {expected_count}x3 landmarks, got {values.shape}")
    width, height = frame_size
    valid = (
        np.isfinite(values[:, :2]).all(axis=1)
        & (values[:, 0] >= 0.0)
        & (values[:, 0] <= 1.0)
        & (values[:, 1] >= 0.0)
        & (values[:, 1] <= 1.0)
    )
    points: list[tuple[int, int] | None] = []
    for index, item in enumerate(values):
        if not valid[index]:
            points.append(None)
            continue
        points.append(
            (
                min(width - 1, max(0, round(float(item[0]) * (width - 1)))),
                min(height - 1, max(0, round(float(item[1]) * (height - 1)))),
            )
        )
    return points, valid


def draw_landmark_graph(
    frame_bgr: np.ndarray,
    pose: np.ndarray | None,
    left_hand: np.ndarray | None,
    right_hand: np.ndarray | None,
    *,
    cv2_module: Any,
    color: tuple[int, int, int] = SKY_BLUE_BGR,
    alpha: float = 0.42,
    line_thickness: int = 1,
    node_radius: int = 2,
) -> np.ndarray:
    """Draw a subtle MediaPipe graph on a copy used only for display."""
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR frame, got {frame.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    base = frame.copy()
    overlay = base.copy()
    line_type = int(getattr(cv2_module, "LINE_AA", 8))
    groups = (
        (pose, 33, POSE_CONNECTIONS, POSE_DISPLAY_INDICES),
        (left_hand, 21, HAND_CONNECTIONS, None),
        (right_hand, 21, HAND_CONNECTIONS, None),
    )
    drew_anything = False
    for landmarks, count, connections, visible_indices in groups:
        points, valid = _display_points(
            landmarks,
            count,
            (int(frame.shape[1]), int(frame.shape[0])),
        )
        if visible_indices is not None:
            valid = valid.copy()
            for index in range(count):
                if index not in visible_indices:
                    valid[index] = False
        for start, end in connections:
            if valid[start] and valid[end]:
                cv2_module.line(
                    overlay,
                    points[start],
                    points[end],
                    color,
                    int(line_thickness),
                    line_type,
                )
                drew_anything = True
        for index, point in enumerate(points):
            if valid[index] and point is not None:
                cv2_module.circle(
                    overlay,
                    point,
                    int(node_radius),
                    color,
                    -1,
                    line_type,
                )
                drew_anything = True
    if not drew_anything:
        return base
    return cv2_module.addWeighted(overlay, float(alpha), base, 1.0 - float(alpha), 0.0)


@lru_cache(maxsize=8)
def _load_panel_font(size: int):
    from PIL import ImageFont

    candidates = (
        Path(r"C:\Windows\Fonts\msjh.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\mingliu.ttc"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    raise RuntimeError(
        "no CJK font found; expected msjh.ttc, msyh.ttc, or mingliu.ttc in C:\\Windows\\Fonts"
    )


@lru_cache(maxsize=8)
def _load_editorial_font(size: int):
    from PIL import ImageFont

    candidates = (
        Path(r"C:\Windows\Fonts\mingliu.ttc"),
        Path(r"C:\Windows\Fonts\msjh.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    raise RuntimeError(
        "no editorial CJK font found; expected mingliu.ttc, msjh.ttc, or msyh.ttc"
    )


def wrap_panel_lines(
    draw: Any,
    lines: Sequence[str],
    font: Any,
    *,
    max_width: int,
) -> list[str]:
    """Wrap mixed Chinese/ASCII text by rendered width while preserving characters."""
    if max_width <= 0:
        raise ValueError("max_width must be positive")
    rows: list[str] = []
    for logical_line in (str(item) for item in lines):
        if not logical_line:
            rows.append("")
            continue
        current = ""
        for character in logical_line:
            candidate = current + character
            bounds = draw.textbbox((0, 0), candidate, font=font)
            width = bounds[2] - bounds[0]
            if current and width > max_width:
                rows.append(current)
                current = character
            else:
                current = candidate
        rows.append(current)
    return rows


def draw_information_panel(
    canvas_bgr: np.ndarray,
    layout: ApplicationLayout,
    lines: Sequence[str],
    *,
    cv2_module: Any,
    font_loader: Callable[[int], Any] | None = None,
) -> np.ndarray:
    """Draw operator text only inside the layout's black panel rectangle."""
    from PIL import Image, ImageDraw

    canvas = np.asarray(canvas_bgr)
    expected_width, expected_height = layout.canvas_size
    if canvas.shape != (expected_height, expected_width, 3):
        raise ValueError(
            f"canvas shape {canvas.shape} does not match layout {layout.canvas_size}"
        )
    result = canvas.copy()
    if not lines:
        return result
    panel_x, panel_y, panel_width, panel_height = layout.panel_rect
    padding = max(10, min(20, panel_width // 24))
    available_width = max(1, panel_width - padding * 2)
    available_height = max(1, panel_height - padding * 2)
    loader = font_loader or _load_panel_font
    font_size = max(16, min(26, round(panel_width * 0.055)))
    rgb = cv2_module.cvtColor(result, cv2_module.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    while font_size > 12:
        font = loader(font_size)
        rows = wrap_panel_lines(
            draw,
            lines,
            font,
            max_width=available_width,
        )
        line_spacing = max(18, round(font_size * 1.42))
        if line_spacing * len(rows) <= available_height:
            break
        font_size -= 1
    font = loader(font_size)
    rows = wrap_panel_lines(
        draw,
        lines,
        font,
        max_width=available_width,
    )
    line_spacing = max(17, min(round(font_size * 1.42), available_height // max(len(rows), 1)))
    for index, line in enumerate(rows):
        y = panel_y + padding + index * line_spacing
        if y + line_spacing > panel_y + panel_height:
            break
        draw.text(
            (panel_x + padding, y),
            str(line),
            font=font,
            fill=(240, 245, 250),
        )
    return cv2_module.cvtColor(np.asarray(image), cv2_module.COLOR_RGB2BGR)


def _draw_rgb_crop(
    result: np.ndarray,
    rect: tuple[int, int, int, int],
    *,
    cv2_module: Any,
    painter: Callable[[Any, int, int], None],
) -> None:
    """Round-trip one UI crop through Pillow without touching the video canvas."""
    from PIL import Image, ImageDraw

    x, y, width, height = (int(item) for item in rect)
    if width <= 0 or height <= 0:
        return
    crop = result[y : y + height, x : x + width].copy()
    rgb = cv2_module.cvtColor(crop, cv2_module.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    painter(ImageDraw.Draw(image), width, height)
    result[y : y + height, x : x + width] = cv2_module.cvtColor(
        np.asarray(image),
        cv2_module.COLOR_RGB2BGR,
    )


def draw_e2_information_panel(
    canvas_bgr: np.ndarray,
    layout: ApplicationLayout,
    data: DisplayPanelData,
    *,
    cv2_module: Any,
    font_loader: Callable[[int], Any] | None = None,
) -> np.ndarray:
    """Draw the approved E2 right panel, footer, and model badge on small crops."""
    canvas = np.asarray(canvas_bgr)
    expected_width, expected_height = layout.canvas_size
    if canvas.shape != (expected_height, expected_width, 3):
        raise ValueError(
            f"canvas shape {canvas.shape} does not match layout {layout.canvas_size}"
        )
    result = canvas.copy()

    def ui_font(size: int):
        return (font_loader or _load_panel_font)(max(8, int(size)))

    def editorial_font(size: int):
        if font_loader is not None:
            return font_loader(max(8, int(size)))
        return _load_editorial_font(max(8, int(size)))

    panel_x, panel_y, panel_width, panel_height = layout.panel_rect
    scale = max(0.55, min(1.35, panel_width / 720.0))

    def paint_panel(draw: Any, width: int, height: int) -> None:
        draw.rectangle((0, 0, width, height), fill=(12, 13, 15))
        padding = max(14, round(24 * scale))
        y = max(16, round(25 * scale))
        overline_font = ui_font(round(12 * scale))
        answer_font = editorial_font(max(48, round(58 * scale)))
        id_font = ui_font(round(17 * scale))
        confidence_font = ui_font(round(27 * scale))
        candidate_font = ui_font(round(17 * scale))
        candidate_id_font = ui_font(round(10 * scale))
        score_font = ui_font(round(13 * scale))
        draw.text((padding, y), "RECOGNITION · TOP 1", font=overline_font, fill=(119, 133, 140))
        y += max(30, round(38 * scale))
        if data.top1 is None:
            answer_text = "等待中"
            label_text = "—"
            confidence_text = "—"
        else:
            answer_text = data.top1.display_text
            label_text = data.top1.label_id
            confidence_text = f"{data.top1.confidence:.1%}"
        draw.text((padding, y), answer_text, font=answer_font, fill=(153, 220, 243))
        answer_box = draw.textbbox((padding, y), answer_text, font=answer_font)
        y = answer_box[3] + max(10, round(10 * scale))
        draw.text((padding, y), label_text, font=id_font, fill=(168, 180, 186))
        y += max(28, round(34 * scale))
        draw.text((padding, y), confidence_text, font=confidence_font, fill=(255, 255, 255))
        y += max(44, round(54 * scale))
        draw.line((padding, y, width - padding, y), fill=(58, 66, 71), width=1)
        y += max(17, round(20 * scale))
        candidates = data.top3[1:3] if len(data.top3) >= 3 else data.top3[:2]
        for rank, candidate in enumerate(candidates, 2):
            row_top = y
            draw.text((padding, row_top), str(rank), font=score_font, fill=(111, 124, 130))
            text_x = padding + max(28, round(34 * scale))
            draw.text(
                (text_x, row_top),
                candidate.display_text,
                font=candidate_font,
                fill=(220, 227, 230),
            )
            draw.text(
                (text_x, row_top + max(27, round(31 * scale))),
                candidate.label_id,
                font=candidate_id_font,
                fill=(115, 127, 133),
            )
            score_text = f"{candidate.confidence:.1%}"
            score_box = draw.textbbox((0, 0), score_text, font=score_font)
            draw.text(
                (width - padding - (score_box[2] - score_box[0]), row_top),
                score_text,
                font=score_font,
                fill=(167, 178, 183),
            )
            y += max(62, round(76 * scale))
            draw.line((padding, y, width - padding, y), fill=(39, 41, 44), width=1)
            y += max(14, round(18 * scale))

    _draw_rgb_crop(
        result,
        layout.panel_rect,
        cv2_module=cv2_module,
        painter=paint_panel,
    )

    footer_x, footer_y, footer_width, footer_height = layout.footer_rect

    def paint_footer(draw: Any, width: int, height: int) -> None:
        draw.rectangle((0, 0, width, height), fill=(8, 11, 13))
        font = ui_font(max(9, round(height * 0.27)))
        baseline_y = max(4, (height - draw.textbbox((0, 0), "WAITING", font=font)[3]) // 2)
        lamp_radius = max(4, round(height * 0.085))
        lamp_x = max(12, round(height * 0.36))
        lamp_y = height // 2
        bgr = state_lamp_bgr(data.state)
        lamp_rgb = (bgr[2], bgr[1], bgr[0])
        draw.ellipse(
            (lamp_x - lamp_radius, lamp_y - lamp_radius, lamp_x + lamp_radius, lamp_y + lamp_radius),
            fill=lamp_rgb,
        )
        state_x = lamp_x + lamp_radius + max(7, round(height * 0.14))
        draw.text((state_x, baseline_y), data.state, font=font, fill=(224, 231, 234))
        entries = (
            data.mode.upper(),
            f"{data.fps:.1f} FPS",
            f"REC {'ON' if data.recording else 'OFF'} · {data.recorded_segments}",
        )
        x = state_x + max(100, round(width * 0.10))
        for entry in entries:
            draw.text((x, baseline_y), entry, font=font, fill=(169, 180, 185))
            box = draw.textbbox((x, baseline_y), entry, font=font)
            x = box[2] + max(20, round(width * 0.018))
        shortcuts = "S 稽核   M 模式   F 全螢幕"
        shortcut_box = draw.textbbox((0, 0), shortcuts, font=font)
        draw.text(
            (width - max(12, round(height * 0.35)) - (shortcut_box[2] - shortcut_box[0]), baseline_y),
            shortcuts,
            font=font,
            fill=(119, 133, 140),
        )

    _draw_rgb_crop(
        result,
        layout.footer_rect,
        cv2_module=cv2_module,
        painter=paint_footer,
    )

    badge_font = ui_font(max(9, round(min(layout.canvas_size) * 0.011)))
    badge_text = f"MODEL · {data.model_version or 'unknown'}"
    from PIL import Image, ImageDraw

    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    badge_box = measure.textbbox((0, 0), badge_text, font=badge_font)
    badge_width = badge_box[2] - badge_box[0] + 20
    badge_height = badge_box[3] - badge_box[1] + 14
    video_x, video_y, video_width, video_height = layout.video_rect
    badge_rect = (
        max(video_x, video_x + video_width - badge_width - 14),
        max(video_y, video_y + video_height - badge_height - 13),
        badge_width,
        badge_height,
    )

    def paint_badge(draw: Any, width: int, height: int) -> None:
        draw.rounded_rectangle(
            (0, 0, width - 1, height - 1),
            radius=max(3, height // 6),
            fill=(8, 12, 15),
            outline=(62, 74, 81),
            width=1,
        )
        draw.text((10, 6 - badge_box[1]), badge_text, font=badge_font, fill=(168, 180, 186))

    _draw_rgb_crop(
        result,
        badge_rect,
        cv2_module=cv2_module,
        painter=paint_badge,
    )
    return result


def render_application_view(
    frame_bgr: np.ndarray,
    target_size: tuple[int, int],
    content: Sequence[str] | DisplayPanelData,
    *,
    pose: np.ndarray | None,
    left_hand: np.ndarray | None,
    right_hand: np.ndarray | None,
    cv2_module: Any,
    font_loader: Callable[[int], Any] | None = None,
) -> tuple[np.ndarray, ApplicationLayout]:
    """Build a display-only UI while keeping the supplied raw frame untouched."""
    landmark_view = draw_landmark_graph(
        frame_bgr,
        pose,
        left_hand,
        right_hand,
        cv2_module=cv2_module,
    )
    canvas, layout = render_application_canvas(
        landmark_view,
        target_size,
        cv2_module=cv2_module,
    )
    if isinstance(content, DisplayPanelData):
        rendered = draw_e2_information_panel(
            canvas,
            layout,
            content,
            cv2_module=cv2_module,
            font_loader=font_loader,
        )
    else:
        rendered = draw_information_panel(
            canvas,
            layout,
            content,
            cv2_module=cv2_module,
            font_loader=font_loader,
        )
    return rendered, layout


def windows_primary_screen_size(
    *,
    user32: Any | None = None,
    fallback: tuple[int, int] = (1920, 1080),
) -> tuple[int, int]:
    """Enable per-monitor DPI awareness and return physical primary-screen pixels."""
    if user32 is None:
        try:
            user32 = ctypes.windll.user32
        except (AttributeError, OSError):
            return fallback
    try:
        user32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(-4)
        )  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
    except (AttributeError, OSError, ctypes.ArgumentError):
        pass
    try:
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
    except (AttributeError, OSError, TypeError, ValueError):
        return fallback
    return (width, height) if width > 0 and height > 0 else fallback


@dataclass
class ResizableDisplay:
    """Own OpenCV window sizing while keeping rendering policy testable."""

    title: str
    screen_size: tuple[int, int]
    coverage: float = 0.85
    fullscreen: bool = field(default=False, init=False)

    @property
    def windowed_size(self) -> tuple[int, int]:
        return initial_window_size(self.screen_size, coverage=self.coverage)

    def create(self, cv2_module: Any) -> None:
        flags = int(cv2_module.WINDOW_NORMAL) | int(
            getattr(cv2_module, "WINDOW_KEEPRATIO", 0)
        )
        cv2_module.namedWindow(self.title, flags)
        self._restore_window(cv2_module)

    def content_size(self, cv2_module: Any) -> tuple[int, int]:
        try:
            _x, _y, width, height = cv2_module.getWindowImageRect(self.title)
            if int(width) > 0 and int(height) > 0:
                return int(width), int(height)
        except (AttributeError, TypeError):
            pass
        return self.windowed_size

    def toggle_fullscreen(self, cv2_module: Any) -> bool:
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            cv2_module.setWindowProperty(
                self.title,
                cv2_module.WND_PROP_FULLSCREEN,
                cv2_module.WINDOW_FULLSCREEN,
            )
        else:
            cv2_module.setWindowProperty(
                self.title,
                cv2_module.WND_PROP_FULLSCREEN,
                cv2_module.WINDOW_NORMAL,
            )
            self._restore_window(cv2_module)
        return self.fullscreen

    def _restore_window(self, cv2_module: Any) -> None:
        width, height = self.windowed_size
        screen_width, screen_height = self.screen_size
        cv2_module.resizeWindow(self.title, width, height)
        cv2_module.moveWindow(
            self.title,
            max(0, (screen_width - width) // 2),
            max(0, (screen_height - height) // 2),
        )


def initial_window_size(
    screen_size: tuple[int, int],
    *,
    coverage: float = 0.85,
    minimum: tuple[int, int] = (960, 540),
) -> tuple[int, int]:
    """Choose a large initial window without exceeding the primary screen."""
    screen_width, screen_height = (int(screen_size[0]), int(screen_size[1]))
    if screen_width <= 0 or screen_height <= 0:
        raise ValueError("screen dimensions must be positive")
    if coverage <= 0:
        raise ValueError("coverage must be positive")
    if screen_width < minimum[0] or screen_height < minimum[1]:
        return screen_width, screen_height
    bounded_coverage = min(float(coverage), 1.0)
    return (
        min(screen_width, max(minimum[0], round(screen_width * bounded_coverage))),
        min(screen_height, max(minimum[1], round(screen_height * bounded_coverage))),
    )


def letterbox_rect(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return centered x/y/width/height preserving the source aspect ratio."""
    source_width, source_height = (int(source_size[0]), int(source_size[1]))
    target_width, target_height = (int(target_size[0]), int(target_size[1]))
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("source and target dimensions must be positive")
    scale = min(target_width / source_width, target_height / source_height)
    width = max(1, min(target_width, round(source_width * scale)))
    height = max(1, min(target_height, round(source_height * scale)))
    return (target_width - width) // 2, (target_height - height) // 2, width, height


def application_layout(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    *,
    panel_fraction: float = 0.31,
    minimum_panel_width: int = 300,
    maximum_panel_width: int = 720,
    footer_fraction: float = 0.055,
    minimum_footer_height: int = 36,
    maximum_footer_height: int = 64,
) -> ApplicationLayout:
    """Reserve an E2 right panel/footer and fit source video in the left region."""
    target_width, target_height = (int(target_size[0]), int(target_size[1]))
    if target_width <= 1 or target_height <= 0:
        raise ValueError("target dimensions must leave room for panel and video")
    if not 0 < panel_fraction < 1:
        raise ValueError("panel_fraction must be between zero and one")
    if not 0 < footer_fraction < 1:
        raise ValueError("footer_fraction must be between zero and one")
    footer_height = max(
        int(minimum_footer_height),
        min(int(maximum_footer_height), round(target_height * footer_fraction)),
    )
    footer_height = min(footer_height, max(1, target_height // 3))
    content_height = target_height - footer_height
    preferred = max(
        int(minimum_panel_width),
        min(int(maximum_panel_width), round(target_width * panel_fraction)),
    )
    panel_width = min(preferred, max(1, target_width // 2))
    viewport = (0, 0, target_width - panel_width, content_height)
    panel = (viewport[2], 0, panel_width, content_height)
    footer = (0, content_height, target_width, footer_height)
    local_video = letterbox_rect(source_size, (viewport[2], viewport[3]))
    video = (
        viewport[0] + local_video[0],
        viewport[1] + local_video[1],
        local_video[2],
        local_video[3],
    )
    return ApplicationLayout(
        canvas_size=(target_width, target_height),
        panel_rect=panel,
        video_viewport_rect=viewport,
        video_rect=video,
        footer_rect=footer,
    )


def resize_interpolation(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    cv2_module: Any,
) -> int:
    """Use area filtering for reduction and cubic filtering for enlargement."""
    if target_size[0] < source_size[0] or target_size[1] < source_size[1]:
        return int(cv2_module.INTER_AREA)
    return int(cv2_module.INTER_CUBIC)


def render_letterboxed(
    frame_bgr: np.ndarray,
    target_size: tuple[int, int],
    *,
    cv2_module: Any,
) -> np.ndarray:
    """Render a BGR frame onto a black target canvas without mutating the input."""
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR frame, got {frame.shape}")
    target_width, target_height = (int(target_size[0]), int(target_size[1]))
    x, y, width, height = letterbox_rect(
        (int(frame.shape[1]), int(frame.shape[0])),
        (target_width, target_height),
    )
    resized = cv2_module.resize(
        frame,
        (width, height),
        interpolation=resize_interpolation(
            (int(frame.shape[1]), int(frame.shape[0])),
            (width, height),
            cv2_module,
        ),
    )
    canvas = np.zeros((target_height, target_width, 3), dtype=frame.dtype)
    canvas[y : y + height, x : x + width] = resized
    return canvas


def render_application_canvas(
    frame_bgr: np.ndarray,
    target_size: tuple[int, int],
    *,
    cv2_module: Any,
) -> tuple[np.ndarray, ApplicationLayout]:
    """Compose a raw-frame copy beside a black panel; return canvas and geometry."""
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR frame, got {frame.shape}")
    layout = application_layout(
        (int(frame.shape[1]), int(frame.shape[0])),
        target_size,
    )
    x, y, width, height = layout.video_rect
    resized = cv2_module.resize(
        frame,
        (width, height),
        interpolation=resize_interpolation(
            (int(frame.shape[1]), int(frame.shape[0])),
            (width, height),
            cv2_module,
        ),
    )
    target_width, target_height = layout.canvas_size
    canvas = np.zeros((target_height, target_width, 3), dtype=frame.dtype)
    canvas[y : y + height, x : x + width] = resized
    return canvas, layout
