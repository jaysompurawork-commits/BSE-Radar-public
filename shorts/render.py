"""
Video renderer for YouTube Shorts.

Creates real animated videos using FFmpeg — NOT a slideshow of static images.

What this produces:
  - 1080x1920 (9:16 shorts format)
  - Animated gradient backgrounds that change per scene
  - Kinetic typography (text slides, fades, zooms)
  - Scene-specific color themes
  - Smooth transitions between scenes (fade/wipe)
  - Burned-in captions at the bottom (white text, semi-transparent background)
  - Narration audio mixed in

NO external image downloads required. 100% FFmpeg + Pillow for graphics.
"""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
import numpy as np

log = logging.getLogger("shorts.render")

# Video parameters: full Shorts spec
WIDTH = 1080
HEIGHT = 1920
FPS = 30
FONT_SIZE_TITLE = 72
FONT_SIZE_CAPTION = 52
FONT_SIZE_HOOK = 62
CAPTION_LINE_MAX = 38  # characters per line
CAPTION_Y = HEIGHT - 260  # where captions appear


# ---------------------------------------------------------------------------
# Color palette helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))  # type: ignore


def _lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))  # type: ignore


# ---------------------------------------------------------------------------
# Frame generators
# ---------------------------------------------------------------------------

def _make_gradient_frame(
    color_top: str,
    color_bot: str,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> Image.Image:
    """Create a vertical gradient background frame."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    top = _hex_to_rgb(color_top)
    bot = _hex_to_rgb(color_bot)
    for y in range(height):
        t = y / height
        r, g, b = _lerp_color(top, bot, t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    return img


def _find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a font. Try system fonts, fall back to PIL default."""
    candidates = []
    if bold:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    # PIL default (bitmap) — always works but no anti-aliasing
    return ImageFont.load_default()


def _wrap_text(text: str, max_chars: int = CAPTION_LINE_MAX) -> List[str]:
    """Wrap text to fit within max_chars per line."""
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = (current + " " + word).strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def _draw_caption_box(draw: ImageDraw.ImageDraw, text: str, font: Any, y_top: int, img: Image.Image) -> None:
    """Draw a semi-transparent box with white caption text."""
    lines = _wrap_text(text, CAPTION_LINE_MAX)
    line_h = FONT_SIZE_CAPTION + 8
    box_h = len(lines) * line_h + 24
    box_y = y_top

    # Semi-transparent overlay (we'll draw a filled rect with alpha)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rectangle(
        [(0, box_y - 12), (WIDTH, box_y + box_h)],
        fill=(0, 0, 0, 160),
    )
    # Composite onto frame
    img_rgba = img.convert("RGBA")
    img_rgba = Image.alpha_composite(img_rgba, overlay)
    img.paste(img_rgba.convert("RGB"))

    # Draw text centered on each line
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        x = (WIDTH - tw) // 2
        y = box_y + i * line_h + 4
        # Shadow
        draw.text((x + 2, y + 2), line, font=font, fill=(0, 0, 0, 200))
        # Main text
        draw.text((x, y), line, font=font, fill=(255, 255, 255))


def _draw_scene_number(draw: ImageDraw.ImageDraw, n: int, total: int, font: Any) -> None:
    """Draw a small progress indicator at top."""
    dots = "".join("●" if i < n else "○" for i in range(total))
    bbox = draw.textbbox((0, 0), dots, font=font)
    tw = bbox[2] - bbox[0]
    x = (WIDTH - tw) // 2
    draw.text((x + 1, 41), dots, font=font, fill=(0, 0, 0, 120))
    draw.text((x, 40), dots, font=font, fill=(255, 255, 255, 200))


def _draw_accent_line(img: Image.Image, accent_hex: str, y: int = 80, thickness: int = 6) -> None:
    """Draw a horizontal accent line below the top area."""
    draw = ImageDraw.Draw(img)
    color = _hex_to_rgb(accent_hex)
    draw.rectangle([(80, y), (WIDTH - 80, y + thickness)], fill=color)


def _draw_large_text_block(
    img: Image.Image,
    text: str,
    accent: str,
    font_large: Any,
    font_small: Any,
    y_center: int = 800,
) -> None:
    """Draw the main scene text — large, centered, with accent color highlight."""
    lines = _wrap_text(text, 26)  # fewer chars per line for large text
    draw = ImageDraw.Draw(img)
    line_h = FONT_SIZE_TITLE + 12
    total_h = len(lines) * line_h
    y_start = y_center - total_h // 2
    accent_rgb = _hex_to_rgb(accent)

    for i, line in enumerate(lines):
        y = y_start + i * line_h
        # Slight color variation: first word in accent, rest in white
        words = line.split(" ", 1)
        if len(words) == 2 and i == 0:
            # Measure first word
            bbox_w = draw.textbbox((0, 0), words[0] + " ", font=font_large)
            tw0 = bbox_w[2] - bbox_w[0]
            bbox_line = draw.textbbox((0, 0), line, font=font_large)
            total_w = bbox_line[2] - bbox_line[0]
            x0 = (WIDTH - total_w) // 2
            # Shadow
            draw.text((x0 + 2, y + 2), words[0] + " ", font=font_large, fill=(0, 0, 0))
            draw.text((x0, y), words[0] + " ", font=font_large, fill=accent_rgb)
            # Rest of first line
            draw.text((x0 + tw0 + 2, y + 2), words[1], font=font_large, fill=(0, 0, 0))
            draw.text((x0 + tw0, y), words[1], font=font_large, fill=(255, 255, 255))
        else:
            bbox = draw.textbbox((0, 0), line, font=font_large)
            tw = bbox[2] - bbox[0]
            x = (WIDTH - tw) // 2
            draw.text((x + 2, y + 2), line, font=font_large, fill=(0, 0, 0))
            draw.text((x, y), line, font=font_large, fill=(255, 255, 255))


def _make_scene_frame(
    scene: Dict[str, Any],
    scene_index: int,
    total_scenes: int,
    frame_in_scene: int,
    total_frames_in_scene: int,
    hook_text: str,
    show_hook: bool = False,
) -> Image.Image:
    """
    Render a single video frame for a scene.

    Animates:
    - Sliding text (text moves in from bottom over first 15 frames)
    - Zoom effect on gradient (subtle Ken-Burns via crop+resize alternative: shift bg)
    - Caption at bottom
    - Progress dots at top
    """
    color_main = scene.get("color", "#1a1a2e")
    accent = scene.get("accent", "#e94560")
    narration = scene.get("narration", "")

    # Gradient top→dark version of color_main, bottom→darker
    top_r, top_g, top_b = _hex_to_rgb(color_main)
    bot_r = max(0, top_r - 30)
    bot_g = max(0, top_g - 30)
    bot_b = max(0, top_b - 30)
    color_bot = f"#{bot_r:02x}{bot_g:02x}{bot_b:02x}"

    img = _make_gradient_frame(color_main, color_bot)

    # Animate: background slow shift (simulates parallax)
    shift_y = int(20 * frame_in_scene / max(1, total_frames_in_scene - 1))
    if shift_y > 0 and shift_y < HEIGHT:
        shifted = Image.new("RGB", (WIDTH, HEIGHT))
        region = img.crop((0, shift_y, WIDTH, HEIGHT))
        shifted.paste(region, (0, 0))
        bottom_fill = _make_gradient_frame(color_bot, color_bot, WIDTH, shift_y)
        shifted.paste(bottom_fill, (0, HEIGHT - shift_y))
        img = shifted

    # Decorative geometric shapes
    draw = ImageDraw.Draw(img)
    accent_rgb = _hex_to_rgb(accent)

    # Large decorative circles (semi-transparent feel via layered outlines)
    for radius in [300, 240, 180]:
        r, g, b = _lerp_color(accent_rgb, _hex_to_rgb(color_main), 0.7)
        draw.ellipse(
            [WIDTH // 2 - radius,
             HEIGHT // 2 - radius - 200,
             WIDTH // 2 + radius,
             HEIGHT // 2 + radius - 200],
            outline=(r, g, b),
        )

    # Horizontal accent lines
    _draw_accent_line(img, accent, y=120, thickness=4)
    _draw_accent_line(img, accent, y=HEIGHT - 300, thickness=2)

    # Channel name / branding at top
    font_tiny = _find_font(32, bold=False)
    font_small = _find_font(40, bold=False)
    font_main = _find_font(FONT_SIZE_TITLE, bold=True)
    font_caption = _find_font(FONT_SIZE_CAPTION, bold=False)

    brand_text = "STRANGE BUT TRUE"
    draw = ImageDraw.Draw(img)
    draw.text((80, 44), brand_text, font=font_small, fill=accent_rgb)

    # Progress dots
    _draw_scene_number(draw, scene_index, total_scenes, font_tiny)

    # Animate text sliding in from bottom
    t_progress = min(1.0, frame_in_scene / 12)  # fully in by frame 12
    text_offset = int((1.0 - t_progress) * 80)  # starts 80px lower

    if show_hook and frame_in_scene < total_frames_in_scene // 2:
        # Show hook in first part of first scene
        text_y = 700 + text_offset
        font_hook = _find_font(FONT_SIZE_HOOK, bold=True)
        lines = _wrap_text(hook_text, 22)
        lh = FONT_SIZE_HOOK + 14
        for i, line in enumerate(lines[:4]):
            bbox = draw.textbbox((0, 0), line, font=font_hook)
            tw = bbox[2] - bbox[0]
            x = (WIDTH - tw) // 2
            y = text_y + i * lh
            draw.text((x + 2, y + 2), line, font=font_hook, fill=(0, 0, 0))
            draw.text((x, y), line, font=font_hook,
                      fill=accent_rgb if i == 0 else (255, 255, 255))
    else:
        # Main narration text
        _draw_large_text_block(img, narration, accent, font_main, font_small,
                               y_center=HEIGHT // 2 - 80 + text_offset)

    # Caption / subtitle at bottom (always visible once text is in)
    if t_progress > 0.3:
        _draw_caption_box(draw, narration, font_caption, CAPTION_Y, img)

    return img


# ---------------------------------------------------------------------------
# PNG sequence → video via FFmpeg
# ---------------------------------------------------------------------------

def _frames_to_video(
    frame_dir: str,
    audio_path: Optional[str],
    output_path: str,
    fps: int = FPS,
    duration: float = 60.0,
) -> bool:
    """
    Combine PNG frames + audio into an MP4 using FFmpeg.
    """
    frame_pattern = os.path.join(frame_dir, "frame_%05d.png")

    if audio_path and os.path.exists(audio_path):
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-i", audio_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ]
    else:
        # No audio — still render video (for testing / dry run)
        n_frames = int(duration * fps)
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-t", str(duration),
            "-movflags", "+faststart",
            output_path,
        ]

    log.info("Running ffmpeg: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        log.error("FFmpeg failed:\n%s\n%s", result.stdout[-2000:], result.stderr[-2000:])
        return False

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 10_000:
        log.error("FFmpeg output missing or too small: %s", output_path)
        return False

    size_mb = os.path.getsize(output_path) / 1_048_576
    log.info("Video rendered: %s (%.2f MB)", output_path, size_mb)
    return True


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_video(
    story: Dict[str, Any],
    audio_path: Optional[str],
    output_path: str,
    audio_duration: float = 60.0,
) -> bool:
    """
    Render a complete Shorts video for the given story.

    story: dict from stories.py (has 'scenes', 'hook', 'title')
    audio_path: path to narration MP3/WAV
    output_path: where to write the final MP4
    audio_duration: actual duration of audio in seconds
    """
    scenes = story.get("scenes", [])
    if not scenes:
        log.error("No scenes in story, cannot render")
        return False

    hook_text = story.get("hook", "")
    n_scenes = len(scenes)

    # Calculate frame count per scene from scene duration field
    total_declared = sum(s.get("duration", 8) for s in scenes)
    video_duration = max(audio_duration, total_declared, 30.0)

    # Scale scene durations proportionally to match audio
    scale = video_duration / total_declared if total_declared > 0 else 1.0
    scene_durations = [s.get("duration", 8) * scale for s in scenes]

    log.info(
        "Rendering %d scenes, total duration %.1fs (audio: %.1fs)",
        n_scenes, video_duration, audio_duration,
    )

    with tempfile.TemporaryDirectory() as frame_dir:
        frame_num = 0

        for scene_idx, (scene, dur) in enumerate(zip(scenes, scene_durations)):
            n_frames = max(1, int(dur * FPS))
            is_first = scene_idx == 0

            for fi in range(n_frames):
                show_hook = is_first and fi < n_frames // 3
                img = _make_scene_frame(
                    scene=scene,
                    scene_index=scene_idx + 1,
                    total_scenes=n_scenes,
                    frame_in_scene=fi,
                    total_frames_in_scene=n_frames,
                    hook_text=hook_text,
                    show_hook=show_hook,
                )

                # Cross-fade transition: last 8 frames fade to black
                if fi >= n_frames - 8 and scene_idx < n_scenes - 1:
                    fade_t = (fi - (n_frames - 8)) / 8.0
                    black = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
                    img = Image.blend(img, black, fade_t)
                # Cross-fade in: first 8 frames fade from black
                elif fi < 8 and scene_idx > 0:
                    fade_t = 1.0 - fi / 8.0
                    black = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
                    img = Image.blend(img, black, fade_t)

                frame_path = os.path.join(frame_dir, f"frame_{frame_num:05d}.png")
                img.save(frame_path, "PNG", optimize=False)
                frame_num += 1

                if frame_num % 100 == 0:
                    log.info("  Rendered %d frames of ~%d total",
                             frame_num, int(video_duration * FPS))

        log.info("All %d frames rendered. Encoding video...", frame_num)
        return _frames_to_video(
            frame_dir=frame_dir,
            audio_path=audio_path,
            output_path=output_path,
            fps=FPS,
            duration=video_duration,
        )
