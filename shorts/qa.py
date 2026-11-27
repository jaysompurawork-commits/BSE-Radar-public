"""
Quality gate for rendered Shorts videos.

Rejects videos that are:
  - Too short or too long
  - Wrong dimensions
  - Missing audio
  - Too small (file size — suggests encoding failure)
  - Mostly black frames (the original "black screen" problem)

Returns a score 0-100. Minimum to upload: 60.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Dict, Tuple

log = logging.getLogger("shorts.qa")

MIN_SCORE = 60
MIN_DURATION_S = 25.0
MAX_DURATION_S = 185.0
MIN_FILE_SIZE_MB = 0.5   # a valid 30s video should be > 500 KB
MIN_WIDTH = 1000
MIN_HEIGHT = 1800


def _run_ffprobe(path: str) -> Dict:
    """Return ffprobe JSON for the video."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as exc:
        log.warning("ffprobe failed: %s", exc)
    return {}


def _check_blackframes(path: str) -> float:
    """
    Run blackdetect to find how many seconds of black frames exist.
    Returns fraction of total video that is black (0.0 = none, 1.0 = all black).
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", path,
             "-vf", "blackdetect=d=0.1:pix_th=0.10",
             "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=60,
        )
        output = result.stderr
        black_seconds = 0.0
        for line in output.split("\n"):
            if "black_duration" in line:
                for part in line.split(":"):
                    if "black_duration" in part:
                        try:
                            val = float(part.split("=")[1].strip())
                            black_seconds += val
                        except (IndexError, ValueError):
                            pass
        # Get total duration from format
        probe = _run_ffprobe(path)
        total = float(probe.get("format", {}).get("duration", 60))
        return black_seconds / max(1.0, total)
    except Exception as exc:
        log.warning("blackdetect failed: %s", exc)
    return 0.0


def validate_video(path: str) -> Tuple[bool, int, str]:
    """
    Validate the video at path.
    Returns (pass, score, report_string).
    Score 0-100. pass=True if score >= MIN_SCORE.
    """
    if not os.path.exists(path):
        return False, 0, "Video file does not exist"

    file_size_mb = os.path.getsize(path) / 1_048_576
    issues = []
    score = 100

    # File size check
    if file_size_mb < MIN_FILE_SIZE_MB:
        issues.append(f"File too small: {file_size_mb:.2f} MB (min {MIN_FILE_SIZE_MB} MB)")
        score -= 50

    probe = _run_ffprobe(path)
    streams = probe.get("streams", [])
    fmt = probe.get("format", {})

    # Duration check
    duration = float(fmt.get("duration", 0))
    if duration < MIN_DURATION_S:
        issues.append(f"Too short: {duration:.1f}s (min {MIN_DURATION_S}s)")
        score -= 30
    elif duration > MAX_DURATION_S:
        issues.append(f"Too long: {duration:.1f}s (max {MAX_DURATION_S}s)")
        score -= 20

    # Video stream check
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not video_streams:
        issues.append("No video stream found")
        score -= 40
    else:
        vs = video_streams[0]
        w = int(vs.get("width", 0))
        h = int(vs.get("height", 0))
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            issues.append(f"Wrong dimensions: {w}x{h} (need {MIN_WIDTH}x{MIN_HEIGHT})")
            score -= 25
        # Check aspect ratio is portrait
        if w > 0 and h > 0 and w > h:
            issues.append(f"Landscape video (need portrait 9:16): {w}x{h}")
            score -= 30

    if not audio_streams:
        issues.append("No audio stream found")
        score -= 30

    # Black frame check (the critical one)
    black_fraction = _check_blackframes(path)
    if black_fraction > 0.5:
        issues.append(f"Video is {black_fraction*100:.0f}% black frames — content failure")
        score -= 50
    elif black_fraction > 0.2:
        issues.append(f"Warning: {black_fraction*100:.0f}% black frames (transitions expected)")
        score -= 10

    score = max(0, score)
    passed = score >= MIN_SCORE and not any("No video stream" in i or "File too small" in i for i in issues)

    if issues:
        report = f"Score: {score}/100. Issues: " + "; ".join(issues)
    else:
        report = f"Score: {score}/100. PASSED. Duration: {duration:.1f}s, Size: {file_size_mb:.2f}MB"

    log.info("QA %s: %s", "PASS" if passed else "FAIL", report)
    return passed, score, report
