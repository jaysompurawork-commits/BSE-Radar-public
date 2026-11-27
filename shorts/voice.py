"""
Voice synthesis for YouTube Shorts.

Uses edge-tts (Microsoft's neural TTS, free, no key required) with gTTS fallback.
Produces high-quality MP3 narration that matches the script duration target.

Voice selection:
  - English: en-US-AriaNeural (warm, conversational, documentary tone)
  - Fallback: en-US-GuyNeural (male voice, clear)
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("shorts.voice")

# Best voices for documentary-style content
VOICE_PRIMARY = "en-US-AriaNeural"
VOICE_FALLBACK = "en-US-GuyNeural"

# SSML rate: slightly slower = more gravitas, better comprehension
TTS_RATE = "+0%"
TTS_PITCH = "+0Hz"
TTS_VOLUME = "+0%"


def _edge_tts_available() -> bool:
    try:
        import edge_tts  # noqa: F401
        return True
    except ImportError:
        return False


async def _synthesize_edge_tts(text: str, output_path: str, voice: str) -> bool:
    """Synthesize using edge-tts async API."""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice, rate=TTS_RATE, pitch=TTS_PITCH)
        await communicate.save(output_path)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
    except Exception as exc:
        log.warning("edge-tts error with voice %s: %s", voice, exc)
        return False


def synthesize_voice(text: str, output_path: str) -> bool:
    """
    Synthesize narration text to output_path (MP3 or WAV).
    Returns True on success.
    """
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Try edge-tts (best quality)
    if _edge_tts_available():
        log.info("Synthesizing with edge-tts voice %s ...", VOICE_PRIMARY)
        ok = asyncio.run(_synthesize_edge_tts(text, output_path, VOICE_PRIMARY))
        if ok:
            log.info("Voice synthesis OK: %s (%.1f KB)", output_path,
                     os.path.getsize(output_path) / 1024)
            return True

        # Try fallback voice
        log.warning("Primary voice failed, trying fallback %s", VOICE_FALLBACK)
        ok = asyncio.run(_synthesize_edge_tts(text, output_path, VOICE_FALLBACK))
        if ok:
            log.info("Voice synthesis OK (fallback): %s", output_path)
            return True

    # Try gTTS as last resort
    try:
        from gtts import gTTS
        log.info("Falling back to gTTS ...")
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(output_path)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            log.info("gTTS OK: %s", output_path)
            return True
    except Exception as exc:
        log.error("gTTS failed: %s", exc)

    log.error("All voice synthesis methods failed.")
    return False


def get_audio_duration(audio_path: str) -> float:
    """Return duration of audio file in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", audio_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            for stream in data.get("streams", []):
                dur = stream.get("duration")
                if dur:
                    return float(dur)
    except Exception as exc:
        log.warning("ffprobe duration check failed: %s", exc)
    return 0.0
