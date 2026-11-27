"""
YouTube Shorts Factory — main entry point.

Called by GitHub Actions every day. Does the full pipeline:
  1. Pick today's story
  2. Synthesize narration voice
  3. Render animated video
  4. Quality check
  5. Upload to YouTube (unless --dry-run)
  6. Send Telegram notification

Usage:
  python shorts/make_short.py
  python shorts/make_short.py --dry-run=true
  python shorts/make_short.py --story-id napoleon_height
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Make sure we can import from the shorts/ package
sys.path.insert(0, str(Path(__file__).parent))

from stories import get_story_for_today, get_story_by_id, STORIES
from voice import synthesize_voice, get_audio_duration
from render import render_video
from qa import validate_video

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shorts.main")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "shorts" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# YouTube upload
# ---------------------------------------------------------------------------

def _youtube_upload(
    mp4_path: str,
    title: str,
    description: str,
    tags: list,
    client_id: str,
    client_secret: str,
    refresh_token: str,
    privacy: str = "public",
) -> dict:
    """Upload video via YouTube Data API v3 resumable upload."""
    import requests

    TOKEN_URL = "https://oauth2.googleapis.com/token"
    UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"

    # Get access token
    resp = requests.post(TOKEN_URL, data={
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Token refresh failed ({resp.status_code}): {resp.text}")
    access_token = resp.json()["access_token"]

    # Ensure #Shorts in title
    if "#Shorts" not in title:
        title = title + " #Shorts"
    if len(title) > 100:
        title = title[:97] + "..."

    file_size = os.path.getsize(mp4_path)

    # Initiate resumable upload
    meta = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "22",  # People & Blogs — good for edu shorts
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }
    init_resp = requests.post(
        UPLOAD_URL,
        params={"part": "snippet,status", "uploadType": "resumable"},
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": "video/mp4",
            "X-Upload-Content-Length": str(file_size),
        },
        json=meta,
        timeout=60,
    )
    if init_resp.status_code not in (200, 201):
        raise RuntimeError(f"Upload init failed ({init_resp.status_code}): {init_resp.text}")

    session_url = init_resp.headers["Location"]

    # Upload the file
    with open(mp4_path, "rb") as fh:
        data = fh.read()

    for attempt in range(1, 5):
        put_resp = requests.put(
            session_url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Length": str(file_size),
                "Content-Type": "video/mp4",
            },
            data=data,
            timeout=300,
        )
        if put_resp.status_code in (200, 201):
            result = put_resp.json() if put_resp.text else {}
            video_id = result.get("id", "unknown")
            log.info("Upload complete! Video ID: %s", video_id)
            log.info("URL: https://www.youtube.com/watch?v=%s", video_id)
            return result
        if put_resp.status_code in (308, 429, 500, 502, 503, 504):
            wait = min(2 ** attempt, 30)
            log.warning("Upload retry %d/4 in %ds (status %d)", attempt, wait, put_resp.status_code)
            time.sleep(wait)
        else:
            raise RuntimeError(f"Upload failed ({put_resp.status_code}): {put_resp.text[:300]}")

    raise RuntimeError("Upload failed after 4 attempts")


# ---------------------------------------------------------------------------
# Telegram notification
# ---------------------------------------------------------------------------

def _telegram_notify(token: str, chat_id: str, message: str) -> None:
    """Send a Telegram message (best-effort, never fatal)."""
    try:
        import requests
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=15,
        )
        if resp.status_code == 200:
            log.info("Telegram notification sent")
        else:
            log.warning("Telegram notification failed: %s", resp.text[:200])
    except Exception as exc:
        log.warning("Telegram notification error: %s", exc)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(story_id: str | None, dry_run: bool) -> dict:
    """Run the full pipeline. Returns a manifest dict."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # 1. Pick story
    if story_id:
        story = get_story_by_id(story_id)
        log.info("Using requested story: %s", story["id"])
    else:
        story = get_story_for_today()
        log.info("Using today's story: %s  (%s)", story["id"], story["title"][:60])

    title = story["title"]
    description = story.get("description", "") + "\n\n" + " ".join(story.get("hashtags", []))

    # Build full narration text from all scenes
    narration_parts = []
    if story.get("hook"):
        narration_parts.append(story["hook"])
    for scene in story.get("scenes", []):
        narration_parts.append(scene.get("narration", ""))
    full_narration = " ".join(p for p in narration_parts if p)

    log.info("Story: %s", title)
    log.info("Narration length: %d words", len(full_narration.split()))

    # File paths
    safe_id = story["id"]
    voice_path = str(OUTPUT_DIR / f"{safe_id}_{ts}_voice.mp3")
    video_path = str(OUTPUT_DIR / f"{safe_id}_{ts}_short.mp4")
    manifest_path = str(OUTPUT_DIR / "manifest.json")

    # 2. Voice synthesis
    log.info("=== Step 1/4: Voice synthesis ===")
    ok = synthesize_voice(full_narration, voice_path)
    if not ok:
        raise RuntimeError("Voice synthesis failed — no audio produced")

    audio_dur = get_audio_duration(voice_path)
    log.info("Audio duration: %.1fs", audio_dur)
    if audio_dur < 20:
        # Probably ffprobe not available; estimate from word count
        wpm = 145
        audio_dur = len(full_narration.split()) / wpm * 60
        log.info("Estimated audio duration from word count: %.1fs", audio_dur)

    # 3. Render video
    log.info("=== Step 2/4: Rendering video ===")
    ok = render_video(
        story=story,
        audio_path=voice_path,
        output_path=video_path,
        audio_duration=audio_dur,
    )
    if not ok:
        raise RuntimeError("Video rendering failed")

    # 4. Quality check
    log.info("=== Step 3/4: Quality check ===")
    passed, score, report = validate_video(video_path)
    log.info("QA result: %s (score %d)", "PASS" if passed else "FAIL", score)
    log.info("QA report: %s", report)

    if not passed:
        raise RuntimeError(f"Video failed quality check (score {score}/100): {report}")

    # 5. Upload
    youtube_video_id = None
    youtube_url = None

    if dry_run:
        log.info("=== Step 4/4: DRY RUN — skipping upload ===")
        log.info("Video ready at: %s", video_path)
    else:
        log.info("=== Step 4/4: Uploading to YouTube ===")
        client_id = os.environ.get("YOUTUBE_CLIENT_ID", "")
        client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET", "")
        refresh_token = os.environ.get("YOUTUBE_REFRESH_TOKEN", "")

        if not all([client_id, client_secret, refresh_token]):
            raise RuntimeError(
                "Missing YouTube OAuth secrets. Set YOUTUBE_CLIENT_ID, "
                "YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN as GitHub secrets."
            )

        result = _youtube_upload(
            mp4_path=video_path,
            title=title,
            description=description,
            tags=[t.lstrip("#") for t in story.get("hashtags", ["Shorts"])],
            client_id=client_id,
            client_secret=client_secret,
            refresh_token=refresh_token,
            privacy="public",
        )

        youtube_video_id = result.get("id")
        youtube_url = f"https://www.youtube.com/watch?v={youtube_video_id}"
        log.info("YouTube URL: %s", youtube_url)

        # Telegram notification
        tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
        if tg_token and tg_chat:
            msg = (
                f"🎬 *New Short uploaded!*\n"
                f"*{title}*\n\n"
                f"📊 QA Score: {score}/100\n"
                f"🔗 {youtube_url}"
            )
            _telegram_notify(tg_token, tg_chat, msg)

    # Write manifest
    manifest = {
        "story_id": story["id"],
        "title": title,
        "timestamp": ts,
        "dry_run": dry_run,
        "voice_path": voice_path,
        "video_path": video_path,
        "video_size_mb": round(os.path.getsize(video_path) / 1_048_576, 2),
        "qa_score": score,
        "qa_report": report,
        "qa_passed": passed,
        "youtube_video_id": youtube_video_id,
        "youtube_url": youtube_url,
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("Manifest written: %s", manifest_path)

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="YouTube Shorts Factory — make and upload a Short")
    parser.add_argument(
        "--dry-run",
        default="false",
        help="Build only, do not upload (true/false)",
    )
    parser.add_argument(
        "--story-id",
        default=None,
        help="Specific story ID to use (default: today's story)",
    )
    parser.add_argument(
        "--list-stories",
        action="store_true",
        help="Print available story IDs and exit",
    )
    args = parser.parse_args()

    if args.list_stories:
        print("\nAvailable stories:")
        for s in STORIES:
            print(f"  {s['id']:<35} {s['title'][:60]}")
        return 0

    # Accept "true"/"false" string from GitHub Actions
    dry_run = str(args.dry_run).lower() in ("true", "1", "yes")

    log.info("=" * 60)
    log.info("YouTube Shorts Factory — %s | dry_run=%s",
             datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), dry_run)
    log.info("=" * 60)

    try:
        manifest = run_pipeline(story_id=args.story_id, dry_run=dry_run)
        log.info("=" * 60)
        log.info("SUCCESS")
        log.info("Story: %s", manifest["title"][:70])
        log.info("Video: %s (%.2f MB)", manifest["video_path"], manifest["video_size_mb"])
        log.info("QA:    %d/100 — %s", manifest["qa_score"], "PASSED" if manifest["qa_passed"] else "FAILED")
        if manifest.get("youtube_url"):
            log.info("URL:   %s", manifest["youtube_url"])
        log.info("=" * 60)
        return 0
    except Exception as exc:
        log.error("=" * 60)
        log.error("PIPELINE FAILED: %s", exc)
        log.error("=" * 60)

        # Telegram failure notification
        tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
        if tg_token and tg_chat:
            _telegram_notify(tg_token, tg_chat,
                             f"❌ *Shorts pipeline FAILED*\n`{exc}`")

        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
