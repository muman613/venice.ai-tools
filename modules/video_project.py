"""Project data and FFmpeg helpers for the scene-queue video tool."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from modules.music_video import run_cancellable


APP_DATA_DIR = Path.home() / ".local" / "share" / "venice-ai-video"
CHARACTERS_PATH = APP_DATA_DIR / "characters.json"
TAIL_DURATIONS = (10, 15, 30)
SCENE_DURATION_TOLERANCE_SECONDS = 1.0


@dataclass
class Character:
    """A reusable character identity and optional visual reference."""

    name: str
    description: str = ""
    image_path: str = ""

    @classmethod
    def from_dict(cls, value):
        return cls(
            name=str(value.get("name", "")).strip(),
            description=str(value.get("description", "")).strip(),
            image_path=str(value.get("image_path", "")).strip(),
        )


@dataclass
class Scene:
    """One queued video clip."""

    number: int
    duration_seconds: int
    prompt: str = ""
    character_names: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value):
        return cls(
            number=int(value.get("number", 0)),
            duration_seconds=int(value.get("duration_seconds", 30)),
            prompt=str(value.get("prompt", "")),
            character_names=[
                str(name) for name in value.get("character_names", [])
            ],
        )


def load_characters(path=CHARACTERS_PATH):
    """Load the persistent character library."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load character library:\n{path}\n\n{exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"Character library must contain a JSON array:\n{path}")
    return [Character.from_dict(item) for item in data if isinstance(item, dict)]


def save_characters(characters, path=CHARACTERS_PATH):
    """Atomically save the persistent character library."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps([asdict(character) for character in characters], indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def probe_audio_duration(path):
    """Return an audio file duration in seconds using FFprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("FFprobe was not found on PATH. Install FFmpeg first.")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist:\n{path}")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe could not read the audio file.\n\n{result.stderr[-3000:]}")
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("FFprobe did not return a valid audio duration.") from exc
    if duration <= 0:
        raise ValueError("The selected audio file has no positive duration.")
    return duration


def probe_video_duration(path):
    """Return a playable video file's duration in seconds using FFprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("FFprobe was not found on PATH. Install FFmpeg first.")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file does not exist:\n{path}")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe could not read the video file.\n\n{result.stderr[-3000:]}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("FFprobe did not return valid JSON for the video file.") from exc
    video_streams = [
        stream
        for stream in data.get("streams", [])
        if isinstance(stream, dict) and stream.get("codec_type") == "video"
    ]
    if not video_streams:
        raise ValueError(f"The file contains no video stream:\n{path}")
    candidates = [data.get("format", {}).get("duration")]
    candidates.extend(stream.get("duration") for stream in video_streams)
    for value in candidates:
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0:
            return duration
    raise ValueError(f"FFprobe did not report a positive video duration:\n{path}")


def scene_duration_matches(actual_seconds, expected_seconds, tolerance=SCENE_DURATION_TOLERANCE_SECONDS):
    """Return whether a scene clip duration is close enough to its plan."""
    return abs(float(actual_seconds) - float(expected_seconds)) <= float(tolerance)


def plan_scene_durations(audio_seconds):
    """Plan 30-second scenes and round the tail to 10, 15, or 30 seconds."""
    seconds = float(audio_seconds)
    if seconds <= 0:
        raise ValueError("Audio duration must be positive.")
    full_scenes = int(seconds // 30)
    remainder = seconds - (full_scenes * 30)
    durations = [30] * full_scenes
    if remainder > 0.05 or not durations:
        tail = next((value for value in TAIL_DURATIONS if remainder <= value), 30)
        durations.append(tail)
    return durations


def project_slug(value):
    """Return a filesystem-safe project prefix."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    slug = re.sub(r"-+", "-", slug).strip("-._")
    if not slug:
        raise ValueError("Enter a project name containing letters or numbers.")
    return slug


def build_scene_prompt(overall_style, scene, characters, model_id):
    """Compose the complete prompt sent for one scene."""
    character_map = {character.name: character for character in characters}
    selected = [
        character_map[name]
        for name in scene.character_names
        if name in character_map
    ]
    lines = []
    if overall_style.strip():
        lines.append("OVERALL VIDEO STYLE (apply consistently):\n" + overall_style.strip())
    if selected:
        details = []
        image_index = 1
        for character in selected:
            detail = character.name
            if character.description:
                detail += ": " + character.description
            if character.image_path:
                tag = f"<Image {image_index}>" if "seedance" in model_id.lower() else f"@Image{image_index}"
                detail += f". Match identity and appearance to {tag}."
                image_index += 1
            details.append(detail)
        lines.append("CHARACTER CONTINUITY:\n" + "\n".join(f"- {item}" for item in details))
    lines.append(f"SCENE {scene.number} ({scene.duration_seconds} seconds):\n{scene.prompt.strip()}")
    lines.append(
        "Keep every named character visually consistent with prior scenes. "
        "Do not render captions, subtitles, logos, watermarks, or other visible text."
    )
    return "\n\n".join(lines)


def save_project(path, payload):
    """Atomically save a project JSON document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def mux_project_audio(
    input_video,
    output_video,
    audio_path="",
    remove_generated_audio=True,
    audio_delay_seconds=0,
    cancel_event=None,
    timeout=1800,
):
    """Create the final MP4, replacing or mixing audio and forcing stereo."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found on PATH.")
    input_video = Path(input_video)
    output_video = Path(output_video)
    output_video.unlink(missing_ok=True)

    command = [ffmpeg, "-nostdin", "-y", "-i", str(input_video)]
    if audio_path:
        audio_seconds = probe_audio_duration(audio_path)
        audio_delay_seconds = max(0.0, float(audio_delay_seconds))
        output_seconds = audio_seconds + audio_delay_seconds
        command.extend(["-i", str(audio_path)])
        if remove_generated_audio or not media_has_audio(input_video):
            if audio_delay_seconds:
                delay_ms = round(audio_delay_seconds * 1000)
                command.extend(
                    [
                        "-filter_complex", f"[1:a:0]adelay={delay_ms}:all=1[audio]",
                        "-map", "0:v:0", "-map", "[audio]",
                    ]
                )
            else:
                command.extend(["-map", "0:v:0", "-map", "1:a:0"])
            command.extend(
                [
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-shortest",
                ]
            )
        else:
            music_filter = "[1:a:0]aformat=channel_layouts=stereo[music]"
            if audio_delay_seconds:
                delay_ms = round(audio_delay_seconds * 1000)
                music_filter = (
                    "[1:a:0]aformat=channel_layouts=stereo,"
                    f"adelay={delay_ms}:all=1[music]"
                )
            command.extend(
                [
                    "-filter_complex",
                    "[0:a:0]aformat=channel_layouts=stereo[clip];"
                    + music_filter + ";"
                    "[clip][music]amix=inputs=2:duration=shortest:normalize=0[mix]",
                    "-map", "0:v:0", "-map", "[mix]",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-shortest",
                ]
            )
        command.extend(["-t", f"{output_seconds:.6f}"])
    elif remove_generated_audio:
        command.extend(["-map", "0:v:0", "-c:v", "copy", "-an"])
    else:
        command.extend(["-map", "0:v:0", "-map", "0:a?", "-c", "copy"])
    command.extend(["-movflags", "+faststart", str(output_video)])

    if cancel_event is not None:
        try:
            run_cancellable(command, cancel_event)
        except Exception:
            output_video.unlink(missing_ok=True)
            raise
        result = None
    else:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    if (result is not None and result.returncode != 0) or not output_video.exists() or output_video.stat().st_size == 0:
        output_video.unlink(missing_ok=True)
        stderr = result.stderr[-6000:] if result is not None else ""
        raise RuntimeError(f"FFmpeg could not create the final audio mix.\n\n{stderr}")


def media_has_audio(path):
    """Return whether FFprobe finds an audio stream in a media file."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("FFprobe was not found on PATH. Install FFmpeg first.")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe could not inspect media streams.\n\n{result.stderr[-3000:]}")
    return bool(result.stdout.strip())
