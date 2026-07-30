"""Text-to-speech helpers for Venice AI audio generation."""

import os
import time
from pathlib import Path

import requests


API_BASE_URL = "https://api.venice.ai/api/v1"
DEFAULT_MODEL = "tts-kokoro"
DEFAULT_CHUNK_SIZE = 4000
DEFAULT_DELAY_SECONDS = 1.0
VALID_VOICES = ("default", "alloy", "echo", "fable", "onyx", "nova", "shimmer")
VENICE_VOICE_ALIASES = {
    "default": "af_sky",
    "alloy": "af_alloy",
    "echo": "am_echo",
    "fable": "bm_fable",
    "onyx": "am_onyx",
    "nova": "af_nova",
    "shimmer": "af_sky",
}


class TextToSpeechError(RuntimeError):
    """Raised when text-to-speech conversion fails."""


def get_venice_api_key(api_key=None):
    """Return the Venice API key from an argument, environment, or local env file."""
    if api_key:
        return api_key

    api_key = os.environ.get("VENICE_API_KEY")
    if api_key:
        return api_key

    for env_file in env_file_candidates():
        loaded_key = read_env_file_value(env_file, "VENICE_API_KEY")
        if loaded_key:
            return loaded_key

    return None


def env_file_candidates():
    """Return likely local env files without requiring python-dotenv."""
    project_root = Path(__file__).resolve().parent.parent
    return (
        Path.cwd() / ".env",
        Path.cwd() / "venice.env",
        project_root / ".env",
        project_root / "venice.env",
    )


def read_env_file_value(env_file, key):
    """Read one KEY=value setting from a shell-style env file."""
    if not env_file.is_file():
        return None

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        name, separator, value = line.partition("=")
        if separator and name.strip() == key:
            return value.strip().strip("\"'")
    return None


def chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE):
    """Split text into chunks, preferring paragraph and whitespace boundaries."""
    text = text.strip()
    if not text:
        return []

    chunks = []
    remaining = text
    while len(remaining) > chunk_size:
        split_at = max(
            remaining.rfind("\n\n", 0, chunk_size),
            remaining.rfind(". ", 0, chunk_size),
            remaining.rfind("? ", 0, chunk_size),
            remaining.rfind("! ", 0, chunk_size),
            remaining.rfind(" ", 0, chunk_size),
        )
        if split_at <= 0:
            split_at = chunk_size
        else:
            split_at += 1

        chunk = remaining[:split_at].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


def synthesize_text_to_mp3(
    text,
    output_path,
    voice_id="default",
    *,
    api_key=None,
    model=DEFAULT_MODEL,
    api_base_url=API_BASE_URL,
    timeout=120,
):
    """Send one text chunk to Venice TTS and save the returned MP3 bytes."""
    voice_id = normalize_voice(voice_id)
    api_key = get_venice_api_key(api_key)
    if not api_key:
        raise TextToSpeechError("VENICE_API_KEY is not set in the process environment or a local .env/venice.env file.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.post(
        f"{api_base_url.rstrip('/')}/audio/speech",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": text,
            "voice": voice_id,
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        raise TextToSpeechError(
            f"Venice TTS request failed with HTTP {response.status_code}: {response.text}"
        )

    output_path.write_bytes(response.content)
    return output_path


def text_file_to_mp3(
    text_file_path,
    output_path=None,
    voice_id="default",
    *,
    api_key=None,
    chunk_size=DEFAULT_CHUNK_SIZE,
    delay_seconds=DEFAULT_DELAY_SECONDS,
    progress_callback=None,
):
    """
    Convert a UTF-8 text file to one MP3 file using Venice TTS.

    Long files are split into request-sized chunks. Returned MP3 chunks are
    appended into the output file in order.
    """
    voice_id = normalize_voice(voice_id)
    text_file_path = Path(text_file_path)
    if not text_file_path.is_file():
        raise TextToSpeechError(f"Text file does not exist: {text_file_path}")

    text = text_file_path.read_text(encoding="utf-8")
    chunks = chunk_text(text, chunk_size)
    if not chunks:
        raise TextToSpeechError(f"Text file is empty: {text_file_path}")

    output_path = Path(output_path) if output_path else text_file_path.with_suffix(".mp3")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = get_venice_api_key(api_key)
    if not api_key:
        raise TextToSpeechError("VENICE_API_KEY is not set in the process environment or a local .env/venice.env file.")

    with output_path.open("wb") as output_file:
        for index, chunk in enumerate(chunks, start=1):
            if progress_callback:
                progress_callback(index, len(chunks), f"Converting chunk {index} of {len(chunks)}")

            chunk_path = output_path.with_suffix(f".part{index:03d}.mp3")
            try:
                synthesize_text_to_mp3(
                    chunk,
                    chunk_path,
                    voice_id,
                    api_key=api_key,
                )
                output_file.write(chunk_path.read_bytes())
            finally:
                chunk_path.unlink(missing_ok=True)

            if index < len(chunks) and delay_seconds:
                time.sleep(delay_seconds)

    if progress_callback:
        progress_callback(len(chunks), len(chunks), "Conversion complete")
    return output_path


def normalize_voice(voice_id):
    """Return a supported Venice voice name."""
    voice_id = (voice_id or "default").strip().lower()
    if voice_id in VENICE_VOICE_ALIASES:
        return VENICE_VOICE_ALIASES[voice_id]
    if voice_id in VENICE_VOICE_ALIASES.values():
        return voice_id
    if voice_id.startswith("vv"):
        return voice_id
    if voice_id not in VALID_VOICES:
        raise TextToSpeechError(
            f"Unsupported voice '{voice_id}'. Choose one of: {', '.join(VALID_VOICES)}"
        )
    return voice_id
