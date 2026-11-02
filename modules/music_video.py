"""Music-video project paths, vocal isolation, transcription, and subtitle helpers."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


class OperationCancelled(RuntimeError):
    """Raised when a background media operation is cancelled."""


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical resource directories beneath a music-video project root."""

    root: Path

    @property
    def audio(self):
        return self.root / "audio"

    @property
    def images(self):
        return self.root / "images"

    @property
    def scenes(self):
        return self.root / "video" / "scenes"

    @property
    def output(self):
        return self.root / "video" / "output"

    @property
    def temporary(self):
        return self.root / "tmp"

    def create(self):
        for path in (self.audio, self.images, self.scenes, self.output, self.temporary):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def clean_temporary(self):
        shutil.rmtree(self.temporary, ignore_errors=True)


@dataclass(frozen=True)
class SubtitleCue:
    """One SRT subtitle cue, measured in seconds."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TimedWord:
    """One timestamped speech-to-text token."""

    start: float
    end: float
    text: str


def stored_project_path(value, project_root):
    """Store project-owned resources relative to the project root."""
    if not str(value).strip():
        return ""
    path = Path(value).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def resolve_project_path(value, project_root, project_file, version=2):
    """Resolve v2 root-relative paths and legacy project-file-relative paths."""
    if not str(value).strip():
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        base = Path(project_root) if int(version) >= 2 else Path(project_file).parent
        path = base / path
    return str(path.resolve())


def copy_project_asset(source, destination_directory, preferred_name=""):
    """Copy an external resource into a project directory without overwriting itself."""
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination_directory = Path(destination_directory)
    destination_directory.mkdir(parents=True, exist_ok=True)
    name = preferred_name or source.name
    destination = destination_directory / name
    if source != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _cancelled(cancel_event):
    return cancel_event is not None and cancel_event.is_set()


def run_cancellable(command, cancel_event=None, cwd=None, on_process=None):
    """Run a subprocess and terminate it promptly when cancellation is requested."""
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as diagnostic:
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=str(cwd) if cwd else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=diagnostic,
            text=True,
        )
        if on_process:
            on_process(process)
        try:
            while process.poll() is None:
                if _cancelled(cancel_event):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise OperationCancelled("Operation cancelled.")
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if on_process:
                on_process(None)
        diagnostic.seek(0)
        stderr = diagnostic.read()
        if process.returncode:
            raise RuntimeError(
                f"Command failed ({process.returncode}): {' '.join(map(str, command))}\n\n"
                f"{stderr[-6000:]}"
            )
    return ""


def extract_vocals(audio_path, output_directory, cancel_event=None, on_process=None):
    """Run Demucs two-stem separation and return the retained vocal WAV path."""
    audio_path = Path(audio_path).resolve()
    output_directory = Path(output_directory).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems=vocals",
        "-n",
        "htdemucs",
        "-o",
        output_directory,
        audio_path,
    ]
    try:
        run_cancellable(command, cancel_event, on_process=on_process)
    except RuntimeError as exc:
        if "No module named demucs" in str(exc):
            raise RuntimeError(
                "Demucs is not installed in this Python environment. Install it with: pip install demucs"
            ) from exc
        raise
    vocals = output_directory / "htdemucs" / audio_path.stem / "vocals.wav"
    if not vocals.is_file():
        matches = list(output_directory.glob(f"*/{audio_path.stem}/vocals.wav"))
        if not matches:
            raise RuntimeError("Demucs completed but did not produce vocals.wav.")
        vocals = matches[0]
    retained = output_directory / f"{audio_path.stem}-vocals.wav"
    shutil.copy2(vocals, retained)
    shutil.rmtree(output_directory / "htdemucs", ignore_errors=True)
    return retained


def transcribe_words(vocal_path, model_name="small", cancel_event=None):
    """Transcribe vocals with faster-whisper and return timestamped words."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install it with: pip install faster-whisper"
        ) from exc
    model = WhisperModel(model_name, device="auto", compute_type="int8")
    segments, _info = model.transcribe(str(vocal_path), word_timestamps=True)
    words = []
    for segment in segments:
        if _cancelled(cancel_event):
            raise OperationCancelled("Transcription cancelled.")
        for word in segment.words or ():
            text = str(word.word).strip()
            if text and word.start is not None and word.end is not None:
                words.append(TimedWord(float(word.start), float(word.end), text))
    if not words:
        raise RuntimeError("Speech-to-text did not identify any timestamped words in the vocal track.")
    return words


def lyric_lines(text):
    """Return non-empty authoritative lyric lines."""
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def align_lyrics_to_words(lyrics, words):
    """Map authoritative lyric lines onto ordered speech-to-text word timestamps.

    The transcript supplies timing only. Tokens are assigned in proportion to the
    authoritative line word counts, which keeps repeated/misrecognized sung text
    from replacing the user's lyrics.
    """
    lines = lyric_lines(lyrics)
    words = list(words)
    if not lines:
        raise ValueError("Enter or import lyrics before creating subtitles.")
    if not words:
        raise ValueError("No timestamped transcription words are available.")
    if len(words) < len(lines):
        raise ValueError(
            "Speech-to-text returned fewer timestamped words than lyric lines; "
            "combine short lyric lines or edit an SRT manually."
        )
    weights = [max(1, len(re.findall(r"\S+", line))) for line in lines]
    total_weight = sum(weights)
    total_words = len(words)
    boundaries = [0]
    cumulative = 0
    for weight in weights[:-1]:
        cumulative += weight
        target = round(total_words * cumulative / total_weight)
        minimum = boundaries[-1] + 1
        remaining_lines = len(lines) - len(boundaries)
        maximum = total_words - remaining_lines
        boundaries.append(max(minimum, min(target, maximum)))
    boundaries.append(total_words)

    cues = []
    for index, line in enumerate(lines):
        start_index = min(boundaries[index], total_words - 1)
        end_index = max(start_index, min(boundaries[index + 1] - 1, total_words - 1))
        start = words[start_index].start
        end = max(words[end_index].end, start + 0.2)
        cues.append(SubtitleCue(start, end, line))
    return cues


def cues_from_words(words, words_per_cue=6):
    """Group a speech-to-text result into readable draft subtitle cues."""
    words = list(words)
    if not words:
        raise ValueError("No timestamped transcription words are available.")
    group_size = max(1, int(words_per_cue))
    cues = []
    for index in range(0, len(words), group_size):
        group = words[index : index + group_size]
        cues.append(
            SubtitleCue(
                group[0].start,
                max(group[-1].end, group[0].start + 0.2),
                " ".join(word.text.strip() for word in group if word.text.strip()),
            )
        )
    return cues


def validate_cues(cues, media_duration=None):
    """Validate cue ordering, content, overlap, and optional media bounds."""
    previous_end = -1.0
    for index, cue in enumerate(cues, 1):
        if not cue.text.strip():
            raise ValueError(f"Subtitle cue {index} has no text.")
        if cue.start < 0 or cue.end <= cue.start:
            raise ValueError(f"Subtitle cue {index} has invalid timestamps.")
        if cue.start < previous_end:
            raise ValueError(f"Subtitle cue {index} overlaps the preceding cue.")
        if media_duration is not None and cue.end > float(media_duration) + 0.05:
            raise ValueError(f"Subtitle cue {index} ends after the soundtrack.")
        previous_end = cue.end
    return True


def _srt_timestamp(seconds):
    milliseconds = max(0, round(float(seconds) * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def format_srt(cues):
    """Serialize validated cues as UTF-8 SRT text."""
    validate_cues(cues)
    blocks = []
    for index, cue in enumerate(cues, 1):
        blocks.append(
            f"{index}\n{_srt_timestamp(cue.start)} --> {_srt_timestamp(cue.end)}\n{cue.text.strip()}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def offset_cues(cues, seconds):
    """Shift subtitle cues for a title card placed before the soundtrack."""
    offset = float(seconds)
    return [SubtitleCue(cue.start + offset, cue.end + offset, cue.text) for cue in cues]


def parse_srt(text):
    """Parse basic SRT content into SubtitleCue values."""
    timestamp = re.compile(
        r"(?P<sh>\d+):(?P<sm>\d{2}):(?P<ss>\d{2})[,.](?P<sms>\d{3})\s*-->\s*"
        r"(?P<eh>\d+):(?P<em>\d{2}):(?P<es>\d{2})[,.](?P<ems>\d{3})"
    )

    def seconds(match, prefix):
        return (
            int(match.group(prefix + "h")) * 3600
            + int(match.group(prefix + "m")) * 60
            + int(match.group(prefix + "s"))
            + int(match.group(prefix + "ms")) / 1000
        )

    cues = []
    normalized = str(text).replace("\r\n", "\n").strip()
    for block in re.split(r"\n\s*\n", normalized) if normalized else []:
        lines = block.splitlines()
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), -1)
        if timing_index < 0:
            raise ValueError("Malformed SRT block: missing timestamp line.")
        match = timestamp.fullmatch(lines[timing_index].strip())
        if not match:
            raise ValueError(f"Malformed SRT timestamp: {lines[timing_index]}")
        cue_text = "\n".join(lines[timing_index + 1 :]).strip()
        cues.append(SubtitleCue(seconds(match, "s"), seconds(match, "e"), cue_text))
    validate_cues(cues)
    return cues


def ffmpeg_filter_path(path):
    """Escape a resolved path for an FFmpeg filter option."""
    value = str(Path(path).resolve())
    for old, new in (("\\", "\\\\"), (":", "\\:"), ("'", "\\'"), (",", "\\,"), ("[", "\\["), ("]", "\\]")):
        value = value.replace(old, new)
    return value


def subtitle_filter(srt_path, font_name, font_size):
    """Return the libass filter expression for a user-selected font and size."""
    safe_font = str(font_name).replace("'", "").replace(",", " ").strip() or "DejaVu Sans"
    size = max(8, min(int(font_size), 144))
    style = (
        f"FontName={safe_font},FontSize={size},PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,Outline=2,Shadow=1,Alignment=2,MarginV=65"
    )
    return f"subtitles=filename='{ffmpeg_filter_path(srt_path)}':force_style='{style}'"


def burn_subtitles(input_video, srt_path, output_video, font_name, font_size, cancel_event=None):
    """Burn an SRT into a video with FFmpeg/libass."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found on PATH.")
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_video.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-i",
        input_video,
        "-vf",
        subtitle_filter(srt_path, font_name, font_size),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        output_video,
    ]
    run_cancellable(command, cancel_event)
    return output_video


def make_title_clip(image_path, output_path, duration, cancel_event=None):
    """Create a silent 1920x1080 H.264 title-card clip."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found on PATH.")
    duration = float(duration)
    if not 5 <= duration <= 10:
        raise ValueError("Title-card duration must be between 5 and 10 seconds.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-loop",
        "1",
        "-i",
        image_path,
        "-t",
        f"{duration:.3f}",
        "-vf",
        "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        "-r",
        "30",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        output_path,
    ]
    run_cancellable(command, cancel_event)
    return output_path


def prepend_title_card(input_video, image_path, output_path, duration, cancel_event=None):
    """Prepend a 1920x1080 still title card and normalize the video to H.264."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found on PATH.")
    duration = float(duration)
    if not 5 <= duration <= 10:
        raise ValueError("Title-card duration must be between 5 and 10 seconds.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    normalize = (
        "scale=1920:1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p"
    )
    filter_graph = (
        f"[0:v]{normalize},trim=duration={duration:.3f},setpts=PTS-STARTPTS[title];"
        f"[1:v]{normalize},setpts=PTS-STARTPTS[video];"
        "[title][video]concat=n=2:v=1:a=0[outv]"
    )
    command = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-loop",
        "1",
        "-framerate",
        "30",
        "-i",
        image_path,
        "-i",
        input_video,
        "-filter_complex",
        filter_graph,
        "-map",
        "[outv]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]
    run_cancellable(command, cancel_event)
    return output_path


def normalize_video_1080p(input_video, output_path, cancel_event=None):
    """Normalize a video to 1920x1080 H.264 without retaining its audio."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found on PATH.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-i",
        input_video,
        "-vf",
        "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        output_path,
    ]
    run_cancellable(command, cancel_event)
    return output_path
