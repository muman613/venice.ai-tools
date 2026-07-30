#!/usr/bin/env python3
"""
Generate a one-minute Venice AI video as four sequential 15-second
video clips.

Workflow:
1. Split the one-minute prompt into four prompts.
2. Generate four sequential image-to-video or text-to-video segments.
3. In image-to-video mode, feed each segment's final frame into the next.
4. Concatenate all four MP4 files with FFmpeg.
"""

import base64
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests
from PySide6.QtCore import QObject, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Venice API configuration
# ---------------------------------------------------------------------------

API_BASE_URL = "https://api.venice.ai/api/v1"

CHAT_URL = f"{API_BASE_URL}/chat/completions"
VIDEO_QUEUE_URL = f"{API_BASE_URL}/video/queue"
VIDEO_RETRIEVE_URL = f"{API_BASE_URL}/video/retrieve"
MODELS_URL = f"{API_BASE_URL}/models"

SEGMENT_MODEL = os.environ.get(
    "VENICE_SEGMENT_MODEL",
    "venice-uncensored-1-2",
).strip()

IMAGE_TO_VIDEO_MODE = "image-to-video"
TEXT_TO_VIDEO_MODE = "text-to-video"

DEFAULT_IMAGE_VIDEO_MODEL = os.environ.get(
    "VENICE_IMAGE_VIDEO_MODEL",
    "wan-2-7-image-to-video",
).strip()
DEFAULT_TEXT_VIDEO_MODEL = os.environ.get(
    "VENICE_TEXT_VIDEO_MODEL",
    "wan-2.5-preview-text-to-video",
).strip()
VIDEO_DURATION = "15s"
VIDEO_RESOLUTION = "1080p"
VIDEO_ASPECT_RATIO = "16:9"

SEGMENT_COUNT = 4

HTTP_TIMEOUT = (30, 180)
DOWNLOAD_TIMEOUT = (30, 600)

VIDEO_JOB_TIMEOUT_SECONDS = int(
    os.environ.get("VENICE_VIDEO_TIMEOUT", "1800")
)

VIDEO_POLL_INTERVAL_SECONDS = float(
    os.environ.get("VENICE_VIDEO_POLL_INTERVAL", "5")
)

MAX_SOURCE_IMAGE_BYTES = 25 * 1024 * 1024
SETTINGS_ORG = "VeniceAI"
SETTINGS_APP = "OneMinuteVideo"
RETAIN_INTERMEDIATE_FILES_KEY = "retain_intermediate_files"

COMPLETED_STATUSES = {
    "COMPLETED",
    "COMPLETE",
    "SUCCEEDED",
    "SUCCESS",
    "DONE",
}

FAILED_STATUSES = {
    "FAILED",
    "FAILURE",
    "ERROR",
    "CANCELLED",
    "CANCELED",
    "REJECTED",
}


# ---------------------------------------------------------------------------
# Data and exceptions
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """One 15-second video segment."""

    number: int
    description: str
    transition_note: str = ""


@dataclass
class VideoModelOption:
    """One selectable Venice video model."""

    model_id: str
    label: str
    uncensored: bool = False


class VeniceAPIError(RuntimeError):
    """An HTTP error returned by the Venice API."""

    def __init__(self, status_code, url, body):
        self.status_code = status_code
        self.url = url
        self.body = body

        super().__init__(
            f"Venice API returned HTTP {status_code} for:\n"
            f"{url}\n\n"
            f"{body or '<empty response>'}"
        )


def model_text(model):
    """Return searchable text for a model metadata object."""
    return json.dumps(
        model,
        ensure_ascii=False,
        sort_keys=True,
    ).lower()


def model_identifier(model):
    """Return the Venice model ID from a model metadata object."""
    if not isinstance(model, dict):
        return ""

    value = model.get("id") or model.get("model")
    return str(value).strip() if value else ""


def model_display_name(model):
    """Return the human-readable model name when available."""
    if not isinstance(model, dict):
        return ""

    model_spec = model.get("model_spec")
    if isinstance(model_spec, dict):
        value = model_spec.get("name")
        if value:
            return str(value).strip()

    value = model.get("name")
    return str(value).strip() if value else ""


def normalize_model_list(data):
    """Return a list from Venice's model-list response shapes."""
    models = data.get("data", data) if isinstance(data, dict) else data

    if isinstance(models, dict):
        return list(models.values())

    if isinstance(models, list):
        return models

    return []


def is_text_to_video_model(model):
    """Return True for video models that accept text prompts."""
    if not isinstance(model, dict):
        return False

    model_type = str(model.get("type", "")).lower()
    text = model_text(model)

    if model_type and model_type != "video":
        return False

    text_to_video_markers = {
        "text-to-video",
        "text to video",
        "text_to_video",
        "text2video",
        "t2v",
    }

    return any(marker in text for marker in text_to_video_markers)


def is_image_to_video_model(model):
    """Return True for video models that accept an input image."""
    if not isinstance(model, dict):
        return False

    model_type = str(model.get("type", "")).lower()
    text = model_text(model)

    if model_type and model_type != "video":
        return False

    image_to_video_markers = {
        "image-to-video",
        "image to video",
        "image_to_video",
        "image2video",
        "i2v",
    }

    return any(marker in text for marker in image_to_video_markers)


def is_uncensored_model(model):
    """Return True when the model metadata advertises uncensored behavior."""
    text = model_text(model)
    return (
        "uncensored" in text
        or "unrestricted" in text
    )


def model_option(model):
    """Build a GUI option from model metadata."""
    model_id = model_identifier(model)
    name = model_display_name(model)
    uncensored = is_uncensored_model(model)

    if name and name != model_id:
        label = f"{name} ({model_id})"
    else:
        label = model_id

    if uncensored:
        label = f"{label} - uncensored"

    return VideoModelOption(
        model_id=model_id,
        label=label,
        uncensored=uncensored,
    )


def extract_video_models(data, generation_mode):
    """Extract selectable video model options from API data."""
    if generation_mode == IMAGE_TO_VIDEO_MODE:
        predicate = is_image_to_video_model
        mode_label = "image-to-video"
    else:
        predicate = is_text_to_video_model
        mode_label = "text-to-video"

    options = [
        model_option(model)
        for model in normalize_model_list(data)
        if predicate(model) and model_identifier(model)
    ]

    options.sort(
        key=lambda option: (
            not option.uncensored,
            option.label.lower(),
        )
    )

    uncensored_options = [
        option
        for option in options
        if option.uncensored
    ]

    if uncensored_options:
        return (
            uncensored_options,
            f"Loaded uncensored {mode_label} models.",
        )

    return (
        options,
        (
            "No models were explicitly marked uncensored; "
            f"showing {mode_label} models."
        ),
    )


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class VeniceVideoModelsWorker(QObject):
    """Load Venice video models outside the Qt event loop."""

    finished = Signal(object, str)
    failed = Signal(str)

    def __init__(self, api_key, generation_mode):
        super().__init__()
        self.api_key = api_key
        self.generation_mode = generation_mode

    def run(self):
        """Fetch current video models from Venice."""
        try:
            response = requests.get(
                MODELS_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                },
                params={
                    "type": "video",
                },
                timeout=HTTP_TIMEOUT,
            )

            diagnostic = VeniceOneMinuteWorker.response_diagnostic(
                response
            )

            if not response.ok:
                raise VeniceAPIError(
                    response.status_code,
                    MODELS_URL,
                    diagnostic,
            )

            data = response.json()
            options, message = extract_video_models(
                data,
                self.generation_mode,
            )

            if not options:
                raise RuntimeError(
                    "The models API did not return any "
                    f"{self.generation_mode} models."
                )

            self.finished.emit(options, message)

        except (
            requests.RequestException,
            RuntimeError,
            ValueError,
        ) as exc:
            self.failed.emit(
                f"{type(exc).__name__}: {exc}"
            )


class VeniceOneMinuteWorker(QObject):
    """Perform API and FFmpeg work outside the Qt event loop."""

    progress = Signal(int, str)
    segment_ready = Signal(str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        api_key,
        prompt,
        source_file,
        output_file,
        video_model,
        generation_mode,
        retain_intermediate_files,
    ):
        super().__init__()

        self.api_key = api_key
        self.prompt = prompt
        self.source_file = Path(source_file) if source_file else None
        self.output_file = Path(output_file)
        self.video_model = video_model
        self.generation_mode = generation_mode
        self.retain_intermediate_files = retain_intermediate_files
        self.destination_dir = self.output_file.parent
        self.output_stem = self.output_file.stem
        self.intermediate_files = []

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            }
        )

    def run(self):
        """Run segmentation, generation, frame extraction, and stitching."""
        try:
            self.destination_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.verify_dependencies()

            if (
                self.is_image_to_video
                and self.source_file is None
            ):
                raise RuntimeError(
                    "Image-to-video mode requires a starting "
                    "reference image."
                )

            self.progress.emit(
                0,
                "Segmenting the one-minute prompt...",
            )

            segments = self.segment_story()
            self.save_segments_json(segments)

            current_image_path = self.source_file
            segment_files = []

            for segment in segments:
                self.progress.emit(
                    segment.number - 1,
                    (
                        f"Preparing segment {segment.number} "
                        f"of {SEGMENT_COUNT}..."
                    ),
                )

                output_path = (
                    self.destination_dir
                    / (
                        f"{self.output_stem}-segment-"
                        f"{segment.number:02d}.mp4"
                    )
                )

                output_path.unlink(missing_ok=True)
                self.track_intermediate_file(output_path)

                source_data_url = None
                if self.is_image_to_video:
                    source_data_url = self.image_to_data_url(
                        current_image_path
                    )

                self.generate_video_segment(
                    segment=segment,
                    output_path=output_path,
                    source_data_url=source_data_url,
                )

                self.validate_media_file(
                    output_path,
                    f"segment {segment.number}",
                )

                segment_files.append(output_path)
                self.segment_ready.emit(str(output_path))

                if (
                    self.is_image_to_video
                    and segment.number < SEGMENT_COUNT
                ):
                    continuation_frame = (
                        self.destination_dir
                        / (
                            f"{self.output_stem}-"
                            f"continuation-frame-"
                            f"{segment.number:02d}.jpg"
                        )
                    )

                    continuation_frame.unlink(missing_ok=True)
                    self.track_intermediate_file(
                        continuation_frame
                    )

                    self.progress.emit(
                        segment.number,
                        (
                            f"Extracting segment {segment.number}'s "
                            "final frame for visual continuity..."
                        ),
                    )

                    self.extract_final_frame(
                        output_path,
                        continuation_frame,
                    )

                    current_image_path = continuation_frame

                self.progress.emit(
                    segment.number,
                    (
                        f"Saved segment {segment.number} "
                        f"of {SEGMENT_COUNT}."
                    ),
                )

            self.output_file.unlink(missing_ok=True)

            self.progress.emit(
                SEGMENT_COUNT,
                "Stitching the four segments with FFmpeg...",
            )

            self.stitch_segments(
                segment_files,
                self.output_file,
            )

            self.validate_media_file(
                self.output_file,
                "final video",
            )

            self.cleanup_intermediate_files()

            self.finished.emit(str(self.output_file))

        except Exception as exc:
            self.failed.emit(
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            self.session.close()

    @property
    def is_image_to_video(self):
        """Return True when using image-to-video generation."""
        return self.generation_mode == IMAGE_TO_VIDEO_MODE

    def track_intermediate_file(self, path):
        """Track a temporary artifact for optional cleanup."""
        path = Path(path)
        if path not in self.intermediate_files:
            self.intermediate_files.append(path)

    def cleanup_intermediate_files(self):
        """Delete intermediate artifacts when retention is disabled."""
        if self.retain_intermediate_files:
            self.progress.emit(
                SEGMENT_COUNT,
                "Retaining intermediate files.",
            )
            return

        removed = 0
        for path in self.intermediate_files:
            try:
                if path.exists():
                    path.unlink()
                    removed += 1
            except OSError as exc:
                self.progress.emit(
                    SEGMENT_COUNT,
                    (
                        f"Could not delete intermediate file "
                        f"{path.name}: {exc}"
                    ),
                )

        self.progress.emit(
            SEGMENT_COUNT,
            f"Removed {removed} intermediate file(s).",
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def verify_dependencies():
        """Verify that FFmpeg is available."""
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "FFmpeg was not found on PATH. Install FFmpeg before "
                "running video generation."
            )

    @staticmethod
    def validate_media_file(path, description):
        """Verify that a generated media file exists and is non-empty."""
        if not path.exists():
            raise RuntimeError(
                f"The {description} file was not created:\n{path}"
            )

        if path.stat().st_size == 0:
            raise RuntimeError(
                f"The {description} file is empty:\n{path}"
            )

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    @staticmethod
    def response_diagnostic(response):
        """Return a readable HTTP response body."""
        try:
            return json.dumps(
                response.json(),
                indent=2,
                ensure_ascii=False,
            )[:6000]
        except ValueError:
            return response.text[:6000]

    def post_json(self, url, payload):
        """POST JSON and return the decoded response."""
        try:
            response = self.session.post(
                url,
                headers={
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=HTTP_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not connect to Venice:\n{url}\n\n{exc}"
            ) from exc

        diagnostic = self.response_diagnostic(response)

        if not response.ok:
            raise VeniceAPIError(
                response.status_code,
                url,
                diagnostic,
            )

        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Venice returned a non-JSON response from:\n"
                f"{url}\n\n"
                f"{diagnostic}"
            ) from exc

    @staticmethod
    def find_first_value(data, names):
        """Recursively find a scalar value associated with a candidate key."""
        if isinstance(data, dict):
            for key, value in data.items():
                if (
                    key in names
                    and value is not None
                    and value != ""
                    and not isinstance(value, (dict, list))
                ):
                    return value

            for value in data.values():
                found = VeniceOneMinuteWorker.find_first_value(
                    value,
                    names,
                )

                if found is not None:
                    return found

        elif isinstance(data, list):
            for item in data:
                found = VeniceOneMinuteWorker.find_first_value(
                    item,
                    names,
                )

                if found is not None:
                    return found

        return None

    @classmethod
    def get_queue_id(cls, data):
        """Extract the queue identifier from a queue response."""
        value = cls.find_first_value(
            data,
            {
                "queue_id",
                "queueId",
            },
        )

        return str(value) if value else None

    @classmethod
    def get_status(cls, data):
        """Extract and normalize a video job status."""
        value = cls.find_first_value(
            data,
            {
                "status",
                "state",
            },
        )

        if value is None:
            return "UNKNOWN"

        return str(value).strip().upper()

    @classmethod
    def get_download_url(cls, data):
        """Extract a completed-video URL."""
        value = cls.find_first_value(
            data,
            {
                "download_url",
                "downloadUrl",
                "video_url",
                "videoUrl",
                "output_url",
                "outputUrl",
                "file_url",
                "fileUrl",
                "asset_url",
                "assetUrl",
                "result_url",
                "resultUrl",
                "url",
            },
        )

        return str(value) if value else None

    # ------------------------------------------------------------------
    # Story segmentation
    # ------------------------------------------------------------------

    def segment_story(self):
        """Split the one-minute concept into four 15-second prompts."""
        payload = {
            "model": SEGMENT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You create precise prompts for sequential "
                        f"{self.generation_mode} generation. Return only valid JSON "
                        "without Markdown or commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": self.segmentation_prompt(),
                },
            ],
            "temperature": 0.6,
            "max_tokens": 2400,
        }

        data = self.post_json(
            CHAT_URL,
            payload,
        )

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "The segmentation response did not contain "
                "choices[0].message.content.\n\n"
                f"{json.dumps(data, indent=2)[:6000]}"
            ) from exc

        return self.parse_segments(str(content))

    def segmentation_prompt(self):
        """Create the segmentation-model instruction."""
        if self.is_image_to_video:
            mode_context = """
The first segment starts from a reference image supplied separately.
Each later segment starts from the extracted final frame of the preceding
segment.
""".strip()
            mode_requirement = (
                "Each description must work as an independent "
                "image-to-video prompt."
            )
        else:
            mode_context = (
                "All segments are generated from text prompts only; "
                "there is no source image."
            )
            mode_requirement = (
                "Each description must work as an independent "
                "text-to-video prompt."
            )

        return f"""
Divide the following one-minute video concept into exactly four sequential
15-second {self.generation_mode} prompts.

{mode_context}

Requirements:

1. Return exactly four segments.
2. Each segment must describe approximately 15 seconds of action.
3. Segment 2 must continue directly from the end of segment 1.
4. Segment 3 must continue directly from the end of segment 2.
5. Segment 4 must continue directly from the end of segment 3.
6. Do not restart or recap the story in later segments.
7. Repeat important character, clothing, environment, lighting, camera,
   lens, and visual-style details when needed for consistency.
8. Describe subject movement, environmental movement, camera movement,
   composition, lighting, pacing, and the desired ending frame.
9. Do not request captions, subtitles, logos, watermarks, or visible text.
10. Segment 4 must end with a satisfying final shot.
11. {mode_requirement}
12. Do not mention API parameters, duration fields, or source-image URLs
    inside the descriptions.

Return only a JSON array in exactly this form:

[
  {{
    "segment": 1,
    "description": "Complete prompt for the first 15-second video.",
    "transition_note": "Exact visual state of the final frame."
  }},
  {{
    "segment": 2,
    "description": "Complete prompt continuing from segment 1.",
    "transition_note": "Exact visual state of the final frame."
  }},
  {{
    "segment": 3,
    "description": "Complete prompt continuing from segment 2.",
    "transition_note": "Exact visual state of the final frame."
  }},
  {{
    "segment": 4,
    "description": "Complete prompt continuing from segment 3.",
    "transition_note": "Description of the final shot."
  }}
]

Original one-minute concept:

{self.prompt}
""".strip()

    def parse_segments(self, content):
        """Parse and validate the segmentation JSON."""
        cleaned = content.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()

            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            cleaned = "\n".join(lines).strip()

        first_bracket = cleaned.find("[")
        last_bracket = cleaned.rfind("]")

        if first_bracket < 0 or last_bracket < first_bracket:
            raise ValueError(
                "The segmentation model did not return a JSON array.\n\n"
                f"{content[:6000]}"
            )

        cleaned = cleaned[
            first_bracket:last_bracket + 1
        ]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "The segmentation model returned invalid JSON.\n\n"
                f"JSON error: {exc}\n\n"
                f"{cleaned[:6000]}"
            ) from exc

        if not isinstance(data, list):
            raise ValueError(
                "The segmentation response must be a JSON array."
            )

        if len(data) != SEGMENT_COUNT:
            raise ValueError(
                f"The segmentation response contained {len(data)} "
                f"segments instead of {SEGMENT_COUNT}."
            )

        segments = []

        for expected_number, item in enumerate(
            data,
            start=1,
        ):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Segment {expected_number} is not a JSON object."
                )

            description = str(
                item.get("description", "")
            ).strip()

            transition_note = str(
                item.get("transition_note", "")
            ).strip()

            if not description:
                raise ValueError(
                    f"Segment {expected_number} has no description."
                )

            # Derive numbering from array order to avoid model mistakes.
            segments.append(
                Segment(
                    number=expected_number,
                    description=description,
                    transition_note=transition_note,
                )
            )

        return segments

    def save_segments_json(self, segments):
        """Save generated prompts for inspection and reuse."""
        data = [
            {
                "segment": segment.number,
                "mode": self.generation_mode,
                "model": self.video_model,
                "duration": VIDEO_DURATION,
                "description": segment.description,
                "transition_note": segment.transition_note,
            }
            for segment in segments
        ]

        output_path = (
            self.destination_dir
            / f"{self.output_stem}-segments.json"
        )

        output_path.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.track_intermediate_file(output_path)

    # ------------------------------------------------------------------
    # Image handling
    # ------------------------------------------------------------------

    @staticmethod
    def image_to_data_url(path):
        """Convert an image file to a base64 data URL."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Reference image does not exist:\n{path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Reference path is not a file:\n{path}"
            )

        if path.stat().st_size > MAX_SOURCE_IMAGE_BYTES:
            raise ValueError(
                "The reference image exceeds the application's "
                "25 MB size limit."
            )

        mime_type, _ = mimetypes.guess_type(
            str(path)
        )

        # Extracted continuation frames are always JPEG.
        if not mime_type and path.suffix.lower() in {
            ".jpg",
            ".jpeg",
        }:
            mime_type = "image/jpeg"

        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(
                f"The selected source is not a recognized image:\n{path}"
            )

        encoded = base64.b64encode(
            path.read_bytes()
        ).decode("ascii")

        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def extract_final_frame(video_path, output_path):
        """Extract a frame near the end of a video for the next segment."""
        ffmpeg_path = shutil.which("ffmpeg")

        if not ffmpeg_path:
            raise RuntimeError(
                "FFmpeg is required to extract continuation frames."
            )

        command = [
            ffmpeg_path,
            "-y",
            "-sseof",
            "-0.10",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg could not extract the final frame.\n\n"
                f"{result.stderr[-6000:]}"
            )

        if (
            not output_path.exists()
            or output_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "FFmpeg completed without creating a continuation frame."
            )

    # ------------------------------------------------------------------
    # Video generation
    # ------------------------------------------------------------------

    def generate_video_segment(
        self,
        segment,
        output_path,
        source_data_url=None,
    ):
        """Queue, poll, download, and save one video segment."""
        payload = {
            "model": self.video_model,
            "prompt": segment.description,
            "duration": VIDEO_DURATION,
            "resolution": VIDEO_RESOLUTION,
        }

        if self.is_image_to_video:
            payload["image_url"] = source_data_url
        else:
            payload["aspect_ratio"] = VIDEO_ASPECT_RATIO

        # Deliberately omitted: audio, because model support varies.

        queue_data = self.post_json(
            VIDEO_QUEUE_URL,
            payload,
        )

        queue_id = self.get_queue_id(queue_data)
        queued_model = str(
            queue_data.get("model") or self.video_model
        )
        queued_download_url = self.get_download_url(
            queue_data
        )

        if not queue_id:
            if queued_download_url:
                self.download_video(
                    queued_download_url,
                    output_path,
                )
                return

            raise RuntimeError(
                "Venice accepted the video request but did not return "
                "a queue_id or download URL.\n\n"
                f"{json.dumps(queue_data, indent=2)[:6000]}"
            )

        self.progress.emit(
            segment.number - 1,
            (
                f"Segment {segment.number} queued. "
                f"Queue ID: {queue_id}"
            ),
        )

        self.poll_video_job(
            queue_id=queue_id,
            segment=segment,
            output_path=output_path,
            queued_model=queued_model,
            queued_download_url=queued_download_url,
        )

    def poll_video_job(
        self,
        queue_id,
        segment,
        output_path,
        queued_model,
        queued_download_url,
    ):
        """Poll Venice until a queued video is available."""
        deadline = (
            time.monotonic()
            + VIDEO_JOB_TIMEOUT_SECONDS
        )

        payload = {
            "model": queued_model,
            "queue_id": queue_id,
        }

        last_status = None
        transient_failures = 0
        completed_without_url = 0

        while time.monotonic() < deadline:
            try:
                response = self.session.post(
                    VIDEO_RETRIEVE_URL,
                    headers={
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=HTTP_TIMEOUT,
                )
            except requests.RequestException as exc:
                transient_failures += 1

                delay = min(
                    30,
                    VIDEO_POLL_INTERVAL_SECONDS
                    * transient_failures,
                )

                self.progress.emit(
                    segment.number - 1,
                    (
                        f"Segment {segment.number}: network error; "
                        f"retrying in {delay:g} seconds..."
                    ),
                )

                time.sleep(delay)
                continue

            content_type = response.headers.get(
                "Content-Type",
                "",
            ).lower()

            # Support a future/direct binary response.
            if (
                response.ok
                and (
                    "video/" in content_type
                    or "application/octet-stream" in content_type
                )
            ):
                temporary_path = output_path.with_suffix(
                    output_path.suffix + ".part"
                )

                temporary_path.write_bytes(
                    response.content
                )

                if temporary_path.stat().st_size == 0:
                    temporary_path.unlink(missing_ok=True)

                    raise RuntimeError(
                        "Venice returned an empty video response."
                    )

                temporary_path.replace(output_path)
                return

            if not response.ok:
                diagnostic = self.response_diagnostic(
                    response
                )

                if response.status_code in {
                    408,
                    409,
                    425,
                    429,
                    500,
                    502,
                    503,
                    504,
                }:
                    transient_failures += 1

                    delay = min(
                        30,
                        VIDEO_POLL_INTERVAL_SECONDS
                        * transient_failures,
                    )

                    self.progress.emit(
                        segment.number - 1,
                        (
                            f"Segment {segment.number}: temporary "
                            f"HTTP {response.status_code}; retrying "
                            f"in {delay:g} seconds..."
                        ),
                    )

                    time.sleep(delay)
                    continue

                raise VeniceAPIError(
                    response.status_code,
                    VIDEO_RETRIEVE_URL,
                    diagnostic,
                )

            transient_failures = 0

            try:
                data = response.json()
            except ValueError as exc:
                raise RuntimeError(
                    "The video retrieval response was neither JSON "
                    "nor video data.\n\n"
                    f"{response.text[:3000]}"
                ) from exc

            status = self.get_status(data)

            if status != last_status:
                self.progress.emit(
                    segment.number - 1,
                    (
                        f"Segment {segment.number}: "
                        f"status {status}, queue {queue_id}"
                    ),
                )

                last_status = status

            if status in FAILED_STATUSES:
                error_value = self.find_first_value(
                    data,
                    {
                        "error",
                        "message",
                        "detail",
                        "failure_reason",
                        "failureReason",
                    },
                )

                diagnostic = (
                    str(error_value)
                    if error_value
                    else json.dumps(
                        data,
                        indent=2,
                        ensure_ascii=False,
                    )[:6000]
                )

                raise RuntimeError(
                    f"Video segment {segment.number} failed.\n\n"
                    f"{diagnostic}"
                )

            download_url = self.get_download_url(data)

            if download_url or (
                queued_download_url
                and status in COMPLETED_STATUSES
            ):
                self.progress.emit(
                    segment.number - 1,
                    (
                        f"Segment {segment.number} completed. "
                        "Downloading MP4..."
                    ),
                )

                self.download_video(
                    download_url or queued_download_url,
                    output_path,
                )

                return

            if status in COMPLETED_STATUSES:
                completed_without_url += 1

                # Permit several additional polls in case status and
                # media availability are not updated atomically.
                if completed_without_url >= 4:
                    raise RuntimeError(
                        "Venice marked the video as completed but did "
                        "not return a download URL.\n\n"
                        f"{json.dumps(data, indent=2)[:6000]}"
                    )
            else:
                completed_without_url = 0

            time.sleep(VIDEO_POLL_INTERVAL_SECONDS)

        raise TimeoutError(
            f"Segment {segment.number}, queue {queue_id}, did not "
            f"complete within {VIDEO_JOB_TIMEOUT_SECONDS} seconds."
        )

    # ------------------------------------------------------------------
    # Video download
    # ------------------------------------------------------------------

    @staticmethod
    def download_video(url, output_path):
        """Download a completed MP4 without forwarding the API key."""
        url = str(url)

        if url.startswith("data:"):
            VeniceOneMinuteWorker.write_data_url(
                url,
                output_path,
            )
            return

        parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                f"Unsupported video download URL:\n{url}"
            )

        temporary_path = output_path.with_suffix(
            output_path.suffix + ".part"
        )

        last_error = None

        for attempt in range(1, 5):
            temporary_path.unlink(missing_ok=True)

            try:
                # Use requests.get rather than the authenticated API
                # session so the Venice API key is not forwarded to a
                # third-party or signed media-storage URL.
                with requests.get(
                    url,
                    stream=True,
                    timeout=DOWNLOAD_TIMEOUT,
                    allow_redirects=True,
                ) as response:
                    response.raise_for_status()

                    content_type = response.headers.get(
                        "Content-Type",
                        "",
                    ).lower()

                    if "application/json" in content_type:
                        raise RuntimeError(
                            "The download URL returned JSON instead "
                            "of video data:\n"
                            f"{response.text[:3000]}"
                        )

                    with temporary_path.open("wb") as handle:
                        for chunk in response.iter_content(
                            chunk_size=1024 * 1024,
                        ):
                            if chunk:
                                handle.write(chunk)

                if (
                    not temporary_path.exists()
                    or temporary_path.stat().st_size == 0
                ):
                    raise RuntimeError(
                        "The downloaded video file was empty."
                    )

                temporary_path.replace(output_path)
                return

            except (
                requests.RequestException,
                OSError,
                RuntimeError,
            ) as exc:
                last_error = exc
                temporary_path.unlink(missing_ok=True)

                if attempt < 4:
                    time.sleep(min(10, attempt * 2))

        raise RuntimeError(
            "Could not download the completed video after four "
            f"attempts.\n\n{last_error}"
        )

    @staticmethod
    def write_data_url(data_url, output_path):
        """Decode and save a base64 video data URL."""
        if "," not in data_url:
            raise ValueError(
                "The returned video data URL is malformed."
            )

        metadata, encoded = data_url.split(",", 1)

        if ";base64" not in metadata.lower():
            raise ValueError(
                "The returned video data URL is not base64 encoded."
            )

        try:
            decoded = base64.b64decode(
                encoded,
                validate=True,
            )
        except ValueError as exc:
            raise ValueError(
                "The returned video contains invalid base64 data."
            ) from exc

        output_path.write_bytes(decoded)

    # ------------------------------------------------------------------
    # FFmpeg stitching
    # ------------------------------------------------------------------

    @staticmethod
    def stitch_segments(segment_files, final_path):
        """Concatenate all segment files into one final MP4."""
        ffmpeg_path = shutil.which("ffmpeg")

        if not ffmpeg_path:
            raise RuntimeError(
                "FFmpeg was not found on PATH."
            )

        for segment_file in segment_files:
            if (
                not segment_file.exists()
                or segment_file.stat().st_size == 0
            ):
                raise RuntimeError(
                    f"Missing or empty segment:\n{segment_file}"
                )

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            delete=False,
        ) as list_handle:
            list_path = Path(list_handle.name)

            for segment_file in segment_files:
                path_text = (
                    segment_file.resolve()
                    .as_posix()
                    .replace("'", "'\\''")
                )

                list_handle.write(
                    f"file '{path_text}'\n"
                )

        try:
            # First attempt a lossless concatenation. This preserves the
            # original embedded video and audio streams.
            copy_command = [
                ffmpeg_path,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(final_path),
            ]

            copy_result = subprocess.run(
                copy_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

            if (
                copy_result.returncode == 0
                and final_path.exists()
                and final_path.stat().st_size > 0
            ):
                return

            final_path.unlink(missing_ok=True)

            # If stream-copy concatenation fails, normalize the video and
            # preserve/re-encode an optional embedded audio stream.
            reencode_command = [
                ffmpeg_path,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-fflags",
                "+genpts",
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(final_path),
            ]

            reencode_result = subprocess.run(
                reencode_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

            if reencode_result.returncode != 0:
                raise RuntimeError(
                    "FFmpeg could not stitch the generated segments.\n\n"
                    f"{reencode_result.stderr[-6000:]}"
                )

        finally:
            list_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Qt GUI
# ---------------------------------------------------------------------------

class PromptEdit(QTextEdit):
    """Plain-text prompt editor."""

    def __init__(self):
        super().__init__()

        self.setAcceptRichText(False)
        self.setPlaceholderText(
            "Describe the complete one-minute video..."
        )


class OneMinuteWindow(QMainWindow):
    """Main GUI window."""

    def __init__(self):
        super().__init__()

        self.thread = None
        self.worker = None
        self.model_thread = None
        self.model_worker = None
        self.generating = False

        self.api_key = os.environ.get(
            "VENICE_API_KEY",
            "",
        ).strip()
        self.settings = QSettings(
            SETTINGS_ORG,
            SETTINGS_APP,
        )

        self.setWindowTitle(
            "Venice One-Minute Video"
        )
        self.resize(980, 720)

        self.source_file_edit = QLineEdit()
        self.source_file_edit.setPlaceholderText(
            "Required for image-to-video"
        )

        self.source_file_button = QPushButton(
            "Browse..."
        )
        self.source_file_button.clicked.connect(
            self.choose_source_file
        )

        self.output_file_edit = QLineEdit()
        self.output_file_edit.setPlaceholderText(
            "Final output MP4 path"
        )

        self.output_file_button = QPushButton(
            "Save As..."
        )
        self.output_file_button.clicked.connect(
            self.choose_output_file
        )

        self.retain_intermediate_checkbox = QCheckBox(
            "Retain intermediate files"
        )
        self.retain_intermediate_checkbox.setChecked(
            self.load_retain_intermediate_files()
        )
        self.retain_intermediate_checkbox.stateChanged.connect(
            self.save_retain_intermediate_files
        )

        self.generation_mode_combo = QComboBox()
        self.generation_mode_combo.addItem(
            "Image-to-video",
            IMAGE_TO_VIDEO_MODE,
        )
        self.generation_mode_combo.addItem(
            "Text-to-video",
            TEXT_TO_VIDEO_MODE,
        )
        self.generation_mode_combo.currentIndexChanged.connect(
            self.generation_mode_changed
        )

        self.video_model_combo = QComboBox()
        self.video_model_combo.addItem(
            DEFAULT_IMAGE_VIDEO_MODEL,
            DEFAULT_IMAGE_VIDEO_MODEL,
        )

        self.refresh_models_button = QPushButton(
            "Refresh Models"
        )
        self.refresh_models_button.clicked.connect(
            self.load_video_models
        )

        self.prompt_input = PromptEdit()

        self.log_display = QTextBrowser()
        self.log_display.setReadOnly(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(
            0,
            SEGMENT_COUNT,
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(
            "Segment %v of %m"
        )

        self.generate_button = QPushButton(
            "Generate"
        )
        self.generate_button.clicked.connect(
            self.generate
        )

        self.status_label = QLabel("Ready")
        self.status_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        self.build_layout()
        self.apply_style()

        QTimer.singleShot(
            0,
            self.startup,
        )

    def build_layout(self):
        """Build the GUI layout."""
        source_row = QHBoxLayout()
        source_row.addWidget(
            self.source_file_edit,
            1,
        )
        source_row.addWidget(
            self.source_file_button
        )

        output_row = QHBoxLayout()
        output_row.addWidget(
            self.output_file_edit,
            1,
        )
        output_row.addWidget(
            self.output_file_button
        )

        model_row = QHBoxLayout()
        model_row.addWidget(
            self.video_model_combo,
            1,
        )
        model_row.addWidget(
            self.refresh_models_button
        )

        form = QFormLayout()
        form.addRow(
            "Mode",
            self.generation_mode_combo,
        )
        form.addRow(
            "Video model",
            model_row,
        )
        form.addRow(
            "Starting image",
            source_row,
        )
        form.addRow(
            "Output MP4",
            output_row,
        )
        form.addRow(
            "",
            self.retain_intermediate_checkbox,
        )

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(
            self.generate_button
        )

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(
            QLabel("One-minute prompt")
        )
        layout.addWidget(
            self.prompt_input,
            2,
        )
        layout.addWidget(
            QLabel("Status log")
        )
        layout.addWidget(
            self.log_display,
            1,
        )
        layout.addWidget(
            self.progress_bar
        )
        layout.addLayout(
            button_row
        )
        layout.addWidget(
            self.status_label
        )

        central_widget = QWidget()
        central_widget.setLayout(layout)

        self.setCentralWidget(
            central_widget
        )

    def apply_style(self):
        """Apply dark GUI styling."""
        self.setStyleSheet(
            """
            QMainWindow,
            QWidget {
                background: #202124;
                color: #f1f3f4;
                font-size: 14px;
            }

            QLineEdit,
            QComboBox,
            QTextBrowser,
            QTextEdit {
                background: #2f3136;
                border: 1px solid #4b4f58;
                border-radius: 4px;
                color: #f1f3f4;
                padding: 7px;
                selection-background-color: #2f80ed;
            }

            QPushButton {
                background: #3c4043;
                border: 1px solid #5f6368;
                border-radius: 4px;
                color: #f1f3f4;
                padding: 8px 14px;
            }

            QPushButton:hover {
                background: #4b4f58;
            }

            QPushButton:disabled {
                color: #8b9098;
                background: #2a2c30;
            }

            QLabel {
                color: #c9d1d9;
            }

            QProgressBar {
                background: #2f3136;
                border: 1px solid #4b4f58;
                border-radius: 4px;
                color: #f1f3f4;
                min-height: 22px;
                text-align: center;
            }

            QProgressBar::chunk {
                background: #2f80ed;
                border-radius: 3px;
            }
            """
        )

    def choose_source_file(self):
        """Choose the starting reference image."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Choose starting image",
            "",
            (
                "Image files "
                "(*.png *.jpg *.jpeg *.webp);;"
                "All files (*)"
            ),
        )

        if filename:
            self.source_file_edit.setText(
                filename
            )

    def choose_output_file(self):
        """Choose the final MP4 output path."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save final video as",
            "",
            "MP4 video (*.mp4);;All files (*)",
        )

        if filename:
            path = Path(filename)

            if path.suffix.lower() != ".mp4":
                path = path.with_suffix(".mp4")

            self.output_file_edit.setText(
                str(path)
            )

    def ensure_api_key(self):
        """Load or request the Venice API key."""
        if self.api_key:
            return True

        api_key, accepted = QInputDialog.getText(
            self,
            "Venice API key",
            "Enter VENICE_API_KEY:",
            QLineEdit.Password,
        )

        if accepted and api_key.strip():
            self.api_key = api_key.strip()

            self.status_label.setText(
                "API key loaded for this session."
            )

            return True

        self.status_label.setText(
            "VENICE_API_KEY is required."
        )

        return False

    def load_retain_intermediate_files(self):
        """Return the persisted intermediate-file retention setting."""
        value = self.settings.value(
            RETAIN_INTERMEDIATE_FILES_KEY,
            False,
        )

        if isinstance(value, bool):
            return value

        return str(value).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def save_retain_intermediate_files(self, *_args):
        """Persist the intermediate-file retention setting."""
        self.settings.setValue(
            RETAIN_INTERMEDIATE_FILES_KEY,
            self.retain_intermediate_checkbox.isChecked(),
        )
        self.settings.sync()

    def startup(self):
        """Load session prerequisites after the window is visible."""
        self.update_source_controls()
        if self.ensure_api_key():
            self.load_video_models()

    def selected_generation_mode(self):
        """Return the selected generation mode."""
        mode = self.generation_mode_combo.currentData()
        return str(mode or IMAGE_TO_VIDEO_MODE)

    def default_video_model(self):
        """Return the default model for the selected generation mode."""
        if self.selected_generation_mode() == TEXT_TO_VIDEO_MODE:
            return DEFAULT_TEXT_VIDEO_MODEL

        return DEFAULT_IMAGE_VIDEO_MODEL

    def generation_mode_changed(self):
        """Handle image/text video mode changes."""
        self.video_model_combo.clear()
        default_model = self.default_video_model()
        self.video_model_combo.addItem(
            default_model,
            default_model,
        )
        self.update_source_controls()
        self.load_video_models()

    def update_source_controls(self):
        """Enable source-image controls only when the mode needs them."""
        is_image_mode = (
            self.selected_generation_mode()
            == IMAGE_TO_VIDEO_MODE
        )

        self.source_file_edit.setEnabled(
            is_image_mode and not self.generating
        )
        self.source_file_button.setEnabled(
            is_image_mode and not self.generating
        )

        if is_image_mode:
            self.source_file_edit.setPlaceholderText(
                "Required reference image for segment 1"
            )
        else:
            self.source_file_edit.setPlaceholderText(
                "Not used in text-to-video mode"
            )

    def load_video_models(self):
        """Load current video models into the dropdown."""
        if self.model_thread is not None:
            return

        if not self.ensure_api_key():
            return

        generation_mode = self.selected_generation_mode()

        self.status_label.setText(
            f"Loading {generation_mode} models..."
        )
        self.refresh_models_button.setDisabled(True)

        current_model = self.selected_video_model()

        self.model_thread = QThread(self)
        self.model_worker = VeniceVideoModelsWorker(
            self.api_key,
            generation_mode,
        )
        self.model_worker.current_model = current_model

        self.model_worker.moveToThread(
            self.model_thread
        )

        self.model_thread.started.connect(
            self.model_worker.run
        )

        self.model_worker.finished.connect(
            self.video_models_loaded
        )

        self.model_worker.failed.connect(
            self.video_models_failed
        )

        self.model_worker.finished.connect(
            self.model_thread.quit
        )

        self.model_worker.failed.connect(
            self.model_thread.quit
        )

        self.model_thread.finished.connect(
            self.model_worker.deleteLater
        )

        self.model_thread.finished.connect(
            self.model_thread.deleteLater
        )

        self.model_thread.finished.connect(
            self.clear_model_worker
        )

        self.model_thread.start()

    def video_models_loaded(self, options, message):
        """Populate the video model dropdown with API results."""
        current_model = self.selected_video_model()

        self.video_model_combo.clear()

        for option in options:
            self.video_model_combo.addItem(
                option.label,
                option.model_id,
            )

        index = self.video_model_combo.findData(
            current_model
        )

        if index < 0:
            index = self.video_model_combo.findData(
                self.default_video_model()
            )

        if index >= 0:
            self.video_model_combo.setCurrentIndex(index)

        self.status_label.setText(message)
        self.append_log(message)

    def video_models_failed(self, message):
        """Report model-loading failures while preserving fallback model."""
        self.status_label.setText(
            "Could not load video models."
        )
        self.append_log(
            f"Model load error: {message}"
        )

    def clear_model_worker(self):
        """Clear model-loader references after thread termination."""
        self.model_thread = None
        self.model_worker = None
        if self.thread is None:
            self.refresh_models_button.setDisabled(False)

    def selected_video_model(self):
        """Return the selected Venice video model ID."""
        model = self.video_model_combo.currentData()
        if model:
            return str(model)

        return (
            self.video_model_combo
            .currentText()
            .strip()
            or self.default_video_model()
        )

    def generate(self):
        """Validate input and start generation."""
        if self.thread is not None:
            return

        prompt = (
            self.prompt_input
            .toPlainText()
            .strip()
        )

        source_file = (
            self.source_file_edit
            .text()
            .strip()
        )

        generation_mode = self.selected_generation_mode()
        video_model = self.selected_video_model()
        retain_intermediate_files = (
            self.retain_intermediate_checkbox.isChecked()
        )

        output_file = (
            self.output_file_edit
            .text()
            .strip()
        )

        if not self.ensure_api_key():
            return

        if generation_mode == IMAGE_TO_VIDEO_MODE and not source_file:
            QMessageBox.warning(
                self,
                "Missing starting image",
                (
                    "Image-to-video mode requires a starting "
                    "reference image."
                ),
            )
            return

        if source_file and not Path(source_file).is_file():
            QMessageBox.warning(
                self,
                "Invalid starting image",
                f"The selected file does not exist:\n{source_file}",
            )
            return

        if not prompt:
            QMessageBox.warning(
                self,
                "Missing prompt",
                "Enter a one-minute video prompt.",
            )
            return

        if not video_model:
            QMessageBox.warning(
                self,
                "Missing video model",
                "Choose a video model first.",
            )
            return

        if not output_file:
            QMessageBox.warning(
                self,
                "Missing output file",
                "Choose the final output MP4 path.",
            )
            return

        output_path = Path(output_file)

        if output_path.suffix.lower() != ".mp4":
            output_path = output_path.with_suffix(".mp4")
            self.output_file_edit.setText(str(output_path))

        self.log_display.clear()
        self.progress_bar.setValue(0)
        self.set_generating(True)

        self.append_log(
            f"Text model: {SEGMENT_MODEL}"
        )
        self.append_log(
            f"Video model: {video_model}"
        )
        self.append_log(
            f"Plan: {SEGMENT_COUNT} × {VIDEO_DURATION} "
            f"at {VIDEO_RESOLUTION}"
        )
        self.append_log(
            f"Mode: {generation_mode}"
        )
        if generation_mode == TEXT_TO_VIDEO_MODE:
            self.append_log(
                f"Aspect ratio: {VIDEO_ASPECT_RATIO}"
            )
        else:
            self.append_log(
                "Aspect ratio: derived from source image"
            )
        self.append_log(
            "Audio field: omitted; model default is used"
        )
        if generation_mode == IMAGE_TO_VIDEO_MODE:
            self.append_log(
                "Continuity: each segment uses the preceding final frame"
            )
        else:
            self.append_log(
                "Continuity: each segment prompt continues the previous one"
            )
        self.append_log(
            "Intermediate files: "
            + (
                "retained"
                if retain_intermediate_files
                else "removed after final MP4 is saved"
            )
        )
        self.append_log(
            "Starting generation..."
        )

        self.thread = QThread(self)

        self.worker = VeniceOneMinuteWorker(
            api_key=self.api_key,
            prompt=prompt,
            source_file=source_file,
            output_file=str(output_path),
            video_model=video_model,
            generation_mode=generation_mode,
            retain_intermediate_files=retain_intermediate_files,
        )

        self.worker.moveToThread(
            self.thread
        )

        self.thread.started.connect(
            self.worker.run
        )

        self.worker.progress.connect(
            self.update_progress
        )

        self.worker.segment_ready.connect(
            self.segment_ready
        )

        self.worker.finished.connect(
            self.generation_finished
        )

        self.worker.failed.connect(
            self.generation_failed
        )

        self.worker.finished.connect(
            self.thread.quit
        )

        self.worker.failed.connect(
            self.thread.quit
        )

        self.thread.finished.connect(
            self.worker.deleteLater
        )

        self.thread.finished.connect(
            self.thread.deleteLater
        )

        self.thread.finished.connect(
            self.clear_worker
        )

        self.thread.start()

    def update_progress(self, value, message):
        """Update GUI progress."""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        self.append_log(message)

    def segment_ready(self, path):
        """Log a completed segment."""
        self.append_log(
            f"Saved {Path(path).name}"
        )

    def generation_finished(self, final_path):
        """Handle successful generation."""
        self.progress_bar.setValue(
            SEGMENT_COUNT
        )

        self.status_label.setText(
            f"Finished: {final_path}"
        )

        self.append_log(
            f"Final output: {final_path}"
        )

        self.set_generating(False)

        QMessageBox.information(
            self,
            "Generation complete",
            f"Saved final MP4:\n{final_path}",
        )

    def generation_failed(self, message):
        """Handle a worker failure."""
        self.status_label.setText(
            "Generation failed."
        )

        self.append_log(
            f"Error: {message}"
        )

        self.set_generating(False)

        QMessageBox.critical(
            self,
            "Generation failed",
            message,
        )

    def clear_worker(self):
        """Clear worker references after thread termination."""
        self.thread = None
        self.worker = None
        self.update_source_controls()

    def set_generating(self, generating):
        """Enable or disable controls."""
        self.generating = generating

        self.generate_button.setDisabled(
            generating
        )

        self.output_file_button.setDisabled(
            generating
        )

        self.output_file_edit.setDisabled(
            generating
        )

        self.generation_mode_combo.setDisabled(
            generating
        )

        self.video_model_combo.setDisabled(
            generating
        )

        self.refresh_models_button.setDisabled(
            generating or self.model_thread is not None
        )

        self.retain_intermediate_checkbox.setDisabled(
            generating
        )

        self.prompt_input.setDisabled(
            generating
        )

        self.update_source_controls()

    def append_log(self, message):
        """Append a line to the status log."""
        self.log_display.append(
            str(message)
        )

    def closeEvent(self, event):
        """Prevent closing while generation is active."""
        if (
            self.thread is not None
            and self.thread.isRunning()
        ):
            QMessageBox.warning(
                self,
                "Generation in progress",
                (
                    "Wait for video generation to finish "
                    "before closing the application."
                ),
            )

            event.ignore()
            return

        if (
            self.model_thread is not None
            and self.model_thread.isRunning()
        ):
            QMessageBox.warning(
                self,
                "Model loading in progress",
                (
                    "Wait for model loading to finish "
                    "before closing the application."
                ),
            )

            event.ignore()
            return

        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Start the GUI application."""
    app = QApplication(sys.argv)

    window = OneMinuteWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
