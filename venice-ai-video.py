#!/usr/bin/env python3
"""
Generate a Venice AI video as sequential API-sized clips.

Workflow:
1. Split the requested video prompt across the requested output duration.
2. Generate sequential image-to-video or text-to-video segments.
3. In image-to-video mode, feed each segment's final frame into the next.
4. Concatenate all segment MP4 files with FFmpeg.
"""

import base64
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import threading
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
    QSpinBox,
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
VIDEO_QUOTE_URL = f"{API_BASE_URL}/video/quote"
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
VIDEO_RESOLUTION = "1080p"
VIDEO_ASPECT_RATIO = "16:9"

SEEDANCE_ENHANCED_R2V_MODEL = (
    "seedance-2-0-enhanced-reference-to-video"
)

PREFERRED_VIDEO_MODEL_IDS = {
    SEEDANCE_ENHANCED_R2V_MODEL,
}

HTTP_TIMEOUT = (30, 180)
DOWNLOAD_TIMEOUT = (30, 600)
FFMPEG_FRAME_TIMEOUT_SECONDS = int(
    os.environ.get("VENICE_FFMPEG_FRAME_TIMEOUT", "120")
)
FFMPEG_STITCH_TIMEOUT_SECONDS = int(
    os.environ.get("VENICE_FFMPEG_STITCH_TIMEOUT", "900")
)
SEEDANCE_CONSENT_TIMEOUT_SECONDS = int(
    os.environ.get("VENICE_SEEDANCE_CONSENT_TIMEOUT", "900")
)

VIDEO_JOB_TIMEOUT_SECONDS = int(
    os.environ.get("VENICE_VIDEO_TIMEOUT", "1800")
)

VIDEO_POLL_INTERVAL_SECONDS = float(
    os.environ.get("VENICE_VIDEO_POLL_INTERVAL", "5")
)

MAX_RETRIEVE_HTTP_500_RETRIES = int(
    os.environ.get("VENICE_VIDEO_MAX_HTTP_500_RETRIES", "3")
)

MAX_SOURCE_IMAGE_BYTES = 25 * 1024 * 1024
SETTINGS_ORG = "VeniceAI"
SETTINGS_APP = "Video"
RETAIN_INTERMEDIATE_FILES_KEY = "retain_intermediate_files"
STARTING_IMAGE_PATH_KEY = "starting_image_path"
PROMPT_DIRECTORY_KEY = "prompt_directory"

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
    """One generated video segment."""

    number: int
    duration_seconds: int
    description: str
    transition_note: str = ""
    first_frame_continuity: str = ""
    motion_continuation: str = ""
    final_frame: str = ""

    @property
    def duration(self):
        """Return Venice's duration string for this segment."""
        return f"{self.duration_seconds}s"


@dataclass
class VideoModelOption:
    """One selectable Venice video model."""

    model_id: str
    label: str
    uncensored: bool = False
    preferred: bool = False
    durations: tuple = ()
    resolutions: tuple = ()
    aspect_ratios: tuple = ()
    max_reference_images: int = 0


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
        "reference-to-video",
        "reference to video",
        "reference_to_video",
        "r2v",
    }

    return any(marker in text for marker in image_to_video_markers)


def is_uncensored_model(model):
    """Return True when the model metadata advertises uncensored behavior."""
    text = model_text(model)
    return (
        "uncensored" in text
        or "unrestricted" in text
    )


def is_preferred_video_model_id(model_id):
    """Return True for models kept visible despite metadata labels."""
    return model_id.lower() in PREFERRED_VIDEO_MODEL_IDS


def is_preferred_video_model(model):
    """Return True for a locally preferred model."""
    return is_preferred_video_model_id(
        model_identifier(model).lower()
    )


def reference_image_tag(model_id, index):
    """Return the prompt tag for one flat reference image."""
    if "seedance" in model_id.lower():
        return f"<Image {index}>"

    return f"@Image{index}"


def model_constraints(model):
    """Return provider-advertised constraints for a model."""
    if not isinstance(model, dict):
        return {}

    model_spec = model.get("model_spec")
    if isinstance(model_spec, dict):
        constraints = model_spec.get("constraints")
        if isinstance(constraints, dict):
            return constraints

    constraints = model.get("constraints")
    if isinstance(constraints, dict):
        return constraints

    return {}


def tuple_constraint(constraints, name):
    """Return one list-valued model constraint as a tuple of strings."""
    value = constraints.get(name)
    if not isinstance(value, list):
        return ()

    return tuple(str(item) for item in value if item)


def reference_image_limit(model):
    """Return supported flat reference-image count for one model."""
    model_id = model_identifier(model).lower()
    text = model_text(model)

    if (
        "reference-to-video" not in model_id
        and "r2v" not in text
    ):
        return 0

    if "seedance-2-5" in model_id:
        return 30

    if "seedance" in model_id:
        return 9

    if "grok" in model_id:
        return 7

    # Venice reference-to-video docs cap combined visual inputs at 7 for
    # Kling-style models; use that as a conservative flat-reference limit.
    return 7


def model_option(model):
    """Build a GUI option from model metadata."""
    model_id = model_identifier(model)
    name = model_display_name(model)
    uncensored = is_uncensored_model(model)
    preferred = is_preferred_video_model_id(
        model_id.lower()
    )
    constraints = model_constraints(model)

    if name and name != model_id:
        label = f"{name} ({model_id})"
    else:
        label = model_id

    if uncensored:
        label = f"{label} - uncensored"
    elif preferred:
        label = f"{label} - preferred/permissive"

    return VideoModelOption(
        model_id=model_id,
        label=label,
        uncensored=uncensored,
        preferred=preferred,
        durations=tuple_constraint(constraints, "durations"),
        resolutions=tuple_constraint(constraints, "resolutions"),
        aspect_ratios=tuple_constraint(constraints, "aspect_ratios"),
        max_reference_images=reference_image_limit(model),
    )


def known_video_model_options(generation_mode):
    """Return local fallback model options not always advertised as uncensored."""
    if generation_mode != IMAGE_TO_VIDEO_MODE:
        return []

    return [
        VideoModelOption(
            model_id=SEEDANCE_ENHANCED_R2V_MODEL,
            label=(
                "Seedance 2.0 R2V Enhanced "
                f"({SEEDANCE_ENHANCED_R2V_MODEL}) - preferred/permissive"
            ),
            uncensored=False,
            preferred=True,
            durations=(
                "10s",
                "9s",
                "8s",
                "7s",
                "6s",
                "5s",
                "4s",
                "3s",
                "2s",
                "1s",
            ),
            resolutions=("1080p", "720p"),
            aspect_ratios=(
                "16:9",
                "4:3",
                "3:2",
                "1:1",
                "2:3",
                "3:4",
                "9:16",
            ),
            max_reference_images=9,
        ),
    ]


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

    existing_model_ids = {
        option.model_id
        for option in options
    }
    for known_option in known_video_model_options(generation_mode):
        if known_option.model_id not in existing_model_ids:
            options.append(known_option)

    options.sort(
        key=lambda option: (
            not (option.uncensored or option.preferred),
            option.label.lower(),
        )
    )

    preferred_count = sum(
        1
        for option in options
        if option.uncensored or option.preferred
    )

    if preferred_count:
        return (
            options,
            (
                f"Loaded {len(options)} {mode_label} models "
                f"({preferred_count} uncensored/preferred first)."
            ),
        )

    return (
        options,
        f"Loaded {len(options)} {mode_label} models.",
    )


def fallback_video_durations(model_id):
    """Return conservative durations for a manually entered model ID."""
    model_id_lower = model_id.lower()

    if "seedance-2-5" in model_id_lower:
        return tuple(
            f"{seconds}s"
            for seconds in range(30, 3, -1)
        )

    if "seedance" in model_id_lower:
        return tuple(
            f"{seconds}s"
            for seconds in range(15, 3, -1)
        )

    return ("15s", "10s", "5s")


def fallback_video_resolutions(model_id):
    """Return conservative resolutions for a manually entered model ID."""
    model_id_lower = model_id.lower()

    if "seedance-2-5" in model_id_lower:
        return ("720p", "480p")

    if "seedance" in model_id_lower:
        return ("1080p", "720p", "480p")

    return (VIDEO_RESOLUTION, "720p")


def fallback_reference_image_limit(model_id):
    """Return a flat-reference-image limit for a manual model option."""
    model_id_lower = model_id.lower()

    if (
        "reference-to-video" not in model_id_lower
        and "r2v" not in model_id_lower
    ):
        return 0

    if "seedance-2-5" in model_id_lower:
        return 30

    if "seedance" in model_id_lower:
        return 9

    return 7


def fallback_video_aspect_ratios(model_id):
    """Return conservative aspect ratios for a manually entered model ID."""
    model_id_lower = model_id.lower()

    if "seedance" in model_id_lower:
        return (
            "16:9",
            "9:16",
            "1:1",
            "4:3",
            "3:4",
            "21:9",
        )

    return (VIDEO_ASPECT_RATIO,)

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

            diagnostic = VeniceVideoWorker.response_diagnostic(
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


class VeniceVideoWorker(QObject):
    """Perform API and FFmpeg work outside the Qt event loop."""

    progress = Signal(int, str)
    segment_ready = Signal(str)
    seedance_consent_required = Signal(int, str, str, str)
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
        total_seconds,
        segment_durations,
        video_resolution,
        video_aspect_ratio,
        reference_files,
        retain_intermediate_files,
    ):
        super().__init__()

        self.api_key = api_key
        self.prompt = prompt
        self.source_file = Path(source_file) if source_file else None
        self.output_file = Path(output_file)
        self.video_model = video_model
        self.generation_mode = generation_mode
        self.total_seconds = total_seconds
        self.segment_durations = list(segment_durations)
        self.video_resolution = video_resolution
        self.video_aspect_ratio = video_aspect_ratio
        self.reference_files = [
            Path(path)
            for path in reference_files
        ]
        self.retain_intermediate_files = retain_intermediate_files
        self.destination_dir = self.output_file.parent
        self.output_stem = self.output_file.stem
        self.intermediate_files = []
        self.seedance_consent_event = threading.Event()
        self.seedance_consent_accepted = False

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

            self.estimate_generation_cost()

            self.progress.emit(
                0,
                "Segmenting the video prompt...",
            )

            segments = self.segment_story()
            self.save_segments_json(segments)

            current_image_path = self.source_file
            reference_data_urls = [
                self.image_to_data_url(path)
                for path in self.reference_files
            ]
            segment_files = []
            segment_count = len(segments)

            for segment in segments:
                self.progress.emit(
                    segment.number - 1,
                    (
                        f"Preparing segment {segment.number} "
                        f"of {segment_count}..."
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
                    reference_data_urls=reference_data_urls,
                )

                self.validate_media_file(
                    output_path,
                    f"segment {segment.number}",
                )

                segment_files.append(output_path)
                self.segment_ready.emit(str(output_path))

                if (
                    self.is_image_to_video
                    and segment.number < segment_count
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
                        f"of {segment_count}."
                    ),
                )

            self.output_file.unlink(missing_ok=True)

            self.progress.emit(
                segment_count,
                "Stitching the segments with FFmpeg...",
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

    @staticmethod
    def plan_segment_durations(
        total_seconds,
        supported_durations=None,
    ):
        """Split requested seconds into supported segment durations."""
        total_seconds = int(total_seconds)

        allowed = []
        for duration in supported_durations or ():
            text = str(duration).strip().lower()
            if not text.endswith("s"):
                continue

            try:
                seconds = int(text[:-1])
            except ValueError:
                continue

            if seconds > 0:
                allowed.append(seconds)

        if not allowed:
            allowed = [15, 10, 5]

        allowed = sorted(set(allowed), reverse=True)
        possible = {0: []}

        for seconds in range(1, total_seconds + 1):
            best = None
            for duration in allowed:
                prior = possible.get(seconds - duration)
                if prior is None:
                    continue

                candidate = prior + [duration]
                if (
                    best is None
                    or len(candidate) < len(best)
                    or candidate > best
                ):
                    best = candidate

            if best is not None:
                possible[seconds] = best

        if total_seconds not in possible:
            supported = ", ".join(f"{duration}s" for duration in allowed)
            raise ValueError(
                f"The selected model cannot generate exactly "
                f"{total_seconds} seconds. Supported segment durations: "
                f"{supported}."
            )

        return possible[total_seconds]

    def track_intermediate_file(self, path):
        """Track a temporary artifact for optional cleanup."""
        path = Path(path)
        if path not in self.intermediate_files:
            self.intermediate_files.append(path)

    def cleanup_intermediate_files(self):
        """Delete intermediate artifacts when retention is disabled."""
        if self.retain_intermediate_files:
            self.progress.emit(
                len(self.segment_durations),
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
                    len(self.segment_durations),
                    (
                        f"Could not delete intermediate file "
                        f"{path.name}: {exc}"
                    ),
                )

        self.progress.emit(
            len(self.segment_durations),
            f"Removed {removed} intermediate file(s).",
        )

    def quote_payload(self, duration_seconds):
        """Return Venice video quote inputs for one segment."""
        payload = {
            "model": self.video_model,
            "duration": f"{duration_seconds}s",
        }

        if self.video_resolution:
            payload["resolution"] = self.video_resolution

        if not self.is_image_to_video and self.video_aspect_ratio:
            payload["aspect_ratio"] = self.video_aspect_ratio

        return payload

    def estimate_generation_cost(self):
        """Log a Venice video quote total when the quote API is available."""
        total = 0.0

        try:
            for duration_seconds in self.segment_durations:
                data = self.post_json(
                    VIDEO_QUOTE_URL,
                    self.quote_payload(duration_seconds),
                )
                quote = self.find_first_value(
                    data,
                    {"quote", "cost", "price", "amount"},
                )

                if quote is None:
                    raise RuntimeError(
                        "The quote response did not include a quote."
                    )

                total += float(quote)

        except (
            VeniceAPIError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            self.progress.emit(
                0,
                f"Cost estimate unavailable: {exc}",
            )
            return

        self.progress.emit(
            0,
            f"Estimated video generation cost: ${total:.4f}",
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
    def describe_payload(payload):
        """Return a log-safe description of a video request payload."""
        description = {}

        for key, value in payload.items():
            if key in {
                "image_url",
                "reference_image_urls",
                "referenceImageUrls",
            }:
                if isinstance(value, list):
                    description[key] = [
                        VeniceVideoWorker.describe_image_value(item)
                        for item in value
                    ]
                    continue

                description[key] = VeniceVideoWorker.describe_image_value(
                    value
                )
            elif key == "prompt":
                text = str(value or "")
                description["prompt_chars"] = len(text)
                description["prompt_preview"] = text[:240]
            else:
                description[key] = value

        return json.dumps(
            description,
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def describe_image_value(value):
        """Return a log-safe description of an image URL or data URL."""
        text = str(value or "")
        if text.startswith("data:"):
            header = text.split(",", 1)[0]
            return f"{header},... ({len(text):,} chars)"

        return text

    @staticmethod
    def parse_json_diagnostic(text):
        """Parse a diagnostic JSON string if Venice returned one."""
        try:
            value = json.loads(text or "")
        except ValueError:
            return {}

        return value if isinstance(value, dict) else {}

    @classmethod
    def seedance_consent_info(cls, error):
        """Return consent info from a Venice needs_consent response."""
        if error.status_code != 409:
            return None

        data = cls.parse_json_diagnostic(error.body)
        error_data = data.get("error")
        if not isinstance(error_data, dict):
            error_data = {}

        code = str(
            error_data.get("code")
            or data.get("code")
            or data.get("error")
            or ""
        )
        consent_flow = str(
            data.get("consent_flow")
            or data.get("consentFlow")
            or ""
        )

        if (
            code != "needs_consent"
            and consent_flow != "seedance"
        ):
            return None

        consent = data.get("consent")
        if not isinstance(consent, dict):
            consent = {}

        roles = data.get("face_media_roles")
        if isinstance(roles, list):
            roles_text = ", ".join(
                str(role)
                for role in roles
            )
        else:
            roles_text = ""

        policy_text = str(
            consent.get("policy_text")
            or error_data.get("message")
            or "Seedance consent is required for this request."
        )
        docs_url = str(
            data.get("docs_url")
            or "https://docs.venice.ai/guides/media/seedance-face-consent"
        )

        return {
            "policy_text": policy_text,
            "roles_text": roles_text,
            "docs_url": docs_url,
        }

    @staticmethod
    def seedance_consent_payload():
        """Return Venice's required Seedance consent attestation."""
        return {
            "confirmed_terms_and_privacy": True,
            "confirmed_legal_right": True,
            "confirmed_screening_acknowledged": True,
        }

    def set_seedance_consent_response(self, accepted):
        """Receive the GUI consent decision."""
        self.seedance_consent_accepted = bool(accepted)
        self.seedance_consent_event.set()

    def request_seedance_consent(self, segment_number, info):
        """Ask the GUI thread for a Seedance consent decision."""
        self.seedance_consent_accepted = False
        self.seedance_consent_event.clear()
        self.seedance_consent_required.emit(
            segment_number,
            info["policy_text"],
            info["roles_text"],
            info["docs_url"],
        )
        if not self.seedance_consent_event.wait(
            SEEDANCE_CONSENT_TIMEOUT_SECONDS
        ):
            raise RuntimeError(
                "Timed out waiting for Seedance consent confirmation."
            )

        return self.seedance_consent_accepted

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
                found = VeniceVideoWorker.find_first_value(
                    value,
                    names,
                )

                if found is not None:
                    return found

        elif isinstance(data, list):
            for item in data:
                found = VeniceVideoWorker.find_first_value(
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
                "media_url",
                "mediaUrl",
                "url",
            },
        )

        return str(value) if value else None

    @staticmethod
    def describe_video_timing(average_execution_time, execution_duration):
        """Return a compact elapsed/estimated processing-time summary."""
        values = []

        for label, value in (
            ("elapsed", execution_duration),
            ("p80", average_execution_time),
        ):
            try:
                milliseconds = float(value)
            except (TypeError, ValueError):
                continue

            if milliseconds <= 0:
                continue

            values.append(
                f"{label} {milliseconds / 1000:.0f}s"
            )

        return ", ".join(values)

    # ------------------------------------------------------------------
    # Story segmentation
    # ------------------------------------------------------------------

    def segmentation_system_prompt(self):
        """Return the system prompt used to split video segments."""
        return (
            "You are an expert video generation prompt engineer "
            "specializing in temporal consistency. Your task is to break "
            "the provided story into a JSON array of sequential video "
            f"segments. Use the '{self.video_model}' model for best "
            "results. Return ONLY valid JSON without Markdown, code "
            "blocks, or commentary.\n\n"
            "CRITICAL RULES FOR CONTINUITY:\n"
            "1. MEMORY: The video model does not have memory. You must "
            "manually bridge the gap between segments.\n"
            "2. VISUAL RECAP: For every segment index > 1, the "
            "'description' field MUST begin with a summary of the "
            "previous segment's visual state (positions, clothing, "
            "camera angle, props).\n"
            "3. ANCHORING: The 'first_frame_continuity' field must "
            "EXACTLY match the 'final_frame' of the previous segment to "
            "force the model to start from that moment.\n\n"
            "JSON Structure:\n"
            "{\n"
            "  'segment': 1,\n"
            f"  'mode': '{self.generation_mode}',\n"
            f"  'model': '{self.video_model}',\n"
            "  'duration': '<segment duration, for example 15s>',\n"
            "  'description': 'Visual recap of previous state + current "
            "action',\n"
            "  'first_frame_continuity': 'Exact match of previous "
            "final_frame',\n"
            "  'motion_continuation': 'Specific movements',\n"
            "  'final_frame': 'The specific ending image you want',\n"
            "  'transition_note': 'Internal note for context'\n"
            "}"
        )

    def segment_story(self):
        """Split the video concept into timed segment prompts."""
        payload = {
            "model": SEGMENT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": self.segmentation_system_prompt(),
                },
                {
                    "role": "user",
                    "content": self.segmentation_prompt(),
                },
            ],
            "temperature": 0.6,
            "max_tokens": min(
                12000,
                max(
                    2400,
                    len(self.segment_durations) * 700,
                ),
            ),
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
        segment_count = len(self.segment_durations)
        segment_lines = "\n".join(
            (
                f"- Segment {index}: {duration} seconds"
            )
            for index, duration in enumerate(
                self.segment_durations,
                start=1,
            )
        )
        example_items = []
        for index, duration in enumerate(
            self.segment_durations,
            start=1,
        ):
            continuity = (
                "Match the supplied starting reference image."
                if index == 1
                else "Match the exact final frame of the previous segment."
            )
            final_note = (
                "Description of the final shot."
                if index == segment_count
                else "Exact visual state of the final frame."
            )
            example_items.append(
                "\n".join(
                    [
                        "  {",
                        f'    "segment": {index},',
                        (
                            '    "description": '
                            f'"Complete prompt for this {duration}-second segment.",'
                        ),
                        (
                            '    "first_frame_continuity": '
                            f'"{continuity}",'
                        ),
                        (
                            '    "motion_continuation": '
                            '"How the action continues from the first frame.",'
                        ),
                        (
                            '    "final_frame": '
                            f'"{final_note}",'
                        ),
                        f'    "transition_note": "{final_note}"',
                        "  }",
                    ]
                )
            )
        example_json = ",\n".join(example_items)

        if self.is_image_to_video:
            mode_context = """
The first segment starts from a reference image supplied separately.
Each later segment starts from the extracted final frame of the preceding
segment. The supplied image for each segment must be treated as the exact
first frame, not as loose inspiration.
""".strip()
            mode_requirement = (
                "Each description must work as a continuation-focused "
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
Divide the following video concept into exactly {segment_count} sequential
{self.generation_mode} prompts covering exactly {self.total_seconds} seconds.

Segment durations:
{segment_lines}

{mode_context}

Requirements:

1. Return exactly {segment_count} segments.
2. Each segment must describe action for its assigned duration.
3. Segment 2 must continue directly from the end of segment 1.
4. Each later segment must continue directly from the end of the prior segment.
5. Do not skip or compress the requested timeline.
6. Do not restart or recap the story in later segments.
7. Repeat important character, clothing, environment, lighting, camera,
   lens, and visual-style details when needed for consistency.
8. Describe subject movement, environmental movement, camera movement,
   composition, lighting, pacing, and the desired ending frame.
9. For every segment after 1, the first visible moment must match the prior
   segment's final frame: same character identity, clothing, pose,
   expression, body position, object positions, camera angle, lens,
   composition, lighting, color grade, and environment.
10. Avoid phrases that imply a new establishing shot, a scene reset,
    a recap, or a time jump unless the original concept explicitly asks
    for one.
11. Do not request captions, subtitles, logos, watermarks, or visible text.
12. The final segment must end with a satisfying final shot.
13. {mode_requirement}
14. Do not mention API parameters, duration fields, or source-image URLs
    inside the descriptions.

Return only a JSON array in exactly this form:

[
{example_json}
]

Original video concept:

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

        if len(data) != len(self.segment_durations):
            raise ValueError(
                f"The segmentation response contained {len(data)} "
                f"segments instead of {len(self.segment_durations)}."
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

            first_frame_continuity = str(
                item.get("first_frame_continuity", "")
            ).strip()

            motion_continuation = str(
                item.get("motion_continuation", "")
            ).strip()

            final_frame = str(
                item.get("final_frame", "")
            ).strip()

            if not description:
                raise ValueError(
                    f"Segment {expected_number} has no description."
                )

            # Derive numbering from array order to avoid model mistakes.
            segments.append(
                Segment(
                    number=expected_number,
                    duration_seconds=self.segment_durations[
                        expected_number - 1
                    ],
                    description=description,
                    transition_note=transition_note,
                    first_frame_continuity=first_frame_continuity,
                    motion_continuation=motion_continuation,
                    final_frame=final_frame,
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
                "duration": segment.duration,
                "description": segment.description,
                "first_frame_continuity": segment.first_frame_continuity,
                "motion_continuation": segment.motion_continuation,
                "final_frame": segment.final_frame,
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
    def detect_image_mime_type(path):
        """Return an image MIME type from file bytes or filename."""
        path = Path(path)
        header = path.read_bytes()[:16]

        if header.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"

        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"

        if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
            return "image/webp"

        mime_type, _ = mimetypes.guess_type(
            str(path)
        )

        if mime_type and mime_type.startswith("image/"):
            return mime_type

        return None

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

        mime_type = VeniceVideoWorker.detect_image_mime_type(
            path
        )

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

        errors = []

        commands = [
            [
                ffmpeg_path,
                "-nostdin",
                "-y",
                "-sseof",
                "-0.50",
                "-i",
                str(video_path),
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(output_path),
            ],
            [
                ffmpeg_path,
                "-nostdin",
                "-y",
                "-i",
                str(video_path),
                "-map",
                "0:v:0",
                "-vf",
                "reverse",
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(output_path),
            ],
        ]

        for command in commands:
            output_path.unlink(missing_ok=True)

            try:
                result = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=FFMPEG_FRAME_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                errors.append(
                    "FFmpeg timed out while extracting a frame."
                )
                continue

            if (
                result.returncode == 0
                and output_path.exists()
                and output_path.stat().st_size > 0
            ):
                return

            errors.append(result.stderr[-3000:])

        raise RuntimeError(
            "FFmpeg could not extract a continuation frame.\n\n"
            + "\n\n".join(errors)[-6000:]
        )

    # ------------------------------------------------------------------
    # Video generation
    # ------------------------------------------------------------------

    def video_prompt_for_segment(self, segment):
        """Build the Venice prompt with explicit continuity guidance."""
        lines = []

        if self.is_image_to_video:
            lines.extend(
                [
                    (
                        "Continue from the supplied input image as the "
                        "exact first frame."
                    ),
                    (
                        "Preserve the same character identity, clothing, "
                        "pose, expression, body position, object positions, "
                        "camera angle, lens, composition, lighting, color "
                        "grade, and environment."
                    ),
                    (
                        "Do not restart the scene, introduce a new "
                        "establishing shot, or change the visual style."
                    ),
                ]
            )
            if self.reference_files:
                tags = ", ".join(
                    reference_image_tag(
                        self.video_model,
                        index,
                    )
                    for index in range(
                        1,
                        len(self.reference_files) + 1,
                    )
                )
                lines.append(
                    "Use the uploaded reference images as identity, "
                    "object, scene, and style anchors. They are available "
                    f"in order as {tags}; preserve their appearance when "
                    "the prompt refers to them."
                )
        elif segment.number > 1:
            lines.extend(
                [
                    (
                        "Continue directly from the prior segment with no "
                        "scene reset, recap, or time jump."
                    ),
                    (
                        "Preserve character identity, clothing, location, "
                        "lighting, camera style, and pacing from the "
                        "previous segment."
                    ),
                ]
            )

        if segment.first_frame_continuity:
            lines.append(
                "First-frame continuity: "
                + segment.first_frame_continuity
            )

        lines.append("Action: " + segment.description)

        if segment.motion_continuation:
            lines.append(
                "Motion continuation: "
                + segment.motion_continuation
            )

        final_frame = (
            segment.final_frame
            or segment.transition_note
        )
        if final_frame:
            lines.append(
                "End frame: " + final_frame
            )

        return "\n\n".join(lines)

    def reference_image_payload_key(self):
        """Return the model-specific flat reference-image field name."""
        model_id = self.video_model.lower()
        if "grok" in model_id:
            return "referenceImageUrls"

        return "reference_image_urls"

    def generate_video_segment(
        self,
        segment,
        output_path,
        source_data_url=None,
        reference_data_urls=None,
    ):
        """Queue, poll, download, and save one video segment."""
        video_prompt = self.video_prompt_for_segment(
            segment
        )
        payload = {
            "model": self.video_model,
            "prompt": video_prompt,
            "duration": segment.duration,
        }

        if self.video_resolution:
            payload["resolution"] = self.video_resolution

        if self.is_image_to_video:
            payload["image_url"] = source_data_url
        elif self.video_aspect_ratio:
            payload["aspect_ratio"] = self.video_aspect_ratio

        if reference_data_urls:
            payload[
                self.reference_image_payload_key()
            ] = reference_data_urls

        # Deliberately omitted: audio, because model support varies.

        self.progress.emit(
            segment.number - 1,
            (
                f"Queue payload for segment {segment.number}: "
                f"{self.describe_payload(payload)}"
            ),
        )

        try:
            queue_data = self.post_json(
                VIDEO_QUEUE_URL,
                payload,
            )
        except VeniceAPIError as exc:
            consent_info = self.seedance_consent_info(exc)
            if consent_info is None:
                raise

            self.progress.emit(
                segment.number - 1,
                (
                    f"Segment {segment.number}: Seedance face-media "
                    "consent is required before queuing."
                ),
            )

            if not self.request_seedance_consent(
                segment.number,
                consent_info,
            ):
                raise RuntimeError(
                    "Seedance consent was not confirmed; generation stopped."
                ) from exc

            payload.setdefault("consents", {})
            payload["consents"]["seedance"] = (
                self.seedance_consent_payload()
            )
            self.progress.emit(
                segment.number - 1,
                (
                    f"Segment {segment.number}: consent confirmed; "
                    "resubmitting queue request."
                ),
            )
            queue_data = self.post_json(
                VIDEO_QUEUE_URL,
                payload,
            )

        queue_id = self.get_queue_id(queue_data)
        # The retrieve endpoint requires the model ID used to queue the
        # generation. Some queue responses include nested or provider-facing
        # model values, so keep the submitted Venice model ID.
        queued_model = self.video_model
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
        retrieve_http_500_failures = 0
        completed_without_url = 0
        last_status_progress_time = 0

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
                    if response.status_code == 500:
                        retrieve_http_500_failures += 1
                    else:
                        retrieve_http_500_failures = 0

                    delay = min(
                        30,
                        VIDEO_POLL_INTERVAL_SECONDS
                        * transient_failures,
                    )

                    diagnostic_summary = (
                        f"{diagnostic[:500].strip()}; "
                        if diagnostic.strip()
                        else ""
                    )

                    if (
                        response.status_code == 500
                        and retrieve_http_500_failures
                        >= MAX_RETRIEVE_HTTP_500_RETRIES
                    ):
                        raise RuntimeError(
                            f"Video segment {segment.number} failed after "
                            f"{retrieve_http_500_failures} consecutive "
                            "HTTP 500 retrieval errors.\n\n"
                            f"{diagnostic or '<empty response>'}"
                        )

                    self.progress.emit(
                        segment.number - 1,
                        (
                            f"Segment {segment.number}: temporary "
                            f"HTTP {response.status_code}; "
                            f"{diagnostic_summary}"
                            f"retrying "
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
            retrieve_http_500_failures = 0

            try:
                data = response.json()
            except ValueError as exc:
                raise RuntimeError(
                    "The video retrieval response was neither JSON "
                    "nor video data.\n\n"
                    f"{response.text[:3000]}"
                ) from exc

            status = self.get_status(data)

            average_execution_time = self.find_first_value(
                data,
                {
                    "average_execution_time",
                    "averageExecutionTime",
                },
            )
            execution_duration = self.find_first_value(
                data,
                {
                    "execution_duration",
                    "executionDuration",
                },
            )

            timing = self.describe_video_timing(
                average_execution_time,
                execution_duration,
            )
            now = time.monotonic()

            if (
                status != last_status
                or now - last_status_progress_time >= 60
            ):
                message = (
                    f"Segment {segment.number}: "
                    f"status {status}, queue {queue_id}"
                )
                if timing:
                    message += f" ({timing})"

                self.progress.emit(
                    segment.number - 1,
                    message,
                )

                last_status = status
                last_status_progress_time = now

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
            VeniceVideoWorker.write_data_url(
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
                "-nostdin",
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

            try:
                copy_result = subprocess.run(
                    copy_command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=FFMPEG_STITCH_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "FFmpeg timed out while stream-copy stitching "
                    "the generated segments."
                ) from exc

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
                "-nostdin",
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

            try:
                reencode_result = subprocess.run(
                    reencode_command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=FFMPEG_STITCH_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "FFmpeg timed out while re-encoding and stitching "
                    "the generated segments."
                ) from exc

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
            "Describe the complete video..."
        )


class VideoWindow(QMainWindow):
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
            "Venice AI Video"
        )
        self.resize(980, 720)

        self.source_file_edit = QLineEdit()
        self.source_file_edit.setPlaceholderText(
            "Required for image-to-video"
        )
        self.source_file_edit.setText(
            self.load_starting_image_path()
        )
        self.source_file_edit.textChanged.connect(
            self.save_starting_image_path
        )

        self.source_file_button = QPushButton(
            "Browse..."
        )
        self.source_file_button.clicked.connect(
            self.choose_source_file
        )

        self.reference_files_edit = QLineEdit()
        self.reference_files_edit.setReadOnly(True)
        self.reference_files_edit.setPlaceholderText(
            "Optional for reference-to-video models"
        )
        self.reference_files = []

        self.reference_files_button = QPushButton(
            "Browse..."
        )
        self.reference_files_button.clicked.connect(
            self.choose_reference_files
        )

        self.clear_reference_files_button = QPushButton(
            "Clear"
        )
        self.clear_reference_files_button.clicked.connect(
            self.clear_reference_files
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

        self.output_seconds_spin = QSpinBox()
        self.output_seconds_spin.setRange(1, 3600)
        self.output_seconds_spin.setSingleStep(5)
        self.output_seconds_spin.setValue(60)
        self.output_seconds_spin.setSuffix(" seconds")

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
        self.video_model_combo.currentIndexChanged.connect(
            self.update_source_controls
        )
        self.video_model_combo.currentIndexChanged.connect(
            lambda _index: self.update_reference_files_text()
        )

        self.refresh_models_button = QPushButton(
            "Refresh Models"
        )
        self.refresh_models_button.clicked.connect(
            self.load_video_models
        )

        self.prompt_input = PromptEdit()

        self.load_prompt_button = QPushButton(
            "Load Prompt"
        )
        self.load_prompt_button.clicked.connect(
            self.load_prompt_text
        )

        self.save_prompt_button = QPushButton(
            "Save Prompt"
        )
        self.save_prompt_button.clicked.connect(
            self.save_prompt_text
        )

        self.log_display = QTextBrowser()
        self.log_display.setReadOnly(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(
            0,
            len(
                VeniceVideoWorker.plan_segment_durations(
                    self.output_seconds_spin.value()
                )
            ),
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

        reference_row = QHBoxLayout()
        reference_row.addWidget(
            self.reference_files_edit,
            1,
        )
        reference_row.addWidget(
            self.reference_files_button
        )
        reference_row.addWidget(
            self.clear_reference_files_button
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
            "Reference images",
            reference_row,
        )
        form.addRow(
            "Output MP4",
            output_row,
        )
        form.addRow(
            "Output length",
            self.output_seconds_spin,
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

        prompt_button_row = QHBoxLayout()
        prompt_button_row.addStretch(1)
        prompt_button_row.addWidget(
            self.load_prompt_button
        )
        prompt_button_row.addWidget(
            self.save_prompt_button
        )

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(
            QLabel("Video prompt")
        )
        layout.addLayout(
            prompt_button_row
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
        current_path = Path(
            self.source_file_edit.text().strip()
        )
        start_directory = (
            str(current_path.parent)
            if current_path.parent.exists()
            else ""
        )

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Choose starting image",
            start_directory,
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
            self.save_starting_image_path()

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

    def choose_reference_files(self):
        """Choose ordered reference images for R2V models."""
        option = self.selected_video_option()
        max_images = option.max_reference_images
        if max_images <= 0:
            QMessageBox.information(
                self,
                "Reference images unavailable",
                (
                    "The selected video model does not advertise "
                    "flat reference-image support."
                ),
            )
            return

        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            f"Choose up to {max_images} reference images",
            "",
            (
                "Image files "
                "(*.png *.jpg *.jpeg *.webp);;"
                "All files (*)"
            ),
        )

        if not filenames:
            return

        if len(filenames) > max_images:
            QMessageBox.warning(
                self,
                "Too many reference images",
                (
                    f"The selected model supports up to {max_images} "
                    "reference image(s). Extra images were ignored."
                ),
            )
            filenames = filenames[:max_images]

        self.reference_files = [
            str(Path(filename))
            for filename in filenames
        ]
        self.update_reference_files_text()

    def clear_reference_files(self):
        """Clear selected reference images."""
        self.reference_files = []
        self.update_reference_files_text()

    def update_reference_files_text(self):
        """Update the reference image summary field."""
        if not self.reference_files:
            self.reference_files_edit.clear()
            self.update_source_controls()
            return

        option = self.selected_video_option()
        names = [
            (
                f"{reference_image_tag(option.model_id, index)}="
                f"{Path(path).name}"
            )
            for index, path in enumerate(
                self.reference_files,
                start=1,
            )
        ]
        self.reference_files_edit.setText(
            "; ".join(names)
        )
        self.update_source_controls()

    def load_prompt_text(self):
        """Load prompt text from a plain-text file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load prompt",
            self.load_prompt_directory(),
            "Text files (*.txt);;Markdown files (*.md);;All files (*)",
        )

        if not filename:
            return

        path = Path(filename)

        try:
            text = path.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Load prompt failed",
                f"Could not read prompt file:\n{path}\n\n{exc}",
            )
            return

        self.prompt_input.setPlainText(text)
        self.save_prompt_directory(path.parent)
        self.status_label.setText(
            f"Loaded prompt: {path.name}"
        )

    def save_prompt_text(self):
        """Save the current prompt text to a plain-text file."""
        prompt = self.prompt_input.toPlainText()

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save prompt",
            self.load_prompt_directory(),
            "Text files (*.txt);;Markdown files (*.md);;All files (*)",
        )

        if not filename:
            return

        path = Path(filename)

        if not path.suffix:
            path = path.with_suffix(".txt")

        try:
            path.write_text(
                prompt,
                encoding="utf-8",
            )
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Save prompt failed",
                f"Could not write prompt file:\n{path}\n\n{exc}",
            )
            return

        self.save_prompt_directory(path.parent)
        self.status_label.setText(
            f"Saved prompt: {path.name}"
        )

    def load_prompt_directory(self):
        """Return the persisted prompt file directory."""
        directory = str(
            self.settings.value(
                PROMPT_DIRECTORY_KEY,
                "",
            )
            or ""
        )

        if directory and Path(directory).is_dir():
            return directory

        return ""

    def save_prompt_directory(self, directory):
        """Persist the prompt file directory."""
        self.settings.setValue(
            PROMPT_DIRECTORY_KEY,
            str(directory),
        )
        self.settings.sync()

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

    def load_starting_image_path(self):
        """Return the persisted starting image path."""
        return str(
            self.settings.value(
                STARTING_IMAGE_PATH_KEY,
                "",
            )
            or ""
        )

    def save_starting_image_path(self, *_args):
        """Persist the starting image path."""
        self.settings.setValue(
            STARTING_IMAGE_PATH_KEY,
            self.source_file_edit.text().strip(),
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
        if self.selected_generation_mode() == TEXT_TO_VIDEO_MODE:
            self.source_file_edit.clear()
            self.clear_reference_files()
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

        option = self.selected_video_option()
        reference_enabled = (
            is_image_mode
            and option.max_reference_images > 0
            and not self.generating
        )
        self.reference_files_edit.setEnabled(
            reference_enabled
        )
        self.reference_files_button.setEnabled(
            reference_enabled
        )
        self.clear_reference_files_button.setEnabled(
            reference_enabled
            and bool(self.reference_files)
        )

        if option.max_reference_images > 0:
            first_tag = reference_image_tag(
                option.model_id,
                1,
            )
            second_tag = reference_image_tag(
                option.model_id,
                2,
            )
            self.reference_files_edit.setPlaceholderText(
                f"Optional; up to {option.max_reference_images} images "
                f"as {first_tag}, {second_tag}, ..."
            )
        else:
            self.reference_files_edit.setPlaceholderText(
                "Only available for reference-to-video models"
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
                option,
            )

        index = self.find_video_model_index(
            current_model
        )

        if index < 0:
            index = self.find_video_model_index(
                self.default_video_model()
            )

        if index >= 0:
            self.video_model_combo.setCurrentIndex(index)

        self.status_label.setText(message)
        self.append_log(message)
        self.update_source_controls()

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

    def find_video_model_index(self, model_id):
        """Find a model option by model ID."""
        for index in range(self.video_model_combo.count()):
            option = self.video_model_combo.itemData(index)
            if isinstance(option, VideoModelOption):
                if option.model_id == model_id:
                    return index
            elif option == model_id:
                return index

        return -1

    def selected_video_option(self):
        """Return the selected model option with constraints."""
        option = self.video_model_combo.currentData()
        if isinstance(option, VideoModelOption):
            return option

        model_id = (
            str(option).strip()
            if option
            else self.video_model_combo.currentText().strip()
        )

        if not model_id:
            model_id = self.default_video_model()

        model_id_lower = model_id.lower()
        return VideoModelOption(
            model_id=model_id,
            label=model_id,
            preferred=is_preferred_video_model_id(model_id_lower),
            durations=fallback_video_durations(model_id),
            resolutions=fallback_video_resolutions(model_id),
            aspect_ratios=fallback_video_aspect_ratios(model_id),
            max_reference_images=fallback_reference_image_limit(model_id),
        )

    def selected_video_model(self):
        """Return the selected Venice video model ID."""
        return self.selected_video_option().model_id

    @staticmethod
    def choose_preferred_value(values, preferred):
        """Return a preferred supported value or the first supported value."""
        values = tuple(value for value in values if value)
        if not values:
            return None

        if preferred in values:
            return preferred

        return values[0]

    def generate(self):
        """Validate input and start generation."""
        if self.thread is not None:
            return

        prompt = (
            self.prompt_input
            .toPlainText()
            .strip()
        )

        generation_mode = self.selected_generation_mode()
        video_option = self.selected_video_option()
        video_model = video_option.model_id
        source_file = ""
        reference_files = []
        if generation_mode == IMAGE_TO_VIDEO_MODE:
            source_file = (
                self.source_file_edit
                .text()
                .strip()
            )
            reference_files = list(self.reference_files)
        total_seconds = self.output_seconds_spin.value()
        try:
            segment_durations = (
                VeniceVideoWorker.plan_segment_durations(
                    total_seconds,
                    video_option.durations,
                )
            )
        except ValueError as exc:
            QMessageBox.warning(
                self,
                "Unsupported video length",
                str(exc),
            )
            return

        video_resolution = self.choose_preferred_value(
            video_option.resolutions,
            VIDEO_RESOLUTION,
        )
        video_aspect_ratio = self.choose_preferred_value(
            video_option.aspect_ratios,
            VIDEO_ASPECT_RATIO,
        )

        if (
            generation_mode == TEXT_TO_VIDEO_MODE
            and video_option.aspect_ratios
            and not video_aspect_ratio
        ):
            QMessageBox.warning(
                self,
                "Unsupported aspect ratio",
                (
                    "The selected text-to-video model does not support "
                    f"{VIDEO_ASPECT_RATIO}."
                ),
            )
            return

        if (
            video_option.resolutions
            and not video_resolution
        ):
            QMessageBox.warning(
                self,
                "Unsupported resolution",
                (
                    "The selected video model does not support "
                    f"{VIDEO_RESOLUTION} or a usable fallback."
                ),
            )
            return

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

        if (
            generation_mode == IMAGE_TO_VIDEO_MODE
            and source_file
            and not Path(source_file).is_file()
        ):
            QMessageBox.warning(
                self,
                "Invalid starting image",
                f"The selected file does not exist:\n{source_file}",
            )
            return

        if reference_files:
            if generation_mode != IMAGE_TO_VIDEO_MODE:
                QMessageBox.warning(
                    self,
                    "Reference images unavailable",
                    (
                        "Reference images can only be used in "
                        "image-to-video mode."
                    ),
                )
                return

            if video_option.max_reference_images <= 0:
                QMessageBox.warning(
                    self,
                    "Reference images unavailable",
                    (
                        "The selected video model does not advertise "
                        "flat reference-image support."
                    ),
                )
                return

            if len(reference_files) > video_option.max_reference_images:
                QMessageBox.warning(
                    self,
                    "Too many reference images",
                    (
                        f"The selected model supports up to "
                        f"{video_option.max_reference_images} "
                        "reference image(s)."
                    ),
                )
                return

            missing_reference_files = [
                path
                for path in reference_files
                if not Path(path).is_file()
            ]
            if missing_reference_files:
                QMessageBox.warning(
                    self,
                    "Invalid reference image",
                    (
                        "The selected reference file does not exist:\n"
                        + missing_reference_files[0]
                    ),
                )
                return

        if not prompt:
            QMessageBox.warning(
                self,
                "Missing prompt",
                "Enter a video prompt.",
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
        self.progress_bar.setRange(
            0,
            len(segment_durations),
        )
        self.progress_bar.setValue(0)
        self.set_generating(True)

        self.append_log(
            f"Text model: {SEGMENT_MODEL}"
        )
        self.append_log(
            f"Video model: {video_model}"
        )
        self.append_log(
            (
                f"Plan: {total_seconds} seconds across "
                f"{len(segment_durations)} segment(s): "
                + ", ".join(
                    f"{duration}s"
                    for duration in segment_durations
                )
            )
        )
        self.append_log(
            "Resolution: " + (video_resolution or "model default")
        )
        if (
            video_resolution
            and video_resolution != VIDEO_RESOLUTION
        ):
            self.append_log(
                (
                    f"Selected model does not support {VIDEO_RESOLUTION}; "
                    f"using {video_resolution}."
                )
            )
        self.append_log(
            f"Mode: {generation_mode}"
        )
        if generation_mode == TEXT_TO_VIDEO_MODE:
            self.append_log(
                "Aspect ratio: "
                + (video_aspect_ratio or "model default")
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
            if reference_files:
                self.append_log(
                    "Reference images: "
                    + "; ".join(
                        (
                            f"{reference_image_tag(video_model, index)}="
                            f"{Path(path).name}"
                        )
                        for index, path in enumerate(
                            reference_files,
                            start=1,
                        )
                    )
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

        self.worker = VeniceVideoWorker(
            api_key=self.api_key,
            prompt=prompt,
            source_file=source_file,
            output_file=str(output_path),
            video_model=video_model,
            generation_mode=generation_mode,
            total_seconds=total_seconds,
            segment_durations=segment_durations,
            video_resolution=video_resolution,
            video_aspect_ratio=video_aspect_ratio,
            reference_files=reference_files,
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

        self.worker.seedance_consent_required.connect(
            self.handle_seedance_consent_required
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

    def handle_seedance_consent_required(
        self,
        segment_number,
        policy_text,
        roles_text,
        docs_url,
    ):
        """Show Seedance face-media consent and notify the worker."""
        checkbox = QCheckBox(
            (
                "I confirm these terms, legal likeness rights, "
                "and screening acknowledgement."
            )
        )
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Warning)
        message.setWindowTitle("Seedance consent required")
        message.setText(
            f"Seedance detected face-bearing media in segment {segment_number}."
        )
        details = policy_text
        if roles_text:
            details += f"\n\nDetected media roles: {roles_text}"
        if docs_url:
            details += f"\n\nDocs: {docs_url}"
        message.setInformativeText(details)
        message.setCheckBox(checkbox)
        message.setStandardButtons(
            QMessageBox.Ok | QMessageBox.Cancel
        )
        message.setDefaultButton(QMessageBox.Cancel)

        accepted = (
            message.exec() == QMessageBox.Ok
            and checkbox.isChecked()
        )

        if not accepted:
            self.append_log(
                "Seedance consent was not confirmed."
            )

        if self.worker is not None:
            self.worker.set_seedance_consent_response(
                accepted
            )

    def generation_finished(self, final_path):
        """Handle successful generation."""
        self.progress_bar.setValue(
            self.progress_bar.maximum()
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

        self.output_seconds_spin.setDisabled(
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

        self.load_prompt_button.setDisabled(
            generating
        )

        self.save_prompt_button.setDisabled(
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

    window = VideoWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
