#!/usr/bin/env python3
"""PySide6 GUI for Venice AI image generation models."""

import base64
import json
import struct
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests
from PySide6.QtCore import QObject, QSettings, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from modules.tts import get_venice_api_key


API_BASE_URL = "https://api.venice.ai/api/v1"
IMAGE_GENERATE_URL = f"{API_BASE_URL}/image/generate"
IMAGE_STYLES_URL = f"{API_BASE_URL}/image/styles"
MODELS_URL = f"{API_BASE_URL}/models"

DEFAULT_MODEL = "venice-sd35"
HTTP_TIMEOUT = (30, 240)
SETTINGS_ORG = "VeniceAI"
SETTINGS_APP = "ImageGeneration"
OUTPUT_DIR_SETTING = "paths/output_dir"
PROMPT_DIR_SETTING = "paths/prompt_dir"
GENERATION_LOG_FILENAME = "venice-image-gen.log"

SIZING_WIDTH_HEIGHT = "width_height"
SIZING_ASPECT_RATIO = "aspect_ratio"
SIZING_RESOLUTION_ASPECT = "resolution_aspect"

FORMAT_EXTENSIONS = {
    "jpeg": "jpg",
    "png": "png",
    "webp": "webp",
}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass
class ImageModelOption:
    """One selectable Venice image model."""

    model_id: str
    label: str
    pricing: object = None


class VeniceAPIError(RuntimeError):
    """An HTTP error returned by the Venice API."""

    def __init__(self, status_code, url, body):
        super().__init__(
            f"Venice API returned HTTP {status_code} for:\n"
            f"{url}\n\n"
            f"{body or '<empty response>'}"
        )


def response_diagnostic(response):
    """Return readable response details for API errors."""
    content_type = response.headers.get("content-type", "")
    text = response.text.strip()
    if "application/json" not in content_type.lower():
        return text

    try:
        return json.dumps(response.json(), indent=2, sort_keys=True)
    except ValueError:
        return text


def normalize_model_list(data):
    """Return a list from Venice's model-list response shapes."""
    models = data.get("data", data) if isinstance(data, dict) else data
    if isinstance(models, dict):
        return list(models.values())
    if isinstance(models, list):
        return models
    return []


def model_identifier(model):
    """Return the model ID from a Venice model metadata object."""
    if not isinstance(model, dict):
        return ""
    value = model.get("id") or model.get("model")
    return str(value).strip() if value else ""


def model_display_name(model):
    """Return the human-readable model name when available."""
    if not isinstance(model, dict):
        return ""

    model_spec = model.get("model_spec")
    if isinstance(model_spec, dict) and model_spec.get("name"):
        return str(model_spec["name"]).strip()

    value = model.get("name")
    return str(value).strip() if value else ""


def model_pricing(model):
    """Return image generation pricing metadata from a model object."""
    if not isinstance(model, dict):
        return None

    model_spec = model.get("model_spec")
    if isinstance(model_spec, dict) and model_spec.get("pricing"):
        return model_spec["pricing"]

    return model.get("pricing")


def extract_image_models(data):
    """Extract image-generation models from the models response."""
    options = []
    for model in normalize_model_list(data):
        if not isinstance(model, dict):
            continue
        if str(model.get("type", "")).lower() not in {"", "image"}:
            continue

        model_id = model_identifier(model)
        if not model_id:
            continue

        display_name = model_display_name(model)
        label = model_id if not display_name else f"{display_name} ({model_id})"
        options.append(ImageModelOption(model_id, label, model_pricing(model)))

    options.sort(key=lambda option: option.label.lower())
    return options


def normalize_styles(data):
    """Return image style names from Venice's style-list response."""
    styles = data.get("data", data) if isinstance(data, dict) else data
    if not isinstance(styles, list):
        return []
    return sorted(str(style) for style in styles if str(style).strip())


def decode_image_data(value):
    """Decode a base64 image string, accepting plain base64 or data URLs."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Image response contained an empty image value.")

    image_data = value.strip()
    if image_data.startswith("data:"):
        _, separator, image_data = image_data.partition(",")
        if not separator:
            raise ValueError("Image response contained an invalid data URL.")

    return base64.b64decode(image_data, validate=True)


def png_chunk(chunk_type, payload):
    """Return one encoded PNG chunk."""
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xffffffff)
    )


def add_png_description(image_bytes, description):
    """Return PNG bytes with a Description iTXt chunk inserted before IEND."""
    if not description or not image_bytes.startswith(PNG_SIGNATURE):
        return image_bytes

    offset = len(PNG_SIGNATURE)
    while offset + 12 <= len(image_bytes):
        chunk_length = struct.unpack(">I", image_bytes[offset:offset + 4])[0]
        chunk_type = image_bytes[offset + 4:offset + 8]
        chunk_end = offset + 12 + chunk_length
        if chunk_end > len(image_bytes):
            return image_bytes

        if chunk_type == b"IEND":
            payload = (
                b"Description\x00"
                b"\x00"
                b"\x00"
                b"\x00"
                b"\x00"
                + description.encode("utf-8")
            )
            return image_bytes[:offset] + png_chunk(b"iTXt", payload) + image_bytes[offset:]

        offset = chunk_end

    return image_bytes


def usd_amount(value):
    """Return a USD float from Venice pricing metadata."""
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value.strip().lstrip("$"))
        except ValueError:
            return None

    if isinstance(value, dict):
        for key in ("usd", "USD", "amount", "price", "cost"):
            amount = usd_amount(value.get(key))
            if amount is not None:
                return amount

    return None


def image_unit_price(pricing, payload):
    """Return a per-image USD price and label for a generation payload."""
    if not isinstance(pricing, dict):
        return None, "No pricing metadata loaded for this model."

    resolution = str(payload.get("resolution", "")).strip()
    quality = str(payload.get("quality") or "high").strip().lower()

    quality_prices = pricing.get("quality")
    if isinstance(quality_prices, dict):
        if not resolution:
            return None, "Choose a resolution tier to estimate this model."

        resolution_prices = quality_prices.get(resolution)
        if isinstance(resolution_prices, dict):
            amount = usd_amount(resolution_prices.get(quality))
            if amount is not None:
                return amount, f"{resolution} {quality}"

            available = ", ".join(sorted(str(key) for key in resolution_prices))
            return None, f"Quality {quality!r} is not priced. Available: {available}."

        return None, f"Resolution {resolution!r} is not priced for this model."

    resolution_prices = pricing.get("resolutions")
    if isinstance(resolution_prices, dict):
        if not resolution:
            return None, "Choose a resolution tier to estimate this model."

        amount = usd_amount(resolution_prices.get(resolution))
        if amount is not None:
            return amount, resolution

        return None, f"Resolution {resolution!r} is not priced for this model."

    amount = usd_amount(pricing.get("generation"))
    if amount is not None:
        return amount, "per image"

    amount = usd_amount(pricing)
    if amount is not None:
        return amount, "per image"

    return None, "No generation price found in model metadata."


def generation_cost_summary(pricing, payload):
    """Return estimated cost details for an image generation payload."""
    unit_price, basis = image_unit_price(pricing, payload)
    variants = max(1, int(payload.get("variants") or 1))
    if unit_price is None:
        return {
            "available": False,
            "basis": basis,
            "variants": variants,
            "unit_price": None,
            "total": None,
            "display": f"unavailable ({basis})",
        }

    total = unit_price * variants
    if variants == 1:
        display = f"${total:.4f} ({basis})"
    else:
        display = f"${total:.4f} ({variants} x ${unit_price:.4f}, {basis})"

    return {
        "available": True,
        "basis": basis,
        "variants": variants,
        "unit_price": unit_price,
        "total": total,
        "display": display,
    }


def append_generation_log(output_dir, entry):
    """Append a completed generation entry to the output directory log."""
    log_path = Path(output_dir) / GENERATION_LOG_FILENAME
    lines = [
        "Venice image generation",
        f"Timestamp: {entry['timestamp']}",
        f"Model: {entry['model']}",
        f"Estimated cost: {entry['estimated_cost']['display']}",
    ]

    if entry.get("enhanced_prompt"):
        lines.append(f"Enhanced prompt: {entry['enhanced_prompt']}")

    lines.extend(
        (
            "Parameters:",
            json.dumps(entry["parameters"], indent=2, sort_keys=True),
            "Generated files:",
        )
    )
    lines.extend(f"- {path}" for path in entry["generated_files"])
    lines.append("")
    lines.append("")

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines))

    return log_path


class MetadataWorker(QObject):
    """Load model and style metadata without blocking the GUI."""

    finished = Signal(object, object)
    failed = Signal(str)

    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key

    def run(self):
        """Fetch image models and style presets."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            }

            models_response = requests.get(
                MODELS_URL,
                headers=headers,
                params={"type": "image"},
                timeout=HTTP_TIMEOUT,
            )
            if not models_response.ok:
                raise VeniceAPIError(
                    models_response.status_code,
                    MODELS_URL,
                    response_diagnostic(models_response),
                )
            models = extract_image_models(models_response.json())

            styles_response = requests.get(
                IMAGE_STYLES_URL,
                headers=headers,
                timeout=HTTP_TIMEOUT,
            )
            if not styles_response.ok:
                raise VeniceAPIError(
                    styles_response.status_code,
                    IMAGE_STYLES_URL,
                    response_diagnostic(styles_response),
                )
            styles = normalize_styles(styles_response.json())

            self.finished.emit(models, styles)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ImageGenerationWorker(QObject):
    """Generate images without blocking the GUI."""

    finished = Signal(object, str)
    failed = Signal(str)

    def __init__(self, api_key, payload, output_dir, filename_prefix, output_format, description):
        super().__init__()
        self.api_key = api_key
        self.payload = payload
        self.output_dir = Path(output_dir)
        self.filename_prefix = filename_prefix
        self.output_format = output_format
        self.description = description

    def run(self):
        """Call Venice image generation and save returned image variants."""
        try:
            response = requests.post(
                IMAGE_GENERATE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json=self.payload,
                timeout=HTTP_TIMEOUT,
            )
            if not response.ok:
                raise VeniceAPIError(
                    response.status_code,
                    IMAGE_GENERATE_URL,
                    response_diagnostic(response),
                )

            data = response.json()
            images = data.get("images")
            if not isinstance(images, list) or not images:
                raise RuntimeError("The image generation response did not include any images.")

            self.output_dir.mkdir(parents=True, exist_ok=True)
            extension = FORMAT_EXTENSIONS.get(self.output_format, self.output_format)
            paths = []
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            for index, image in enumerate(images, start=1):
                image_bytes = decode_image_data(image)
                suffix = f"-{index}" if len(images) > 1 else ""
                output_path = self.output_dir / f"{self.filename_prefix}-{timestamp}{suffix}.{extension}"
                if extension == "png" and self.description:
                    image_bytes = add_png_description(image_bytes, self.description)
                output_path.write_bytes(image_bytes)
                paths.append(str(output_path))

            enhanced_prompt = response.headers.get("x-venice-enhanced-prompt", "")
            self.finished.emit(paths, enhanced_prompt)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ImageGenerationWindow(QMainWindow):
    """Main window for Venice AI image generation."""

    def __init__(self):
        super().__init__()
        self.metadata_thread = None
        self.metadata_worker = None
        self.generation_thread = None
        self.generation_worker = None
        self.active_generation_log = None
        self.generated_paths = []
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)

        self.setWindowTitle("Venice AI Image Generator")
        self.resize(1280, 720)
        self.build_menu()

        self.api_key_edit = QLineEdit(get_venice_api_key() or "")
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("VENICE_API_KEY")

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItem(DEFAULT_MODEL, ImageModelOption(DEFAULT_MODEL, DEFAULT_MODEL))
        self.model_combo.currentIndexChanged.connect(self.update_cost_estimate)
        self.model_combo.editTextChanged.connect(self.update_cost_estimate)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_metadata)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setAcceptRichText(False)
        self.prompt_edit.setMinimumHeight(150)
        self.prompt_edit.setPlaceholderText("Describe the image to generate.")

        self.negative_prompt_edit = QTextEdit()
        self.negative_prompt_edit.setAcceptRichText(False)
        self.negative_prompt_edit.setMaximumHeight(70)
        self.negative_prompt_edit.setPlaceholderText("Optional negative prompt.")

        self.description_check = QCheckBox("Add PNG description metadata")
        self.description_check.toggled.connect(self.update_description_controls)

        self.description_edit = QTextEdit()
        self.description_edit.setAcceptRichText(False)
        self.description_edit.setMaximumHeight(70)
        self.description_edit.setPlaceholderText("Description saved as PNG metadata when the output format is PNG.")

        self.style_combo = QComboBox()
        self.style_combo.addItem("None", "")

        self.sizing_mode_combo = QComboBox()
        self.sizing_mode_combo.addItem("Size", SIZING_WIDTH_HEIGHT)
        self.sizing_mode_combo.addItem("Aspect ratio", SIZING_ASPECT_RATIO)
        self.sizing_mode_combo.addItem("Res + aspect", SIZING_RESOLUTION_ASPECT)
        self.sizing_mode_combo.currentIndexChanged.connect(self.update_sizing_controls)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 1280)
        self.width_spin.setSingleStep(64)
        self.width_spin.setValue(1024)
        self.width_spin.valueChanged.connect(self.update_cost_estimate)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 1280)
        self.height_spin.setSingleStep(64)
        self.height_spin.setValue(1024)
        self.height_spin.valueChanged.connect(self.update_cost_estimate)

        self.aspect_ratio_combo = QComboBox()
        self.aspect_ratio_combo.addItems(("1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"))
        self.aspect_ratio_combo.currentIndexChanged.connect(self.update_cost_estimate)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(("1K", "2K", "4K"))
        self.resolution_combo.currentIndexChanged.connect(self.update_cost_estimate)

        self.format_combo = QComboBox()
        self.format_combo.addItems(("webp", "png", "jpeg"))

        self.quality_combo = QComboBox()
        self.quality_combo.addItem("Default", "")
        self.quality_combo.addItems(("low", "medium", "high"))
        self.quality_combo.currentIndexChanged.connect(self.update_cost_estimate)

        self.variants_spin = QSpinBox()
        self.variants_spin.setRange(1, 4)
        self.variants_spin.setValue(1)
        self.variants_spin.valueChanged.connect(self.update_cost_estimate)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 150)
        self.steps_spin.setValue(8)

        self.cfg_spin = QSpinBox()
        self.cfg_spin.setRange(1, 20)
        self.cfg_spin.setValue(7)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-999999999, 999999999)
        self.seed_spin.setValue(0)

        self.safe_mode_check = QCheckBox("Safe mode")
        self.safe_mode_check.setChecked(False)
        self.hide_watermark_check = QCheckBox("Hide watermark")
        self.enhance_prompt_check = QCheckBox("Enhance prompt")
        self.embed_exif_check = QCheckBox("Embed EXIF metadata")

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_edit.setText(self.settings.value(OUTPUT_DIR_SETTING, "", str))

        self.output_dir_button = QPushButton("Choose")
        self.output_dir_button.clicked.connect(self.choose_output_dir)

        self.filename_prefix_edit = QLineEdit("venice-image")

        self.generate_button = QPushButton("Generate Image")
        self.generate_button.clicked.connect(self.generate_image)

        self.open_output_button = QPushButton("Open Selected")
        self.open_output_button.clicked.connect(self.open_selected_image)
        self.open_output_button.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

        self.status_label = QLabel("Ready")
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.cost_label = QLabel("Estimated cost: unavailable until model metadata loads.")
        self.cost_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.preview_label = QLabel("Generated image preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(320, 240)

        self.output_list = QListWidget()
        self.output_list.currentRowChanged.connect(self.preview_selected_image)

        self.build_layout()
        self.apply_style()
        self.update_sizing_controls()
        self.update_description_controls()
        self.update_cost_estimate()
        # Automatically refresh model/style metadata on startup when an API key is available
        try:
            if self.api_key_edit.text().strip():
                self.refresh_metadata()
        except Exception:
            # Do not let metadata refresh failures prevent the UI from starting
            pass

    def build_menu(self):
        """Create prompt file actions."""
        prompt_menu = self.menuBar().addMenu("Prompt")

        save_action = QAction("Save Prompt...", self)
        save_action.triggered.connect(self.save_prompt)
        prompt_menu.addAction(save_action)

        save_txt_action = QAction("Save Prompt as TXT...", self)
        save_txt_action.triggered.connect(self.save_prompt_txt)
        prompt_menu.addAction(save_txt_action)

        load_action = QAction("Load Prompt...", self)
        load_action.triggered.connect(self.load_prompt)
        prompt_menu.addAction(load_action)

    def build_layout(self):
        """Assemble the form and preview area."""
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.refresh_button)

        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit, 1)
        output_row.addWidget(self.output_dir_button)

        api_form = QFormLayout()
        api_form.addRow("API key", self.api_key_edit)
        api_form.addRow("Model", self.wrap_layout(model_row))

        api_group = QGroupBox("Connection")
        api_group.setLayout(api_form)

        prompt_form = QFormLayout()
        prompt_form.addRow("Prompt", self.prompt_edit)
        prompt_form.addRow("Negative", self.negative_prompt_edit)
        prompt_form.addRow("Description", self.description_edit)

        prompt_group = QGroupBox("Prompt")
        prompt_group.setLayout(prompt_form)

        params_grid = QGridLayout()
        params_grid.addWidget(QLabel("Style"), 0, 0)
        params_grid.addWidget(self.style_combo, 0, 1)
        params_grid.addWidget(QLabel("Sizing"), 0, 2)
        params_grid.addWidget(self.sizing_mode_combo, 0, 3)
        params_grid.addWidget(QLabel("Width"), 0, 4)
        params_grid.addWidget(self.width_spin, 0, 5)
        params_grid.addWidget(QLabel("Height"), 1, 0)
        params_grid.addWidget(self.height_spin, 1, 1)
        params_grid.addWidget(QLabel("Aspect"), 1, 2)
        params_grid.addWidget(self.aspect_ratio_combo, 1, 3)
        params_grid.addWidget(QLabel("Resolution"), 1, 4)
        params_grid.addWidget(self.resolution_combo, 1, 5)
        params_grid.addWidget(QLabel("Format"), 2, 0)
        params_grid.addWidget(self.format_combo, 2, 1)
        params_grid.addWidget(QLabel("Quality"), 2, 2)
        params_grid.addWidget(self.quality_combo, 2, 3)
        params_grid.addWidget(QLabel("Variants"), 2, 4)
        params_grid.addWidget(self.variants_spin, 2, 5)
        params_grid.addWidget(QLabel("Steps"), 3, 0)
        params_grid.addWidget(self.steps_spin, 3, 1)
        params_grid.addWidget(QLabel("CFG"), 3, 2)
        params_grid.addWidget(self.cfg_spin, 3, 3)
        params_grid.addWidget(QLabel("Seed"), 3, 4)
        params_grid.addWidget(self.seed_spin, 3, 5)
        params_grid.addWidget(QLabel("Cost"), 4, 0)
        params_grid.addWidget(self.cost_label, 4, 1, 1, 5)
        params_grid.setColumnStretch(1, 1)
        params_grid.setColumnStretch(3, 1)
        params_grid.setColumnStretch(5, 1)

        params_group = QGroupBox("Image Parameters")
        params_group.setLayout(params_grid)

        output_form = QFormLayout()
        output_form.addRow("Directory", self.wrap_layout(output_row))
        output_form.addRow("Filename prefix", self.filename_prefix_edit)

        output_group = QGroupBox("Output")
        output_group.setLayout(output_form)

        toggles = QHBoxLayout()
        toggles.addWidget(self.description_check)
        toggles.addWidget(self.safe_mode_check)
        toggles.addWidget(self.hide_watermark_check)
        toggles.addWidget(self.enhance_prompt_check)
        toggles.addWidget(self.embed_exif_check)
        toggles.addStretch(1)

        controls = QHBoxLayout()
        controls.addWidget(self.generate_button)
        controls.addWidget(self.open_output_button)
        controls.addStretch(1)

        top_controls = QHBoxLayout()
        top_controls.addWidget(api_group, 1)
        top_controls.addWidget(output_group, 1)

        left_layout = QVBoxLayout()
        left_layout.addLayout(top_controls)
        left_layout.addWidget(params_group)
        left_layout.addWidget(prompt_group, 1)
        left_layout.addLayout(toggles)
        left_layout.addWidget(self.progress)
        left_layout.addWidget(self.status_label)
        left_layout.addLayout(controls)

        left = QWidget()
        left.setLayout(left_layout)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.NoFrame)
        left_scroll.setWidget(left)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.preview_label, 1)
        right_layout.addWidget(QLabel("Generated files"))
        right_layout.addWidget(self.output_list)

        right = QWidget()
        right.setLayout(right_layout)
        right.setMaximumWidth(560)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes((980, 360))

        layout = QVBoxLayout()
        layout.addWidget(splitter)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    @staticmethod
    def wrap_layout(layout):
        """Return a widget wrapper for a row layout."""
        widget = QWidget()
        widget.setLayout(layout)
        return widget

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
            QTextEdit,
            QComboBox,
            QSpinBox,
            QListWidget {
                background: #2f3136;
                border: 1px solid #4b4f58;
                border-radius: 4px;
                color: #f1f3f4;
                padding: 7px;
                selection-background-color: #2f80ed;
            }

            QLineEdit:read-only {
                background: #292b2f;
                color: #d6d9df;
            }

            QTextEdit:disabled {
                background: #292b2f;
                color: #8b9098;
            }

            QMenuBar {
                background: #202124;
                color: #f1f3f4;
            }

            QMenuBar::item:selected,
            QMenu {
                background: #2f3136;
                color: #f1f3f4;
            }

            QMenu::item:selected {
                background: #2f80ed;
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

            QLabel,
            QCheckBox {
                color: #c9d1d9;
            }

            QGroupBox {
                border: 1px solid #4b4f58;
                border-radius: 4px;
                color: #f1f3f4;
                margin-top: 12px;
                padding: 10px 8px 8px 8px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
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

            QSplitter::handle {
                background: #3c4043;
                width: 4px;
            }

            QScrollArea {
                border: none;
            }
            """
        )

    def refresh_metadata(self):
        """Load current image models and style presets."""
        if self.metadata_thread:
            return

        api_key = self.api_key_edit.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Missing API key", "Enter your Venice AI API key first.")
            return

        self.status_label.setText("Loading image models and styles...")
        self.refresh_button.setEnabled(False)

        self.metadata_thread = QThread(self)
        self.metadata_worker = MetadataWorker(api_key)
        self.metadata_worker.moveToThread(self.metadata_thread)
        self.metadata_thread.started.connect(self.metadata_worker.run)
        self.metadata_worker.finished.connect(self.metadata_loaded)
        self.metadata_worker.failed.connect(self.metadata_failed)
        self.metadata_worker.finished.connect(self.metadata_thread.quit)
        self.metadata_worker.failed.connect(self.metadata_thread.quit)
        self.metadata_thread.finished.connect(self.metadata_worker.deleteLater)
        self.metadata_thread.finished.connect(self.metadata_thread.deleteLater)
        self.metadata_thread.finished.connect(self.clear_metadata_worker)
        self.metadata_thread.start()

    def metadata_loaded(self, models, styles):
        """Populate model and style controls."""
        current_model = self.selected_model()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        if models:
            for option in models:
                self.model_combo.addItem(option.label, option)
            if current_model:
                index = self.find_model_index(current_model)
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)
        else:
            self.model_combo.addItem(DEFAULT_MODEL, ImageModelOption(DEFAULT_MODEL, DEFAULT_MODEL))
        self.model_combo.blockSignals(False)

        self.style_combo.clear()
        self.style_combo.addItem("None", "")
        for style in styles:
            self.style_combo.addItem(style, style)

        self.refresh_button.setEnabled(True)
        self.status_label.setText(f"Loaded {self.model_combo.count()} image models and {len(styles)} styles.")
        self.update_cost_estimate()

    def metadata_failed(self, message):
        """Show metadata loading errors."""
        self.refresh_button.setEnabled(True)
        self.status_label.setText("Metadata load failed.")
        QMessageBox.critical(self, "Metadata load failed", message)

    def clear_metadata_worker(self):
        """Clear finished metadata worker references."""
        self.metadata_thread = None
        self.metadata_worker = None

    def choose_output_dir(self):
        """Prompt for an output directory."""
        dirname = QFileDialog.getExistingDirectory(
            self,
            "Choose output directory",
            self.output_dir_edit.text() or str(Path.cwd()),
        )
        if not dirname:
            return
        self.output_dir_edit.setText(dirname)
        self.settings.setValue(OUTPUT_DIR_SETTING, dirname)

    def save_prompt(self):
        """Save prompt fields to a JSON or text file."""
        prompt_dir = self.settings.value(PROMPT_DIR_SETTING, str(Path.cwd()), str)
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save prompt",
            str(Path(prompt_dir) / "image-prompt.json"),
            "Prompt JSON (*.json);;Text files (*.txt);;All files (*)",
        )
        if not filename:
            return

        try:
            path = Path(filename)
            if not path.suffix:
                if selected_filter.startswith("Text"):
                    path = path.with_suffix(".txt")
                elif selected_filter.startswith("Prompt JSON"):
                    path = path.with_suffix(".json")
            self.settings.setValue(PROMPT_DIR_SETTING, str(path.parent))
            if path.suffix.lower() == ".txt":
                path.write_text(self.prompt_edit.toPlainText(), encoding="utf-8")
            else:
                data = {
                    "prompt": self.prompt_edit.toPlainText(),
                    "negative_prompt": self.negative_prompt_edit.toPlainText(),
                    "use_description": self.description_check.isChecked(),
                    "description": self.description_edit.toPlainText(),
                }
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            self.status_label.setText(f"Saved prompt: {path}")
        except OSError as exc:
            QMessageBox.critical(self, "Save prompt failed", str(exc))

    def save_prompt_txt(self):
        """Save only the image prompt as a plain text file."""
        prompt_dir = self.settings.value(PROMPT_DIR_SETTING, str(Path.cwd()), str)
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save prompt as TXT",
            str(Path(prompt_dir) / "image-prompt.txt"),
            "Text files (*.txt);;All files (*)",
        )
        if not filename:
            return

        try:
            path = Path(filename)
            if not path.suffix:
                path = path.with_suffix(".txt")
            self.settings.setValue(PROMPT_DIR_SETTING, str(path.parent))
            path.write_text(self.prompt_edit.toPlainText(), encoding="utf-8")
            self.status_label.setText(f"Saved prompt TXT: {path}")
        except OSError as exc:
            QMessageBox.critical(self, "Save prompt TXT failed", str(exc))

    def load_prompt(self):
        """Load prompt fields from a JSON or plain text file."""
        prompt_dir = self.settings.value(PROMPT_DIR_SETTING, str(Path.cwd()), str)
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load prompt",
            prompt_dir,
            "Prompt files (*.json *.txt);;All files (*)",
        )
        if not filename:
            return

        try:
            path = Path(filename)
            self.settings.setValue(PROMPT_DIR_SETTING, str(path.parent))
            text = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".json":
                data = json.loads(text)
                if not isinstance(data, dict):
                    raise ValueError("Prompt JSON must contain an object.")
                self.prompt_edit.setPlainText(str(data.get("prompt", "")))
                self.negative_prompt_edit.setPlainText(str(data.get("negative_prompt", "")))
                self.description_check.setChecked(bool(data.get("use_description", False)))
                self.description_edit.setPlainText(str(data.get("description", "")))
            else:
                self.prompt_edit.setPlainText(text)

            self.status_label.setText(f"Loaded prompt: {path}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Load prompt failed", str(exc))

    def update_sizing_controls(self):
        """Enable only the controls used by the selected sizing mode."""
        mode = self.sizing_mode_combo.currentData()
        self.width_spin.setEnabled(mode == SIZING_WIDTH_HEIGHT)
        self.height_spin.setEnabled(mode == SIZING_WIDTH_HEIGHT)
        self.aspect_ratio_combo.setEnabled(mode in {SIZING_ASPECT_RATIO, SIZING_RESOLUTION_ASPECT})
        self.resolution_combo.setEnabled(mode == SIZING_RESOLUTION_ASPECT)
        self.update_cost_estimate()

    def update_description_controls(self):
        """Enable description text only when PNG metadata is requested."""
        self.description_edit.setEnabled(self.description_check.isChecked())

    def selected_model(self):
        """Return the selected or typed model ID."""
        data = self.model_combo.currentData()
        if isinstance(data, ImageModelOption):
            current_text = self.model_combo.currentText().strip()
            if current_text and current_text not in {data.label, data.model_id}:
                return current_text
            return data.model_id
        if data:
            return str(data).strip()
        return self.model_combo.currentText().strip()

    def selected_model_option(self):
        """Return pricing metadata for the selected model when available."""
        model_id = self.selected_model()
        data = self.model_combo.currentData()
        if isinstance(data, ImageModelOption) and data.model_id == model_id:
            return data

        index = self.find_model_index(model_id)
        if index >= 0:
            option = self.model_combo.itemData(index)
            if isinstance(option, ImageModelOption):
                return option

        return ImageModelOption(model_id, model_id)

    def find_model_index(self, model_id):
        """Find a loaded model option by ID."""
        for index in range(self.model_combo.count()):
            option = self.model_combo.itemData(index)
            if isinstance(option, ImageModelOption):
                if option.model_id == model_id:
                    return index
            elif option == model_id:
                return index
        return -1

    def update_cost_estimate(self, *_args):
        """Display the estimated image generation cost."""
        if not hasattr(self, "cost_label"):
            return

        payload = self.generation_payload()
        option = self.selected_model_option()
        cost = generation_cost_summary(option.pricing, payload)
        self.cost_label.setText(f"Estimated cost: {cost['display']}")

    def generation_payload(self):
        """Build a Venice /image/generate JSON payload from the form."""
        payload = {
            "model": self.selected_model(),
            "prompt": self.prompt_edit.toPlainText().strip(),
            "format": self.format_combo.currentText(),
            "return_binary": False,
            "variants": self.variants_spin.value(),
            "safe_mode": self.safe_mode_check.isChecked(),
            "hide_watermark": self.hide_watermark_check.isChecked(),
            "embed_exif_metadata": self.embed_exif_check.isChecked(),
            "steps": self.steps_spin.value(),
            "cfg_scale": self.cfg_spin.value(),
        }

        negative_prompt = self.negative_prompt_edit.toPlainText().strip()
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        style = self.style_combo.currentData()
        if style:
            payload["style_preset"] = str(style)

        quality = self.quality_combo.currentText().strip().lower()
        if quality and quality != "default":
            payload["quality"] = quality

        seed = self.seed_spin.value()
        if seed:
            payload["seed"] = seed

        if self.enhance_prompt_check.isChecked():
            payload["enhance_prompt"] = True

        mode = self.sizing_mode_combo.currentData()
        if mode == SIZING_WIDTH_HEIGHT:
            payload["width"] = self.width_spin.value()
            payload["height"] = self.height_spin.value()
        elif mode == SIZING_ASPECT_RATIO:
            payload["aspect_ratio"] = self.aspect_ratio_combo.currentText()
        elif mode == SIZING_RESOLUTION_ASPECT:
            payload["aspect_ratio"] = self.aspect_ratio_combo.currentText()
            payload["resolution"] = self.resolution_combo.currentText()

        return payload

    def generate_image(self):
        """Validate input and start image generation."""
        if self.generation_thread:
            return

        api_key = self.api_key_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        prefix = self.filename_prefix_edit.text().strip() or "venice-image"
        payload = self.generation_payload()
        description = ""
        if self.description_check.isChecked():
            description = self.description_edit.toPlainText().strip()

        if not api_key:
            QMessageBox.warning(self, "Missing API key", "Enter your Venice AI API key first.")
            return
        if not payload["model"]:
            QMessageBox.warning(self, "Missing model", "Choose or enter a Venice image model first.")
            return
        if not payload["prompt"]:
            QMessageBox.warning(self, "Missing prompt", "Enter an image prompt first.")
            return
        if not output_dir:
            QMessageBox.warning(self, "Missing output directory", "Choose an output directory first.")
            return

        option = self.selected_model_option()
        self.active_generation_log = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model": payload["model"],
            "parameters": dict(payload),
            "estimated_cost": generation_cost_summary(option.pricing, payload),
            "output_dir": output_dir,
        }

        self.set_generating(True)
        self.progress.setRange(0, 0)
        self.status_label.setText("Generating image...")

        self.generation_thread = QThread(self)
        self.generation_worker = ImageGenerationWorker(
            api_key,
            payload,
            output_dir,
            prefix,
            payload["format"],
            description,
        )
        self.generation_worker.moveToThread(self.generation_thread)
        self.generation_thread.started.connect(self.generation_worker.run)
        self.generation_worker.finished.connect(self.generation_finished)
        self.generation_worker.failed.connect(self.generation_failed)
        self.generation_worker.finished.connect(self.generation_thread.quit)
        self.generation_worker.failed.connect(self.generation_thread.quit)
        self.generation_thread.finished.connect(self.generation_worker.deleteLater)
        self.generation_thread.finished.connect(self.generation_thread.deleteLater)
        self.generation_thread.finished.connect(self.clear_generation_worker)
        self.generation_thread.start()

    def generation_finished(self, paths, enhanced_prompt):
        """Handle successful image generation."""
        new_paths = list(paths)
        self.generated_paths.extend(new_paths)

        first_new_row = self.output_list.count()
        self.output_list.blockSignals(True)
        self.output_list.addItems(new_paths)
        self.output_list.blockSignals(False)

        if new_paths:
            self.output_list.setCurrentRow(first_new_row)
            self.output_list.scrollToItem(self.output_list.item(first_new_row))

        log_path = None
        log_error = None
        if self.active_generation_log and new_paths:
            try:
                entry = dict(self.active_generation_log)
                entry["enhanced_prompt"] = enhanced_prompt
                entry["generated_files"] = new_paths
                log_path = append_generation_log(entry["output_dir"], entry)
            except OSError as exc:
                log_error = str(exc)

        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.open_output_button.setEnabled(bool(self.output_list.currentItem()))
        self.set_generating(False)

        message = f"Saved {len(new_paths)} image file(s)."
        if log_path:
            message += f" Log: {log_path}"
        if enhanced_prompt:
            message += " Enhanced prompt returned in the response headers."
        if log_error:
            message += " Log write failed."
        self.status_label.setText(message)
        if log_error:
            QMessageBox.warning(self, "Log write failed", log_error)

    def generation_failed(self, message):
        """Display image generation errors."""
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.set_generating(False)
        self.status_label.setText("Generation failed.")
        QMessageBox.critical(self, "Generation failed", message)

    def clear_generation_worker(self):
        """Clear finished generation worker references."""
        self.generation_thread = None
        self.generation_worker = None
        self.active_generation_log = None

    def preview_selected_image(self, _row=None):
        """Preview the selected generated image."""
        item = self.output_list.currentItem()
        if not item:
            self.preview_label.setText("Generated image preview")
            self.open_output_button.setEnabled(False)
            return

        self.display_image(item.text())

    def display_image(self, image_path):
        """Load an image path into the preview."""
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.preview_label.setText(f"Could not preview:\n{image_path}")
            self.open_output_button.setEnabled(False)
            return False

        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)
        self.open_output_button.setEnabled(True)
        return True

    def open_selected_image(self):
        """Open the selected generated image with the desktop handler."""
        item = self.output_list.currentItem()
        if not item:
            return
        path = Path(item.text())
        if not path.is_file():
            QMessageBox.warning(self, "Missing image", f"The selected image does not exist:\n{path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def set_generating(self, generating):
        """Enable or disable controls while generation is running."""
        self.generate_button.setEnabled(not generating)
        self.refresh_button.setEnabled(not generating and self.metadata_thread is None)
        self.open_output_button.setEnabled(False if generating else bool(self.output_list.currentItem()))

    def closeEvent(self, event):
        """Prevent closing while generation is active."""
        if self.generation_thread:
            QMessageBox.warning(
                self,
                "Generation in progress",
                "Wait for image generation to finish before closing.",
            )
            event.ignore()
            return
        if self.metadata_thread:
            QMessageBox.warning(
                self,
                "Metadata refresh in progress",
                "Wait for image model refresh to finish before closing.",
            )
            event.ignore()
            return
        super().closeEvent(event)


def main():
    """Run the Venice AI image-generation GUI."""
    app = QApplication(sys.argv)
    window = ImageGenerationWindow()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
