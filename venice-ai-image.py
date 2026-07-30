#!/usr/bin/env python3
"""PySide6 GUI for Venice AI image generation models."""

import base64
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests
from PySide6.QtCore import QObject, QSettings, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
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

SIZING_WIDTH_HEIGHT = "width_height"
SIZING_ASPECT_RATIO = "aspect_ratio"
SIZING_RESOLUTION_ASPECT = "resolution_aspect"

FORMAT_EXTENSIONS = {
    "jpeg": "jpg",
    "png": "png",
    "webp": "webp",
}


@dataclass
class ImageModelOption:
    """One selectable Venice image model."""

    model_id: str
    label: str


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
        options.append(ImageModelOption(model_id, label))

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

    def __init__(self, api_key, payload, output_dir, filename_prefix, output_format):
        super().__init__()
        self.api_key = api_key
        self.payload = payload
        self.output_dir = Path(output_dir)
        self.filename_prefix = filename_prefix
        self.output_format = output_format

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
        self.generated_paths = []
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)

        self.setWindowTitle("Venice AI Image Generator")
        self.resize(1120, 780)

        self.api_key_edit = QLineEdit(get_venice_api_key() or "")
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("VENICE_API_KEY")

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItem(DEFAULT_MODEL, DEFAULT_MODEL)

        self.refresh_button = QPushButton("Refresh Models")
        self.refresh_button.clicked.connect(self.refresh_metadata)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setAcceptRichText(False)
        self.prompt_edit.setPlaceholderText("Describe the image to generate.")

        self.negative_prompt_edit = QTextEdit()
        self.negative_prompt_edit.setAcceptRichText(False)
        self.negative_prompt_edit.setMaximumHeight(90)
        self.negative_prompt_edit.setPlaceholderText("Optional negative prompt.")

        self.style_combo = QComboBox()
        self.style_combo.addItem("None", "")

        self.sizing_mode_combo = QComboBox()
        self.sizing_mode_combo.addItem("Width and height", SIZING_WIDTH_HEIGHT)
        self.sizing_mode_combo.addItem("Aspect ratio", SIZING_ASPECT_RATIO)
        self.sizing_mode_combo.addItem("Resolution and aspect ratio", SIZING_RESOLUTION_ASPECT)
        self.sizing_mode_combo.currentIndexChanged.connect(self.update_sizing_controls)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 1280)
        self.width_spin.setSingleStep(64)
        self.width_spin.setValue(1024)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 1280)
        self.height_spin.setSingleStep(64)
        self.height_spin.setValue(1024)

        self.aspect_ratio_combo = QComboBox()
        self.aspect_ratio_combo.addItems(("1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"))

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(("1K", "2K", "4K"))

        self.format_combo = QComboBox()
        self.format_combo.addItems(("webp", "png", "jpeg"))

        self.quality_combo = QComboBox()
        self.quality_combo.addItem("Default", "")
        self.quality_combo.addItems(("low", "medium", "high"))

        self.variants_spin = QSpinBox()
        self.variants_spin.setRange(1, 4)
        self.variants_spin.setValue(1)

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
        self.safe_mode_check.setChecked(True)
        self.hide_watermark_check = QCheckBox("Hide watermark")
        self.enhance_prompt_check = QCheckBox("Enhance prompt")
        self.embed_exif_check = QCheckBox("Embed EXIF metadata")

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_edit.setText(self.settings.value(OUTPUT_DIR_SETTING, "", str))

        self.output_dir_button = QPushButton("Choose Output Directory")
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

        self.preview_label = QLabel("Generated image preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(520, 420)

        self.output_list = QListWidget()
        self.output_list.currentRowChanged.connect(self.preview_selected_image)

        self.build_layout()
        self.apply_style()
        self.update_sizing_controls()

    def build_layout(self):
        """Assemble the form and preview area."""
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.refresh_button)

        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit, 1)
        output_row.addWidget(self.output_dir_button)

        settings_form = QFormLayout()
        settings_form.addRow("API key", self.api_key_edit)
        settings_form.addRow("Model", self.wrap_layout(model_row))
        settings_form.addRow("Prompt", self.prompt_edit)
        settings_form.addRow("Negative prompt", self.negative_prompt_edit)
        settings_form.addRow("Style", self.style_combo)
        settings_form.addRow("Sizing", self.sizing_mode_combo)
        settings_form.addRow("Width", self.width_spin)
        settings_form.addRow("Height", self.height_spin)
        settings_form.addRow("Aspect ratio", self.aspect_ratio_combo)
        settings_form.addRow("Resolution", self.resolution_combo)
        settings_form.addRow("Format", self.format_combo)
        settings_form.addRow("Quality", self.quality_combo)
        settings_form.addRow("Variants", self.variants_spin)
        settings_form.addRow("Steps", self.steps_spin)
        settings_form.addRow("CFG scale", self.cfg_spin)
        settings_form.addRow("Seed", self.seed_spin)
        settings_form.addRow("Output directory", self.wrap_layout(output_row))
        settings_form.addRow("Filename prefix", self.filename_prefix_edit)

        toggles = QHBoxLayout()
        toggles.addWidget(self.safe_mode_check)
        toggles.addWidget(self.hide_watermark_check)
        toggles.addWidget(self.enhance_prompt_check)
        toggles.addWidget(self.embed_exif_check)
        toggles.addStretch(1)

        controls = QHBoxLayout()
        controls.addWidget(self.generate_button)
        controls.addWidget(self.open_output_button)
        controls.addStretch(1)

        left_layout = QVBoxLayout()
        left_layout.addLayout(settings_form)
        left_layout.addLayout(toggles)
        left_layout.addWidget(self.progress)
        left_layout.addWidget(self.status_label)
        left_layout.addLayout(controls)

        left = QWidget()
        left.setLayout(left_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.preview_label, 1)
        right_layout.addWidget(QLabel("Generated files"))
        right_layout.addWidget(self.output_list)

        right = QWidget()
        right.setLayout(right_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

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
        self.model_combo.clear()
        if models:
            for option in models:
                self.model_combo.addItem(option.label, option.model_id)
            if current_model:
                index = self.model_combo.findData(current_model)
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)
        else:
            self.model_combo.addItem(DEFAULT_MODEL, DEFAULT_MODEL)

        self.style_combo.clear()
        self.style_combo.addItem("None", "")
        for style in styles:
            self.style_combo.addItem(style, style)

        self.refresh_button.setEnabled(True)
        self.status_label.setText(f"Loaded {self.model_combo.count()} image models and {len(styles)} styles.")

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

    def update_sizing_controls(self):
        """Enable only the controls used by the selected sizing mode."""
        mode = self.sizing_mode_combo.currentData()
        self.width_spin.setEnabled(mode == SIZING_WIDTH_HEIGHT)
        self.height_spin.setEnabled(mode == SIZING_WIDTH_HEIGHT)
        self.aspect_ratio_combo.setEnabled(mode in {SIZING_ASPECT_RATIO, SIZING_RESOLUTION_ASPECT})
        self.resolution_combo.setEnabled(mode == SIZING_RESOLUTION_ASPECT)

    def selected_model(self):
        """Return the selected or typed model ID."""
        data = self.model_combo.currentData()
        if data:
            return str(data).strip()
        return self.model_combo.currentText().strip()

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
        self.generated_paths = list(paths)
        self.output_list.clear()
        self.output_list.addItems(self.generated_paths)
        if self.generated_paths:
            self.output_list.setCurrentRow(0)

        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.open_output_button.setEnabled(bool(self.generated_paths))
        self.set_generating(False)

        message = f"Saved {len(self.generated_paths)} image file(s)."
        if enhanced_prompt:
            message += " Enhanced prompt returned in the response headers."
        self.status_label.setText(message)

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

    def preview_selected_image(self, _row=None):
        """Load the selected generated image into the preview."""
        item = self.output_list.currentItem()
        if not item:
            self.preview_label.setText("Generated image preview")
            self.open_output_button.setEnabled(False)
            return

        image_path = item.text()
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.preview_label.setText(f"Could not preview:\n{image_path}")
            self.open_output_button.setEnabled(False)
            return

        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)
        self.open_output_button.setEnabled(True)

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
        super().closeEvent(event)


def main():
    """Run the Venice AI image-generation GUI."""
    app = QApplication(sys.argv)
    window = ImageGenerationWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
