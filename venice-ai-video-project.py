#!/usr/bin/env python3
"""Build an audio-length video from an editable queue of 30-second scenes."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path

import requests
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from modules.video_project import (
    CHARACTERS_PATH,
    Character,
    Scene,
    build_scene_prompt,
    load_characters,
    mux_project_audio,
    plan_scene_durations,
    probe_audio_duration,
    probe_video_duration,
    project_slug,
    save_characters,
    save_project,
    scene_duration_matches,
)


def load_video_module():
    """Load the existing generator so this tool shares its API plumbing."""
    path = Path(__file__).with_name("venice-ai-video.py")
    spec = importlib.util.spec_from_file_location("venice_ai_video_base", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


video = load_video_module()
DEFAULT_PROJECT_MODEL = "seedance-2-5-reference-to-video-basic"


def model_kind(model):
    """Return the supported project mode for live model metadata."""
    model_id = video.model_identifier(model).lower()
    text = video.model_text(model)
    if "reference-to-video" in model_id or "reference to video" in text or "r2v" in model_id:
        return "reference"
    if video.is_text_to_video_model(model):
        return "text"
    return ""


class ProjectModelsWorker(QObject):
    finished = Signal(object, str)
    failed = Signal(str)

    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key

    def run(self):
        try:
            response = requests.get(
                video.MODELS_URL,
                headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"},
                params={"type": "video"},
                timeout=video.HTTP_TIMEOUT,
            )
            if not response.ok:
                raise video.VeniceAPIError(
                    response.status_code,
                    video.MODELS_URL,
                    video.VeniceVideoWorker.response_diagnostic(response),
                )
            options = []
            for model in video.normalize_model_list(response.json()):
                kind = model_kind(model)
                option = video.model_option(model)
                if kind and "30s" in option.durations:
                    options.append((option, kind))
            options.sort(
                key=lambda item: (
                    item[1] != "reference",
                    item[0].model_id != DEFAULT_PROJECT_MODEL,
                    item[0].label.lower(),
                )
            )
            if not options:
                raise RuntimeError("No text/reference video model currently advertises 30-second clips.")
            self.finished.emit(options, f"Loaded {len(options)} models supporting 30-second clips.")
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ProjectCostWorker(QObject):
    """Request a quote for the current scene plan without queuing video."""

    finished = Signal(float)
    failed = Signal(str)

    def __init__(self, api_key, model_id, durations, resolution, aspect_ratio):
        super().__init__()
        self.api_key = api_key
        self.model_id = model_id
        self.durations = list(durations)
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio

    def run(self):
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        try:
            total = 0.0
            for duration in self.durations:
                payload = {
                    "model": self.model_id,
                    "duration": f"{duration}s",
                }
                if self.resolution:
                    payload["resolution"] = self.resolution
                if self.aspect_ratio:
                    payload["aspect_ratio"] = self.aspect_ratio
                try:
                    response = session.post(
                        video.VIDEO_QUOTE_URL,
                        json=payload,
                        timeout=video.HTTP_TIMEOUT,
                    )
                except requests.RequestException as exc:
                    raise RuntimeError(
                        f"Could not connect to Venice:\n{video.VIDEO_QUOTE_URL}\n\n{exc}"
                    ) from exc
                if not response.ok:
                    raise video.VeniceAPIError(
                        response.status_code,
                        video.VIDEO_QUOTE_URL,
                        video.VeniceVideoWorker.response_diagnostic(response),
                    )
                try:
                    data = response.json()
                except ValueError as exc:
                    raise RuntimeError("Venice returned a non-JSON quote response.") from exc
                quote = video.VeniceVideoWorker.find_first_value(
                    data,
                    {"quote", "cost", "price", "amount"},
                )
                if quote is None:
                    raise RuntimeError("The quote response did not include a quote.")
                total += float(quote)
            self.finished.emit(total)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            session.close()


class ProjectGenerationWorker(video.VeniceVideoWorker):
    """Generate user-authored scenes, stitch them, and install final audio."""

    scene_ready = Signal(int, str, bool, float)

    @staticmethod
    def format_generation_cost(total):
        """Display the USD quote in equivalent Venice account credits."""
        return f"Estimated video generation cost: {total:.4f} credits"

    def __init__(
        self,
        api_key,
        scenes,
        characters,
        overall_style,
        output_dir,
        project_name,
        audio_path,
        remove_generated_audio,
        video_option,
        model_kind_value,
        resolution,
        aspect_ratio,
    ):
        self.scenes = scenes
        self.characters = characters
        self.overall_style = overall_style
        self.project_name = project_name
        self.audio_path = audio_path
        self.remove_generated_audio = remove_generated_audio
        self.model_kind_value = model_kind_value
        final_path = Path(output_dir) / f"{project_name}.mp4"
        super().__init__(
            api_key=api_key,
            prompt=overall_style,
            source_file="",
            output_file=str(final_path),
            video_model=video_option.model_id,
            generation_mode=video.TEXT_TO_VIDEO_MODE,
            total_seconds=sum(scene.duration_seconds for scene in scenes),
            segment_durations=[scene.duration_seconds for scene in scenes],
            video_resolution=resolution,
            video_aspect_ratio=aspect_ratio,
            reference_files=[],
            retain_intermediate_files=True,
        )

    def run(self):
        temp_dir = None
        try:
            self.destination_dir.mkdir(parents=True, exist_ok=True)
            self.verify_dependencies()
            temp_root = Path(tempfile.gettempdir()) / "venice-ai-video"
            temp_root.mkdir(parents=True, exist_ok=True)
            temp_dir = temp_root / f"{self.project_name}-{uuid.uuid4().hex[:10]}"
            temp_dir.mkdir()
            segment_files = {}
            scenes_to_generate = []
            character_map = {character.name: character for character in self.characters}

            for scene in self.scenes:
                output_path = self.destination_dir / f"{self.project_name}-scene-{scene.number}.mp4"
                if output_path.exists():
                    try:
                        duration = probe_video_duration(output_path)
                        if not scene_duration_matches(duration, scene.duration_seconds):
                            raise ValueError(
                                f"duration is {duration:.2f}s; expected {scene.duration_seconds}s"
                            )
                    except Exception as exc:
                        self.progress.emit(
                            0,
                            f"Scene {scene.number}: existing {output_path.name} is not reusable ({exc}); regenerating.",
                        )
                        output_path.unlink(missing_ok=True)
                    else:
                        segment_files[scene.number] = output_path
                        self.scene_ready.emit(scene.number, str(output_path), True, duration)
                        self.progress.emit(
                            0,
                            f"Scene {scene.number}: reusing {output_path.name} ({duration:.2f}s).",
                        )
                        continue
                scenes_to_generate.append(scene)

            self.segment_durations = [scene.duration_seconds for scene in scenes_to_generate]
            if scenes_to_generate:
                self.estimate_generation_cost()
            else:
                self.progress.emit(0, "Estimated video generation cost: 0.0000 credits (all scene clips reused).")

            for scene in scenes_to_generate:
                self.progress.emit(scene.number - 1, f"Preparing scene {scene.number} of {len(self.scenes)}...")
                output_path = self.destination_dir / f"{self.project_name}-scene-{scene.number}.mp4"
                selected = [
                    character_map[name]
                    for name in scene.character_names
                    if name in character_map
                ]
                reference_paths = [
                    Path(character.image_path)
                    for character in selected
                    if character.image_path
                ]
                reference_urls = []
                if self.model_kind_value == "reference":
                    reference_urls = [self.image_to_data_url(path) for path in reference_paths]
                prompt = build_scene_prompt(
                    self.overall_style,
                    scene,
                    selected,
                    self.video_model,
                )
                segment = video.Segment(
                    number=scene.number,
                    duration_seconds=scene.duration_seconds,
                    description=prompt,
                )
                self.reference_files = reference_paths if reference_urls else []
                self.generate_video_segment(
                    segment=segment,
                    output_path=output_path,
                    reference_data_urls=reference_urls,
                )
                self.validate_media_file(output_path, f"scene {scene.number}")
                duration = probe_video_duration(output_path)
                if not scene_duration_matches(duration, scene.duration_seconds):
                    raise RuntimeError(
                        f"Generated scene {scene.number} is {duration:.2f}s; expected "
                        f"{scene.duration_seconds}s (±1.00s):\n{output_path}"
                    )
                segment_files[scene.number] = output_path
                self.scene_ready.emit(scene.number, str(output_path), False, duration)
                self.progress.emit(scene.number, f"Saved {output_path.name} ({duration:.2f}s)")

            stitched = temp_dir / "stitched-scenes.mp4"
            self.progress.emit(len(self.scenes), "Stitching generated scenes...")
            ordered_segment_files = [segment_files[scene.number] for scene in self.scenes]
            self.stitch_segments(ordered_segment_files, stitched)
            self.progress.emit(len(self.scenes), "Installing the project audio track...")
            mux_project_audio(
                stitched,
                self.output_file,
                self.audio_path,
                self.remove_generated_audio,
                timeout=video.FFMPEG_STITCH_TIMEOUT_SECONDS,
            )
            self.validate_media_file(self.output_file, "final video")
            self.finished.emit(str(self.output_file))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.session.close()
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)


class CharacterDialog(QDialog):
    """Edit one reusable character."""

    def __init__(self, parent=None, character=None):
        super().__init__(parent)
        self.setWindowTitle("Character")
        self.name_edit = QLineEdit(character.name if character else "")
        self.description_edit = QTextEdit(character.description if character else "")
        self.description_edit.setMinimumHeight(130)
        self.image_edit = QLineEdit(character.image_path if character else "")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self.choose_image)
        image_row = QHBoxLayout()
        image_row.addWidget(self.image_edit, 1)
        image_row.addWidget(browse)
        form = QFormLayout()
        form.addRow("Name", self.name_edit)
        form.addRow("Description", self.description_edit)
        form.addRow("Reference image", image_row)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose Character Image", self.image_edit.text(), "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if path:
            self.image_edit.setText(path)

    def validate_and_accept(self):
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Missing name", "Enter a character name.")
            return
        path = self.image_edit.text().strip()
        if path and not Path(path).is_file():
            QMessageBox.warning(self, "Invalid image", f"The reference image does not exist:\n{path}")
            return
        self.accept()

    def character(self):
        return Character(
            name=self.name_edit.text().strip(),
            description=self.description_edit.toPlainText().strip(),
            image_path=self.image_edit.text().strip(),
        )


class ProjectWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Venice AI Video Project Builder")
        self.resize(1180, 820)
        self.api_key = os.environ.get("VENICE_API_KEY", "").strip()
        self.characters = []
        self.scenes = []
        self.ready_scene_numbers = set()
        self.current_scene_index = -1
        self.thread = None
        self.worker = None
        self.model_thread = None
        self.model_worker = None
        self.cost_thread = None
        self.cost_worker = None
        self.audio_seconds = 0.0
        self.build_controls()
        self.build_layout()
        self.apply_style()
        self.reload_characters()
        self.add_scene(30)
        if self.api_key:
            self.load_models()

    def build_controls(self):
        self.project_name_edit = QLineEdit()
        self.project_name_edit.setPlaceholderText("tides-of-hate")
        self.project_name_edit.textChanged.connect(self.clear_scene_file_status)
        self.output_dir_edit = QLineEdit(str(Path.home() / "Videos"))
        self.output_dir_edit.textChanged.connect(self.clear_scene_file_status)
        output_button = QPushButton("Browse...")
        output_button.clicked.connect(self.choose_output_dir)
        self.output_row = QHBoxLayout()
        self.output_row.addWidget(self.output_dir_edit, 1)
        self.output_row.addWidget(output_button)
        self.audio_edit = QLineEdit()
        audio_button = QPushButton("Choose MP3...")
        audio_button.clicked.connect(self.choose_audio)
        self.audio_row = QHBoxLayout()
        self.audio_row.addWidget(self.audio_edit, 1)
        self.audio_row.addWidget(audio_button)
        self.audio_info = QLabel("No audio selected")
        self.remove_audio_checkbox = QCheckBox("Remove generated clip audio; use MP3 as the stereo soundtrack")
        self.remove_audio_checkbox.setChecked(True)
        self.model_combo = QComboBox()
        self.model_combo.addItem(DEFAULT_PROJECT_MODEL, None)
        refresh = QPushButton("Refresh Models")
        refresh.clicked.connect(self.load_models)
        self.model_row = QHBoxLayout()
        self.model_row.addWidget(self.model_combo, 1)
        self.model_row.addWidget(refresh)
        self.style_edit = QTextEdit()
        self.style_edit.setPlaceholderText("Cinematography, visual language, palette, period, lens, lighting, pacing...")
        self.style_edit.setMaximumHeight(130)

        self.scene_list = QListWidget()
        self.scene_list.currentRowChanged.connect(self.scene_changed)
        add_scene_button = QPushButton("Add Scene")
        add_scene_button.clicked.connect(lambda: self.add_scene(30))
        remove_scene_button = QPushButton("Remove Scene")
        remove_scene_button.clicked.connect(self.remove_scene)
        rebuild_button = QPushButton("Rebuild from Audio")
        rebuild_button.clicked.connect(self.rebuild_from_audio)
        self.scene_buttons = QHBoxLayout()
        self.scene_buttons.addWidget(add_scene_button)
        self.scene_buttons.addWidget(remove_scene_button)
        self.scene_buttons.addWidget(rebuild_button)

        self.duration_combo = QComboBox()
        for seconds in (30, 15, 10):
            self.duration_combo.addItem(f"{seconds} seconds", seconds)
        self.duration_combo.currentIndexChanged.connect(self.scene_editor_changed)
        self.scene_prompt_edit = QTextEdit()
        self.scene_prompt_edit.setPlaceholderText("Describe this cut-scene's action, camera movement, composition, and end state...")
        self.scene_prompt_edit.textChanged.connect(self.scene_editor_changed)
        self.scene_characters = QListWidget()
        self.scene_characters.itemChanged.connect(self.scene_editor_changed)

        self.character_list = QListWidget()
        self.character_list.itemDoubleClicked.connect(lambda _item: self.edit_character())
        new_character = QPushButton("New")
        new_character.clicked.connect(self.new_character)
        edit_character = QPushButton("Inspect / Edit")
        edit_character.clicked.connect(self.edit_character)
        delete_character = QPushButton("Delete")
        delete_character.clicked.connect(self.delete_character)
        self.character_buttons = QHBoxLayout()
        self.character_buttons.addWidget(new_character)
        self.character_buttons.addWidget(edit_character)
        self.character_buttons.addWidget(delete_character)

        self.log = QTextBrowser()
        self.progress = QProgressBar()
        self.save_button = QPushButton("Save Project")
        self.save_button.clicked.connect(self.save_project_dialog)
        self.load_button = QPushButton("Load Project")
        self.load_button.clicked.connect(self.load_project_dialog)
        self.estimate_button = QPushButton("Estimate Cost")
        self.estimate_button.clicked.connect(self.estimate_cost)
        self.generate_button = QPushButton("Start Generation")
        self.generate_button.clicked.connect(self.generate)

    def build_layout(self):
        project_form = QFormLayout()
        project_form.addRow("Project name", self.project_name_edit)
        project_form.addRow("Output directory", self.output_row)
        project_form.addRow("Soundtrack", self.audio_row)
        project_form.addRow("Audio duration", self.audio_info)
        project_form.addRow("", self.remove_audio_checkbox)
        project_form.addRow("30-second model", self.model_row)

        scene_left = QWidget()
        scene_left_layout = QVBoxLayout(scene_left)
        scene_left_layout.addWidget(QLabel("Scene queue"))
        scene_left_layout.addWidget(self.scene_list, 1)
        scene_left_layout.addLayout(self.scene_buttons)
        scene_right = QWidget()
        scene_right_layout = QVBoxLayout(scene_right)
        scene_right_layout.addWidget(QLabel("Selected scene duration"))
        scene_right_layout.addWidget(self.duration_combo)
        scene_right_layout.addWidget(QLabel("Scene prompt"))
        scene_right_layout.addWidget(self.scene_prompt_edit, 2)
        scene_right_layout.addWidget(QLabel("Characters in this scene"))
        scene_right_layout.addWidget(self.scene_characters, 1)
        scenes_splitter = QSplitter(Qt.Horizontal)
        scenes_splitter.addWidget(scene_left)
        scenes_splitter.addWidget(scene_right)
        scenes_splitter.setStretchFactor(1, 2)

        character_widget = QWidget()
        character_layout = QVBoxLayout(character_widget)
        character_layout.addWidget(QLabel(f"Character library — {CHARACTERS_PATH}"))
        character_layout.addWidget(self.character_list, 1)
        character_layout.addLayout(self.character_buttons)

        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.addWidget(scenes_splitter)
        content_splitter.addWidget(character_widget)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 1)

        actions = QHBoxLayout()
        actions.addWidget(self.load_button)
        actions.addWidget(self.save_button)
        actions.addStretch(1)
        actions.addWidget(self.estimate_button)
        actions.addWidget(self.generate_button)
        layout = QVBoxLayout()
        layout.addLayout(project_form)
        layout.addWidget(QLabel("Overall video style (prepended to every scene prompt)"))
        layout.addWidget(self.style_edit)
        layout.addWidget(content_splitter, 3)
        layout.addWidget(QLabel("Status log"))
        layout.addWidget(self.log, 1)
        layout.addWidget(self.progress)
        layout.addLayout(actions)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #202124; color: #f1f3f4; font-size: 14px; }
            QLineEdit, QComboBox, QTextEdit, QTextBrowser, QListWidget {
                background: #2f3136; border: 1px solid #4b4f58; border-radius: 4px;
                color: #f1f3f4; padding: 6px; selection-background-color: #2f80ed;
            }
            QPushButton { background: #3c4043; border: 1px solid #5f6368; border-radius: 4px; padding: 7px 12px; }
            QPushButton:hover { background: #4b4f58; }
            QPushButton:disabled { color: #8b9098; }
            QProgressBar { background: #2f3136; border: 1px solid #4b4f58; text-align: center; min-height: 21px; }
            QProgressBar::chunk { background: #2f80ed; }
            """
        )

    def ensure_api_key(self):
        if self.api_key:
            return True
        value, accepted = QInputDialog.getText(self, "Venice API Key", "Enter VENICE_API_KEY:", QLineEdit.Password)
        if accepted and value.strip():
            self.api_key = value.strip()
            return True
        return False

    def choose_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choose Output Directory", self.output_dir_edit.text())
        if path:
            self.output_dir_edit.setText(path)

    def choose_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose MP3 Soundtrack", self.audio_edit.text(), "MP3 audio (*.mp3)")
        if not path:
            return
        try:
            self.audio_seconds = probe_audio_duration(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not read audio", str(exc))
            return
        self.audio_edit.setText(path)
        self.audio_info.setText(f"{self.audio_seconds:.2f} seconds")
        self.rebuild_from_audio()

    def add_scene(self, duration, prompt="", character_names=None):
        self.commit_scene_editor()
        self.scenes.append(Scene(len(self.scenes) + 1, duration, prompt, list(character_names or [])))
        self.refresh_scene_list()
        self.scene_list.setCurrentRow(len(self.scenes) - 1)

    def remove_scene(self):
        row = self.scene_list.currentRow()
        if row < 0 or len(self.scenes) <= 1:
            return
        self.scenes.pop(row)
        self.ready_scene_numbers.clear()
        for index, scene in enumerate(self.scenes, 1):
            scene.number = index
        self.current_scene_index = -1
        self.refresh_scene_list()
        self.scene_list.setCurrentRow(min(row, len(self.scenes) - 1))

    def rebuild_from_audio(self):
        path = self.audio_edit.text().strip()
        if not path:
            QMessageBox.information(self, "No audio", "Choose an MP3 soundtrack first.")
            return
        try:
            self.audio_seconds = probe_audio_duration(path)
            durations = plan_scene_durations(self.audio_seconds)
        except Exception as exc:
            QMessageBox.critical(self, "Could not plan scenes", str(exc))
            return
        self.commit_scene_editor()
        old = self.scenes
        self.scenes = []
        self.ready_scene_numbers.clear()
        for index, duration in enumerate(durations, 1):
            previous = old[index - 1] if index <= len(old) else None
            self.scenes.append(
                Scene(index, duration, previous.prompt if previous else "", list(previous.character_names) if previous else [])
            )
        self.audio_info.setText(
            f"{self.audio_seconds:.2f} seconds → " + ", ".join(f"{value}s" for value in durations)
        )
        self.current_scene_index = -1
        self.refresh_scene_list()
        self.scene_list.setCurrentRow(0)

    def refresh_scene_list(self):
        self.scene_list.blockSignals(True)
        self.scene_list.clear()
        for scene in self.scenes:
            if scene.number in self.ready_scene_numbers:
                marker = "✓"
            else:
                marker = "•" if scene.prompt.strip() else "○"
            self.scene_list.addItem(f"{marker} Scene {scene.number} — {scene.duration_seconds}s")
        self.scene_list.blockSignals(False)

    def clear_scene_file_status(self, *_args):
        """Clear clip checkmarks when their project path may have changed."""
        if not self.ready_scene_numbers:
            return
        self.ready_scene_numbers.clear()
        if hasattr(self, "scene_list"):
            self.refresh_scene_list_preserving_row()

    def scan_existing_scene_clips(self, log_results=True):
        """Inventory reusable scene files for the current project."""
        self.commit_scene_editor()
        reusable = set()
        missing = []
        invalid = []
        try:
            slug = project_slug(self.project_name_edit.text())
        except ValueError:
            self.ready_scene_numbers.clear()
            self.refresh_scene_list_preserving_row()
            return list(self.scenes)
        output_text = self.output_dir_edit.text().strip()
        if not output_text:
            self.ready_scene_numbers.clear()
            self.refresh_scene_list_preserving_row()
            return list(self.scenes)
        output_dir = Path(output_text)
        for scene in self.scenes:
            path = output_dir / f"{slug}-scene-{scene.number}.mp4"
            if not path.is_file():
                missing.append(scene)
                continue
            try:
                duration = probe_video_duration(path)
                if not scene_duration_matches(duration, scene.duration_seconds):
                    raise ValueError(
                        f"duration is {duration:.2f}s; expected {scene.duration_seconds}s"
                    )
            except Exception as exc:
                invalid.append((scene, path, exc))
            else:
                reusable.add(scene.number)
        self.ready_scene_numbers = reusable
        self.refresh_scene_list_preserving_row()
        pending = [scene for scene in self.scenes if scene.number not in reusable]
        if log_results:
            self.append_log(
                f"Scene file scan: {len(reusable)} reusable, {len(pending)} need generation "
                f"({len(missing)} missing, {len(invalid)} invalid)."
            )
            for scene, path, exc in invalid:
                self.append_log(f"Scene {scene.number} is not reusable ({path.name}): {exc}")
        return pending

    def scene_changed(self, row):
        self.commit_scene_editor()
        self.current_scene_index = row
        enabled = 0 <= row < len(self.scenes)
        self.duration_combo.setEnabled(enabled)
        self.scene_prompt_edit.setEnabled(enabled)
        self.scene_characters.setEnabled(enabled)
        if not enabled:
            return
        scene = self.scenes[row]
        self.duration_combo.blockSignals(True)
        index = self.duration_combo.findData(scene.duration_seconds)
        self.duration_combo.setCurrentIndex(max(0, index))
        self.duration_combo.blockSignals(False)
        self.scene_prompt_edit.blockSignals(True)
        self.scene_prompt_edit.setPlainText(scene.prompt)
        self.scene_prompt_edit.blockSignals(False)
        self.populate_scene_characters(scene.character_names)

    def scene_editor_changed(self, *_args):
        self.commit_scene_editor()
        self.refresh_scene_list_preserving_row()

    def commit_scene_editor(self):
        row = self.current_scene_index
        if row < 0 or row >= len(self.scenes):
            return
        scene = self.scenes[row]
        duration = int(self.duration_combo.currentData() or 30)
        if duration != scene.duration_seconds:
            self.ready_scene_numbers.discard(scene.number)
        scene.duration_seconds = duration
        scene.prompt = self.scene_prompt_edit.toPlainText().strip()
        scene.character_names = [
            self.scene_characters.item(index).text()
            for index in range(self.scene_characters.count())
            if self.scene_characters.item(index).checkState() == Qt.Checked
        ]

    def refresh_scene_list_preserving_row(self):
        row = self.current_scene_index
        self.refresh_scene_list()
        if row >= 0:
            self.scene_list.blockSignals(True)
            self.scene_list.setCurrentRow(row)
            self.scene_list.blockSignals(False)

    def populate_scene_characters(self, selected):
        self.scene_characters.blockSignals(True)
        self.scene_characters.clear()
        for character in self.characters:
            item = QListWidgetItem(character.name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if character.name in selected else Qt.Unchecked)
            self.scene_characters.addItem(item)
        self.scene_characters.blockSignals(False)

    def reload_characters(self):
        try:
            self.characters = load_characters()
        except Exception as exc:
            QMessageBox.critical(self, "Character library error", str(exc))
            self.characters = []
        self.character_list.clear()
        for character in self.characters:
            suffix = f" — {Path(character.image_path).name}" if character.image_path else ""
            self.character_list.addItem(character.name + suffix)
        selected = self.scenes[self.current_scene_index].character_names if 0 <= self.current_scene_index < len(self.scenes) else []
        self.populate_scene_characters(selected)

    def new_character(self):
        dialog = CharacterDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        character = dialog.character()
        if any(item.name.casefold() == character.name.casefold() for item in self.characters):
            QMessageBox.warning(self, "Duplicate character", "A character with that name already exists.")
            return
        self.characters.append(character)
        save_characters(self.characters)
        self.reload_characters()

    def edit_character(self):
        row = self.character_list.currentRow()
        if row < 0:
            return
        old_name = self.characters[row].name
        dialog = CharacterDialog(self, self.characters[row])
        if dialog.exec() != QDialog.Accepted:
            return
        updated = dialog.character()
        if any(index != row and item.name.casefold() == updated.name.casefold() for index, item in enumerate(self.characters)):
            QMessageBox.warning(self, "Duplicate character", "A character with that name already exists.")
            return
        self.characters[row] = updated
        for scene in self.scenes:
            scene.character_names = [updated.name if name == old_name else name for name in scene.character_names]
        save_characters(self.characters)
        self.reload_characters()

    def delete_character(self):
        row = self.character_list.currentRow()
        if row < 0:
            return
        character = self.characters[row]
        if QMessageBox.question(self, "Delete character", f"Delete '{character.name}' from the library?") != QMessageBox.Yes:
            return
        self.characters.pop(row)
        for scene in self.scenes:
            scene.character_names = [name for name in scene.character_names if name != character.name]
        save_characters(self.characters)
        self.reload_characters()

    def load_models(self):
        if self.model_thread is not None or not self.ensure_api_key():
            return
        self.model_thread = QThread(self)
        self.model_worker = ProjectModelsWorker(self.api_key)
        self.model_worker.moveToThread(self.model_thread)
        self.model_thread.started.connect(self.model_worker.run)
        self.model_worker.finished.connect(self.models_loaded)
        self.model_worker.failed.connect(self.models_failed)
        self.model_worker.finished.connect(self.model_thread.quit)
        self.model_worker.failed.connect(self.model_thread.quit)
        self.model_thread.finished.connect(self.clear_model_worker)
        self.model_thread.start()

    def models_loaded(self, options, message):
        self.model_combo.clear()
        for option, kind in options:
            self.model_combo.addItem(f"{option.label} [{kind}]", (option, kind))
        self.append_log(message)

    def models_failed(self, message):
        self.append_log("Model refresh failed: " + message)
        QMessageBox.warning(self, "Could not load models", message)

    def clear_model_worker(self):
        if self.model_worker:
            self.model_worker.deleteLater()
        if self.model_thread:
            self.model_thread.deleteLater()
        self.model_worker = None
        self.model_thread = None

    def project_payload(self):
        self.commit_scene_editor()
        model_data = self.model_combo.currentData()
        model_id = model_data[0].model_id if model_data else self.model_combo.currentText()
        return {
            "version": 1,
            "project_name": self.project_name_edit.text().strip(),
            "output_directory": self.output_dir_edit.text().strip(),
            "audio_path": self.audio_edit.text().strip(),
            "audio_seconds": self.audio_seconds,
            "remove_generated_audio": self.remove_audio_checkbox.isChecked(),
            "video_model": model_id,
            "overall_style": self.style_edit.toPlainText(),
            "scenes": [asdict(scene) for scene in self.scenes],
        }

    def save_project_dialog(self):
        try:
            slug = project_slug(self.project_name_edit.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid project name", str(exc))
            return
        default = Path(self.output_dir_edit.text() or ".") / f"{slug}.venice-video-project.json"
        path, _ = QFileDialog.getSaveFileName(self, "Save Video Project", str(default), "Video project (*.json)")
        if path:
            save_project(path, self.project_payload())
            self.append_log(f"Saved project: {path}")

    def load_project_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Video Project", self.output_dir_edit.text(), "Video project (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            scenes = [Scene.from_dict(item) for item in data.get("scenes", [])]
            if not scenes:
                raise ValueError("The project contains no scenes.")
        except Exception as exc:
            QMessageBox.critical(self, "Could not load project", str(exc))
            return
        self.project_name_edit.setText(str(data.get("project_name", "")))
        self.output_dir_edit.setText(str(data.get("output_directory", "")))
        self.audio_edit.setText(str(data.get("audio_path", "")))
        self.audio_seconds = float(data.get("audio_seconds", 0))
        self.audio_info.setText(f"{self.audio_seconds:.2f} seconds" if self.audio_seconds else "No audio selected")
        self.remove_audio_checkbox.setChecked(bool(data.get("remove_generated_audio", True)))
        self.style_edit.setPlainText(str(data.get("overall_style", "")))
        self.scenes = scenes
        self.ready_scene_numbers.clear()
        for index, scene in enumerate(self.scenes, 1):
            scene.number = index
        self.current_scene_index = -1
        self.refresh_scene_list()
        self.scene_list.setCurrentRow(0)
        wanted_model = str(data.get("video_model", ""))
        for index in range(self.model_combo.count()):
            value = self.model_combo.itemData(index)
            if value and value[0].model_id == wanted_model:
                self.model_combo.setCurrentIndex(index)
                break
        self.append_log(f"Loaded project: {path}")
        self.scan_existing_scene_clips()

    def validate_generation(self):
        self.commit_scene_editor()
        slug = project_slug(self.project_name_edit.text())
        output_text = self.output_dir_edit.text().strip()
        if not output_text:
            raise ValueError("Choose an output directory.")
        output_dir = Path(output_text)
        audio_path = self.audio_edit.text().strip()
        if audio_path and not Path(audio_path).is_file():
            raise FileNotFoundError(f"The soundtrack does not exist:\n{audio_path}")
        if not self.scenes:
            raise ValueError("Add at least one scene to the queue.")
        generation_scenes = [
            scene for scene in self.scenes if scene.number not in self.ready_scene_numbers
        ]
        if any(not scene.prompt.strip() for scene in generation_scenes):
            raise ValueError("Enter a prompt for every scene that needs generation.")
        model_data = self.model_combo.currentData()
        if not model_data:
            raise ValueError("Refresh and select a current 30-second video model.")
        option, kind = model_data
        unsupported = [
            scene.duration_seconds
            for scene in generation_scenes
            if f"{scene.duration_seconds}s" not in option.durations
        ]
        if unsupported:
            raise ValueError(f"The selected model does not support {unsupported[0]}-second scenes.")
        selected_names = {name for scene in generation_scenes for name in scene.character_names}
        selected_characters = [character for character in self.characters if character.name in selected_names]
        missing = [character.image_path for character in selected_characters if character.image_path and not Path(character.image_path).is_file()]
        if missing:
            raise FileNotFoundError(f"A character reference image is missing:\n{missing[0]}")
        max_scene_images = max(
            (
                sum(
                    1
                    for character in selected_characters
                    if character.name in scene.character_names and character.image_path
                )
                for scene in generation_scenes
            ),
            default=0,
        )
        if kind == "reference" and option.max_reference_images and max_scene_images > option.max_reference_images:
            raise ValueError(f"The selected model supports at most {option.max_reference_images} character images per scene.")
        return slug, output_dir, audio_path, option, kind, selected_characters

    def quote_configuration(self, scenes=None):
        """Return the live model settings needed for a preflight quote."""
        self.commit_scene_editor()
        target_scenes = list(self.scenes if scenes is None else scenes)
        if not target_scenes:
            raise ValueError("Add at least one scene before estimating cost.")
        model_data = self.model_combo.currentData()
        if not model_data:
            raise ValueError("Refresh and select a current 30-second video model.")
        option, _kind = model_data
        unsupported = [
            scene.duration_seconds
            for scene in target_scenes
            if f"{scene.duration_seconds}s" not in option.durations
        ]
        if unsupported:
            raise ValueError(f"The selected model does not support {unsupported[0]}-second scenes.")
        resolution = "1080p" if "1080p" in option.resolutions else (option.resolutions[0] if option.resolutions else None)
        aspect_ratio = "16:9" if "16:9" in option.aspect_ratios else (option.aspect_ratios[0] if option.aspect_ratios else None)
        return option, resolution, aspect_ratio

    def estimate_cost(self):
        if self.cost_thread is not None or self.thread is not None:
            return
        pending_scenes = self.scan_existing_scene_clips()
        if not pending_scenes:
            message = "Estimated video generation cost: 0.0000 credits"
            self.append_log(message + " (all scene clips are reusable).")
            QMessageBox.information(
                self,
                "Cost estimate",
                message + "\n\nAll scene clips already exist and passed validation.",
            )
            return
        if not self.ensure_api_key():
            return
        try:
            option, resolution, aspect_ratio = self.quote_configuration(pending_scenes)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot estimate cost", str(exc))
            return
        durations = [scene.duration_seconds for scene in pending_scenes]
        self.append_log(
            f"Requesting estimate for {len(durations)} scenes needing generation with {option.model_id}: "
            + ", ".join(f"{duration}s" for duration in durations)
        )
        self.estimate_button.setDisabled(True)
        self.generate_button.setDisabled(True)
        self.cost_thread = QThread(self)
        self.cost_worker = ProjectCostWorker(
            self.api_key,
            option.model_id,
            durations,
            resolution,
            aspect_ratio,
        )
        self.cost_worker.moveToThread(self.cost_thread)
        self.cost_thread.started.connect(self.cost_worker.run)
        self.cost_worker.finished.connect(self.cost_estimate_finished)
        self.cost_worker.failed.connect(self.cost_estimate_failed)
        self.cost_worker.finished.connect(self.cost_thread.quit)
        self.cost_worker.failed.connect(self.cost_thread.quit)
        self.cost_thread.finished.connect(self.clear_cost_worker)
        self.cost_thread.start()

    def cost_estimate_finished(self, total):
        message = f"Estimated video generation cost: {total:.4f} credits"
        self.append_log(message)
        QMessageBox.information(
            self,
            "Cost estimate",
            message
            + "\n\nVenice quotes video in USD; account credits are valued 1:1 with USD. "
            "This is a live estimate, so the amount charged may differ.",
        )

    def cost_estimate_failed(self, message):
        self.append_log("Cost estimate unavailable: " + message)
        QMessageBox.warning(self, "Cost estimate unavailable", message)

    def clear_cost_worker(self):
        if self.cost_worker:
            self.cost_worker.deleteLater()
        if self.cost_thread:
            self.cost_thread.deleteLater()
        self.cost_worker = None
        self.cost_thread = None
        self.estimate_button.setEnabled(True)
        self.generate_button.setEnabled(True)

    def generate(self):
        if self.thread is not None or self.cost_thread is not None or not self.ensure_api_key():
            return
        self.scan_existing_scene_clips()
        try:
            slug, output_dir, audio_path, option, kind, selected_characters = self.validate_generation()
        except Exception as exc:
            QMessageBox.warning(self, "Cannot start generation", str(exc))
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        final_output = output_dir / f"{slug}.mp4"
        if final_output.exists():
            accepted = QMessageBox.question(
                self,
                "Replace existing final video?",
                "Generation will replace the final project video:\n\n"
                + final_output.name
                + "\n\nExisting valid scene clips will be reused.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            ) == QMessageBox.Yes
            if not accepted:
                return
        save_project(output_dir / f"{slug}.venice-video-project.json", self.project_payload())
        resolution = "1080p" if "1080p" in option.resolutions else (option.resolutions[0] if option.resolutions else None)
        aspect_ratio = "16:9" if "16:9" in option.aspect_ratios else (option.aspect_ratios[0] if option.aspect_ratios else None)
        self.log.clear()
        self.progress.setRange(0, len(self.scenes))
        self.progress.setValue(0)
        self.refresh_scene_list_preserving_row()
        self.set_generating(True)
        self.append_log(f"Model: {option.model_id} ({kind})")
        self.append_log(f"Plan: {len(self.scenes)} scenes, " + ", ".join(f"{scene.duration_seconds}s" for scene in self.scenes))
        self.append_log("Final audio: " + ("MP3 replaces generated audio (stereo)" if self.remove_audio_checkbox.isChecked() else "MP3 mixed with generated audio (stereo)"))
        if kind == "text" and any(character.image_path for character in selected_characters):
            self.append_log("Note: text-to-video model uses character descriptions; reference images are not uploaded.")
        self.thread = QThread(self)
        self.worker = ProjectGenerationWorker(
            self.api_key,
            list(self.scenes),
            list(self.characters),
            self.style_edit.toPlainText().strip(),
            output_dir,
            slug,
            audio_path,
            self.remove_audio_checkbox.isChecked(),
            option,
            kind,
            resolution,
            aspect_ratio,
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.scene_ready.connect(self.scene_clip_ready)
        self.worker.seedance_consent_required.connect(self.handle_consent)
        self.worker.finished.connect(self.generation_finished)
        self.worker.failed.connect(self.generation_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.clear_worker)
        self.thread.start()

    def handle_consent(self, scene_number, message, terms_url, privacy_url):
        detail = message
        if terms_url:
            detail += f"\n\nTerms: {terms_url}"
        if privacy_url:
            detail += f"\nPrivacy: {privacy_url}"
        accepted = QMessageBox.question(
            self,
            f"Consent required for scene {scene_number}",
            detail + "\n\nDo you consent and want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) == QMessageBox.Yes
        if self.worker:
            self.worker.set_seedance_consent_response(accepted)

    def update_progress(self, value, message):
        self.progress.setValue(value)
        self.append_log(message)

    def scene_clip_ready(self, scene_number, path, reused, duration):
        """Record a validated scene clip on the main GUI thread."""
        self.ready_scene_numbers.add(scene_number)
        self.refresh_scene_list_preserving_row()
        action = "Reused" if reused else "Generated"
        self.append_log(f"{action}: {path} ({duration:.2f}s)")

    def generation_finished(self, path):
        self.progress.setValue(len(self.scenes))
        self.append_log(f"Complete: {path}")
        QMessageBox.information(self, "Video complete", f"Saved final video:\n{path}")

    def generation_failed(self, message):
        self.append_log("FAILED: " + message)
        QMessageBox.critical(self, "Generation failed", message)

    def clear_worker(self):
        if self.worker:
            self.worker.deleteLater()
        if self.thread:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None
        self.set_generating(False)

    def set_generating(self, generating):
        self.generate_button.setDisabled(generating)
        self.estimate_button.setDisabled(generating)
        self.save_button.setDisabled(generating)
        self.load_button.setDisabled(generating)

    def append_log(self, message):
        self.log.append(str(message).replace("\n", "<br>"))

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.information(self, "Generation active", "Wait for the active generation to finish before closing.")
            event.ignore()
            return
        if self.model_thread is not None and self.model_thread.isRunning():
            QMessageBox.information(self, "Models loading", "Wait for model loading to finish before closing.")
            event.ignore()
            return
        if self.cost_thread is not None and self.cost_thread.isRunning():
            QMessageBox.information(self, "Estimate active", "Wait for the cost estimate to finish before closing.")
            event.ignore()
            return
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = ProjectWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
