#!/usr/bin/env python3
"""PySide6 front end for writing stories with Venice AI chat models."""

import html
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import markdown
import requests
from PySide6.QtCore import QObject, QSettings, Qt, QThread, Signal
from PySide6.QtGui import QKeySequence, QShortcut, QTextCursor, QTextDocument
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from modules.tts import get_venice_api_key


API_URL = "https://api.venice.ai/api/v1/chat/completions"
MODELS_URL = "https://api.venice.ai/api/v1/models"
DEFAULT_MODEL = "venice-uncensored-1-2"
SYSTEM_PROMPT = (
    "Help write engaging stories with vivid descriptions, strong continuity, "
    "and compelling narrative momentum.\n\n"
    "Always return a well-formed Markdown document fragment. Use Markdown "
    "headings, paragraphs, emphasis, block quotes, lists, and horizontal rules "
    "when they help the document. Put a blank line between paragraphs. Do not "
    "wrap the response in a fenced code block unless the user specifically asks "
    "for code."
)
SETTINGS_ORG = "VeniceAI"
SETTINGS_APP = "StoryWriter"
SELECTED_GENRE_SETTING = "story/selected_genre"
SAVE_DIRECTORY_SETTING = "paths/save_directory"
GENRES_FILENAME = "venice-ai-story-writer-genres.json"
DEFAULT_GENRES = (
    {
        "name": "General",
        "prompt": (
            "Write an engaging story that follows the user's requested subject, tone, "
            "setting, and audience."
        ),
    },
    {
        "name": "Erotic fetish",
        "prompt": "Write as an erotic fetish story-writing assistant.",
    },
)
MARKDOWN_EXTENSIONS = (
    "extra",
    "sane_lists",
)
MODELS = (
    "venice-uncensored-1-2",
    "venice-uncensored-role-play",
    "e2ee-venice-uncensored-24b-p",
    "qwen-3-6-plus",
    "e2ee-qwen3-6-35b-a3b-uncensored-p",
    "gemma-4-uncensored",
    "e2ee-gemma-4-26b-a4b-uncensored-p",
    "olafangensan-glm-4.7-flash-heretic",
)


@dataclass
class ChatModelOption:
    """One selectable Venice chat model."""

    model_id: str
    label: str
    context_length: int = 0
    pricing: object = None


def model_text(model):
    """Return searchable model metadata text."""
    return json.dumps(model, ensure_ascii=False, sort_keys=True).lower()


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


def model_context_length(model):
    """Return context length when available."""
    if not isinstance(model, dict):
        return 0
    model_spec = model.get("model_spec")
    spec_context = 0
    if isinstance(model_spec, dict):
        spec_context = int(model_spec.get("availableContextTokens") or 0)
    return int(model.get("context_length") or spec_context or 0)


def model_pricing(model):
    """Return token pricing metadata from a Venice model object."""
    if not isinstance(model, dict):
        return None
    model_spec = model.get("model_spec")
    if isinstance(model_spec, dict) and model_spec.get("pricing"):
        return model_spec["pricing"]
    return model.get("pricing")


def is_uncensored_chat_model(model):
    """Return True when model metadata advertises uncensored text chat behavior."""
    if not isinstance(model, dict):
        return False
    if str(model.get("type", "")).lower() != "text":
        return False
    text = model_text(model)
    return "uncensored" in text or "unrestricted" in text


def chat_model_option(model):
    """Build a combo-box option from model metadata."""
    model_id = model_identifier(model)
    name = model_display_name(model)
    context_length = model_context_length(model)
    label = model_id if not name or name == model_id else f"{name} ({model_id})"
    if context_length:
        label = f"{label} - {context_length:,} ctx"
    return ChatModelOption(model_id, label, context_length, model_pricing(model))


def extract_uncensored_chat_models(data):
    """Extract uncensored chat model options from Venice model metadata."""
    options = [
        chat_model_option(model)
        for model in normalize_model_list(data)
        if is_uncensored_chat_model(model) and model_identifier(model)
    ]
    options.sort(key=lambda option: (option.model_id != DEFAULT_MODEL, option.label.lower()))
    return options


@dataclass
class ChatEntry:
    """One visible turn in the story-writing session."""

    timestamp: str
    prompt: str
    response: str


def normalize_genres(data):
    """Validate and normalize genre data loaded from JSON."""
    genres = data.get("genres") if isinstance(data, dict) else data
    if not isinstance(genres, list):
        raise ValueError("The genre file must contain a 'genres' list.")

    normalized = []
    names = set()
    for index, genre in enumerate(genres, start=1):
        if not isinstance(genre, dict):
            raise ValueError(f"Genre {index} must be a JSON object.")
        name = str(genre.get("name", "")).strip()
        prompt = str(genre.get("prompt", "")).strip()
        if not name or not prompt:
            raise ValueError(f"Genre {index} must have a non-empty name and prompt.")
        name_key = name.casefold()
        if name_key in names:
            raise ValueError(f"Duplicate genre name: {name}")
        names.add(name_key)
        normalized.append({"name": name, "prompt": prompt})
    return normalized


def load_genres(path):
    """Load genres from disk, creating the default file on first use."""
    path = Path(path)
    if not path.exists():
        genres = [dict(genre) for genre in DEFAULT_GENRES]
        save_genres(path, genres)
        return genres
    return normalize_genres(json.loads(path.read_text(encoding="utf-8")))


def save_genres(path, genres):
    """Validate and save genres as readable JSON."""
    path = Path(path)
    normalized = normalize_genres(genres)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"genres": normalized}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_system_prompt(genre):
    """Append the selected genre instructions to the stock system prompt."""
    name = str(genre.get("name", "")).strip()
    prompt = str(genre.get("prompt", "")).strip()
    if not name or not prompt:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\n\n## Genre: {name}\n\n{prompt}"


def build_chat_messages(system_prompt, history, prompt):
    """Return the messages included in a story-generation request."""
    messages = [{"role": "system", "content": system_prompt}]
    for entry in history:
        messages.append({"role": "user", "content": entry.prompt})
        messages.append({"role": "assistant", "content": entry.response})
    messages.append({"role": "user", "content": prompt})
    return messages


def estimate_message_tokens(messages):
    """Estimate chat input tokens without requiring a model-specific tokenizer."""
    token_count = 2
    for message in messages:
        content = str(message.get("content", ""))
        token_count += 4 + math.ceil(len(content.encode("utf-8")) / 4)
    return token_count


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


def text_token_prices(pricing):
    """Return input and output USD prices per million tokens."""
    if not isinstance(pricing, dict):
        return None, None
    return usd_amount(pricing.get("input")), usd_amount(pricing.get("output"))


def text_generation_cost(pricing, input_tokens, max_output_tokens):
    """Estimate input cost and maximum completion cost in USD."""
    input_price, output_price = text_token_prices(pricing)
    if input_price is None or output_price is None:
        return {
            "available": False,
            "display": "unavailable (refresh models to load pricing)",
        }

    input_cost = input_tokens * input_price / 1_000_000
    output_cost = max_output_tokens * output_price / 1_000_000
    total = input_cost + output_cost
    return {
        "available": True,
        "input_tokens": input_tokens,
        "max_output_tokens": max_output_tokens,
        "input_price": input_price,
        "output_price": output_price,
        "input_cost": input_cost,
        "max_output_cost": output_cost,
        "total": total,
        "display": (
            f"≈${total:.4f} maximum (≈{input_tokens:,} input + up to "
            f"{max_output_tokens:,} output tokens; ${input_price:g}/${output_price:g} per 1M)"
        ),
    }


class GenreEditorDialog(QDialog):
    """Create, update, and delete persisted story genres."""

    genres_changed = Signal(object)

    def __init__(self, genres, genres_path, parent=None):
        super().__init__(parent)
        self.genres = [dict(genre) for genre in genres]
        self.genres_path = Path(genres_path)
        self.editing_index = None

        self.setWindowTitle("Genre Editor")
        self.resize(700, 440)

        self.genre_list = QListWidget()
        self.genre_list.currentRowChanged.connect(self.genre_selected)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Genre name")
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setAcceptRichText(False)
        self.prompt_edit.setPlaceholderText("Instructions added to the system prompt for this genre...")

        self.new_button = QPushButton("New")
        self.new_button.clicked.connect(self.new_genre)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_genre)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_genre)

        editor_form = QFormLayout()
        editor_form.addRow("Name", self.name_edit)
        editor_form.addRow("Prompt", self.prompt_edit)
        editor_panel = QWidget()
        editor_panel.setLayout(editor_form)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.genre_list)
        splitter.addWidget(editor_panel)
        splitter.setSizes([210, 470])

        actions = QHBoxLayout()
        actions.addWidget(self.new_button)
        actions.addWidget(self.save_button)
        actions.addWidget(self.delete_button)
        actions.addStretch(1)
        close_buttons = QDialogButtonBox(QDialogButtonBox.Close)
        close_buttons.rejected.connect(self.reject)
        actions.addWidget(close_buttons)

        layout = QVBoxLayout()
        layout.addWidget(splitter, 1)
        layout.addLayout(actions)
        self.setLayout(layout)

        self.refresh_list()
        if self.genres:
            self.genre_list.setCurrentRow(0)

    def refresh_list(self, selected_index=None):
        """Rebuild the genre list and optionally restore its selection."""
        self.genre_list.blockSignals(True)
        self.genre_list.clear()
        self.genre_list.addItems(genre["name"] for genre in self.genres)
        self.genre_list.blockSignals(False)
        if selected_index is not None and self.genres:
            self.genre_list.setCurrentRow(min(selected_index, len(self.genres) - 1))

    def genre_selected(self, index):
        """Load the selected genre into the editing controls."""
        if index < 0 or index >= len(self.genres):
            return
        self.editing_index = index
        self.name_edit.setText(self.genres[index]["name"])
        self.prompt_edit.setPlainText(self.genres[index]["prompt"])
        self.delete_button.setEnabled(True)

    def new_genre(self):
        """Clear the form so Save creates a new genre."""
        self.genre_list.clearSelection()
        self.genre_list.setCurrentRow(-1)
        self.editing_index = None
        self.name_edit.clear()
        self.prompt_edit.clear()
        self.delete_button.setEnabled(False)
        self.name_edit.setFocus()

    def save_genre(self):
        """Add a new genre or update the selected one."""
        name = self.name_edit.text().strip()
        prompt = self.prompt_edit.toPlainText().strip()
        if not name or not prompt:
            QMessageBox.warning(self, "Incomplete genre", "Enter both a genre name and prompt.")
            return

        for index, genre in enumerate(self.genres):
            if index != self.editing_index and genre["name"].casefold() == name.casefold():
                QMessageBox.warning(self, "Duplicate genre", f'A genre named "{name}" already exists.')
                return

        previous_genres = [dict(genre) for genre in self.genres]
        genre = {"name": name, "prompt": prompt}
        if self.editing_index is None:
            self.genres.append(genre)
            selected_index = len(self.genres) - 1
        else:
            self.genres[self.editing_index] = genre
            selected_index = self.editing_index

        if not self.persist_changes():
            self.genres = previous_genres
            return
        self.refresh_list(selected_index)

    def delete_genre(self):
        """Delete the selected genre after confirmation."""
        if self.editing_index is None:
            return
        genre = self.genres[self.editing_index]
        if QMessageBox.question(
            self,
            "Delete genre",
            f'Delete the "{genre["name"]}" genre?',
        ) != QMessageBox.Yes:
            return

        deleted_index = self.editing_index
        del self.genres[deleted_index]
        if not self.persist_changes():
            self.genres.insert(deleted_index, genre)
            return
        self.editing_index = None
        self.name_edit.clear()
        self.prompt_edit.clear()
        self.refresh_list(min(deleted_index, len(self.genres) - 1) if self.genres else None)
        self.delete_button.setEnabled(bool(self.genres))

    def persist_changes(self):
        """Write the current genre collection and notify the main window."""
        try:
            save_genres(self.genres_path, self.genres)
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, "Could not save genres", str(exc))
            return False
        self.genres_changed.emit([dict(genre) for genre in self.genres])
        return True


class VeniceChatWorker(QObject):
    """Call Venice AI without blocking the Qt event loop."""

    finished = Signal(str, str, object)
    failed = Signal(str)

    def __init__(self, api_key, model, prompt, history, max_tokens, system_prompt=SYSTEM_PROMPT):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.prompt = prompt
        self.history = history
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def run(self):
        """Send a chat completion request and emit the generated response."""
        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": self.messages(),
                    "temperature": 0.8,
                    "max_tokens": self.max_tokens,
                },
                timeout=90,
            )
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            content = choice["message"]["content"].strip()
            finish_reason = choice.get("finish_reason") or "unknown"
            self.finished.emit(content, finish_reason, data.get("usage"))
        except (KeyError, IndexError, requests.RequestException, ValueError) as exc:
            self.failed.emit(str(exc))

    def messages(self):
        """Return the system prompt, prior turns, and the new user prompt."""
        return build_chat_messages(self.system_prompt, self.history, self.prompt)


class VeniceChatModelsWorker(QObject):
    """Load uncensored Venice chat models without blocking the Qt event loop."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key

    def run(self):
        """Fetch current text models and return uncensored options."""
        try:
            response = requests.get(
                MODELS_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                },
                params={"type": "text"},
                timeout=60,
            )
            response.raise_for_status()
            options = extract_uncensored_chat_models(response.json())
            if not options:
                raise RuntimeError("No uncensored chat completion models were returned by the models API.")
            self.finished.emit(options)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class PromptEdit(QTextEdit):
    """Text edit that submits with Ctrl+Enter."""

    submitted = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and event.modifiers() & Qt.ControlModifier:
            self.submitted.emit()
            return
        super().keyPressEvent(event)


class StoryWriterWindow(QMainWindow):
    """Main window for Venice AI story writing."""

    def __init__(self):
        super().__init__()
        self.thread = None
        self.worker = None
        self.models_thread = None
        self.models_worker = None
        self.models_loaded_once = False
        self.pending_send_after_models = False
        self.pending_prompt = ""
        self.current_entry = None
        self.conversation_history = []
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self.genres_path = Path(self.settings.fileName()).parent / GENRES_FILENAME
        try:
            self.genres = load_genres(self.genres_path)
            self.genres_load_error = ""
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            self.genres = [dict(genre) for genre in DEFAULT_GENRES]
            self.genres_load_error = str(exc)

        self.setWindowTitle("Venice AI Story Writer")
        self.resize(980, 760)

        self.api_key_edit = QLineEdit(get_venice_api_key() or "")
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("VENICE_API_KEY")

        self.model_combo = QComboBox()
        self.model_options = {}
        for model_id in MODELS:
            self.model_options[model_id] = ChatModelOption(model_id, model_id)
            self.model_combo.addItem(model_id, model_id)
        self.model_combo.setCurrentText(DEFAULT_MODEL)
        self.model_combo.currentIndexChanged.connect(self.update_cost_estimate)

        self.refresh_models_button = QPushButton("Refresh Models")
        self.refresh_models_button.clicked.connect(self.refresh_models)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(1024, 32768)
        self.max_tokens_spin.setSingleStep(1024)
        self.max_tokens_spin.setValue(8192)
        self.max_tokens_spin.valueChanged.connect(self.update_cost_estimate)

        self.genre_combo = QComboBox()
        self.populate_genre_combo()
        self.genre_combo.currentIndexChanged.connect(self.genre_changed)
        self.genre_combo.currentIndexChanged.connect(self.update_cost_estimate)
        self.genre_editor_button = QPushButton("Edit Genres")
        self.genre_editor_button.clicked.connect(self.open_genre_editor)

        self.cost_label = QLabel()
        self.cost_label.setWordWrap(True)
        self.cost_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.story_display = QTextBrowser()
        self.story_display.setReadOnly(True)
        self.story_display.setOpenExternalLinks(True)
        self.story_display.setPlaceholderText("The latest response will appear here.")

        self.prompt_input = PromptEdit()
        self.prompt_input.setAcceptRichText(False)
        self.prompt_input.setPlaceholderText("Write the next prompt...")
        self.prompt_input.submitted.connect(self.send_prompt)
        self.prompt_input.textChanged.connect(self.update_cost_estimate)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_prompt)
        self.clear_prompt_button = QPushButton("Clear")
        self.clear_prompt_button.clicked.connect(self.clear_prompt)
        self.clear_story_button = QPushButton("Clear Chat")
        self.clear_story_button.clicked.connect(self.clear_story)
        self.save_story_button = QPushButton("Save Story")
        self.save_story_button.clicked.connect(self.save_story)
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItem("TXT", "txt")
        self.save_format_combo.addItem("MD", "md")
        self.save_format_combo.addItem("HTML", "html")
        self.save_format_combo.setCurrentText("MD")
        self.save_chat_button = QPushButton("Export Chat")
        self.save_chat_button.clicked.connect(self.save_full_chat)
        self.save_html_button = QPushButton("Export HTML")
        self.save_html_button.clicked.connect(self.save_html)

        self.status_label = QLabel("Ready")
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.build_layout()
        self.apply_style()

        shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        shortcut.activated.connect(self.send_prompt)
        self.update_cost_estimate()

        if self.genres_load_error:
            QMessageBox.warning(
                self,
                "Could not load genres",
                f"The genre file could not be loaded. Built-in defaults will be used.\n\n"
                f"{self.genres_path}\n\n{self.genres_load_error}",
            )

        if self.api_key_edit.text().strip():
            self.refresh_models()

    def build_layout(self):
        """Assemble the form, editor, and command controls."""
        settings_form = QFormLayout()
        settings_form.addRow("API key", self.api_key_edit)
        settings_form.addRow("Model", self.row(self.model_combo, self.refresh_models_button))
        settings_form.addRow("Max tokens", self.max_tokens_spin)
        settings_form.addRow("Estimated cost", self.cost_label)
        settings_form.addRow("Genre", self.row(self.genre_combo, self.genre_editor_button))

        settings_panel = QWidget()
        settings_panel.setLayout(settings_form)

        splitter = QSplitter(Qt.Vertical)
        request_panel = QWidget()
        request_layout = QVBoxLayout()
        request_layout.setContentsMargins(0, 0, 0, 0)
        request_layout.addWidget(QLabel("Request"))
        request_layout.addWidget(self.prompt_input)
        request_panel.setLayout(request_layout)

        response_panel = QWidget()
        response_layout = QVBoxLayout()
        response_layout.setContentsMargins(0, 0, 0, 0)
        response_layout.addWidget(QLabel("Response"))
        response_layout.addWidget(self.story_display)
        response_panel.setLayout(response_layout)

        splitter.addWidget(request_panel)
        splitter.addWidget(response_panel)
        splitter.setSizes([190, 460])
        splitter.setChildrenCollapsible(False)

        buttons = QHBoxLayout()
        buttons.addWidget(self.send_button)
        buttons.addWidget(self.clear_prompt_button)
        buttons.addWidget(self.clear_story_button)
        buttons.addStretch(1)
        buttons.addWidget(QLabel("Save as"))
        buttons.addWidget(self.save_format_combo)
        buttons.addWidget(self.save_story_button)
        buttons.addWidget(self.save_chat_button)
        buttons.addWidget(self.save_html_button)

        layout = QVBoxLayout()
        layout.addWidget(settings_panel)
        layout.addWidget(splitter, 1)
        layout.addLayout(buttons)
        layout.addWidget(self.status_label)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    @staticmethod
    def row(*widgets):
        """Return a horizontal row widget for form controls."""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            layout.addWidget(widget)
        container = QWidget()
        container.setLayout(layout)
        return container

    def apply_style(self):
        """Apply a compact dark Qt stylesheet."""
        self.setStyleSheet(
            """
            QMainWindow,
            QWidget {
                background: #202124;
                color: #f1f3f4;
                font-size: 14px;
            }
            QLineEdit,
            QTextBrowser,
            QTextEdit,
            QComboBox,
            QSpinBox {
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
            QSplitter::handle {
                background: #3c4043;
                height: 4px;
            }
            """
        )

    def selected_model(self):
        """Return the selected Venice model ID."""
        data = self.model_combo.currentData()
        if data:
            return str(data).strip()
        return self.model_combo.currentText().strip()

    def selected_model_option(self):
        """Return metadata for the selected model when it has been loaded."""
        model_id = self.selected_model()
        return self.model_options.get(model_id, ChatModelOption(model_id, model_id))

    def estimated_request_messages(self):
        """Build the messages used by the live token and cost estimate."""
        return build_chat_messages(
            build_system_prompt(self.selected_genre()),
            self.conversation_history,
            self.prompt_input.toPlainText().strip(),
        )

    def update_cost_estimate(self, *_args):
        """Show an estimated maximum cost for the next request."""
        if not hasattr(self, "cost_label") or not hasattr(self, "prompt_input"):
            return
        input_tokens = estimate_message_tokens(self.estimated_request_messages())
        option = self.selected_model_option()
        estimate = text_generation_cost(
            option.pricing,
            input_tokens,
            self.max_tokens_spin.value(),
        )
        self.cost_label.setText(estimate["display"])

    def populate_genre_combo(self, preferred_name=None):
        """Fill the genre selector while preserving a named selection."""
        selected_name = preferred_name
        if selected_name is None and hasattr(self, "genre_combo"):
            selected_name = self.genre_combo.currentData()
        if not selected_name:
            selected_name = self.settings.value(SELECTED_GENRE_SETTING, "General", str)

        self.genre_combo.blockSignals(True)
        self.genre_combo.clear()
        for genre in self.genres:
            self.genre_combo.addItem(genre["name"], genre["name"])
        index = self.genre_combo.findData(selected_name)
        if index < 0 and self.genre_combo.count():
            index = 0
        self.genre_combo.setCurrentIndex(index)
        self.genre_combo.blockSignals(False)
        if index >= 0:
            self.genre_changed(index)

    def genre_changed(self, _index):
        """Remember the selected genre."""
        name = self.genre_combo.currentData()
        if name:
            self.settings.setValue(SELECTED_GENRE_SETTING, name)

    def selected_genre(self):
        """Return the complete selected genre record."""
        selected_name = self.genre_combo.currentData()
        for genre in self.genres:
            if genre["name"] == selected_name:
                return genre
        return {}

    def open_genre_editor(self):
        """Open the modal editor for persisted genre instructions."""
        dialog = GenreEditorDialog(self.genres, self.genres_path, self)
        dialog.genres_changed.connect(self.genres_updated)
        dialog.exec()

    def genres_updated(self, genres):
        """Apply genre changes from the editor to the selector."""
        selected_name = self.genre_combo.currentData()
        self.genres = [dict(genre) for genre in genres]
        self.populate_genre_combo(selected_name)
        self.update_cost_estimate()
        self.status_label.setText(f"Saved {len(self.genres)} genres to {self.genres_path.name}.")

    def refresh_models(self):
        """Refresh uncensored chat models from Venice."""
        if self.models_thread:
            return

        api_key = self.api_key_edit.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Missing API key", "Enter your Venice AI API key first.")
            return

        self.status_label.setText("Loading uncensored chat models...")
        self.refresh_models_button.setEnabled(False)

        self.models_thread = QThread(self)
        self.models_worker = VeniceChatModelsWorker(api_key)
        self.models_worker.moveToThread(self.models_thread)
        self.models_thread.started.connect(self.models_worker.run)
        self.models_worker.finished.connect(self.models_loaded)
        self.models_worker.failed.connect(self.models_failed)
        self.models_worker.finished.connect(self.models_thread.quit)
        self.models_worker.failed.connect(self.models_thread.quit)
        self.models_thread.finished.connect(self.models_worker.deleteLater)
        self.models_thread.finished.connect(self.models_thread.deleteLater)
        self.models_thread.finished.connect(self.clear_models_worker)
        self.models_thread.start()

    def models_loaded(self, options):
        """Populate the model selector with uncensored chat models."""
        current_model = self.selected_model()
        self.models_loaded_once = True
        self.model_options = {option.model_id: option for option in options}
        self.model_combo.clear()
        for option in options:
            self.model_combo.addItem(option.label, option.model_id)

        index = self.model_combo.findData(current_model)
        if index < 0:
            index = self.model_combo.findData(DEFAULT_MODEL)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)

        self.update_cost_estimate()
        self.refresh_models_button.setEnabled(True)
        self.status_label.setText(f"Loaded {len(options)} uncensored chat models.")
        if self.pending_send_after_models:
            self.pending_send_after_models = False
            self.send_prompt()

    def models_failed(self, message):
        """Display model refresh errors without clearing fallback models."""
        self.pending_send_after_models = False
        self.refresh_models_button.setEnabled(True)
        self.status_label.setText("Model refresh failed.")
        QMessageBox.critical(self, "Model refresh failed", message)

    def clear_models_worker(self):
        """Clear finished model worker references."""
        self.models_thread = None
        self.models_worker = None

    def send_prompt(self):
        """Validate input and start a background Venice request."""
        prompt = self.prompt_input.toPlainText().strip()
        api_key = self.api_key_edit.text().strip()

        if self.thread:
            return
        if not prompt:
            self.status_label.setText("Enter a prompt first.")
            return
        if not api_key:
            QMessageBox.warning(self, "Missing API key", "Enter your Venice AI API key first.")
            return

        if self.selected_model_option().pricing is None and not self.models_loaded_once:
            self.pending_send_after_models = True
            if not self.models_thread:
                self.refresh_models()
            self.status_label.setText("Loading model pricing before sending...")
            return

        self.update_cost_estimate()
        self.pending_prompt = prompt
        self.story_display.clear()
        self.set_generating(True)
        self.status_label.setText("Generating response...")

        self.thread = QThread(self)
        self.worker = VeniceChatWorker(
            api_key,
            self.selected_model(),
            prompt,
            list(self.conversation_history),
            self.max_tokens_spin.value(),
            build_system_prompt(self.selected_genre()),
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.generation_finished)
        self.worker.failed.connect(self.generation_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.clear_worker)
        self.thread.start()

    def generation_finished(self, response, finish_reason, usage):
        """Record and display the generated response."""
        timestamp = datetime.now().strftime("%H:%M")
        entry = ChatEntry(timestamp, self.pending_prompt, response)
        self.current_entry = entry
        self.conversation_history.append(entry)
        self.refresh_response_display(response)
        self.update_cost_estimate()
        self.status_label.setText(self.response_status(response, finish_reason, usage))
        self.set_generating(False)

    def response_status(self, response, finish_reason, usage):
        """Return a status message that makes length-limited responses obvious."""
        token_text = ""
        if isinstance(usage, dict):
            completion_tokens = usage.get("completion_tokens")
            if completion_tokens is not None:
                token_text = f", {completion_tokens:,} output tokens"

        if finish_reason == "length":
            return (
                f"Response stopped at the max token limit "
                f"({len(response):,} chars{token_text}). Increase Max tokens or ask it to continue."
            )

        return f"Ready ({len(response):,} chars{token_text})"

    def generation_failed(self, message):
        """Display an API or parsing error."""
        self.status_label.setText("Generation failed.")
        self.set_generating(False)
        QMessageBox.critical(self, "API error", f"Failed to get a response:\n{message}")

    def clear_worker(self):
        """Clear finished worker references."""
        self.thread = None
        self.worker = None
        self.pending_prompt = ""

    def refresh_response_display(self, response):
        """Render the latest response as styled HTML in the response display."""
        self.story_display.setHtml(self.response_html(response))
        cursor = self.story_display.textCursor()
        cursor.movePosition(QTextCursor.Start)
        self.story_display.setTextCursor(cursor)
        self.story_display.ensureCursorVisible()

    def response_html(self, response):
        """Return a styled HTML document for the latest model response."""
        body = self.markdown_html(response) or "<p class=\"empty\">No response yet.</p>"

        return "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                "<meta charset=\"utf-8\">",
                "<style>",
                self.document_css(),
                "</style>",
                "</head>",
                "<body>",
                "<article class=\"response\">",
                body,
                "</article>",
                "</body>",
                "</html>",
            ]
        )

    def display_html(self):
        """Return a styled HTML document for the visible conversation."""
        turns = []
        for index, entry in enumerate(self.conversation_history, start=1):
            turns.append(self.entry_html(index, entry))

        body = "\n".join(turns) or "<p class=\"empty\">No story yet.</p>"

        return "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                "<meta charset=\"utf-8\">",
                "<style>",
                self.document_css(),
                "</style>",
                "</head>",
                "<body>",
                body,
                "</body>",
                "</html>",
            ]
        )

    def entry_html(self, index, entry):
        """Return the rendered HTML for one prompt and response pair."""
        prompt_html = self.plain_text_html(entry.prompt)
        response_html = self.markdown_html(entry.response)
        return "\n".join(
            [
                "<section class=\"turn\">",
                f"<div class=\"turn-meta\">Turn {index} - {html.escape(entry.timestamp)}</div>",
                "<div class=\"prompt\">",
                "<div class=\"prompt-label\">Prompt</div>",
                f"<blockquote>{prompt_html}</blockquote>",
                "</div>",
                "<article class=\"response\">",
                response_html,
                "</article>",
                "</section>",
            ]
        )

    def display_markdown(self):
        """Return Markdown with prompts separated from model responses."""
        turns = []
        for entry in self.conversation_history:
            turns.append(
                "\n".join(
                    [
                        f"*[{entry.timestamp}]*",
                        "",
                        "**Prompt**",
                        "",
                        self.blockquote_text(entry.prompt),
                        "",
                        entry.response,
                    ]
                )
            )
        return "\n\n---\n\n".join(turns)

    @staticmethod
    def blockquote_text(text):
        """Render user prompts as Markdown block quotes."""
        return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())

    @staticmethod
    def markdown_html(text):
        """Convert model Markdown into HTML."""
        return markdown.markdown(
            text,
            extensions=MARKDOWN_EXTENSIONS,
            output_format="html5",
        )

    @classmethod
    def markdown_plain_text(cls, text):
        """Convert Markdown to plain text through the same rendered HTML path."""
        document = QTextDocument()
        document.setHtml(cls.markdown_html(text))
        return document.toPlainText()

    @staticmethod
    def plain_text_html(text):
        """Escape plain text while preserving line breaks."""
        return "<br>\n".join(html.escape(line) for line in text.splitlines())

    @staticmethod
    def document_css():
        """Return CSS for readable generated documents inside QTextBrowser."""
        return """
            body {
                background: #2f3136;
                color: #f1f3f4;
                font-family: system-ui, sans-serif;
                font-size: 15px;
                line-height: 1.55;
                margin: 18px;
            }
            .empty {
                color: #aeb4bd;
            }
            .turn {
                border-bottom: 1px solid #4b4f58;
                margin-bottom: 24px;
                padding-bottom: 18px;
            }
            .turn-meta {
                color: #aeb4bd;
                font-size: 12px;
                margin-bottom: 10px;
                text-transform: uppercase;
            }
            .prompt {
                margin-bottom: 18px;
            }
            .prompt-label {
                color: #c9d1d9;
                font-weight: 600;
            }
            blockquote {
                border-left: 3px solid #6ea8fe;
                color: #c9d1d9;
                margin: 10px 0 0;
                padding-left: 12px;
            }
            .response h1,
            .response h2,
            .response h3 {
                color: #ffffff;
                font-weight: 700;
                line-height: 1.2;
                margin: 18px 0 10px;
            }
            .response h1 {
                font-size: 26px;
            }
            .response h2 {
                font-size: 22px;
            }
            .response h3 {
                font-size: 18px;
            }
            .response p,
            .response ul,
            .response ol {
                margin: 0 0 14px;
            }
            .response li {
                margin-bottom: 5px;
            }
            .response hr {
                border: 0;
                border-top: 1px solid #5f6368;
                margin: 20px 0;
            }
            .response code {
                background: #202124;
                border-radius: 3px;
                color: #f8d866;
                padding: 1px 4px;
            }
            .response pre {
                background: #202124;
                border: 1px solid #4b4f58;
                border-radius: 4px;
                color: #f1f3f4;
                padding: 12px;
            }
            .response table {
                border-collapse: collapse;
                margin: 12px 0;
            }
            .response th,
            .response td {
                border: 1px solid #5f6368;
                padding: 6px 8px;
            }
            .response th {
                background: #3c4043;
            }
        """

    def clear_prompt(self):
        """Clear only the request editor, preserving response and chat history."""
        if not self.prompt_input.toPlainText().strip():
            return
        self.prompt_input.clear()
        self.status_label.setText("Prompt cleared.")

    def clear_story(self):
        """Clear all displayed text and structured chat history."""
        if not self.conversation_history and not self.story_display.toPlainText().strip():
            return
        if QMessageBox.question(self, "Clear story", "Clear all story text?") == QMessageBox.Yes:
            self.story_display.clear()
            self.current_entry = None
            self.conversation_history.clear()
            self.update_cost_estimate()
            self.status_label.setText("Story cleared.")

    def save_story(self):
        """Save only the latest generated response in the selected format."""
        if not self.current_entry:
            QMessageBox.warning(self, "No story", "No story text to save.")
            return

        file_format = self.save_format_combo.currentData()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save story",
            str(self.last_save_directory()),
            self.save_file_filter(file_format),
        )
        if not filename:
            return

        path = self.path_with_format_suffix(Path(filename), file_format)
        path.write_text(
            self.story_content_for_format(file_format),
            encoding="utf-8",
        )
        self.remember_save_directory(path)
        self.status_label.setText(f"Saved story to {path.name}")

    def last_save_directory(self):
        """Return the persisted save directory, falling back to the working directory."""
        saved_directory = self.settings.value(SAVE_DIRECTORY_SETTING, "", str).strip()
        if saved_directory:
            path = Path(saved_directory).expanduser()
            if path.is_dir():
                return path
        return Path.cwd()

    def remember_save_directory(self, file_path):
        """Persist the parent directory of a successfully saved file."""
        directory = Path(file_path).expanduser().resolve().parent
        self.settings.setValue(SAVE_DIRECTORY_SETTING, str(directory))
        self.settings.sync()

    @staticmethod
    def save_file_filter(file_format):
        """Return a QFileDialog filter for the selected story save format."""
        filters = {
            "txt": "Text files (*.txt);;All files (*)",
            "md": "Markdown files (*.md);;All files (*)",
            "html": "HTML files (*.html);;All files (*)",
        }
        return filters.get(file_format, filters["md"])

    @staticmethod
    def path_with_format_suffix(path, file_format):
        """Apply a suffix matching the selected save format."""
        suffixes = {
            "txt": ".txt",
            "md": ".md",
            "html": ".html",
        }
        expected_suffix = suffixes.get(file_format, ".md")
        if file_format == "html" and path.suffix.lower() == ".htm":
            return path
        if path.suffix.lower() != expected_suffix:
            return path.with_suffix(expected_suffix)
        return path

    def story_markdown(self):
        """Return the latest generated response as Markdown."""
        return self.current_entry.response if self.current_entry else ""

    def story_html(self):
        """Return generated responses as a styled HTML document."""
        body = self.markdown_html(self.story_markdown())
        return "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                "<meta charset=\"utf-8\">",
                "<style>",
                self.document_css(),
                "</style>",
                "</head>",
                "<body>",
                "<article class=\"response\">",
                body,
                "</article>",
                "</body>",
                "</html>",
            ]
        )

    def story_plain_text(self):
        """Return generated responses as plain text with Markdown syntax removed."""
        return self.markdown_plain_text(self.story_markdown())

    def story_content_for_format(self, file_format):
        """Return story content converted for the selected save format."""
        if file_format == "txt":
            return self.story_plain_text()
        if file_format == "html":
            return self.story_html()
        return self.story_markdown()

    def save_full_chat(self):
        """Save prompts and responses as Markdown."""
        if not self.conversation_history:
            QMessageBox.warning(self, "No chat", "No chat history to export.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export full chat",
            str(self.last_save_directory()),
            "Markdown files (*.md);;Text files (*.txt);;All files (*)",
        )
        if not filename:
            return

        lines = [
            "# Story Writing Session",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Model:** {self.selected_model()}",
            "",
            "---",
            "",
        ]
        for index, entry in enumerate(self.conversation_history, start=1):
            lines.extend(
                [
                    f"### Prompt {index} ({entry.timestamp})",
                    "",
                    entry.prompt,
                    "",
                    "**Response:**",
                    "",
                    entry.response,
                    "",
                    "---",
                    "",
                ]
            )

        path = Path(filename)
        path.write_text("\n".join(lines), encoding="utf-8")
        self.remember_save_directory(path)
        self.status_label.setText(f"Exported chat to {path.name}")

    def save_html(self):
        """Export the latest rendered response as an HTML document."""
        if not self.current_entry:
            QMessageBox.warning(self, "No story", "No story text to export.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export rendered HTML",
            str(self.last_save_directory()),
            "HTML files (*.html);;All files (*)",
        )
        if not filename:
            return

        path = Path(filename)
        if path.suffix.lower() not in (".html", ".htm"):
            path = path.with_suffix(".html")

        path.write_text(self.response_html(self.current_entry.response), encoding="utf-8")
        self.remember_save_directory(path)
        self.status_label.setText(f"Exported HTML to {path.name}")

    def set_generating(self, generating):
        """Enable or disable controls while a request is running."""
        self.send_button.setEnabled(not generating)
        self.clear_prompt_button.setEnabled(not generating)
        self.clear_story_button.setEnabled(not generating)
        self.save_story_button.setEnabled(not generating)
        self.save_format_combo.setEnabled(not generating)
        self.save_chat_button.setEnabled(not generating)
        self.save_html_button.setEnabled(not generating)
        self.api_key_edit.setEnabled(not generating)
        self.model_combo.setEnabled(not generating)
        self.refresh_models_button.setEnabled(not generating and self.models_thread is None)
        self.max_tokens_spin.setEnabled(not generating)
        self.genre_combo.setEnabled(not generating)
        self.genre_editor_button.setEnabled(not generating)
        self.prompt_input.setEnabled(not generating)

    def closeEvent(self, event):
        """Prevent closing while background work is active."""
        if self.thread:
            QMessageBox.warning(
                self,
                "Generation in progress",
                "Wait for the current response to finish before closing.",
            )
            event.ignore()
            return
        if self.models_thread:
            QMessageBox.warning(
                self,
                "Model refresh in progress",
                "Wait for model refresh to finish before closing.",
            )
            event.ignore()
            return
        self.settings.sync()
        super().closeEvent(event)


def main():
    """Run the PySide6 Venice story writer."""
    app = QApplication(sys.argv)
    window = StoryWriterWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
