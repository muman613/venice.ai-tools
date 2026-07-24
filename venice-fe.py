#!/usr/bin/env python3
"""PySide6 front end for writing stories with Venice AI chat models."""

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
DEFAULT_MODEL = "venice-uncensored-1-2"
SYSTEM_PROMPT = (
    "You are a creative writing assistant. Help write engaging stories with "
    "vivid descriptions, strong continuity, and compelling narrative momentum."
)
MODELS = (
    "venice-uncensored-1-2",
    "venice-uncensored",
    "zai-org-glm-5-2",
    "zai-org-glm-4.7",
    "olafangensan-glm-4.7-flash-heretic",
    "llama-3.3-70b",
    "qwen-2.5-72b",
    "dolphin-2.9.4-qwen2-72b",
)


@dataclass
class ChatEntry:
    """One visible turn in the story-writing session."""

    timestamp: str
    prompt: str
    response: str


class VeniceChatWorker(QObject):
    """Call Venice AI without blocking the Qt event loop."""

    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, api_key, model, prompt, history, max_tokens):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.prompt = prompt
        self.history = history
        self.max_tokens = max_tokens

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
            self.finished.emit(data["choices"][0]["message"]["content"].strip())
        except (KeyError, IndexError, requests.RequestException, ValueError) as exc:
            self.failed.emit(str(exc))

    def messages(self):
        """Return the system prompt, prior turns, and the new user prompt."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for entry in self.history:
            messages.append({"role": "user", "content": entry.prompt})
            messages.append({"role": "assistant", "content": entry.response})
        messages.append({"role": "user", "content": self.prompt})
        return messages


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
        self.pending_prompt = ""
        self.conversation_history = []

        self.setWindowTitle("Venice AI Story Writer")
        self.resize(980, 760)

        self.api_key_edit = QLineEdit(get_venice_api_key() or "")
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("VENICE_API_KEY")

        self.model_combo = QComboBox()
        self.model_combo.addItems(MODELS)
        self.model_combo.setCurrentText(DEFAULT_MODEL)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(256, 8192)
        self.max_tokens_spin.setSingleStep(256)
        self.max_tokens_spin.setValue(2000)

        self.story_display = QTextBrowser()
        self.story_display.setReadOnly(True)
        self.story_display.setOpenExternalLinks(True)

        self.prompt_input = PromptEdit()
        self.prompt_input.setAcceptRichText(False)
        self.prompt_input.setPlaceholderText("Write the next prompt...")
        self.prompt_input.submitted.connect(self.send_prompt)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_prompt)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_story)
        self.save_story_button = QPushButton("Save Story")
        self.save_story_button.clicked.connect(self.save_story)
        self.save_chat_button = QPushButton("Export Chat")
        self.save_chat_button.clicked.connect(self.save_full_chat)

        self.status_label = QLabel("Ready")
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.build_layout()
        self.apply_style()

        shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        shortcut.activated.connect(self.send_prompt)

    def build_layout(self):
        """Assemble the form, editor, and command controls."""
        settings_form = QFormLayout()
        settings_form.addRow("API key", self.api_key_edit)
        settings_form.addRow("Model", self.model_combo)
        settings_form.addRow("Max tokens", self.max_tokens_spin)

        settings_panel = QWidget()
        settings_panel.setLayout(settings_form)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.story_display)
        splitter.addWidget(self.prompt_input)
        splitter.setSizes([520, 130])
        splitter.setChildrenCollapsible(False)

        buttons = QHBoxLayout()
        buttons.addWidget(self.send_button)
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)
        buttons.addWidget(self.save_story_button)
        buttons.addWidget(self.save_chat_button)

        layout = QVBoxLayout()
        layout.addWidget(settings_panel)
        layout.addWidget(splitter, 1)
        layout.addLayout(buttons)
        layout.addWidget(self.status_label)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

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

        self.pending_prompt = prompt
        self.set_generating(True)
        self.status_label.setText("Generating response...")

        self.thread = QThread(self)
        self.worker = VeniceChatWorker(
            api_key,
            self.model_combo.currentText(),
            prompt,
            list(self.conversation_history),
            self.max_tokens_spin.value(),
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

    def generation_finished(self, response):
        """Record and display the generated response."""
        timestamp = datetime.now().strftime("%H:%M")
        entry = ChatEntry(timestamp, self.pending_prompt, response)
        self.conversation_history.append(entry)
        self.refresh_story_display()
        self.prompt_input.clear()
        self.status_label.setText("Ready")
        self.set_generating(False)

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

    def refresh_story_display(self):
        """Render the conversation as Markdown in the story display."""
        self.story_display.setMarkdown(self.display_markdown())
        cursor = self.story_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.story_display.setTextCursor(cursor)
        self.story_display.ensureCursorVisible()

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

    def clear_story(self):
        """Clear all displayed text and structured chat history."""
        if not self.conversation_history and not self.story_display.toPlainText().strip():
            return
        if QMessageBox.question(self, "Clear story", "Clear all story text?") == QMessageBox.Yes:
            self.story_display.clear()
            self.conversation_history.clear()
            self.status_label.setText("Story cleared.")

    def save_story(self):
        """Save only generated responses."""
        if not self.conversation_history:
            QMessageBox.warning(self, "No story", "No story text to save.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save story",
            "",
            "Text files (*.txt);;Markdown files (*.md);;All files (*)",
        )
        if not filename:
            return

        story_text = "\n\n".join(entry.response for entry in self.conversation_history)
        Path(filename).write_text(story_text, encoding="utf-8")
        self.status_label.setText(f"Saved story to {Path(filename).name}")

    def save_full_chat(self):
        """Save prompts and responses as Markdown."""
        if not self.conversation_history:
            QMessageBox.warning(self, "No chat", "No chat history to export.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export full chat",
            "",
            "Markdown files (*.md);;Text files (*.txt);;All files (*)",
        )
        if not filename:
            return

        lines = [
            "# Story Writing Session",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Model:** {self.model_combo.currentText()}",
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

        Path(filename).write_text("\n".join(lines), encoding="utf-8")
        self.status_label.setText(f"Exported chat to {Path(filename).name}")

    def set_generating(self, generating):
        """Enable or disable controls while a request is running."""
        self.send_button.setEnabled(not generating)
        self.clear_button.setEnabled(not generating)
        self.save_story_button.setEnabled(not generating)
        self.save_chat_button.setEnabled(not generating)
        self.api_key_edit.setEnabled(not generating)
        self.model_combo.setEnabled(not generating)
        self.max_tokens_spin.setEnabled(not generating)
        self.prompt_input.setEnabled(not generating)


def main():
    """Run the PySide6 Venice story writer."""
    app = QApplication(sys.argv)
    window = StoryWriterWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
