#!/usr/bin/env python3
"""PySide6 front end for writing stories with Venice AI chat models."""

import html
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import markdown
import requests
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QKeySequence, QShortcut, QTextCursor, QTextDocument
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
    "vivid descriptions, strong continuity, and compelling narrative momentum.\n\n"
    "Always return a well-formed Markdown document fragment. Use Markdown headings, "
    "paragraphs, emphasis, block quotes, lists, and horizontal rules when they help "
    "the document. Put a blank line between paragraphs. Do not wrap the response in "
    "a fenced code block unless the user specifically asks for code."
)
MARKDOWN_EXTENSIONS = (
    "extra",
    "sane_lists",
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

    finished = Signal(str, str, object)
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
            choice = data["choices"][0]
            content = choice["message"]["content"].strip()
            finish_reason = choice.get("finish_reason") or "unknown"
            self.finished.emit(content, finish_reason, data.get("usage"))
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
        self.max_tokens_spin.setRange(1024, 32768)
        self.max_tokens_spin.setSingleStep(1024)
        self.max_tokens_spin.setValue(8192)

        self.story_display = QTextBrowser()
        self.story_display.setReadOnly(True)
        self.story_display.setOpenExternalLinks(True)
        self.story_display.setPlaceholderText("The latest response will appear here.")

        self.prompt_input = PromptEdit()
        self.prompt_input.setAcceptRichText(False)
        self.prompt_input.setPlaceholderText("Write the next prompt...")
        self.prompt_input.submitted.connect(self.send_prompt)

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

    def build_layout(self):
        """Assemble the form, editor, and command controls."""
        settings_form = QFormLayout()
        settings_form.addRow("API key", self.api_key_edit)
        settings_form.addRow("Model", self.model_combo)
        settings_form.addRow("Max tokens", self.max_tokens_spin)

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
        self.story_display.clear()
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

    def generation_finished(self, response, finish_reason, usage):
        """Record and display the generated response."""
        timestamp = datetime.now().strftime("%H:%M")
        entry = ChatEntry(timestamp, self.pending_prompt, response)
        self.conversation_history.append(entry)
        self.refresh_response_display(response)
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
            self.conversation_history.clear()
            self.status_label.setText("Story cleared.")

    def save_story(self):
        """Save only generated responses in the selected format."""
        if not self.conversation_history:
            QMessageBox.warning(self, "No story", "No story text to save.")
            return

        file_format = self.save_format_combo.currentData()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save story",
            "",
            self.save_file_filter(file_format),
        )
        if not filename:
            return

        path = self.path_with_format_suffix(Path(filename), file_format)
        path.write_text(
            self.story_content_for_format(file_format),
            encoding="utf-8",
        )
        self.status_label.setText(f"Saved story to {path.name}")

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
        """Return generated responses as one Markdown document."""
        return "\n\n---\n\n".join(entry.response for entry in self.conversation_history)

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

    def save_html(self):
        """Export the rendered conversation as an HTML document."""
        if not self.conversation_history:
            QMessageBox.warning(self, "No chat", "No chat history to export.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export rendered HTML",
            "",
            "HTML files (*.html);;All files (*)",
        )
        if not filename:
            return

        path = Path(filename)
        if path.suffix.lower() not in (".html", ".htm"):
            path = path.with_suffix(".html")

        path.write_text(self.display_html(), encoding="utf-8")
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
