#!/usr/bin/env python3
"""Qt GUI for converting text files to Venice AI speech MP3 files."""

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, QThread, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
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
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from modules.html_text_extractor import extract_text
from modules.tts import VALID_VOICES, TextToSpeechError, text_file_to_mp3


ORGANIZATION_NAME = "venice-ai-tools"
APPLICATION_NAME = "venice-ai-tts"
SOURCE_DIR_SETTING = "paths/source_dir"
OUTPUT_DIR_SETTING = "paths/output_dir"


class TtsWorker(QObject):
    """Run TTS conversion off the UI thread."""

    progress_changed = Signal(int, int, str)
    failed = Signal(str)
    finished = Signal(str)

    def __init__(self, source_file, output_dir, voice_id):
        super().__init__()
        self.source_file = Path(source_file)
        self.output_dir = Path(output_dir)
        self.voice_id = voice_id

    def run(self):
        """Convert the selected file and report completion."""
        try:
            text_file, output_file = prepare_source_text(
                self.source_file,
                self.output_dir,
                self.progress_changed.emit,
            )
            output_path = text_file_to_mp3(
                text_file,
                output_file,
                self.voice_id,
                progress_callback=self.progress_changed.emit,
            )
        except (OSError, subprocess.SubprocessError, TextToSpeechError) as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(str(output_path))


def prepare_source_text(source_file, output_dir, progress_callback=None):
    """Return the text file to convert and the final MP3 output path."""
    if not source_file.is_file():
        raise TextToSpeechError(f"Input file does not exist: {source_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{source_file.stem}.mp3"
    suffix = source_file.suffix.lower()

    if suffix in {".html", ".htm"}:
        editable_text_file = output_dir / f"{source_file.stem}.txt"
        if progress_callback:
            progress_callback(0, 1, f"Extracting text from {source_file.name}")
        editable_text_file.write_text(extract_text(source_file), encoding="utf-8")
        if progress_callback:
            progress_callback(0, 1, f"Edit and save {editable_text_file.name} in gedit")
        subprocess.run(["gedit", "--wait", str(editable_text_file)], check=True)
        return editable_text_file, output_file

    if suffix == ".txt":
        return source_file, output_file

    raise TextToSpeechError(f"Choose a .txt, .html, or .htm file: {source_file}")


class TextToSpeechWindow(QMainWindow):
    """Main window for selecting a text file, voice, and MP3 playback."""

    def __init__(self):
        super().__init__()
        self.thread = None
        self.worker = None
        self.output_file = None
        self.settings = QSettings(ORGANIZATION_NAME, APPLICATION_NAME)

        self.setWindowTitle("Text to Speech")
        self.resize(720, 220)
        self.apply_style()

        self.source_file_edit = QLineEdit()
        self.source_file_edit.setReadOnly(True)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        self.output_file_edit = QLineEdit()
        self.output_file_edit.setReadOnly(True)

        self.voice_combo = QComboBox()
        self.voice_combo.addItems(VALID_VOICES)

        self.browse_button = QPushButton("Choose Input File")
        self.browse_button.clicked.connect(self.choose_source_file)

        self.output_button = QPushButton("Choose Output Directory")
        self.output_button.clicked.connect(self.choose_output_dir)

        self.convert_button = QPushButton("Convert to MP3")
        self.convert_button.clicked.connect(self.convert)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play)
        self.play_button.setEnabled(False)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause)
        self.pause_button.setEnabled(False)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status_label = QLabel("Select a .txt, .html, or .htm file to convert.")

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.playbackStateChanged.connect(self.update_playback_buttons)

        form = QFormLayout()
        form.addRow("Input file", self.row(self.source_file_edit, self.browse_button))
        form.addRow("Output directory", self.row(self.output_dir_edit, self.output_button))
        form.addRow("Output MP3", self.output_file_edit)
        form.addRow("Voice", self.voice_combo)

        controls = QHBoxLayout()
        controls.addWidget(self.convert_button)
        controls.addWidget(self.play_button)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.stop_button)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.progress)
        layout.addWidget(self.status_label)
        layout.addLayout(controls)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.restore_saved_paths()

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
            QComboBox {
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

    @staticmethod
    def row(*widgets):
        """Return a horizontal row for a form field."""
        layout = QHBoxLayout()
        for widget in widgets:
            layout.addWidget(widget)
        container = QWidget()
        container.setLayout(layout)
        return container

    def choose_source_file(self):
        """Prompt for the input story text or HTML file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Choose input file",
            self.saved_source_dir(),
            "Text and HTML files (*.txt *.html *.htm);;Text files (*.txt);;HTML files (*.html *.htm);;All files (*)",
        )
        if not filename:
            return
        self.source_file_edit.setText(filename)
        self.save_source_dir(Path(filename).parent)
        if not self.output_dir_edit.text():
            self.output_dir_edit.setText(str(Path(filename).parent))
            self.save_output_dir(Path(filename).parent)
        self.update_output_file()

    def choose_output_dir(self):
        """Prompt for the destination MP3 directory."""
        dirname = QFileDialog.getExistingDirectory(
            self,
            "Choose output directory",
            self.output_dir_edit.text() or "",
        )
        if dirname:
            self.output_dir_edit.setText(dirname)
            self.save_output_dir(dirname)
            self.update_output_file()

    def update_output_file(self):
        """Display the derived MP3 output path."""
        source_file = self.source_file_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        if source_file and output_dir:
            self.output_file_edit.setText(str(Path(output_dir) / f"{Path(source_file).stem}.mp3"))

    def restore_saved_paths(self):
        """Restore remembered source and destination directories."""
        output_dir = self.settings.value(OUTPUT_DIR_SETTING, "", str)
        if output_dir:
            self.output_dir_edit.setText(output_dir)

    def saved_source_dir(self):
        """Return the last remembered source directory for file browsing."""
        return self.settings.value(SOURCE_DIR_SETTING, "", str)

    def save_source_dir(self, source_dir):
        """Remember the last source directory used by the file picker."""
        self.settings.setValue(SOURCE_DIR_SETTING, str(source_dir))

    def save_output_dir(self, output_dir):
        """Remember the destination directory."""
        self.settings.setValue(OUTPUT_DIR_SETTING, str(output_dir))

    def convert(self):
        """Start conversion in a worker thread."""
        source_file = self.source_file_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        if not source_file:
            QMessageBox.warning(self, "Missing input file", "Choose a .txt, .html, or .htm file first.")
            return
        if not output_dir:
            QMessageBox.warning(self, "Missing output directory", "Choose an output directory first.")
            return
        self.update_output_file()

        self.set_converting(True)
        self.progress.setRange(0, 0)
        self.status_label.setText("Starting conversion...")

        self.thread = QThread(self)
        self.worker = TtsWorker(source_file, output_dir, self.voice_combo.currentText())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress_changed.connect(self.update_progress)
        self.worker.failed.connect(self.conversion_failed)
        self.worker.finished.connect(self.conversion_finished)
        self.worker.failed.connect(self.thread.quit)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.clear_thread)
        self.thread.start()

    def update_progress(self, current, total, message):
        """Update conversion progress."""
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        self.status_label.setText(message)

    def conversion_finished(self, output_file):
        """Enable playback after conversion."""
        self.output_file = output_file
        self.output_file_edit.setText(output_file)
        self.player.setSource(QUrl.fromLocalFile(output_file))
        self.status_label.setText(f"Saved {output_file}")
        self.progress.setValue(self.progress.maximum())
        self.set_converting(False)
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(True)

    def conversion_failed(self, message):
        """Display conversion failure."""
        self.status_label.setText("Conversion failed.")
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.set_converting(False)
        QMessageBox.critical(self, "Conversion failed", message)

    def clear_thread(self):
        """Clear finished worker references."""
        self.thread = None
        self.worker = None

    def play(self):
        """Play the generated MP3."""
        output_file = self.output_file_edit.text().strip()
        if output_file and Path(output_file).is_file():
            self.player.setSource(QUrl.fromLocalFile(output_file))
            self.player.play()

    def pause(self):
        """Pause playback."""
        self.player.pause()

    def stop(self):
        """Stop playback."""
        self.player.stop()

    def update_playback_buttons(self, state):
        """Enable playback controls based on player state."""
        has_file = bool(self.output_file_edit.text().strip())
        self.play_button.setEnabled(has_file and state != QMediaPlayer.PlayingState)
        self.pause_button.setEnabled(has_file and state == QMediaPlayer.PlayingState)
        self.stop_button.setEnabled(has_file and state != QMediaPlayer.StoppedState)

    def set_converting(self, converting):
        """Enable or disable controls while conversion is running."""
        self.convert_button.setEnabled(not converting)
        self.browse_button.setEnabled(not converting)
        self.output_button.setEnabled(not converting)
        self.voice_combo.setEnabled(not converting)
        self.play_button.setEnabled(False if converting else self.play_button.isEnabled())
        self.pause_button.setEnabled(False if converting else self.pause_button.isEnabled())
        self.stop_button.setEnabled(False if converting else self.stop_button.isEnabled())


def main():
    """Run the Venice AI text-to-speech GUI."""
    app = QApplication(sys.argv)
    window = TextToSpeechWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
