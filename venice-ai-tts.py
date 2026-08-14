#!/usr/bin/env python3
"""Qt GUI for queueing text files for Venice AI speech MP3 conversion."""

import sys
import tempfile
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
    QListWidget,
    QListWidgetItem,
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

    progress_changed = Signal(int, int, int, str)
    failed = Signal(int, str)
    finished = Signal(int, str)

    def __init__(self, queue_index, source_file, output_dir, voice_id):
        super().__init__()
        self.queue_index = queue_index
        self.source_file = Path(source_file)
        self.output_dir = Path(output_dir)
        self.voice_id = voice_id

    def run(self):
        """Convert the selected file and report completion."""
        cleanup_text_file = False
        try:
            text_file, output_file, cleanup_text_file = prepare_source_text(
                self.source_file,
                self.output_dir,
                self.emit_progress,
            )
            output_path = text_file_to_mp3(
                text_file,
                output_file,
                self.voice_id,
                progress_callback=self.emit_progress,
            )
        except (OSError, TextToSpeechError) as exc:
            self.failed.emit(self.queue_index, str(exc))
            return
        finally:
            if cleanup_text_file:
                Path(text_file).unlink(missing_ok=True)
        self.finished.emit(self.queue_index, str(output_path))

    def emit_progress(self, current, total, message):
        """Attach the queue item index to worker progress."""
        self.progress_changed.emit(self.queue_index, current, total, message)


def prepare_source_text(source_file, output_dir, progress_callback=None):
    """Return the text file to convert and the final MP3 output path."""
    if not source_file.is_file():
        raise TextToSpeechError(f"Input file does not exist: {source_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{source_file.stem}.mp3"
    suffix = source_file.suffix.lower()

    if suffix in {".html", ".htm"}:
        if progress_callback:
            progress_callback(0, 1, f"Extracting text from {source_file.name}")
        extracted_text = extract_text(source_file)
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            encoding="utf-8",
            prefix=f"{source_file.stem}-",
            suffix=".txt",
        ) as extracted_text_file:
            extracted_text_file.write(extracted_text)
            return Path(extracted_text_file.name), output_file, True

    if suffix == ".txt":
        return source_file, output_file, False

    raise TextToSpeechError(f"Choose a .txt, .html, or .htm file: {source_file}")


class TextToSpeechWindow(QMainWindow):
    """Main window for selecting a text file, voice, and MP3 playback."""

    def __init__(self):
        super().__init__()
        self.thread = None
        self.worker = None
        self.output_file = None
        self.selected_source_files = []
        self.queue_items = []
        self.current_queue_index = None
        self.settings = QSettings(ORGANIZATION_NAME, APPLICATION_NAME)

        self.setWindowTitle("Text to Speech")
        self.resize(820, 440)
        self.apply_style()

        self.source_file_edit = QLineEdit()
        self.source_file_edit.setReadOnly(True)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        self.output_file_edit = QLineEdit()
        self.output_file_edit.setReadOnly(True)

        self.voice_combo = QComboBox()
        self.voice_combo.addItems(VALID_VOICES)

        self.browse_button = QPushButton("Choose Input Files")
        self.browse_button.clicked.connect(self.choose_source_files)

        self.output_button = QPushButton("Choose Output Directory")
        self.output_button.clicked.connect(self.choose_output_dir)

        self.convert_button = QPushButton("Add to Queue")
        self.convert_button.clicked.connect(self.add_selected_to_queue)

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
        self.status_label = QLabel("Select .txt, .html, or .htm files to add to the queue.")

        self.queue_list = QListWidget()
        self.queue_list.itemSelectionChanged.connect(self.queue_selection_changed)

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.playbackStateChanged.connect(self.update_playback_buttons)

        form = QFormLayout()
        form.addRow("Input files", self.row(self.source_file_edit, self.browse_button))
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
        layout.addWidget(QLabel("Queue"))
        layout.addWidget(self.queue_list)
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
            QComboBox,
            QListWidget {
                background: #2f3136;
                border: 1px solid #4b4f58;
                border-radius: 4px;
                color: #f1f3f4;
                padding: 7px;
                selection-background-color: #2f80ed;
            }

            QListWidget {
                min-height: 150px;
            }

            QListWidget::item {
                padding: 6px;
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

    def choose_source_files(self):
        """Prompt for one or more input story text or HTML files."""
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose input files",
            self.saved_source_dir(),
            "Text and HTML files (*.txt *.html *.htm);;"
            "Text files (*.txt);;"
            "HTML files (*.html *.htm);;"
            "All files (*)",
        )
        if not filenames:
            return
        self.selected_source_files = filenames
        self.source_file_edit.setText(self.selected_files_label())
        self.save_source_dir(Path(filenames[0]).parent)
        if not self.output_dir_edit.text():
            self.output_dir_edit.setText(str(Path(filenames[0]).parent))
            self.save_output_dir(Path(filenames[0]).parent)
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
        output_dir = self.output_dir_edit.text().strip()
        if len(self.selected_source_files) == 1 and output_dir:
            source_file = self.selected_source_files[0]
            self.output_file_edit.setText(str(Path(output_dir) / f"{Path(source_file).stem}.mp3"))
        elif len(self.selected_source_files) > 1 and output_dir:
            self.output_file_edit.setText(
                f"{len(self.selected_source_files)} MP3 files in {output_dir}"
            )

    def selected_files_label(self):
        """Return compact text for the selected input files field."""
        if len(self.selected_source_files) == 1:
            return self.selected_source_files[0]
        return f"{len(self.selected_source_files)} files selected"

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

    def add_selected_to_queue(self):
        """Add selected files to the conversion queue."""
        output_dir = self.output_dir_edit.text().strip()
        if not self.selected_source_files:
            QMessageBox.warning(
                self,
                "Missing input files",
                "Choose one or more .txt, .html, or .htm files first.",
            )
            return
        if not output_dir:
            QMessageBox.warning(
                self,
                "Missing output directory",
                "Choose an output directory first.",
            )
            return
        self.update_output_file()

        voice_id = self.voice_combo.currentText()
        for filename in self.selected_source_files:
            source_file = Path(filename)
            output_file = Path(output_dir) / f"{source_file.stem}.mp3"
            queue_item = QListWidgetItem(f"Queued: {source_file.name} -> {output_file.name}")
            self.queue_list.addItem(queue_item)
            self.queue_items.append(
                {
                    "source_file": source_file,
                    "output_dir": Path(output_dir),
                    "voice_id": voice_id,
                    "output_file": output_file,
                    "list_item": queue_item,
                    "status": "queued",
                }
            )

        self.status_label.setText(f"Added {len(self.selected_source_files)} file(s) to the queue.")
        self.process_next_queue_item()

    def process_next_queue_item(self):
        """Start the next queued item if no conversion is currently running."""
        if self.thread is not None:
            return

        next_index = self.next_queued_index()
        if next_index is None:
            self.current_queue_index = None
            self.set_processing(False)
            if self.queue_items:
                self.status_label.setText(self.queue_summary())
            return

        queue_item = self.queue_items[next_index]
        self.current_queue_index = next_index
        self.set_processing(True)
        self.progress.setRange(0, 0)
        self.status_label.setText(f"Starting {queue_item['source_file'].name}...")
        queue_item["status"] = "running"
        queue_item["list_item"].setText(
            f"Running: {queue_item['source_file'].name} -> {queue_item['output_file'].name}"
        )

        self.thread = QThread(self)
        self.worker = TtsWorker(
            next_index,
            queue_item["source_file"],
            queue_item["output_dir"],
            queue_item["voice_id"],
        )
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

    def next_queued_index(self):
        """Return the index of the next queued item."""
        for index, queue_item in enumerate(self.queue_items):
            if queue_item["status"] == "queued":
                return index
        return None

    def update_progress(self, queue_index, current, total, message):
        """Update conversion progress."""
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        self.status_label.setText(message)
        queue_item = self.queue_items[queue_index]
        queue_item["list_item"].setText(
            f"Running: {queue_item['source_file'].name} - {message}"
        )

    def conversion_finished(self, queue_index, output_file):
        """Record item completion and enable playback for the latest MP3."""
        queue_item = self.queue_items[queue_index]
        queue_item["status"] = "done"
        queue_item["output_file"] = Path(output_file)
        queue_item["list_item"].setText(
            f"Done: {queue_item['source_file'].name} -> {Path(output_file).name}"
        )
        self.output_file = output_file
        self.output_file_edit.setText(output_file)
        self.player.setSource(QUrl.fromLocalFile(output_file))
        self.status_label.setText(f"Saved {output_file}")
        self.progress.setValue(self.progress.maximum())
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(True)

    def conversion_failed(self, queue_index, message):
        """Record item failure and continue with later queue entries."""
        queue_item = self.queue_items[queue_index]
        queue_item["status"] = "failed"
        queue_item["list_item"].setText(
            f"Failed: {queue_item['source_file'].name} - {message}"
        )
        self.status_label.setText(f"Conversion failed for {queue_item['source_file'].name}.")
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

    def clear_thread(self):
        """Clear finished worker references."""
        self.thread = None
        self.worker = None
        self.process_next_queue_item()

    def queue_selection_changed(self):
        """Use a completed selected queue item as the playback target."""
        selected_items = self.queue_list.selectedItems()
        if not selected_items:
            return

        selected_item = selected_items[0]
        for queue_item in self.queue_items:
            if queue_item["list_item"] is selected_item and queue_item["status"] == "done":
                output_file = str(queue_item["output_file"])
                self.output_file = output_file
                self.output_file_edit.setText(output_file)
                self.player.setSource(QUrl.fromLocalFile(output_file))
                self.update_playback_buttons(self.player.playbackState())
                return

    def queue_summary(self):
        """Return a compact summary of completed queue work."""
        done_count = sum(1 for item in self.queue_items if item["status"] == "done")
        failed_count = sum(1 for item in self.queue_items if item["status"] == "failed")
        if failed_count:
            return f"Queue complete: {done_count} done, {failed_count} failed."
        return f"Queue complete: {done_count} done."

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

    def set_processing(self, processing):
        """Update controls while the background queue is processing."""
        self.convert_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.output_button.setEnabled(True)
        self.voice_combo.setEnabled(True)
        if processing:
            self.pause_button.setEnabled(False)


def main():
    """Run the Venice AI text-to-speech GUI."""
    app = QApplication(sys.argv)
    window = TextToSpeechWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
