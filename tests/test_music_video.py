"""Tests for music-video resource and subtitle helpers."""

import tempfile
import threading
import sys
import unittest
from pathlib import Path

from modules.music_video import (
    ProjectPaths,
    OperationCancelled,
    SubtitleCue,
    TimedWord,
    align_lyrics_to_words,
    format_srt,
    offset_cues,
    parse_srt,
    resolve_project_path,
    run_cancellable,
    stored_project_path,
    subtitle_filter,
    validate_cues,
)


class MusicVideoTests(unittest.TestCase):
    def test_project_paths_create_expected_layout_and_clean_tmp(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = ProjectPaths(Path(directory)).create()
            self.assertTrue(paths.audio.is_dir())
            self.assertTrue(paths.images.is_dir())
            self.assertTrue(paths.scenes.is_dir())
            self.assertTrue(paths.output.is_dir())
            (paths.temporary / "partial.dat").write_text("partial", encoding="utf-8")
            paths.clean_temporary()
            self.assertFalse(paths.temporary.exists())

    def test_project_paths_round_trip_and_version_one_relative_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            project_file = root / "song.venice-video-project.json"
            audio = root / "audio" / "song.mp3"
            self.assertEqual(stored_project_path(audio, root), "audio/song.mp3")
            self.assertEqual(
                resolve_project_path("audio/song.mp3", root, project_file, 2),
                str(audio.resolve()),
            )
            legacy_file = Path(directory) / "legacy" / "old.json"
            self.assertEqual(
                resolve_project_path("song.mp3", root, legacy_file, 1),
                str((legacy_file.parent / "song.mp3").resolve()),
            )

    def test_authoritative_lyrics_use_transcript_timing(self):
        words = [
            TimedWord(1.0, 1.4, "wrong"),
            TimedWord(1.5, 1.9, "recognized"),
            TimedWord(2.1, 2.5, "words"),
            TimedWord(2.6, 3.0, "here"),
        ]
        cues = align_lyrics_to_words("Actual sung line\nSecond line", words)
        self.assertEqual([cue.text for cue in cues], ["Actual sung line", "Second line"])
        self.assertEqual((cues[0].start, cues[-1].end), (1.0, 3.0))

    def test_srt_round_trip_and_offset(self):
        cues = [SubtitleCue(1.25, 2.5, "First line"), SubtitleCue(3, 4.125, "Second")]
        text = format_srt(cues)
        self.assertEqual(parse_srt(text), cues)
        shifted = offset_cues(cues, 7)
        self.assertEqual((shifted[0].start, shifted[-1].end), (8.25, 11.125))

    def test_validation_rejects_overlap_and_media_overrun(self):
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validate_cues([SubtitleCue(0, 2, "One"), SubtitleCue(1.5, 3, "Two")])
        with self.assertRaisesRegex(ValueError, "after the soundtrack"):
            validate_cues([SubtitleCue(9, 10.2, "Late")], media_duration=10)

    def test_subtitle_filter_uses_font_and_clamps_size(self):
        value = subtitle_filter("lyrics.srt", "Noto Sans", 500)
        self.assertIn("FontName=Noto Sans", value)
        self.assertIn("FontSize=144", value)

    def test_cancellable_process_stops_when_event_is_set(self):
        cancel_event = threading.Event()
        cancel_event.set()
        with self.assertRaises(OperationCancelled):
            run_cancellable(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                cancel_event,
            )


if __name__ == "__main__":
    unittest.main()
