"""Tests for video-project planning and prompt composition."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from modules.video_project import (
    Character,
    Scene,
    build_scene_prompt,
    load_characters,
    mux_project_audio,
    plan_scene_durations,
    probe_video_duration,
    project_slug,
    save_characters,
    scene_duration_matches,
)


class VideoProjectTests(unittest.TestCase):
    def test_audio_duration_plan(self):
        cases = {
            5.1: [10],
            10: [10],
            10.1: [15],
            15: [15],
            15.1: [30],
            30: [30],
            31: [30, 10],
            44.9: [30, 15],
            45.1: [30, 30],
            61: [30, 30, 10],
        }
        for duration, expected in cases.items():
            with self.subTest(duration=duration):
                self.assertEqual(plan_scene_durations(duration), expected)

    def test_project_slug(self):
        self.assertEqual(project_slug(" Tides of Hate! "), "Tides-of-Hate")
        with self.assertRaises(ValueError):
            project_slug("!!!")

    def test_character_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "characters.json"
            characters = [Character("Mara", "Red coat", "/images/mara.png")]
            save_characters(characters, path)
            self.assertEqual(load_characters(path), characters)

    def test_scene_prompt_contains_style_and_reference_tag(self):
        prompt = build_scene_prompt(
            "Neo-noir, anamorphic lens",
            Scene(2, 30, "Mara crosses the flooded street.", ["Mara"]),
            [Character("Mara", "Red coat", "/images/mara.png")],
            "seedance-2-5-reference-to-video-basic",
        )
        self.assertIn("OVERALL VIDEO STYLE", prompt)
        self.assertIn("<Image 1>", prompt)
        self.assertIn("SCENE 2 (30 seconds)", prompt)

    def test_scene_duration_match_uses_one_second_tolerance(self):
        self.assertTrue(scene_duration_matches(29.04, 30))
        self.assertTrue(scene_duration_matches(30.99, 30))
        self.assertFalse(scene_duration_matches(28.99, 30))
        self.assertFalse(scene_duration_matches(31.01, 30))

    @patch("modules.video_project.subprocess.run")
    @patch("modules.video_project.shutil.which", return_value="/usr/bin/ffprobe")
    def test_probe_video_duration_requires_video_stream(self, _which, run):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.mp4"
            path.touch()
            run.return_value = SimpleNamespace(
                returncode=0,
                stdout='{"format":{"duration":"30.023"},"streams":[{"codec_type":"video"}]}',
                stderr="",
            )
            self.assertEqual(probe_video_duration(path), 30.023)
            run.return_value = SimpleNamespace(
                returncode=0,
                stdout='{"format":{"duration":"30.023"},"streams":[{"codec_type":"audio"}]}',
                stderr="",
            )
            with self.assertRaisesRegex(ValueError, "no video stream"):
                probe_video_duration(path)

    @patch("modules.video_project.media_has_audio", return_value=False)
    @patch("modules.video_project.probe_audio_duration", return_value=120.0)
    @patch("modules.video_project.subprocess.run")
    @patch("modules.video_project.shutil.which", return_value="/usr/bin/ffmpeg")
    def test_mux_can_delay_original_audio_for_title_card(self, _which, run, _duration, _has_audio):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.mp4"
            audio_path = Path(directory) / "song.mp3"
            output_path = Path(directory) / "output.mp4"
            input_path.touch()
            audio_path.touch()
            def complete(_command, **_kwargs):
                output_path.write_bytes(b"created")
                return SimpleNamespace(returncode=0, stderr="")

            run.side_effect = complete
            mux_project_audio(input_path, output_path, audio_path, True, audio_delay_seconds=7)
            command = run.call_args.args[0]
            self.assertIn("[1:a:0]adelay=7000:all=1[audio]", command)
            self.assertIn("127.000000", command)


if __name__ == "__main__":
    unittest.main()
