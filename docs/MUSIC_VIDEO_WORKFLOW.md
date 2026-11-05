# Music Video Production Workflow — Suno, Demucs, Venice AI, SRT, FFmpeg

**Target repository:** [muman613/venice.ai-tools](https://github.com/muman613/venice.ai-tools)  
**Primary application:** [`venice-ai-video-project.py`](https://github.com/muman613/venice.ai-tools/blob/main/venice-ai-video-project.py)  
**Supporting module:** [`modules/video_project.py`](https://github.com/muman613/venice.ai-tools/blob/main/modules/video_project.py)  
**Status:** Proposed music-video workflow and integration specification. The current project application supports scene planning, video generation, stitching, and soundtrack muxing; **Demucs, lyric-to-SRT alignment, and subtitle burn-in are proposed additions, not current application features.**

## 1. Objective

Create a repeatable workflow producing a finished **1920×1080, 16:9, H.264/AAC MP4** from a song created with Suno. Lyrics are synchronized to the vocals using a Demucs-isolated vocal stem and transcription/alignment, while scene prompts drive the existing Venice AI 30-second scene generator. FFmpeg burns the approved SRT subtitles onto the final video while retaining the **original full-quality Suno mix**, not the separated vocals.

```text
Lyrics + style prompt -> Suno (online) -> song.mp3
                                         |
                         +---------------+------------------+
                         |                                  |
                    Original MP3                        Demucs
                         |                                  |
                Venice video project                   vocals.wav
                         |                                  |
               30-second scene plan              transcription / lyric alignment
                         |                                  |
           scene prompts + characters                   lyrics.srt
                         |                                  |
              Venice-generated clips                      |
                         |                                  |
                 stitch / mux MP3                          |
                         +---------------+------------------+
                                         |
                             FFmpeg subtitle burn-in
                                         |
                              song-final-1080p.mp4
```

## 2. Dependencies and environment

On Ubuntu:

```bash
sudo apt update
sudo apt install -y ffmpeg python3-venv

git clone https://github.com/muman613/venice.ai-tools.git
cd venice.ai-tools
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional music-video preprocessing tools:
pip install demucs pysubs2 faster-whisper
# Demucs and ASR dependencies (including PyTorch) may be large.
# For word/line-level forced alignment, add an appropriate aligner such as WhisperX.
```

`ffmpeg -filters | grep subtitles` should include a `subtitles` filter built with **libass**. Verify `ffprobe -version`, `ffmpeg -version`, and `python -m demucs --help` in the same runtime environment.

Venice generation requires `VENICE_API_KEY`. Suno song creation/download is a separate, manual online step; this proposal does **not** assume a Suno API integration.

## 3. Stage A — Compose music in Suno

1. Develop lyrics, song structure, and a musical style prompt; save both alongside the project.
2. Generate a draft in [Suno](https://suno.com/) and review pronunciation, lyric changes, repeated lines, instrumental passages, and the actual mix.
3. Download the selected song as `song.mp3` (or the best source format available). **Do not** replace it with the Demucs vocals-only output in the finished video.
4. Save the **actual sung lyrics** as `lyrics.txt`, one phrase or subtitle line per line. Update for repetitions, ad-libs, and changed words.
5. Record the Suno source URL, version, title, rights/license notes, and export date in project metadata. Verify publication rights under the applicable Suno plan and terms.

Example source layout:

```text
projects/my-song/
  source/
    song.mp3
    lyrics.txt
    suno-prompt.md
  stems/
  subtitles/
  prompts/
  clips/
  render/
```

## 4. Stage B — Separate vocals with Demucs

Demucs helps lyric timing by separating the sung vocal from guitars, drums, and bass. It **does not produce an SRT file or automatically identify the correct lyrics**.

```bash
cd projects/my-song
python -m demucs --two-stems=vocals -n htdemucs \
  -o stems source/song.mp3
```

Expected output (default model):

```text
stems/htdemucs/song/vocals.wav
stems/htdemucs/song/no_vocals.wav
```

If running out of VRAM, use CPU processing (`-d cpu`); Demucs performance depends on system hardware. Preserve `source/song.mp3` untouched. The separated vocal may contain artifacts or residual instruments, so all timestamps require review.

## 5. Stage C — Build and validate synchronized SRT lyrics

**Preferred approach:** Use the original written lyrics as the authoritative text, then align them with the isolated vocal. An automatic speech-to-text pass can suggest phrase boundaries, but sung lyrics frequently produce missed words, repeated words, and inaccurate timestamps.

Suggested pipeline:

1. Transcribe `vocals.wav` using a local ASR engine (e.g. faster-whisper) to obtain provisional timestamps.
2. Compare recognized text against `source/lyrics.txt`; repair words and repetition counts.
3. Use a forced aligner (e.g. WhisperX where suitable) or manually adjust lyric-line boundaries against the vocal waveform.
4. Group words into readable subtitle cues, normally **1–2 lines** with about **3–7 words per line**; avoid unnecessarily short flashing captions.
5. Export UTF-8 `subtitles/lyrics.srt` and confirm every start timestamp is earlier than its end, cues are sorted, and cues do not unintentionally overlap.
6. Spot-check the opening, verse-to-chorus changes, every repeated chorus, instrumental breaks, and the final line.

Example format:

```srt
1
00:00:12,100 --> 00:00:15,400
When the storm comes rolling in

2
00:00:15,550 --> 00:00:18,700
I can feel it underneath my skin
```

The text is illustrative only; **do not infer real lyric timestamps from an MP3 duration or the written lyrics alone**. Demucs output makes transcription easier but does not guarantee synchronization. Optional use of `pysubs2` is useful for editing or validating the final SRT programmatically.

### Suggested validation rules

- Every cue has non-empty text and valid `HH:MM:SS,mmm` timecodes.
- All cues lie within the soundtrack duration; apply an explicit timing offset only if measured.
- Normal cues should not overlap; intentional simultaneous lyrics are an exception requiring special handling.
- Review clipping and subtitle safe margins at 1920×1080.
- Ensure the final rendered MP4 and SRT derive from the same untrimmed master soundtrack.

## 6. Stage D — Create a Venice music-video project

Start the **existing** application from the repository root:

```bash
export VENICE_API_KEY='your-venice-api-key'
python venice-ai-video-project.py
```

The current implementation:

- Reads the selected MP3 duration with `probe_audio_duration()`.
- Creates an editable queue using `plan_scene_durations()`, with 30-second scenes and supported shorter tail durations.
- Lets the user edit individual scene prompts and select reusable character descriptions/reference images.
- Queries live video models supporting 30-second clips; the default preferred model identifier is `seedance-2-5-reference-to-video-basic` where available.
- Generates scene MP4s, validates reusable clips, stitches the scenes, and uses `mux_project_audio()` to replace or mix generated audio with the selected MP3.
- Saves project state as `<slug>.venice-video-project.json` and produces `<slug>.mp4`.

**Important:** This application is not yet an SRT-aware lyric-video editor. Prepare the SRT independently and add the subtitle-render stage after its existing finished video.

### Scene-planning guidelines

1. Select the original `source/song.mp3`; use **Rebuild from audio** to populate the scene list.
2. Save a project name and output directory. Save the project JSON early.
3. Write an overall art direction: aspect ratio, cinematography, palette, recurring symbols, mood, camera movement, and continuity rules.
4. Use the SRT as the timing source for **specific imagery within each scene**. For example, scene 3 covering 01:00–01:30 may need several narrative beats; describe the intended beats within that scene prompt, although video generation does **not guarantee frame-accurate adherence**.
5. Prefer concise, strongly visual prompts with explicitly named characters and consistent reference images where supported.
6. Review estimated generation cost before starting, generate, inspect the scene MP4s, and regenerate weak scenes as needed.
7. Keep `Remove generated clip audio; use MP3 as the stereo soundtrack` checked unless clip audio is intentionally being mixed into the song.
8. Produce the stitched, music-backed `<slug>.mp4` and **leave this as an uncaptioned intermediate**.

**Note:** The scene generator produces coherent 30-second scene units, not automatic 10–15-second lyric-precise cuts. For faster cuts, add an optional shot-level edit list and a post-generation cutting/transition pass (or finish in Shotcut).

## 7. Stage E — Burn the SRT into the MP4 with FFmpeg

### Shell command

Run in a directory with simple filenames (or use the Python wrapper below to escape paths):

```bash
ffmpeg -y \
  -i render/my-song.mp4 \
  -vf "subtitles=subtitles/lyrics.srt:force_style='FontName=DejaVu Sans,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=1,Alignment=2,MarginV=65'" \
  -c:v libx264 -preset medium -crf 18 \
  -c:a copy -movflags +faststart \
  render/my-song-final.mp4
```

FFmpeg/libass `FontSize` is ASS-style text sizing, **not guaranteed to be literal output pixels**; inspect a rendered sample and tune for 1080p. `Alignment=2` places text at the bottom center and `MarginV` shifts it above the bottom edge. H.264 video must be re-encoded to burn subtitles; AAC audio can normally be copied unchanged from the existing project output.

### Minimal Python wrapper (proposed new utility)

Save as `render-srt.py` or incorporate the function into `modules/video_project.py`:

```python
#!/usr/bin/env python3
"""Burn an SRT into a video using FFmpeg/libass."""

import argparse
import shutil
import subprocess
from pathlib import Path


def ffmpeg_filter_path(path: Path) -> str:
    """Escape a path for FFmpeg's filter-option parser.

    Prefer relative paths under a predictable work directory when possible.
    """
    value = str(path.resolve())
    # Escape backslashes, colons, single quotes, commas, and brackets in filter paths.
    for old, new in [("\\", "\\\\"), (":", "\\:"), ("'", "\\'"), (",", "\\,"), ("[", "\\["), ("]", "\\]")]:
        value = value.replace(old, new)
    return value


def burn_srt(video_file, srt_file, output_file):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is not installed")
    video_file = Path(video_file)
    srt_file = Path(srt_file)
    output_file = Path(output_file)
    for path in (video_file, srt_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if video_file.resolve() == output_file.resolve():
        raise ValueError("Output must differ from input")

    style = (
        "FontName=DejaVu Sans,FontSize=22,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "Outline=2,Shadow=1,Alignment=2,MarginV=65"
    )
    vf = f"subtitles=filename='{ffmpeg_filter_path(srt_file)}':force_style='{style}'"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_file),
        "-vf", vf, "-c:v", "libx264", "-crf", "18",
        "-preset", "medium", "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-movflags", "+faststart",
        str(output_file),
    ], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("srt")
    parser.add_argument("output")
    args = parser.parse_args()
    burn_srt(args.video, args.srt, args.output)
```

Example:

```bash
python render-srt.py \
  render/my-song.mp4 \
  subtitles/lyrics.srt \
  render/my-song-final.mp4
```

For production use, test escaping with paths containing punctuation, spaces, quotes, and non-ASCII characters; FFmpeg filter escaping varies across input contexts. A temporary ASCII-only SRT copy can be used as a robust fallback. Keep the SRT as an additional downloadable sidecar for platforms that support selectable subtitles.

## 8. Stage F — Quality control

Check the output codecs, dimensions, frame rate, audio stream, and duration:

```bash
ffprobe -v error -show_entries \
  format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate \
  -of json render/my-song-final.mp4
```

Watch the first and last minute, several repeated choruses, and all scene boundaries. Specifically check:

- Words appear while they are sung (not merely while the scene happens).
- No text is cut off or obscured by YouTube/mobile UI safe margins.
- Final song audio matches the original Suno file and plays without gaps.
- Clip transitions do not create blank frames or repeat frames.
- Final duration covers the entire song; the last subtitle ends before the file ends.
- The final video has the intended aspect ratio, resolution, and non-clipping sound.

If using Shotcut, import the **uncaptioned** stitched video for creative corrections, export it, and burn the final subtitles afterward. Burning subtitles last avoids baking lyrics into clips that may still be trimmed or re-timed.

## 9. Proposed integration into `venice-ai-video-project.py`

Implement as an incremental feature, rather than rewriting working Venice generation code.

### A. Extend project schema

Preserve compatibility with existing `version: 1` project files. New optional keys might be:

```json
{
  "music_video": {
    "lyrics_path": "source/lyrics.txt",
    "vocals_path": "stems/htdemucs/song/vocals.wav",
    "srt_path": "subtitles/lyrics.srt",
    "subtitle_enabled": true,
    "subtitle_style": {
      "font_name": "DejaVu Sans",
      "font_size": 22,
      "alignment": 2,
      "margin_v": 65
    },
    "subtitle_render_path": "render/my-song-final.mp4"
  }
}
```

Resolve relative media paths against the project file location and continue loading old projects when `music_video` is absent.

### B. Add GUI controls

Proposed new **Music Video** section:

- **Lyrics file:** select an authoritative UTF-8 TXT file.
- **Separate vocals:** queue Demucs and show progress + final vocal-stem path.
- **Create/Edit SRT:** align lyrics to the vocal track, open a cue editor, and preview timestamps.
- **Subtitle style:** font, size, outline, position, and enable/disable burn-in.
- **Render final:** process the already-stitched video with the approved SRT.
- **Export assets:** save sidecar SRT plus a manifest/edit list.

### C. Add separable worker stages

Recommended helper modules:

```text
modules/
  video_project.py       # existing project and mux helpers
  music_separation.py    # subprocess-based Demucs wrapper
  lyric_alignment.py     # ASR, forced alignment, SRT generation/validation
  subtitle_render.py     # FFmpeg/libass rendering and media QA
```

Use `QThread` workers and progress/error signals, like the existing video-generation worker. Do not run Demucs or FFmpeg on the GUI thread. Cache results keyed by source file fingerprint and processing settings so changing an SRT does not regenerate Venice clips.

### D. Recommended build phases

1. **MVP:** Import an existing SRT, preview it, burn it onto the completed project MP4, and save both outputs.
2. **Audio processing:** Add a Demucs command and retain vocal/instrumental stems.
3. **Alignment:** Add transcription + human-reviewed lyric SRT generation, with editable timing and offset.
4. **Scene cue overlay:** Show lyric lines/timestamps overlapping each 30-second scene to guide prompt writing.
5. **Advanced editing:** Support 10–15-second shot edit lists, cuts/fades/wipes, and resumable final rendering.

## 10. Reproducibility and acceptance criteria

A complete project should preserve the untouched original MP3, exact lyrics, isolated vocals, approved SRT, video project JSON, individual generated scene clips, uncaptioned stitched MP4, and final captioned MP4. Record component versions and key processing choices, but **never store API keys in project files**.

Accept the workflow when:

- It can regenerate the captioned MP4 from an existing video and SRT **without calling Venice**.
- The original Suno audio, rather than Demucs vocals, is the final music track.
- Subtitles are visually legible and accurately aligned after human review.
- Existing Venice project JSONs still open.
- Venice clip generation remains resumable and isolated from subtitle edits.
- Errors from missing Demucs, unsupported FFmpeg builds, malformed SRT, or interrupted renders are clearly reported.

## References

- Venice tools repository: https://github.com/muman613/venice.ai-tools
- Current project builder: https://github.com/muman613/venice.ai-tools/blob/main/venice-ai-video-project.py
- Current video project helpers: https://github.com/muman613/venice.ai-tools/blob/main/modules/video_project.py
- Demucs: https://github.com/facebookresearch/demucs
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- WhisperX: https://github.com/m-bain/whisperX
- pysubs2: https://github.com/tkarabela/pysubs2
- FFmpeg subtitles filter: https://ffmpeg.org/ffmpeg-filters.html#subtitles-1
