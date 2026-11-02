# Venice AI Video Project Builder — User Manual

`venice-ai-video-project.py` is a desktop application for making a multi-scene
music video whose length follows a soundtrack. It can isolate vocals, transcribe
or align user-supplied lyrics, export SRT subtitles, prepend a still title card,
generate or reuse Venice scenes, and optionally burn styled subtitles into a
1920×1080 final render while retaining the original song mix.

The tool is best suited to music videos, narrated visual sequences, trailers,
and other projects that can be designed as independent 10-, 15-, or 30-second
scenes. It does not carry the final frame of one scene into the next; continuity
comes from your overall style, scene prompts, and character definitions.

## 1. Requirements

Before starting, you need:

- Python 3.10 or newer.
- The packages in the repository's `requirements.txt`.
- `ffmpeg` and `ffprobe` available on `PATH`.
- A Venice API key with access to video generation.
- A graphical desktop environment supported by PySide6.
- Enough Venice balance for every scene in the project.
- Optional Demucs and faster-whisper packages for vocal isolation and automatic
  subtitle timing. Manual SRT import/edit/export does not require them.

From the repository root, a typical setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-music-video.txt  # optional music preprocessing
sudo apt install ffmpeg             # Ubuntu or Debian
export VENICE_API_KEY="your-api-key"
```

Do not put an API key in a project file or commit one to source control. If
`VENICE_API_KEY` is not set, the application asks for the key when it first
needs to contact Venice. A key entered in that dialog is retained only for the
current run.

## 2. Start the application

Run this command from the repository root:

```bash
python venice-ai-video-project.py
```

When an API key is available, the application immediately loads current video
models. Only text-to-video and reference-to-video models that advertise
30-second generation are shown. Select **Refresh Models** to load the list
again or to enter an API key.

The window is divided into five tabs: **Project**, **Scenes & Characters**,
**Music & Subtitles**, **Render**, and **Status**.

## 3. Quick-start workflow

1. Enter a short **Project name**, such as `tides-of-hate`.
2. Choose a dedicated **Project directory**.
3. Select **Choose Soundtrack...** and open the soundtrack. The scene queue is rebuilt
   to cover its duration.
4. Select a model. Use a reference-to-video model if you want to upload
   character images; use a text-to-video model if descriptions are sufficient.
5. Enter an **Overall video style**.
6. Create any recurring characters in the character library.
7. Work through the scene queue. Write a prompt for every scene and check the
   characters that appear in it.
8. In **Music & Subtitles**, enter/import the actual sung lyrics. Select
   **Extract Vocals**, then **Transcribe / Align SRT**, and review the editable SRT.
9. In **Render**, optionally select a title image and its 5–10 second duration,
   then choose subtitle font, size, and whether to burn captions.
10. Review the audio option and select **Estimate Cost** to retrieve a live quote
   without generating any video.
11. Save the editable project and select **Start Generation**.
12. Watch the status log. When generation finishes, the application reports the
   path of the completed MP4.

Generation can take a long time because scenes are submitted and completed one
at a time. The application must remain open until the project finishes.

## 4. Project settings

### Project name and output directory

The project name becomes the filename prefix. Spaces and unsafe filename
characters are converted to hyphens when generation begins. For example,
`Tides of Hate!` becomes `Tides-of-Hate`.

For a project whose resulting prefix is `tides-of-hate`, the project directory
contains:

```text
tides-of-hate.venice-video-project.json
audio/song.mp3
audio/song-vocals.wav
audio/lyrics.txt
images/title.png
video/scenes/tides-of-hate-scene-1.mp4
video/scenes/tides-of-hate-scene-2.mp4
video/output/tides-of-hate.srt
video/output/tides-of-hate.mp4
video/output/tides-of-hate-captioned.mp4
tmp/                                  # removed after each run
```

Scene MP4s are retained after a successful build. If any of these video files
already exist, the application checks them with FFprobe before requesting that
scene again. A file that contains a video stream and is within one second of
the scene's planned duration is reused; an unreadable or incorrectly timed file
is replaced by a new generation. The application asks before replacing an
existing final project MP4. The editable JSON project is automatically saved
immediately before generation.

### Soundtrack and automatic scene planning

The soundtrack is optional for generation, but selecting one is the intended
workflow. It must be an MP3 that FFprobe can read.

When you choose a soundtrack, the application creates as many 30-second scenes
as possible and rounds the remaining time up to a supported scene duration:

| MP3 duration | Planned scenes |
| --- | --- |
| 5 seconds | 10 seconds |
| 12 seconds | 15 seconds |
| 29 seconds | 30 seconds |
| 44 seconds | 30 + 15 seconds |
| 61 seconds | 30 + 30 + 10 seconds |

The generated visual track may therefore be slightly longer than the MP3. The
final mux trims it to the exact soundtrack duration.

Selecting another MP3 or **Rebuild from Audio** rebuilds the scene durations.
Existing prompts and character selections are preserved by scene position when
possible. Extra old scenes are discarded; newly required scenes start empty.

You may change a scene to 10, 15, or 30 seconds manually, add scenes, or remove
scenes. The selected model must support every chosen duration. Manual changes
also change the available visual length, so ensure the combined scene duration
still covers the soundtrack.

### Video model, resolution, and aspect ratio

The model list comes from Venice at runtime, so its contents can change. Each
entry is marked `[reference]` or `[text]`:

- A **reference** model receives the checked characters' reference images for
  each scene, along with their written descriptions.
- A **text** model receives the written prompt and character information, but
  character images are not uploaded.

The application automatically requests `1080p` and `16:9` when the selected
model advertises them. Otherwise it uses the first resolution and aspect ratio
advertised by that model. There are no manual resolution or aspect-ratio
controls in this tool.

The default model name shown before a successful refresh is only a placeholder.
You must refresh and select a live model before generation.

## 5. Write the overall style and scene prompts

### Overall video style

The overall style is prepended to every generated scene prompt. Put global
decisions here rather than repeating them in each scene, for example:

```text
Cinematic neo-noir photographed on anamorphic lenses. Rainy night exteriors,
deep blue shadows, warm practical lights, restrained handheld movement, subtle
film grain, realistic anatomy, and consistent wardrobe throughout.
```

Useful global details include visual medium, period, palette, lighting, lens,
camera language, pacing, wardrobe rules, and recurring locations.

### Scene prompts

Select a scene in the queue and describe what should happen during that clip.
An empty scene has a circle, and a scene with a completed prompt has a bullet.
During generation, a scene changes to a check mark after its generated or
reused MP4 passes validation. Only scenes that still require generation need a
non-empty prompt and model-supported duration; reusable scenes are not submitted
again.

A strong scene prompt usually identifies:

- The location and starting composition.
- The characters present and their actions.
- Camera position and movement.
- Lighting, weather, and important props specific to the scene.
- The desired final composition or pose.

Example:

```text
Wide street-level view outside the flooded theater. Mara steps from beneath
the marquee, crosses toward camera through ankle-deep water, and looks over her
shoulder as a tram passes behind her. Slow dolly backward, natural body motion,
reflections rippling in the foreground. End on a medium close-up with Mara
facing frame right.
```

The application adds the scene number and duration, character-continuity
details, and an instruction not to render captions, logos, watermarks, or other
visible text. These instructions improve consistency but cannot guarantee a
particular model's output.

Scenes are generated independently. If one scene must cut naturally into the
next, describe the ending state in the first prompt and repeat the starting
state in the next prompt.

## 6. Use the character library

The character library is shared across projects and application runs. On Linux
it is stored at:

```text
~/.local/share/venice-ai-video/characters.json
```

Select **New** and provide:

- **Name**: required and unique without regard to letter case.
- **Description**: appearance, age, wardrobe, distinguishing features, and
  other identity details that should remain stable.
- **Reference image**: an optional PNG, JPEG, or WebP file.

Use **Inspect / Edit** or double-click a character to revise it. Renaming a
character updates its selections in the current project. **Delete** removes the
character from the shared library and from the current scene selections after
confirmation. It does not delete the referenced image file.

For each scene, check only the characters that appear in it. With a reference
model, their images are uploaded in the displayed character order and prompt
tags such as `<Image 1>` or `@Image1` are generated automatically. A model's
reference-image limit applies separately to each scene. Generation is blocked
if a scene exceeds that limit or a selected reference file is missing.

Reference images work best when each file clearly shows one character, matches
the description, and uses a readable pose and lighting. The application stores
the image path rather than copying the image, so moving or deleting the file
breaks the reference until the character record is updated.

## 7. Choose the final audio behavior

**Remove generated clip audio; use MP3 as the stereo soundtrack** is enabled by
default.

- Enabled, with an MP3: model audio is discarded and the MP3 becomes a stereo
  AAC track. The final video is trimmed to the MP3 duration.
- Disabled, with an MP3: model audio and the MP3 are mixed at their existing
  levels into stereo AAC. The tool does not provide volume controls or ducking.
- Enabled, without an MP3: the final video is silent.
- Disabled, without an MP3: any audio in the generated clips is retained.

The current generator deliberately does not request audio in the Venice scene
payload because audio support varies by model. Consequently, mixing has an
effect only when the returned clips actually contain audio.

### Vocal isolation and subtitle timing

In **Music & Subtitles**, enter the actual sung lyrics with one intended cue per
line, or leave the editor empty to use speech-to-text wording as a draft.
**Extract Vocals** runs Demucs locally and retains `audio/<song>-vocals.wav`.
**Transcribe / Align SRT** runs faster-whisper on that stem. When lyrics were
entered, their text remains authoritative and only the word timestamps are used
for timing. Review every cue, especially repeated choruses and instrumental
breaks. Use **Import SRT** or **Export Edited SRT** for a fully manual workflow.

The isolated vocal is never used as final program audio. Rendering always uses
the original imported song. When a title card is selected, automatic SRT cues
are offset by its duration because the title precedes the song.

In **Render**, select an installed font and size. Subtitle burn-in requires an
FFmpeg build with the libass `subtitles` filter. The uncaptioned MP4 and sidecar
SRT remain available even when a captioned MP4 is produced.

## 8. Save and load editable projects

Select **Save Project** to write a JSON project document. It records the project
name, project directory, soundtrack path and measured duration, audio choice,
selected model ID, overall style, all scene data, lyrics and processing paths,
title-card settings, and subtitle font/size. Version-2 resource paths inside the
project are saved relative to the project root. Existing version-1 JSON files
continue to load and are upgraded when next saved.

Select **Load Project** to restore those fields. Character definitions are not
embedded in the project; loaded scenes resolve character names against the
separate shared character library. Keep that library and all referenced image
and audio files available if you move a project to another machine.

Immediately after loading a project, the application checks the expected scene
MP4s in its saved output directory. Valid, correctly timed clips receive check
marks, and the status log reports how many scenes are reusable, missing, or
invalid. This inventory determines which scenes are included in a cost estimate.

After loading, refresh the model list if necessary. The saved model is selected
only if its ID exists in the current live list. Also verify that the soundtrack
still exists: loading displays the saved duration but does not probe the file
again until you rebuild the scene plan or generate.

Project JSON is human-readable, but editing it by hand is not recommended.
There is no autosave while you edit. Save explicitly if you want to preserve
work before generation.

## 9. Generate the project

### Estimate cost before generation

Select **Estimate Cost** to quote the scenes that still require generation
without saving the project, creating output files, or queueing video jobs. The
tool first scans the expected scene files again, then requests one live Venice
quote per missing or invalid scene using the selected model, scene duration,
and the resolution and aspect ratio that generation will use. Reusable scenes
cost zero and are excluded. The tool displays the total in Venice credits and
adds it to the status log. The quote endpoint
returns USD; the tool converts it at Venice's documented account-credit value
of 1 credit = $1, so the numeric value is unchanged.

An estimate requires a live model and supported scene durations, but does not
require completed prompts, a project name, an output directory, or local media
files. Quotes are informational and the amount ultimately charged by Venice may
differ. Change the model or scene plan and select **Estimate Cost** again to get
an updated total. The estimate logged automatically when generation starts uses
the same credit format.

Before selecting **Start Generation**, confirm that:

- The project has a valid name and output directory.
- Every scene has a prompt.
- A live model is selected and supports every scene duration.
- Referenced audio and image files still exist.
- Each reference-model scene is within the model's image limit.
- You are willing to replace the final project video if prompted.

At the start of a build, the application saves the project JSON and validates
all existing scene MP4s. It asks Venice for a cost quote only for scenes that
still require generation. The total is displayed in the status log. If the
quote service is unavailable, generation continues and the log says that the
estimate is unavailable.

The application then processes each scene in order:

1. Reuse an existing scene file if it has a video stream and its duration is
   within one second of the planned duration.
2. Otherwise, compose the full prompt from the overall style, selected characters, and
   scene prompt.
3. Upload applicable reference images as data URLs.
4. Queue the scene with Venice and poll until it completes.
5. Download and validate the scene MP4 in `video/scenes`.
6. Join all scene files with FFmpeg.
7. Prepend the selected still title card and delay the original soundtrack by
   the same 5–10 second duration.
8. Replace, mix, retain, or remove audio according to the selected audio mode.
9. Optionally burn the approved SRT with the selected font and size.
10. Validate and save outputs in `video/output`.

The progress bar counts completed scenes. The status log shows the model,
duration plan, estimate, queue progress, and paths of completed files. While a
worker is active, all controls except **Cancel Generation** are disabled.
Cancellation stops remaining work and removes `tmp`. If a billed Venice request
has already been queued, the confirmation dialog warns that stopping locally
may not prevent that request from being billed. Successfully downloaded scene
files remain reusable after cancellation or failure.

### Face-media consent

Some Seedance requests involving faces require explicit provider consent. If
Venice returns that requirement, the application displays the explanation and
links supplied by the service. Read them and choose whether to continue. If you
decline, the project stops; if you accept, only the affected request is
resubmitted with the required consent record.

## 10. Prompting and production tips

- Decide character descriptions and the global look before writing individual
  scenes.
- Keep names spelled exactly the same in prose and in the character library.
- Use the character checkboxes as the authoritative list for each scene.
- Give each clip one clear dramatic beat rather than many rapid events.
- State camera movement explicitly and keep it achievable within the duration.
- Describe both the opening and ending composition when continuity matters.
- Avoid asking the model to draw readable titles; add titles in an editor after
  generation.
- Test a short one-scene project before committing to an expensive long build.
- If you mix model sound with music, normalize or adjust the soundtrack in an
  audio editor first because this tool mixes without level balancing controls.
- Keep the project JSON, soundtrack, reference images, scene MP4s, and final MP4
  together when archiving a production.

## 11. Troubleshooting

### No models appear

Check the API key and network connection, then select **Refresh Models**. The
tool only accepts models whose live metadata identifies text-to-video or
reference-to-video support and includes a 30-second duration. A model available
elsewhere may therefore be absent from this list.

### “Refresh and select a current 30-second video model”

The text initially visible in the model box is not a loaded model record. Use
**Refresh Models**, wait for the completion message, and choose an entry.

### A scene duration is unsupported

Select another model or change that scene to one of the durations advertised by
the model. Although the editor offers 10, 15, and 30 seconds, a particular model
may support only some of them.

### FFmpeg or FFprobe is missing

Install FFmpeg and restart the application from an environment where both
`ffmpeg` and `ffprobe` are on `PATH`. FFprobe is used to measure soundtracks and
inspect audio streams; FFmpeg joins clips and creates the final audio mix.

### A reference image is missing

Edit the character and select its current image path, or remove the image from
the character record. Loading a project does not copy or restore referenced
files.

### The final video is shorter than the scene plan

When an MP3 is present, the final file is intentionally limited to the MP3's
duration. If the scene plan is shorter than the soundtrack because of manual
edits, FFmpeg's shortest-input behavior can instead end the result with the
available video.

### The final video is silent

If no MP3 was selected and the remove-audio checkbox was enabled, silence is
expected. If mixing was selected, the chosen model may simply have returned
clips without audio.

### Generation stops partway through

Read the final error in the status log. Common causes include a Venice API
error, insufficient balance, a missing local file, an unsupported model
constraint, a network timeout, or rejected face-media consent. Completed scene
MP4s remain in the output directory, but the application does not resume from
them automatically.

### The window will not close

The application prevents closing while models are loading or a generation is
active, to avoid abandoning a worker in progress. Wait for the operation to
finish. If the process is forcibly terminated, the current API job may continue
remotely and the local project may be incomplete.

## 12. Files and privacy

The application sends scene prompts and, for reference models, selected
character images to Venice. The MP3 soundtrack is processed locally and is not
included in the video-generation request. The final stitching and audio work
also happen locally through FFmpeg.

Project JSON files contain local paths, prompts, and character names. The shared
character JSON contains descriptions and image paths. Treat these files as
production data when sharing or archiving them.
