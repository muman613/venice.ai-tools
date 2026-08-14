# Venice.ai Tools

Small Python utilities for working with the Venice.ai API.

## Tools

- `venice-ai-story-writer.py` - PySide6 desktop chat/story-writing front end for Venice chat models.
- `venice-ai-image.py` - PySide6 desktop tool for generating images with Venice image models.
- `venice-ai-video.py` - PySide6 desktop workflow for generating variable-length videos as sequential image-to-video or text-to-video clips, then stitching them with FFmpeg.
- `venice-ai-tts.py` - PySide6 desktop tool for converting TXT or HTML input files to Venice text-to-speech MP3 output.
- `list-models.py` - Prints Venice model metadata matching Wan 2.7 from the models endpoint.

## Requirements

- Python 3.10 or newer
- A Venice.ai API key
- FFmpeg installed and available on `PATH` for video generation

Python packages are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your Venice API key:

```bash
export VENICE_API_KEY="your-api-key"
```

You can also put the key in a local `.env` or `venice.env` file:

```bash
VENICE_API_KEY="your-api-key"
```

On Ubuntu/Debian, install FFmpeg if you plan to use the video generator:

```bash
sudo apt install ffmpeg
```

## Usage

Run the story-writing GUI:

```bash
python venice-ai-story-writer.py
```

### Story Writer

`venice-ai-story-writer.py` is a desktop writing assistant for Venice chat models.

- Enter or edit your prompt in the `Request` text area.
- Click `Send`, or press `Ctrl+Enter`, to generate a response.
- The previous visible response is cleared when a new request is sent.
- The prompt remains in the `Request` area after generation so you can revise it and send again.
- The `Clear` button clears only the request text.
- `Clear Chat` clears the saved in-memory conversation history and the visible response.
- `Save Story` exports the latest generated response as TXT, Markdown, or HTML.
- `Export Chat` exports prompts and responses together as Markdown.
- `Export HTML` exports the latest rendered response as HTML.

The `Max tokens` control limits how much text Venice can return. If a response stops because it reached that limit, the status bar reports that explicitly. Increase `Max tokens` or ask the model to continue when that happens.

Run the image generator:

```bash
python venice-ai-image.py
```

The image tool loads current Venice image models and style presets, then saves generated variants to the selected output directory. Choose width/height, aspect-ratio, or resolution-tier sizing depending on the selected model.

Run the video generator:

```bash
python venice-ai-video.py
```

The video tool loads current Venice video models from `/api/v1/models?type=video`. Choose `Image-to-video` or `Text-to-video` from the `Mode` dropdown, choose a model from the refreshed `Video model` dropdown, and set the requested output length in seconds. When model metadata marks models as uncensored, the dropdown prefers those models.

Use `Retain intermediate files` to keep or remove generated segment MP4s, continuation frames, and the segment JSON after the final MP4 is saved. The checkbox setting is remembered between runs.

Run the text-to-speech GUI:

```bash
python venice-ai-tts.py
```

Choose one or more `.txt`, `.html`, or `.htm` files, select an output directory and voice, then add them to the queue. Files are converted to MP3s in order in the background, and the queue list shows running, completed, and failed items. HTML files are extracted to text automatically before conversion.

Run the model listing helper:

```bash
python list-models.py
```

## Notes

- `venice-ai-video.py` requires a starting reference image in image-to-video mode and writes the final output as an MP4.
- The video generator creates one segment MP4 per planned segment before producing the final stitched video.
- `venice-ai-story-writer.py` stores no chat history automatically; use the GUI export buttons to save story text or full chat history.
