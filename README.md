# Venice.ai Tools

Small Python utilities for working with the Venice.ai API.

## Tools

- `venice-story-writer.py` - PySide6 desktop chat/story-writing front end for Venice chat models.
- `venice-ai-oneminute.py` - PySide6 desktop workflow for generating a one-minute video as four sequential 15-second Wan 2.7 image-to-video clips, then stitching them with FFmpeg.
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
python venice-story-writer.py
```

Run the one-minute video generator:

```bash
python venice-ai-oneminute.py
```

Run the model listing helper:

```bash
python list-models.py
```

## Notes

- `venice-ai-oneminute.py` requires a starting reference image and writes the final output as an MP4.
- The video generator creates four segment MP4 files before producing the final stitched video.
- `venice-story-writer.py` stores no chat history automatically; use the GUI export buttons to save story text or full chat history.
