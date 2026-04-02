# YouTube -> LPM TreeSeg Preprocessing

This folder contains a standalone local pipeline that converts one YouTube lecture into
TreeSeg-compatible files under a configurable local data root. By default the script
writes to `masters-thesis/custom_data`, without modifying anything in `LPMDataset/`.

## Output Contract

For a target lecture directory:

`<data_root>/<speaker>/<course_dir>/<meeting_id>/`

the pipeline writes:

- `<video_id>_transcripts.csv` with exact header `Word,Start,End`
- `segments.txt` with one ascending slide-end timestamp per line
- `slide_000.jpg`, `slide_001.jpg`, ...
- `slide_000_ocr.csv`, `slide_001_ocr.csv`, ...

These files are directly consumable by existing readers in:

- `masters-thesis/utterances.py`
- `masters-thesis/treeseg_lpm.py`
- `masters-thesis/treeseg_vector_index_modular/lecture_catalog.py`

For the EduVidQA batch workflow, use `masters-thesis/eduvid_evaluation/process_dataset.py`.
That pipeline writes video-centric bundles under
`masters-thesis/eduvid_evaluation/storage/videos/<video_id>/` while preserving the same
transcript / segment / slide / OCR file names inside each bundle.

## Why this pipeline

- Replaces Google ASR with local `openai/whisper` (`word_timestamps=True`)
- Uses automatic slide segmentation via PySceneDetect
- Uses Tesseract OCR to produce `slide_*_ocr.csv` files in the expected schema

## Prerequisites

### 1) System tools

Install these once (macOS/Homebrew example):

```bash
brew install ffmpeg yt-dlp tesseract
```

### 2) Python environment

From `masters-thesis/`:

```bash
python3.11 -m venv .venv_lpm_preproc
source .venv_lpm_preproc/bin/activate
pip install -U pip
pip install -r preprocessing/requirements_preprocessing.txt
```

If YouTube blocks `yt-dlp` with a "Sign in to confirm you're not a bot" error, export one
of these environment variables before running the pipeline:

```bash
export YT_DLP_COOKIES_FROM_BROWSER=chrome
```

or

```bash
export YT_DLP_COOKIES_FILE=~/Downloads/youtube_cookies.txt
```

If YouTube challenge solving still fails, the official yt-dlp recommendation for pip installs
is to install the default dependency group, which includes `yt-dlp-ejs`:

```bash
python -m pip install -U "yt-dlp[default]"
```

If Deno still fails after that, try Node as the JS runtime:

```bash
export YT_DLP_JS_RUNTIMES=node
```

If you explicitly want remote EJS downloads, you can opt in with:

```bash
export YT_DLP_REMOTE_COMPONENTS=ejs:npm
```

## Main command

From `masters-thesis/`:

```bash
python preprocessing/youtube_to_lpm.py \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --speaker ml-1 \
  --course-dir MultimodalMachineLearning \
  --meeting-id 99
```

By default this writes to `custom_data/ml-1/MultimodalMachineLearning/99`. Use
`--data-root` if you want a different destination.

Optional knobs:

- `--whisper-model turbo`
- `--language en`
- `--scene-threshold 25.0`
- `--min-scene-len 3.0`
- `--text-change-filter` / `--no-text-change-filter` (enabled by default)
- `--text-sim-threshold 0.85`
- `--ocr-min-conf 60`
- `--ocr-min-tokens 6`
- `--max-same-text-span 900`
- `--keep-temp-audio`
- `--delete-video-after-processing`
- `--skip-validate`
- `--dry-run`

## Validate a generated lecture directory

```bash
python preprocessing/validate_lpm_meeting.py \
  --meeting-dir custom_data/ml-1/MultimodalMachineLearning/99
```

If validation passes, your meeting directory should be directly usable by current TreeSeg
and utterance extraction code.

## Dry run

Use dry-run to preview actions and paths without downloading/transcribing:

```bash
python preprocessing/youtube_to_lpm.py \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --speaker ml-1 \
  --course-dir MultimodalMachineLearning \
  --meeting-id 99 \
  --dry-run
```
