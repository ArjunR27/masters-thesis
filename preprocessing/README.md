# YouTube -> LPM TreeSeg Preprocessing

This folder contains a standalone local pipeline that converts one YouTube lecture into
TreeSeg-compatible files under `masters-thesis/lpm_data`, without modifying anything in
`LPMDataset/`.

## Output Contract

For a target lecture directory:

`masters-thesis/lpm_data/<speaker>/<course_dir>/<meeting_id>/`

the pipeline writes:

- `<video_id>_transcripts.csv` with exact header `Word,Start,End`
- `segments.txt` with one ascending slide-end timestamp per line
- `slide_000.jpg`, `slide_001.jpg`, ...
- `slide_000_ocr.csv`, `slide_001_ocr.csv`, ...

These files are directly consumable by existing readers in:

- `masters-thesis/utterances.py`
- `masters-thesis/treeseg_lpm.py`
- `masters-thesis/treeseg_vector_index_modular/lecture_catalog.py`

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

## Main command

From `masters-thesis/`:

```bash
python preprocessing/youtube_to_lpm.py \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --speaker ml-1 \
  --course-dir MultimodalMachineLearning \
  --meeting-id 99 \
  --text-change-filter
```

Optional knobs:

- `--whisper-model turbo`
- `--language en`
- `--scene-threshold 2.0`
- `--min-scene-len 1.0`
- `--text-change-filter` (prune scene boundaries when OCR text is mostly unchanged)
- `--text-sim-threshold 0.80`
- `--ocr-min-conf 60`
- `--ocr-min-tokens 6`
- `--max-same-text-span 900`
- `--keep-temp-audio`
- `--skip-validate`
- `--dry-run`

## Validate a generated lecture directory

```bash
python preprocessing/validate_lpm_meeting.py \
  --meeting-dir lpm_data/ml-1/MultimodalMachineLearning/99
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
