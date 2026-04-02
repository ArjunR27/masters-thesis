# EduVidQA Preprocessing And Evaluation

This directory contains the batch preprocessing pipeline for the EduVidQA dataset.
It reads an EduVidQA CSV, deduplicates rows by YouTube `id`, downloads each unique
video once, and stores transcript-first or full ASR/OCR artifacts in a clean per-video bundle.
Successful runs delete temporary source media to save space.
The default batch settings also use stricter scene detection, OCR-based boundary pruning,
edge masking, and visual slide deduplication to reduce noisy lecture-camera transitions.

## Preprocessing

From the workspace root:

```bash
python masters-thesis/eduvid_evaluation/process_dataset.py
```

That defaults to `eduvidqa-emnlp25/data/real_world_test.csv`.

Optional filters:

```bash
python masters-thesis/eduvid_evaluation/process_dataset.py --asr
python masters-thesis/eduvid_evaluation/process_dataset.py --split synthetic_test
python masters-thesis/eduvid_evaluation/process_dataset.py --video-id F9-yqoS7b8w
python masters-thesis/eduvid_evaluation/process_dataset.py --limit 5 --dry-run
```

## Evaluation

`eduvid_evaluation.py` evaluates the `real_world_test.csv` split with the existing
ASR-only transcript bundles and compares two retrievers:

- `leaf`
- `summary_tree`

It generates answers with the same Ollama-backed responder for both retrievers and
writes:

- `storage/evaluation_outputs/leaf_predictions.jsonl`
- `storage/evaluation_outputs/summary_tree_predictions.jsonl`
- `storage/evaluation_outputs/metrics_per_question.csv`
- `storage/evaluation_outputs/summary.json`

Main command:

```bash
python masters-thesis/eduvid_evaluation/eduvid_evaluation.py
```

Optional runs:

```bash
python masters-thesis/eduvid_evaluation/eduvid_evaluation.py --limit 10
python masters-thesis/eduvid_evaluation/eduvid_evaluation.py --leaf
python masters-thesis/eduvid_evaluation/eduvid_evaluation.py --summary_tree
python masters-thesis/eduvid_evaluation/eduvid_evaluation.py --leaf --limit 10
```

The evaluator uses only ASR transcripts and computes:

- `BLEU-1`
- `ROUGE-L`
- `METEOR`
- `FactQA-Precision`
- `FactQA-Recall`

## Storage layout

Outputs are written under:

`masters-thesis/eduvid_evaluation/storage/`

Per unique video:

`storage/videos/<video_id>/`

Each video bundle always contains:

- `<video_id>_transcripts.csv`
- `questions.jsonl` with all dataset questions currently associated with that video
- `metadata.json`

Full mode also adds:

- `segments.txt`
- `slide_000.jpg`, `slide_001.jpg`, ...
- `slide_000_ocr.csv`, `slide_001_ocr.csv`, ...

During processing you may briefly see `<video_id>.mp4` and `<video_id>.wav`, but the
batch pipeline removes both after a successful run.

Global manifests:

`storage/manifests/<split>/`

- `questions.csv`: one row per dataset question with parsed timestamp seconds
- `videos.csv`: one row per unique video with artifact paths and status
- `failures.csv`: failed videos with stage and error message
- `run_summary.json`: aggregate run counts
- `evaluation_outputs/`: answer-generation outputs and metric summaries from `eduvid_evaluation.py`

## Behavior

- Existing valid video bundles are skipped automatically on reruns.
- `--asr` treats a valid transcript as complete and skips segmentation / OCR entirely.
- A later full run can reuse the existing transcript and add `segments.txt`, slide JPGs, and OCR CSVs.
- Successful runs do not retain the downloaded MP4 or WAV.
- Each video directory keeps the existing transcript / slide / OCR file names so it
  stays compatible with the current validation logic.
- Timestamps are normalized from `MM:SS` or `H:MM:SS` into integer seconds for later
  evaluation scripts.

## Requirements

Use the same prerequisites as `masters-thesis/preprocessing/README.md`:

- `yt-dlp`
- `ffmpeg`
- `tesseract`
- Python dependencies from `masters-thesis/preprocessing/requirements_preprocessing.txt`

## YouTube cookies

If YouTube blocks `yt-dlp` with a "Sign in to confirm you're not a bot" error, export one
of these environment variables before running the pipeline:

```bash
export YT_DLP_COOKIES_FROM_BROWSER=chrome
```

or

```bash
export YT_DLP_COOKIES_FILE=~/Downloads/youtube_cookies.txt
```

The batch pipeline will pass that through to `yt-dlp` automatically, so the main Python
command does not need any extra flags.

Whisper now defaults to CPU for stability. If you explicitly want to try a different device,
you can override it with:

```bash
export WHISPER_DEVICE=mps
```

If YouTube challenge solving still fails, the official yt-dlp recommendation for pip installs
is to install the default dependency group, which includes `yt-dlp-ejs`:

```bash
venv_lpm_preproc/bin/python -m pip install -U "yt-dlp[default]"
```

If Deno still fails after that, try Node as the JS runtime:

```bash
export YT_DLP_JS_RUNTIMES=node
```

If you explicitly want remote EJS downloads, you can opt in with:

```bash
export YT_DLP_REMOTE_COMPONENTS=ejs:npm
```
