# Baseline RAG System

This directory contains a simple chunked RAG baseline for comparison against the
existing TreeSeg retrievers.

It supports:

- `utterance_packed` chunking
- `raw_token_window` chunking
- chunk sizes `128`, `256`, `512`
- overlap `0%` and `10%`
- `transcript_only` and `combined_ocr` modes

## Environment Setup

From the repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements/lpm_requirements.txt
pip install ollama
export MPLCONFIGDIR=/tmp/matplotlib
```

If you want answer generation, make sure your Ollama server is running.

## List Available Lectures

```bash
python -m baseline_rag_system.main --list-lectures
```

## Run A Single Baseline Query

Transcript-only example:

```bash
python -m baseline_rag_system.main \
  --lecture "ml-1/MultimodalMachineLearning/01" \
  --query "What is self-attention?" \
  --chunk-strategy utterance_packed \
  --chunk-size 256 \
  --overlap-percent 10 \
  --ocr-mode transcript_only
```

OCR-enabled example:

```bash
python -m baseline_rag_system.main \
  --lecture "ml-1/MultimodalMachineLearning/01" \
  --query "What is self-attention?" \
  --chunk-strategy raw_token_window \
  --chunk-size 128 \
  --overlap-percent 10 \
  --ocr-mode combined_ocr \
  --answer
```

Useful flags:

- `--answer`: generate an answer from retrieved context
- `--rerank`: rerank retrieved chunks with the cross-encoder
- `--rerank-top-n 5`: keep the top `N` chunks after reranking
- `--top-k 10`: retrieve this many chunks before truncation/reranking

## Interactive CLI

You can also run the baseline exactly like the TreeSeg CLI: select a lecture once,
then submit multiple queries in the same session.

```bash
python -m baseline_rag_system
```

Or:

```bash
python -m baseline_rag_system.main
```

If you already know the lecture, start the session directly on it:

```bash
python -m baseline_rag_system \
  --lecture "ml-1/MultimodalMachineLearning/01" \
  --chunk-strategy utterance_packed \
  --chunk-size 256 \
  --overlap-percent 10 \
  --ocr-mode transcript_only
```

The CLI will:

- show the selected lecture
- show the active baseline configuration
- keep the lecture fixed for the session
- accept multiple queries until you enter `exit`, `quit`, `q`, or an empty line

## Run EduVidQA Comparison

Small run with one baseline variant:

```bash
python eduvid_evaluation/eduvid_evaluation.py \
  --baseline \
  --limit 5 \
  --baseline-chunk-strategy utterance_packed \
  --baseline-chunk-size 256 \
  --baseline-overlap-percent 10 \
  --baseline-ocr-mode transcript_only
```

This evaluates:

- TreeSeg `leaf`
- TreeSeg `summary_tree`
- the selected baseline configuration

Run all baseline variants:

```bash
python eduvid_evaluation/eduvid_evaluation.py --baseline
```

Example with reranking and a custom retrieval tolerance:

```bash
python eduvid_evaluation/eduvid_evaluation.py \
  --baseline \
  --limit 20 \
  --rerank \
  --rerank-top-n 5 \
  --retrieval-tolerance-seconds 15
```

## Outputs

EduVid evaluation outputs are written to:

`eduvid_evaluation/storage/evaluation_outputs/`

Key files:

- `summary.json`
- `metrics_per_question.csv`
- `<system_name>_predictions.jsonl`

## OCR Note

The current checked-in EduVid bundles appear transcript-only. That means
baseline configurations using `--baseline-ocr-mode combined_ocr` will be
skipped unless OCR artifacts exist for those videos.

To preprocess EduVid videos with full OCR artifacts, run:

```bash
python eduvid_evaluation/process_dataset.py --limit 5
```

That full mode can produce:

- `<video_id>_transcripts.csv`
- `segments.txt`
- `slide_*_ocr.csv`

## Smoke Test

Run the local baseline smoke checks:

```bash
python baseline_rag_system/test_baseline_rag.py
```
