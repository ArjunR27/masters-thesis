# TreeSeg Vector Index Modular

This directory contains the main TreeSeg-based retrieval system over the lecture
bundles in `lpm_data/`.

It supports:

- TreeSeg `leaf` retrieval
- TreeSeg `summary_tree` retrieval
- `combined` retrieval over ASR segments with OCR attached
- `separate` retrieval with ASR and OCR indexed independently
- optional cross-encoder reranking
- interactive multi-query sessions on a selected lecture

## Environment Setup

From the repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements/lpm_requirements.txt
pip install ollama
export MPLCONFIGDIR=/tmp/matplotlib
```

If you want answer generation in interactive mode, make sure your Ollama server
is running.

## List Available Lectures

```bash
python -m treeseg_vector_index_modular.main --list-lectures
```

## Interactive CLI

Run the main interactive CLI:

```bash
python -m treeseg_vector_index_modular.main
```

The CLI will:

- show or let you select a lecture
- keep that lecture fixed for the session
- accept multiple queries at `search>`
- keep going until you enter `exit`, `quit`, `q`, or an empty line

If you already know the lecture, start directly on it:

```bash
python -m treeseg_vector_index_modular.main \
  --lecture "ml-1/MultimodalMachineLearning/01"
```

## Run A Single Query

Default example:

```bash
python -m treeseg_vector_index_modular.main \
  --lecture "ml-1/MultimodalMachineLearning/01" \
  --query "What is self-attention?"
```

That uses:

- `--retrieval-mode combined`
- `--index-kind leaf`

## Important Flags

### Retrieval Mode

`--retrieval-mode combined`

- builds one TreeSeg ASR index where OCR is attached to the ASR segments
- prints one ranked result list

`--retrieval-mode separate`

- builds one TreeSeg ASR index and one OCR-only slide index
- prints separate ASR and OCR result lists

Examples:

```bash
python -m treeseg_vector_index_modular.main \
  --lecture "ml-1/MultimodalMachineLearning/01" \
  --query "What is self-attention?" \
  --retrieval-mode combined
```

```bash
python -m treeseg_vector_index_modular.main \
  --lecture "ml-1/MultimodalMachineLearning/01" \
  --query "What is self-attention?" \
  --retrieval-mode separate
```

### Index Kind

`--index-kind leaf`

- indexes TreeSeg leaf segments directly
- faster and simpler

`--index-kind summary_tree`

- indexes the TreeSeg summary tree
- retrieves summary nodes and expands them with supporting leaf evidence
- usually better when you want broader semantic coverage

Examples:

```bash
python -m treeseg_vector_index_modular.main \
  --lecture "ml-1/MultimodalMachineLearning/01" \
  --query "What is self-attention?" \
  --index-kind leaf
```

```bash
python -m treeseg_vector_index_modular.main \
  --lecture "ml-1/MultimodalMachineLearning/01" \
  --query "What is self-attention?" \
  --index-kind summary_tree
```

### Reranking

Use `--rerank` to rerank the retrieved results with a cross-encoder.

Useful flags:

- `--rerank`
- `--rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2`
- `--ocr-rerank-model BAAI/bge-reranker-v2-m3`
- `--rerank-top-n 5`

Example:

```bash
python -m treeseg_vector_index_modular.main \
  --lecture "ml-1/MultimodalMachineLearning/01" \
  --query "What is self-attention?" \
  --index-kind summary_tree \
  --retrieval-mode combined \
  --rerank \
  --rerank-top-n 5
```

## Common Run Patterns

### Leaf + Combined

```bash
python -m treeseg_vector_index_modular.main \
  --lecture "ml-1/MultimodalMachineLearning/01" \
  --query "What is self-attention?" \
  --index-kind leaf \
  --retrieval-mode combined
```

### Leaf + Separate

```bash
python -m treeseg_vector_index_modular.main \
  --lecture "ml-1/MultimodalMachineLearning/01" \
  --query "What is self-attention?" \
  --index-kind leaf \
  --retrieval-mode separate
```

### Summary Tree + Combined

```bash
python -m treeseg_vector_index_modular.main \
  --lecture "ml-1/MultimodalMachineLearning/01" \
  --query "What is self-attention?" \
  --index-kind summary_tree \
  --retrieval-mode combined
```

### Summary Tree + Separate

```bash
python -m treeseg_vector_index_modular.main \
  --lecture "ml-1/MultimodalMachineLearning/01" \
  --query "What is self-attention?" \
  --index-kind summary_tree \
  --retrieval-mode separate
```

## What The CLI Prints

For `combined` mode:

- one ranked ASR result list
- in interactive mode, an Ollama-generated answer from the retrieved context

For `separate` mode:

- one ASR result list
- one OCR result list
- in interactive mode, an Ollama-generated answer from the combined ASR + OCR context

## Summary Tree Inspection Tools

This directory also includes helper scripts for inspecting and testing the
summary-tree pipeline.

Build and inspect one summary tree:

```bash
python treeseg_vector_index_modular/test_summary_tree.py
```

List lectures for that script:

```bash
python treeseg_vector_index_modular/test_summary_tree.py --list-lectures
```

Inspect a specific lecture:

```bash
python treeseg_vector_index_modular/test_summary_tree.py \
  --lecture "ml-1/MultimodalMachineLearning/01"
```

Other helper scripts:

- `python treeseg_vector_index_modular/test_summary_tree_retrieval.py`
- `python treeseg_vector_index_modular/test_summary_tree_parallel_cache.py`

## Data Requirements

The main CLI expects lecture bundles under:

`lpm_data/{speaker}/{course}/{meeting}/`

Typical files:

- `<video_id>_transcripts.csv`
- `segments.txt`
- `slide_*_ocr.csv`

If OCR files are missing, `combined` and `separate` OCR behavior will only use
what is available for that lecture.
