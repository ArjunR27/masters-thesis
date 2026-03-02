# Repository Guidelines

## Project Structure & Module Organization
Core retrieval and segmentation scripts live at the repo root:
- `treeseg_vector_index.py`: main retrieval pipeline (TreeSeg + vector search + optional reranking).
- `treeseg_lpm.py`, `utterances.py`: utterance extraction and TreeSeg data prep utilities.
- `test.py`: lightweight top-level smoke script.

Supporting directories:
- `retriever_evaluation/`: evaluation workflow and datasets (`retriever_evaluation.py`, `retriever_evaluation.csv`, `segment_dumps/`).
- `data_exploration/`: analysis scripts for transcript/segmentation behavior.
- `treeseg_exploration/`: TreeSeg experimentation code, dataset adapters, and exploratory tests.
- `lpm_data/`: local lecture/transcript data assets (large; treat as data, not source code).

## Build, Test, and Development Commands
Use Python 3.11+ in a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r lpm_requirements.txt
```

Common run commands:
- `python treeseg_vector_index.py --list-lectures` lists discovered lectures.
- `python treeseg_vector_index.py --lecture <speaker/course/meeting> --query "..."` runs non-interactive retrieval.
- `python retriever_evaluation/retriever_evaluation.py` runs combined/separate retriever evaluation.
- `python data_exploration/utterance_gap_analysis.py` plots transcript word-gap distributions.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation.
- Use `snake_case` for functions/variables and `PascalCase` for classes/dataclasses (for example, `LectureDescriptor`).
- Keep modules script-friendly: clear `main()` entry points and small, composable helpers.
- Prefer explicit paths via `pathlib.Path` for new code.

## Testing Guidelines
- No formal `pytest` suite is configured; tests are currently script-based smoke checks.
- Run `python test.py` and `python treeseg_exploration/test.py` before opening a PR.
- For retrieval changes, include a reproducible check with `python retriever_evaluation/retriever_evaluation.py` and summarize key metric deltas.

## Commit & Pull Request Guidelines
- Existing history favors short, direct commit subjects (often lowercase, action-oriented).
- Recommended format: `<area>: <imperative summary>` (example: `retriever: add OCR reranker fallback`).
- Keep commits focused; separate refactors from behavior changes.
- PRs should include: purpose, affected scripts/data paths, commands run, and before/after outputs for retrieval or evaluation changes.
