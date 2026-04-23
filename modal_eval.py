"""
Run thesis evaluations on Modal cloud machines.

Setup
-----
1.  pip install modal
2.  modal setup                   (one-time authentication)
3.  Create a Modal secret with your OpenAI key:
        modal secret create thesis-openai OPENAI_API_KEY=sk-...

4.  Upload data to Modal Volumes ONE TIME (only redo if data changes):
        modal run modal_eval.py::upload_lpm
        modal run modal_eval.py::upload_eduvid

Usage
-----
# RAGAS + LPM-QA retrieval evaluation (CPU, OpenAI for generation + judging)
modal run modal_eval.py::ragas
modal run modal_eval.py::ragas --limit 20
modal run modal_eval.py::ragas --generator-model gpt-4o

# EduVid retrieval + QA evaluation (T4 GPU, Ollama/llama3.2)
modal run modal_eval.py::eduvid --limit 10
modal run modal_eval.py::eduvid --limit 20 --leaf-only --rerank

Download outputs
----------------
modal volume ls thesis-eval-outputs
modal volume get thesis-eval-outputs ragas/all_systems_summary.json .
modal volume get thesis-eval-outputs eduvid/summary.json .
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent

# ── Volumes ─────────────────────────────────────────────────────────────────────
# Data volumes are uploaded once and reused across every run.
# Source code is baked into the image (fast rebuild via Modal's layer cache).

lpm_vol = modal.Volume.from_name("thesis-lpm-data", create_if_missing=True)
eduvid_vol = modal.Volume.from_name("thesis-eduvid-storage", create_if_missing=True)
output_vol = modal.Volume.from_name("thesis-eval-outputs", create_if_missing=True)

# ── Ignore helper (source code only) ───────────────────────────────────────────

_SKIP = (
    ".git",
    ".venv",
    ".venv_lpm_preproc",
    "__pycache__",
    "lpm_data",
    "eduvid_evaluation/storage",
    "ragas_evaluation/outputs",
)


def _ignore_source(p: Path) -> bool:
    """Return True to exclude the path from the source sync."""
    s = str(p)
    return any(part in s for part in _SKIP)


# ── Images ─────────────────────────────────────────────────────────────────────
# We do NOT use pip_install_from_requirements with the local requirements file
# because that file was pinned on macOS/Python 3.13 and contains packages
# (torch==2.9.0, hf-xet, ir-measures) that may not have Linux wheels.
# Instead we install only what the eval scripts actually need.

_CORE_PKGS = [
    "torch",
    "faiss-cpu",
    "sentence-transformers",
    "transformers",
    "numpy",
    "pandas",
    "scikit-learn",
    "structlog",
    "yake",
    "segtok",
    "tabulate",
    "matplotlib",
    "wordcloud",
    "ollama",   # Python client — imported at module load even for openai backend
]

_RAGAS_PKGS = [
    "ragas>=0.2",
    "langchain-openai>=0.3",
    "openai>=1.0",
    "python-dotenv",
]

# RAGAS eval: CPU-only, OpenAI for generation and RAGAS judging.
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("zstd")
    .pip_install(*_CORE_PKGS)
    .pip_install(*_RAGAS_PKGS)
    .env({"TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(str(REPO_ROOT), remote_path="/repo", ignore=_ignore_source)
)

# EduVid eval: needs Ollama server binary + GPU for llama3.2 inference.
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install(*_CORE_PKGS)
    .pip_install("python-dotenv")
    .env({"TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(str(REPO_ROOT), remote_path="/repo", ignore=_ignore_source)
)

# ── App ─────────────────────────────────────────────────────────────────────────

app = modal.App("thesis-evaluation")


# ── One-time data upload entrypoints ──────────────────────────────────────────

@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/lpm_data": lpm_vol},
    timeout=60 * 30,
)
def _upload_lpm_fn(files: list[tuple[str, bytes]]) -> int:
    from pathlib import Path
    for rel_path, data in files:
        dest = Path("/lpm_data") / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    lpm_vol.commit()
    return len(files)


@app.local_entrypoint()
def upload_lpm():
    """Upload lpm_data/ to Modal Volume (run once; re-run only if data changes)."""
    lpm_dir = REPO_ROOT / "lpm_data"
    if not lpm_dir.exists():
        print("ERROR: lpm_data/ not found at", lpm_dir)
        return

    # Collect only transcript/OCR CSVs and segments.txt — skip large slide images.
    keep_suffixes = {".csv", ".txt"}
    files = []
    for path in sorted(lpm_dir.rglob("*")):
        if path.is_file() and path.suffix in keep_suffixes:
            rel = path.relative_to(lpm_dir)
            files.append((str(rel), path.read_bytes()))

    print(f"Uploading {len(files)} files from lpm_data/ ...")
    count = _upload_lpm_fn.remote(files)
    print(f"Done — {count} files in volume 'thesis-lpm-data'.")


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/eduvid_storage": eduvid_vol},
    timeout=60 * 30,
)
def _upload_eduvid_fn(files: list[tuple[str, bytes]]) -> int:
    from pathlib import Path
    for rel_path, data in files:
        dest = Path("/eduvid_storage") / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    eduvid_vol.commit()
    return len(files)


@app.local_entrypoint()
def upload_eduvid():
    """Upload eduvid_evaluation/storage/ to Modal Volume (run once)."""
    storage_dir = REPO_ROOT / "eduvid_evaluation" / "storage"
    if not storage_dir.exists():
        print("ERROR: eduvid_evaluation/storage/ not found at", storage_dir)
        return

    files = []
    for path in sorted(storage_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(storage_dir)
            files.append((str(rel), path.read_bytes()))

    print(f"Uploading {len(files)} files from eduvid_evaluation/storage/ ...")
    count = _upload_eduvid_fn.remote(files)
    print(f"Done — {count} files in volume 'thesis-eduvid-storage'.")


# ── RAGAS / LPM-QA Retrieval Evaluation ───────────────────────────────────────

@app.function(
    image=cpu_image,
    secrets=[modal.Secret.from_name("thesis-openai")],
    volumes={
        "/repo/lpm_data": lpm_vol,
        "/outputs": output_vol,
    },
    timeout=60 * 120,
    cpu=4.0,
    memory=16384,
)
def _ragas_fn(limit: int, generator_model: str, ragas_judge_model: str) -> dict:
    import json
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, "/repo")
    os.chdir("/repo")

    import ragas_evaluation.ragas_eval as script

    script.LIMIT = limit
    script.GENERATOR_BACKEND = "openai"
    script.GENERATOR_OPENAI_MODEL = generator_model
    script.RAGAS_JUDGE_MODEL = ragas_judge_model
    script.OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

    out_dir = Path("/outputs/ragas")
    out_dir.mkdir(parents=True, exist_ok=True)
    script.OUTPUT_DIR = out_dir

    script.main()
    output_vol.commit()

    summary_path = out_dir / "all_systems_summary.json"
    return json.loads(summary_path.read_text()) if summary_path.exists() else {}


@app.local_entrypoint()
def ragas(
    limit: int = 5,
    generator_model: str = "gpt-4o-mini",
    ragas_judge_model: str = "gpt-4o",
):
    """RAGAS evaluation on the LPM QA dataset (CPU, OpenAI backend).

    Examples:
        modal run modal_eval.py::ragas
        modal run modal_eval.py::ragas --limit 20
        modal run modal_eval.py::ragas --generator-model gpt-4o
    """
    import json

    print(
        f"Starting RAGAS eval  "
        f"limit={limit}  generator={generator_model}  judge={ragas_judge_model}"
    )
    summary = _ragas_fn.remote(
        limit=limit,
        generator_model=generator_model,
        ragas_judge_model=ragas_judge_model,
    )
    print("\n── Scores ──────────────────────────────────────────────────────")
    print(json.dumps(summary, indent=2))
    print("\nFull per-question CSVs: modal volume ls thesis-eval-outputs")


# ── EduVid Retrieval + QA Evaluation ──────────────────────────────────────────

@app.function(
    image=gpu_image,
    volumes={
        "/repo/eduvid_evaluation/storage": eduvid_vol,
        "/outputs": output_vol,
    },
    gpu="T4",
    timeout=60 * 180,
    cpu=4.0,
    memory=32768,
)
def _eduvid_fn(
    limit: int | None,
    leaf_only: bool,
    summary_tree_only: bool,
    rerank: bool,
) -> dict:
    import json
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, "/repo")
    os.chdir("/repo")

    print("Starting Ollama server...")
    subprocess.Popen(["ollama", "serve"])
    time.sleep(3)
    print("Pulling llama3.2 (first run downloads ~2 GB)...")
    subprocess.run(["ollama", "pull", "llama3.2"], check=True)
    print("Model ready.")

    import eduvid_evaluation.eduvid_evaluation as ev

    out_dir = Path("/outputs/eduvid")
    cache_dir = out_dir / "summary_tree_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ev.OUTPUT_DIR = out_dir

    argv: list[str] = ["--summary-tree-cache-dir", str(cache_dir)]
    if limit is not None:
        argv += ["--limit", str(limit)]
    if leaf_only:
        argv.append("--leaf")
    if summary_tree_only:
        argv.append("--summary_tree")
    if rerank:
        argv.append("--rerank")

    ev.main(argv)
    output_vol.commit()

    summary_path = out_dir / "summary.json"
    return json.loads(summary_path.read_text()) if summary_path.exists() else {}


@app.local_entrypoint()
def eduvid(
    limit: int | None = None,
    leaf_only: bool = False,
    summary_tree_only: bool = False,
    rerank: bool = False,
):
    """EduVid retrieval + QA evaluation on a Modal T4 GPU (Ollama/llama3.2).

    Examples:
        modal run modal_eval.py::eduvid --limit 10
        modal run modal_eval.py::eduvid --limit 10 --leaf-only
        modal run modal_eval.py::eduvid --limit 20 --rerank
    """
    import json

    print(
        f"Starting EduVid eval  "
        f"limit={limit}  leaf_only={leaf_only}  "
        f"summary_tree_only={summary_tree_only}  rerank={rerank}"
    )
    summary = _eduvid_fn.remote(
        limit=limit,
        leaf_only=leaf_only,
        summary_tree_only=summary_tree_only,
        rerank=rerank,
    )
    print("\n── Summary ─────────────────────────────────────────────────────")
    print(json.dumps(summary, indent=2))
    print("\nFull outputs: modal volume ls thesis-eval-outputs")
