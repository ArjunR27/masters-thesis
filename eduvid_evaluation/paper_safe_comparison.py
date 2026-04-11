from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = SCRIPT_DIR / "storage"
SUMMARY_PATH = STORAGE_DIR / "evaluation_outputs" / "summary.json"
REAL_WORLD_PATH = STORAGE_DIR / "eduvid_data" / "real_world_test.csv"
SYNTHETIC_TEST_PATH = STORAGE_DIR / "eduvid_data" / "synthetic_test.csv"
SYNTHETIC_TRAIN_PATH = STORAGE_DIR / "eduvid_data" / "synthetic_train.csv"

PAPER_REAL_WORLD_METRICS = {
    "Video LlaVA 7B (paper)": {"bleu": 12.47, "rouge_l": 21.84, "meteor": 17.55},
    "mPLUG Owl 3 8B (paper)": {"bleu": 18.17, "rouge_l": 31.79, "meteor": 19.64},
    "Qwen VL 7B (paper)": {"bleu": 18.36, "rouge_l": 33.00, "meteor": 19.22},
    "Llava 13B (paper)": {"bleu": 17.16, "rouge_l": 20.03, "meteor": 31.41},
    "GPT4o (paper)": {"bleu": 15.03, "rouge_l": 32.87, "meteor": 22.46},
    "Gemini 1.5 (paper)": {"bleu": 13.98, "rouge_l": 30.76, "meteor": 21.25},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a paper-safe comparison between local EduVidQA ASR evaluation "
            "results and the paper's real-world Table 4 text metrics."
        )
    )
    parser.add_argument(
        "--summary-path",
        default=str(SUMMARY_PATH),
        help=f"Path to the local evaluation summary JSON (default: {SUMMARY_PATH}).",
    )
    return parser


def count_csv_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def as_percent(value: float) -> float:
    return value * 100.0


def render_row(label: str, bleu: float, rouge_l: float, meteor: float) -> str:
    return f"| {label} | {bleu:.2f} | {rouge_l:.2f} | {meteor:.2f} |"


def main() -> int:
    args = build_parser().parse_args()
    summary_path = Path(args.summary_path).expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    local_real_world_rows = count_csv_rows(REAL_WORLD_PATH)
    local_synthetic_rows = count_csv_rows(SYNTHETIC_TEST_PATH) + count_csv_rows(
        SYNTHETIC_TRAIN_PATH
    )

    print("# EduVidQA Paper-Safe Comparison")
    print()
    print("## Local Run")
    print(
        f"- Summary file: {summary_path}"
    )
    print(
        f"- Requested questions: {summary.get('requested_question_count', 'n/a')}"
    )
    print(
        f"- Evaluated questions: {summary.get('evaluated_question_count', 'n/a')}"
    )
    print(
        f"- Skipped questions: {summary.get('skipped_question_count', 'n/a')}"
    )
    print(
        f"- Local real-world rows: {local_real_world_rows} vs paper real-world rows: 270"
    )
    print(
        f"- Local synthetic rows: {local_synthetic_rows} vs paper synthetic rows: 4982"
    )
    print()
    print("## Text Metrics")
    print()
    print("| System | BLEU | ROUGE-L | METEOR |")
    print("| --- | ---: | ---: | ---: |")

    systems = summary.get("systems", {})
    for system_name, metrics in systems.items():
        print(
            render_row(
                f"local {system_name}",
                as_percent(float(metrics.get("bleu1", 0.0))),
                as_percent(float(metrics.get("rouge_l", 0.0))),
                as_percent(float(metrics.get("meteor", 0.0))),
            )
        )

    for label, metrics in PAPER_REAL_WORLD_METRICS.items():
        print(
            render_row(
                label,
                float(metrics["bleu"]),
                float(metrics["rouge_l"]),
                float(metrics["meteor"]),
            )
        )

    print()
    print("## Caveats")
    print(
        "- Local results are ASR-only RAG over transcript bundles with a `llama3.2` answer generator, not the paper's VLM/MLLM setup."
    )
    print(
        "- The paper uses a 4-minute timestamp-centered transcript or audio context window for model evaluation."
    )
    print(
        "- Local BLEU/ROUGE-L/METEOR are custom implementations in `eduvid_evaluation.py`; exact metric backend parity with the paper is not verified."
    )
    print(
        "- Compare the local rows only to the paper's real-world table, not the synthetic split."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
