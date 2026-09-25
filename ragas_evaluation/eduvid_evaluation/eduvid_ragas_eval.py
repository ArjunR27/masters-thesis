"""GPT-4o-mini generation + RAGAS judging for the EduVidQA dataset.

Mirrors ragas_evaluation/ragas_eval.py's generation and judging pipeline for
TinyLPM-QA — same generator (GPT-4o-mini via OpenAI Batch API), same judge
(gpt-4.1-mini), same five RAGAS metrics — so results are directly comparable.
All of that generic machinery is imported unchanged from ragas_eval.py, not
duplicated; this file only adds what's actually EduVidQA-specific: loading
questions from storage/videos/*/questions.jsonl and retrieval against EduVidQA's
transcripts, reusing the store/search functions already built and verified in
evaluate_eduvid_temporal_retrieval.py. Neither of those two files is modified
by this one.

Usage:
    python3 eduvid_ragas_eval.py
    python3 eduvid_ragas_eval.py --limit 20
    python3 eduvid_ragas_eval.py --systems treeseg__leaf treeseg__summary_tree

Output: ragas_evaluation/eduvid_evaluation/outputs/eduvid_ragas/
    <system>_per_question.csv, per_question_all_systems.csv,
    system_summary.csv, all_systems_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(PROJECT_DIR / "ragas_evaluation") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "ragas_evaluation"))

_dotenv_path = PROJECT_DIR / ".env"
if _dotenv_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_dotenv_path)

import os  # noqa: E402

# ── Reused, unmodified machinery from ragas_eval.py ───────────────────────────
import ragas_eval as ragas  # noqa: E402

# ── EduVidQA-specific retrieval, already built and verified ──────────────────
import evaluate_eduvid_temporal_retrieval as eduvid_retrieval  # noqa: E402

from treeseg_vector_index_modular.cross_encoder_reranker import CrossEncoderReranker  # noqa: E402
from treeseg_vector_index_modular.rerank_input_builder import RerankInputBuilder  # noqa: E402

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LIMIT = 150  # match TinyLPM-QA's evaluated question count by default

OUTPUT_DIR = HERE / "outputs" / "eduvid_ragas"
RETRIEVAL_CHECKPOINT_DIR = OUTPUT_DIR / "retrieval_checkpoints"
SUMMARY_TREE_CACHE_DIR = HERE / "storage" / "summary_tree_cache"


# ── Checkpointing (same pattern as ragas_eval.py, own output dir) ────────────

def _checkpoint_path(system_name: str) -> Path:
    return RETRIEVAL_CHECKPOINT_DIR / f"{system_name}.jsonl"


def _load_checkpoint(system_name: str) -> tuple[list, set[str]]:
    cp = _checkpoint_path(system_name)
    if not cp.exists():
        return [], set()
    results = []
    done_ids: set[str] = set()
    with cp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ex = {k: record[k] for k in ("example_id", "video_id", "question", "answer")}
            results.append((ex, record["context"], record["context_texts"]))
            done_ids.add(record["example_id"])
    if done_ids:
        print(f"  [checkpoint] Resuming — {len(done_ids)} questions already done.")
    return results, done_ids


def _append_checkpoint(system_name: str, ex: dict, context: str, context_texts: list[str]) -> None:
    RETRIEVAL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    record = {**ex, "context": context, "context_texts": context_texts}
    with _checkpoint_path(system_name).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── Phase A: retrieval (reuses eduvid_retrieval's own store/search machinery) ─

def _tree_store_args() -> argparse.Namespace:
    return argparse.Namespace(
        summary_tree_cache_dir=str(SUMMARY_TREE_CACHE_DIR),
        summary_tree_workers="auto",
        rebuild_summary_tree_cache=False,
    )


def build_rerankers() -> tuple[CrossEncoderReranker, CrossEncoderReranker]:
    # Force CPU: same MPS-memory fix already applied in evaluate_eduvid_temporal_retrieval.py.
    default_reranker = CrossEncoderReranker(eduvid_retrieval.RERANK_MODEL, device="cpu")
    summary_reranker = CrossEncoderReranker(
        eduvid_retrieval.RERANK_MODEL,
        device="cpu",
        input_builder=RerankInputBuilder.build_summary_tree_rerank_input,
    )
    return default_reranker, summary_reranker


def run_retrieval(
    spec: dict,
    lectures,
    by_video: dict,
    examples: list[dict],
    default_reranker: CrossEncoderReranker,
    summary_reranker: CrossEncoderReranker,
) -> list[tuple[dict, str, list[str]]]:
    name = spec["name"]
    kind = spec["kind"]

    results, done_ids = _load_checkpoint(name)
    remaining = [ex for ex in examples if ex["example_id"] not in done_ids]
    if not remaining:
        print(f"  [checkpoint] All {len(results)} questions already done — skipping store build.")
        return results

    print(f"  Building store for {name}...")
    if kind in ("leaf", "summary_tree"):
        store = eduvid_retrieval.build_tree_store(lectures, kind, _tree_store_args())
        reranker = summary_reranker if kind == "summary_tree" else default_reranker
    else:
        build_result = eduvid_retrieval.build_baseline_store(lectures, spec["config"])
        store = build_result.store
        reranker = default_reranker

    context_kind = "summary_tree" if kind == "summary_tree" else "leaf"

    for i, ex in enumerate(remaining, start=1):
        video_id = ex["video_id"]
        lecture = by_video.get(video_id)
        if lecture is None or lecture.key not in store.lecture_indices:
            continue
        if i % 25 == 0:
            print(f"    [{name}] {i}/{len(remaining)}")

        hits = eduvid_retrieval.search_hits(spec, store, ex["question"], lecture.key, reranker)
        context = ragas.build_context(hits, context_kind)
        context_texts = ragas.extract_context_texts(hits, context_kind)

        results.append((ex, context, context_texts))
        _append_checkpoint(name, ex, context, context_texts)

    del store
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-4o-mini + RAGAS eval on EduVidQA")
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--systems", nargs="+", default=None)
    parser.add_argument(
        "--videos-file",
        default=None,
        help="Optional path to a file listing one video_id per line "
        "(lines starting with # are ignored). Restricts evaluation to "
        "only these videos — pass the same file used for the retrieval "
        "evaluation (eduvid_sampled_videos.txt) for a matched comparison.",
    )
    args = parser.parse_args()

    print(f"Generator:   openai / {ragas.GENERATOR_OPENAI_MODEL}")
    print(f"Judge model: {ragas.RAGAS_JUDGE_MODEL}")
    print(f"Videos root: {eduvid_retrieval.VIDEOS_ROOT}")
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY is not set — batch calls will fail.")

    video_whitelist = eduvid_retrieval.load_video_whitelist(args.videos_file)
    if video_whitelist is not None:
        print(f"Video whitelist: {len(video_whitelist)} videos (from {args.videos_file})")

    examples, skipped = eduvid_retrieval.load_all_examples(
        eduvid_retrieval.VIDEOS_ROOT, limit=args.limit, video_whitelist=video_whitelist
    )
    if not examples:
        print("No questions found — exiting.")
        return
    print(f"\nLoaded {len(examples)} questions ({len(skipped)} skipped at load time)")

    video_ids = sorted({e["video_id"] for e in examples})
    lectures, by_video, missing = eduvid_retrieval.build_lecture_descriptors(
        video_ids, eduvid_retrieval.VIDEOS_ROOT
    )
    if missing:
        print(f"WARNING: {len(missing)} video(s) missing transcripts: {missing[:5]}")
    examples = [e for e in examples if e["video_id"] in by_video]
    print(f"Questions with usable transcripts: {len(examples)}")

    system_specs = eduvid_retrieval.build_system_specs()
    if args.systems:
        known = {s["name"] for s in system_specs}
        unknown = set(args.systems) - known
        if unknown:
            print(f"ERROR: unknown system(s): {unknown}")
            print(f"Known: {sorted(known)}")
            return
        system_specs = [s for s in system_specs if s["name"] in args.systems]

    default_reranker, summary_reranker = build_rerankers()

    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # ── Phase A: retrieval (all systems) ─────────────────────────────────────
    all_retrieval: dict[str, list] = {}
    for spec in system_specs:
        print(f"\n{'=' * 60}\nSystem: {spec['name']}  [retrieval]\n{'=' * 60}")
        all_retrieval[spec["name"]] = run_retrieval(
            spec, lectures, by_video, examples, default_reranker, summary_reranker
        )

    all_retrieval = {k: v for k, v in all_retrieval.items() if v}
    if not all_retrieval:
        print("No retrieval results for any system — exiting.")
        return

    # ── Phase B: generation (one batch for all systems) ──────────────────────
    print(f"\n{'=' * 60}\nPhase B — Batch generation\n{'=' * 60}")
    all_gen_results = ragas.generate_all_batch(all_retrieval, openai_client)

    # ── Phase C: judging (one batch for all systems) ─────────────────────────
    print(f"\n{'=' * 60}\nPhase C — Batch judging\n{'=' * 60}")
    all_judge_scores = ragas.judge_all_batch(all_gen_results, openai_client)

    # ── Save results ──────────────────────────────────────────────────────────
    all_scores: dict[str, dict] = {}
    all_per_q_dfs: dict = {}
    for name, gen_results in all_gen_results.items():
        print(f"\n{'=' * 60}\nSystem: {name}  [saving results]\n{'=' * 60}")
        per_q_metrics = all_judge_scores.get(name, {})
        scores, per_q_df = ragas.save_results(name, gen_results, per_q_metrics, OUTPUT_DIR)
        for metric, val in scores.items():
            print(f"    {metric}: {val:.4f}")
        all_scores[name] = scores
        all_per_q_dfs[name] = per_q_df

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "all_systems_summary.json"
    summary_path.write_text(json.dumps(all_scores, indent=2))
    print(f"\nSummary saved to {summary_path}")

    ragas._write_combined_csvs(all_per_q_dfs, all_scores, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
