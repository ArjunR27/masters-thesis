"""Temporal retrieval evaluation on the EduVidQA dataset.

Mirrors lpm_qa_evaluation/evaluate_lpm_temporal_retrieval.py but adapted for
EduVidQA's single-point timestamps (not start/end ranges). Uses the preprocessed
video transcripts already in eduvid_evaluation/storage/videos/.

Metrics (all @k, default k=1,3,5):
  recall@k          – fraction of questions where any top-k hit contains the
                      ground-truth timestamp exactly.
  recall_relaxed@k  – same but with ±TOLERANCE_S (default 15s) window.
  mean_td@k         – mean temporal distance (seconds) from each top-k hit to
                      the ground-truth timestamp; 0 if the hit contains it.
  mrr               – mean reciprocal rank for exact containment (over top-5).
  ndcg@k            – nDCG using binary relevance (1=exact hit, 0=otherwise).

Output: eduvid_evaluation/outputs/eduvid_retrieval_evaluation.csv
        eduvid_evaluation/outputs/eduvid_skipped_rows.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

from ir_measures import Qrel, ScoredDoc, calc_aggregate, nDCG

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from baseline_rag_system.store_builder import BaselineStoreBuilder
from baseline_rag_system.types import build_default_baseline_configs
from treeseg_vector_index_modular.cross_encoder_reranker import CrossEncoderReranker
from treeseg_vector_index_modular.lecture_descriptor import LectureDescriptor
from treeseg_vector_index_modular.lecture_segment_builder import SummaryTreeBuildOptions
from treeseg_vector_index_modular.lpm_config_builder import LpmConfigBuilder
from treeseg_vector_index_modular.rerank_input_builder import RerankInputBuilder
from treeseg_vector_index_modular.vector_store_factory import VectorStoreFactory

VIDEOS_ROOT = HERE / "storage" / "videos"
DEFAULT_OUTPUT_DIR = HERE / "outputs"
DEFAULT_K_VALUES = [1, 3, 5]
TOLERANCE_S = 15.0          # relaxed recall window in seconds
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
INITIAL_TOP_K = 50
FINAL_TOP_N = 5
SUMMARY_TREE_TOP_DESCENDANT_LEAVES = 3


# ── Argument parsing ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate temporal retrieval on the EduVidQA dataset."
    )
    parser.add_argument(
        "--videos-root",
        default=str(VIDEOS_ROOT),
        help="Root directory containing preprocessed video subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where evaluation outputs will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of questions to evaluate.",
    )
    parser.add_argument(
        "--k-values",
        default=",".join(str(v) for v in DEFAULT_K_VALUES),
        help="Comma-separated k values for Recall@k, Mean TD@k, and nDCG@k.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=TOLERANCE_S,
        help="Seconds tolerance for relaxed recall (default 15).",
    )
    parser.add_argument(
        "--summary-tree-workers",
        default="auto",
    )
    parser.add_argument(
        "--summary-tree-cache-dir",
        default=str(HERE / "storage" / "summary_tree_cache"),
        help="Directory for summary-tree cache files.",
    )
    parser.add_argument(
        "--rebuild-summary-tree-cache",
        action="store_true",
    )
    return parser


def parse_k_values(raw: str) -> list[int]:
    values = sorted({int(p.strip()) for p in raw.split(",") if p.strip()})
    if not values:
        raise ValueError("Provide at least one k value.")
    if any(v <= 0 for v in values):
        raise ValueError("All k values must be positive.")
    if max(values) > FINAL_TOP_N:
        raise ValueError(f"k cannot exceed FINAL_TOP_N={FINAL_TOP_N}.")
    return values


def resolve_workers(raw: str) -> int:
    v = (raw or "").strip().lower()
    if v in {"", "auto"}:
        return min(2, max(1, (os.cpu_count() or 1) // 4))
    w = int(v)
    if w < 1:
        raise ValueError("--summary-tree-workers must be 'auto' or a positive integer.")
    return w


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_examples(
    videos_root: Path,
    limit: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load questions from questions.jsonl files in each video subdirectory.

    Returns (examples, skipped) where each example has:
      example_id, video_id, question, answer, timestamp, timestamp_seconds
    """
    examples: list[dict] = []
    skipped: list[dict] = []

    video_dirs = sorted(p for p in videos_root.iterdir() if p.is_dir())
    for video_dir in video_dirs:
        questions_path = video_dir / "questions.jsonl"
        if not questions_path.exists():
            continue
        with questions_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if limit is not None and len(examples) >= limit:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped.append({"reason": "json_parse_error", "video_id": video_dir.name})
                    continue

                video_id = row.get("video_id", "")
                question = (row.get("question") or "").strip()
                ts_seconds = row.get("timestamp_seconds")

                if not question:
                    skipped.append({"example_id": row.get("example_id", ""), "video_id": video_id, "reason": "missing_question", "question": "", "timestamp": row.get("timestamp", "")})
                    continue
                if ts_seconds is None:
                    skipped.append({"example_id": row.get("example_id", ""), "video_id": video_id, "reason": "missing_timestamp", "question": question, "timestamp": row.get("timestamp", "")})
                    continue

                examples.append({
                    "example_id": row.get("example_id", f"{video_id}-unknown"),
                    "video_id": video_id,
                    "question": question,
                    "answer": (row.get("answer") or "").strip(),
                    "timestamp": row.get("timestamp", ""),
                    "timestamp_seconds": int(ts_seconds),
                })

        if limit is not None and len(examples) >= limit:
            break

    return examples, skipped


def build_lecture_descriptors(
    video_ids: list[str],
    videos_root: Path,
) -> tuple[list[LectureDescriptor], dict[str, LectureDescriptor], list[str]]:
    lectures: list[LectureDescriptor] = []
    by_video: dict[str, LectureDescriptor] = {}
    missing: list[str] = []

    for vid in video_ids:
        video_dir = videos_root / vid
        transcript_path = video_dir / f"{vid}_transcripts.csv"
        if not transcript_path.exists():
            missing.append(vid)
            continue
        lecture = LectureDescriptor(
            speaker="eduvid",
            course_dir="eduvid",
            meeting_id=vid,
            video_id=vid,
            transcripts_path=str(transcript_path),
            meeting_dir=str(video_dir),
        )
        lectures.append(lecture)
        by_video[vid] = lecture

    return lectures, by_video, missing


# ── System specs (same 14 systems as LPM eval) ───────────────────────────────

def build_system_specs() -> list[dict]:
    specs = [
        {"name": "treeseg__leaf", "kind": "leaf", "config": None},
        {"name": "treeseg__summary_tree", "kind": "summary_tree", "config": None},
    ]
    baseline_configs = build_default_baseline_configs(ocr_modes=["transcript_only"])
    for config in baseline_configs:
        config = replace(config, embedding_model=EMBEDDING_MODEL)
        specs.append({"name": config.system_name, "kind": "baseline", "config": config})
    return specs


# ── Store building ────────────────────────────────────────────────────────────

def build_tree_store(lectures: list[LectureDescriptor], index_kind: str, args: argparse.Namespace):
    config = LpmConfigBuilder.build_lpm_config(embedding_model=EMBEDDING_MODEL)
    build_options = None
    if index_kind == "summary_tree":
        cache_dir = (args.summary_tree_cache_dir or "").strip() or None
        build_options = SummaryTreeBuildOptions(
            workers=resolve_workers(args.summary_tree_workers),
            cache_dir=cache_dir,
            rebuild_cache=args.rebuild_summary_tree_cache,
        )
    return VectorStoreFactory().build_vector_store(
        lectures=lectures,
        treeseg_config=config,
        embed_model=EMBEDDING_MODEL,
        normalize=True,
        build_global=False,
        max_gap_s="auto",
        lowercase=True,
        attach_ocr=False,
        include_ocr_in_treeseg=False,
        ocr_min_conf=60.0,
        ocr_per_slide=1,
        target_segments=None,
        index_kind=index_kind,
        summary_tree_build_options=build_options,
    )


def build_baseline_store(lectures: list[LectureDescriptor], config):
    return BaselineStoreBuilder().build_store(lectures, config, build_global=False)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def search_hits(spec: dict, store, question: str, lecture_key: str, reranker) -> list[dict]:
    if spec["kind"] == "summary_tree":
        query_embedding = store.encode_query(question)
        hits = store.search_with_embedding(query_embedding, top_k=INITIAL_TOP_K, lecture_key=lecture_key)
        hits = store.expand_summary_tree_results(
            query=question,
            results=hits,
            lecture_key=lecture_key,
            top_descendant_leaves=SUMMARY_TREE_TOP_DESCENDANT_LEAVES,
            query_embedding=query_embedding,
        )
        hits = reranker.rerank(question, hits, top_n=None)
        hits = store.deduplicate_summary_tree_results(hits)
        return hits[:FINAL_TOP_N]
    hits = store.search(question, top_k=INITIAL_TOP_K, lecture_key=lecture_key)
    hits = reranker.rerank(question, hits, top_n=None)
    return hits[:FINAL_TOP_N]


# ── Point-timestamp metrics ───────────────────────────────────────────────────

def _hit_bounds(hit: dict) -> tuple[float, float] | None:
    start, end = hit.get("start"), hit.get("end")
    if start is None or end is None:
        return None
    return float(start), float(end)


def _exact_hit(hit: dict, t: float) -> bool:
    bounds = _hit_bounds(hit)
    return bounds is not None and bounds[0] <= t <= bounds[1]


def _relaxed_hit(hit: dict, t: float, tol: float) -> bool:
    bounds = _hit_bounds(hit)
    return bounds is not None and (bounds[0] - tol) <= t <= (bounds[1] + tol)


def _distance(hit: dict, t: float) -> float:
    bounds = _hit_bounds(hit)
    if bounds is None:
        return float("nan")
    s, e = bounds
    if s <= t <= e:
        return 0.0
    return min(abs(t - s), abs(t - e))


def _make_doc_id(video_id: str, hit: dict) -> str:
    tree_path = hit.get("tree_path")
    if tree_path:
        return f"{video_id}::tree::{tree_path}"
    seg_id = hit.get("segment_id")
    if seg_id is not None:
        return f"{video_id}::seg::{int(seg_id)}"
    bounds = _hit_bounds(hit)
    if bounds:
        return f"{video_id}::span::{bounds[0]}::{bounds[1]}"
    return f"{video_id}::unknown"


def compute_point_metrics(
    hits: list[dict],
    t_seconds: float,
    candidates: list[dict],
    video_id: str,
    query_id: str,
    k_values: list[int],
    ndcg_measures: dict[int, object],
    tol: float,
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    for k in k_values:
        subset = hits[:k]
        exact_hits = [_exact_hit(h, t_seconds) for h in subset]
        relaxed_hits = [_relaxed_hit(h, t_seconds, tol) for h in subset]
        distances = [_distance(h, t_seconds) for h in subset if _hit_bounds(h) is not None]

        metrics[f"recall@{k}"] = 1.0 if any(exact_hits) else 0.0
        metrics[f"recall_relaxed@{k}"] = 1.0 if any(relaxed_hits) else 0.0
        metrics[f"mean_td@{k}"] = sum(distances) / len(distances) if distances else float("nan")

    # MRR (exact, over all top-5 hits)
    mrr = 0.0
    for rank, hit in enumerate(hits, start=1):
        if _exact_hit(hit, t_seconds):
            mrr = 1.0 / rank
            break
    metrics["mrr"] = mrr

    # nDCG using binary exact relevance (1=exact hit, 0=otherwise)
    qrels = [
        Qrel(query_id, _make_doc_id(video_id, c), 1 if _exact_hit(c, t_seconds) else 0)
        for c in candidates
    ]
    run = [
        ScoredDoc(query_id, _make_doc_id(video_id, h), float(h.get("rerank_score", h.get("score", 0.0))))
        for h in hits
    ]
    ndcg_scores = calc_aggregate(list(ndcg_measures.values()), qrels, run)
    for k, measure in ndcg_measures.items():
        v = float(ndcg_scores.get(measure, 0.0))
        metrics[f"ndcg@{k}"] = 0.0 if math.isnan(v) else v

    return metrics


def metric_column_names(k_values: list[int]) -> list[str]:
    cols = []
    for prefix in ("recall", "recall_relaxed", "mean_td"):
        for k in k_values:
            cols.append(f"{prefix}@{k}")
    cols.append("mrr")
    for k in k_values:
        cols.append(f"ndcg@{k}")
    return cols


def mean_numeric(values: list) -> float:
    cleaned = [float(v) for v in values if v is not None and not math.isnan(float(v)) and not math.isinf(float(v))]
    return sum(cleaned) / len(cleaned) if cleaned else float("nan")


# ── CSV helpers ───────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_approach_columns(spec: dict) -> dict:
    row = {
        "system": spec["name"],
        "kind": spec["kind"],
        "approach": "",
        "retrieval_source": "transcript_only",
        "embedding_model": EMBEDDING_MODEL,
        "rerank_model": RERANK_MODEL,
        "initial_top_k": INITIAL_TOP_K,
        "final_top_n": FINAL_TOP_N,
        "baseline_chunk_strategy": "",
        "baseline_chunk_size_tokens": "",
        "baseline_overlap_percent": "",
    }
    if spec["kind"] == "leaf":
        row["approach"] = "TreeSeg leaf"
    elif spec["kind"] == "summary_tree":
        row["approach"] = "TreeSeg summary_tree"
    else:
        c = spec["config"]
        row["approach"] = f"Baseline {c.chunk_strategy} {c.chunk_size_tokens}tok overlap {c.overlap_percent}%"
        row["baseline_chunk_strategy"] = c.chunk_strategy
        row["baseline_chunk_size_tokens"] = c.chunk_size_tokens
        row["baseline_overlap_percent"] = c.overlap_percent
    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    videos_root = Path(args.videos_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    k_values = parse_k_values(args.k_values)
    tol = args.tolerance
    metric_cols = metric_column_names(k_values)
    ndcg_measures = {k: nDCG @ k for k in k_values}

    print(f"Videos root:  {videos_root}")
    print(f"Output dir:   {output_dir}")
    print(f"k values:     {k_values}")
    print(f"Tolerance:    ±{tol}s")

    examples, skipped = load_all_examples(videos_root, limit=args.limit)
    if not examples:
        raise SystemExit("No valid questions found in the videos root.")
    print(f"\nLoaded {len(examples)} questions from {len({e['video_id'] for e in examples})} videos")

    # Build lecture descriptors for all video IDs that appear in the examples
    video_ids = sorted({e["video_id"] for e in examples})
    lectures, by_video, missing_videos = build_lecture_descriptors(video_ids, videos_root)
    if missing_videos:
        print(f"WARNING: {len(missing_videos)} video(s) missing transcripts: {missing_videos[:5]}")

    # Filter examples to only those whose video has been preprocessed
    filtered: list[dict] = []
    for ex in examples:
        if ex["video_id"] not in by_video:
            skipped.append({
                "example_id": ex["example_id"],
                "video_id": ex["video_id"],
                "reason": "transcript_not_found",
                "question": ex["question"],
                "timestamp": ex["timestamp"],
            })
        else:
            filtered.append(ex)

    if not filtered:
        raise SystemExit("All questions were skipped — no transcripts found.")

    print(f"Questions to evaluate: {len(filtered)}  (skipped: {len(skipped)})")

    system_specs = build_system_specs()
    output_dir.mkdir(parents=True, exist_ok=True)

    default_reranker = CrossEncoderReranker(RERANK_MODEL)
    summary_reranker = CrossEncoderReranker(
        RERANK_MODEL,
        input_builder=RerankInputBuilder.build_summary_tree_rerank_input,
    )

    summary_rows: list[dict] = []

    for spec in system_specs:
        system_name = spec["name"]
        system_kind = spec["kind"]
        print(f"\nBuilding {system_name}...")

        skipped_lectures: dict[str, str] = {}
        if system_kind == "leaf":
            store = build_tree_store(lectures, "leaf", args)
            reranker = default_reranker
        elif system_kind == "summary_tree":
            store = build_tree_store(lectures, "summary_tree", args)
            reranker = summary_reranker
        else:
            build_result = build_baseline_store(lectures, spec["config"])
            store = build_result.store
            skipped_lectures = build_result.skipped_lectures
            reranker = default_reranker

        system_metric_rows: list[dict] = []
        system_skips = 0

        for ex in filtered:
            video_id = ex["video_id"]
            lecture_key = by_video[video_id].key if video_id in by_video else video_id

            if lecture_key not in store.lecture_indices:
                skipped.append({
                    "example_id": ex["example_id"],
                    "video_id": video_id,
                    "reason": skipped_lectures.get(lecture_key, "lecture_not_indexed"),
                    "question": ex["question"],
                    "timestamp": ex["timestamp"],
                })
                system_skips += 1
                continue

            hits = search_hits(spec, store, ex["question"], lecture_key, reranker)
            candidates = store.lecture_indices[lecture_key]["segments"]
            metrics = compute_point_metrics(
                hits=hits,
                t_seconds=float(ex["timestamp_seconds"]),
                candidates=candidates,
                video_id=video_id,
                query_id=ex["example_id"],
                k_values=k_values,
                ndcg_measures=ndcg_measures,
                tol=tol,
            )
            system_metric_rows.append(metrics)

        summary_row = {
            **build_approach_columns(spec),
            "question_count": len(system_metric_rows),
            "skipped_count": system_skips,
        }
        for col in metric_cols:
            summary_row[col] = mean_numeric([r[col] for r in system_metric_rows])
        summary_rows.append(summary_row)
        print(f"  {system_name}: evaluated {len(system_metric_rows)}, skipped {system_skips}")

        del store
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    retrieval_fieldnames = [
        "system", "kind", "approach", "retrieval_source",
        "embedding_model", "rerank_model", "initial_top_k", "final_top_n",
        "baseline_chunk_strategy", "baseline_chunk_size_tokens", "baseline_overlap_percent",
        "question_count", "skipped_count",
        *metric_cols,
    ]
    skipped_fieldnames = ["example_id", "video_id", "reason", "question", "timestamp"]

    write_csv(output_dir / "eduvid_retrieval_evaluation.csv", summary_rows, retrieval_fieldnames)
    write_csv(output_dir / "eduvid_skipped_rows.csv", skipped, skipped_fieldnames)

    print(f"\nWrote {output_dir / 'eduvid_retrieval_evaluation.csv'}")
    print(f"Wrote {output_dir / 'eduvid_skipped_rows.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
