"""Analyze token lengths of TreeSeg leaf and summary-tree segments across all lectures.

Usage:
    python -m lpm_qa_custom_eval.analyze_segment_lengths
    python -m lpm_qa_custom_eval.analyze_segment_lengths --leaf-only
    python -m lpm_qa_custom_eval.analyze_segment_lengths --model BAAI/bge-base-en-v1.5
    python -m lpm_qa_custom_eval.analyze_segment_lengths --lecture ml-1/MultimodalMachineLearning/1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from transformers import AutoTokenizer  # noqa: E402

from treeseg_vector_index_modular.lecture_catalog import LectureCatalog  # noqa: E402
from treeseg_vector_index_modular.lpm_config_builder import LpmConfigBuilder  # noqa: E402
from treeseg_vector_index_modular.lecture_segment_builder import (  # noqa: E402
    LectureSegmentBuilder,
    SummaryTreeBuildOptions,
)

LPM_DATA_DIR = PROJECT_DIR / "lpm_data"
SUMMARY_TREE_CACHE_DIR = SCRIPT_DIR / "storage" / "summary_tree_cache"

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def percentile(sorted_vals: list[int], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * p / 100
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def print_stats(label: str, token_counts: list[int], model_max: int) -> None:
    if not token_counts:
        print(f"  {label}: no segments")
        return
    s = sorted(token_counts)
    over = sum(1 for t in s if t > model_max)
    pct_over = 100 * over / len(s)
    print(f"  {label}  (n={len(s)})")
    print(f"    min={s[0]}  max={s[-1]}  mean={sum(s)/len(s):.1f}")
    print(f"    p50={percentile(s,50):.0f}  p75={percentile(s,75):.0f}  "
          f"p90={percentile(s,90):.0f}  p95={percentile(s,95):.0f}  p99={percentile(s,99):.0f}")
    print(f"    exceeds model limit ({model_max} tok): {over}/{len(s)} = {pct_over:.1f}%")


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def segment_text_for_embedding(seg: dict) -> str:
    """Return the text that would actually be embedded for this segment."""
    if not seg.get("is_leaf", True):
        # Internal summary nodes embed their summary text
        return str(seg.get("summary_text") or seg.get("text") or "")
    return str(seg.get("text") or "")


def main():
    parser = argparse.ArgumentParser(description="Analyze segment token lengths.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HuggingFace embedding model to use for tokenization.")
    parser.add_argument("--leaf-only", action="store_true",
                        help="Only analyze leaf segments (skip summary tree).")
    parser.add_argument("--summary-tree-only", action="store_true",
                        help="Only analyze summary-tree segments (skip leaf).")
    parser.add_argument("--lecture", default=None,
                        help="Only analyze a specific lecture key.")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_max = tokenizer.model_max_length
    if model_max > 100_000:
        # Some tokenizers report absurdly large max — use 512 as safe default
        model_max = 512
    print(f"Model max tokens: {model_max}\n")

    treeseg_config = LpmConfigBuilder.build_lpm_config(embedding_model=args.model)

    all_lectures = LectureCatalog.discover_lectures(data_dir=LPM_DATA_DIR)
    if args.lecture:
        all_lectures = [l for l in all_lectures if l.key == args.lecture]
        if not all_lectures:
            print(f"Lecture '{args.lecture}' not found.")
            return

    print(f"Found {len(all_lectures)} lecture(s)\n")

    all_leaf_counts: list[int] = []
    all_tree_leaf_counts: list[int] = []
    all_tree_internal_counts: list[int] = []

    for lecture in all_lectures:
        print(f"=== {lecture.key} ===")

        # --- Leaf segments ---
        if not args.summary_tree_only:
            try:
                utterances = LectureSegmentBuilder.load_lecture_utterances(
                    lecture,
                    max_gap_s="auto",
                    lowercase=True,
                    attach_ocr=False,
                    ocr_min_conf=60.0,
                    ocr_per_slide=1,
                )
                segments = LectureSegmentBuilder.build_segments_for_lecture(
                    lecture,
                    utterances,
                    treeseg_config=treeseg_config,
                    target_segments=None,
                    include_ocr=False,
                )
                leaf_counts = [count_tokens(tokenizer, segment_text_for_embedding(s))
                               for s in segments]
                all_leaf_counts.extend(leaf_counts)
                print_stats("leaf", leaf_counts, model_max)
            except Exception as e:
                print(f"  leaf: ERROR — {e}")

        # --- Summary-tree segments ---
        if not args.leaf_only:
            try:
                build_options = SummaryTreeBuildOptions(
                    workers=1,
                    cache_dir=str(SUMMARY_TREE_CACHE_DIR),
                    rebuild_cache=False,
                )
                segments = LectureSegmentBuilder.build_or_load_summary_tree_index_records_for_lecture(
                    lecture,
                    treeseg_config=treeseg_config,
                    max_gap_s="auto",
                    lowercase=True,
                    attach_ocr=False,
                    ocr_min_conf=60.0,
                    ocr_per_slide=1,
                    target_segments=None,
                    include_ocr=False,
                    normalize_embeddings=True,
                    build_options=build_options,
                )
                all_counts = [count_tokens(tokenizer, segment_text_for_embedding(s))
                              for s in segments]
                leaf_counts = [count_tokens(tokenizer, segment_text_for_embedding(s))
                               for s in segments if s.get("is_leaf", True)]
                internal_counts = [count_tokens(tokenizer, segment_text_for_embedding(s))
                                   for s in segments if not s.get("is_leaf", True)]

                all_tree_leaf_counts.extend(leaf_counts)
                all_tree_internal_counts.extend(internal_counts)

                print_stats("summary_tree (all nodes)", all_counts, model_max)
                if leaf_counts:
                    print_stats("  └─ leaf nodes only", leaf_counts, model_max)
                if internal_counts:
                    print_stats("  └─ internal (summary) nodes", internal_counts, model_max)
            except Exception as e:
                print(f"  summary_tree: ERROR — {e}")

        print()

    # --- Overall summary across all lectures ---
    if len(all_lectures) > 1:
        print("=== OVERALL (all lectures) ===")
        if all_leaf_counts:
            print_stats("leaf", all_leaf_counts, model_max)
        if all_tree_leaf_counts or all_tree_internal_counts:
            print_stats("summary_tree leaf nodes", all_tree_leaf_counts, model_max)
            print_stats("summary_tree internal nodes", all_tree_internal_counts, model_max)
            print_stats("summary_tree all nodes",
                        all_tree_leaf_counts + all_tree_internal_counts, model_max)


if __name__ == "__main__":
    main()
