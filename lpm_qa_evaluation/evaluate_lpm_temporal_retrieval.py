from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import math
import os
import re
import sys
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
from treeseg_vector_index_modular.cross_encoder_reranker import (
    CrossEncoderReranker,
)
from treeseg_vector_index_modular.lecture_catalog import LectureCatalog
from treeseg_vector_index_modular.lecture_segment_builder import ( 
    SummaryTreeBuildOptions,
)
from treeseg_vector_index_modular.lpm_config_builder import LpmConfigBuilder
from treeseg_vector_index_modular.rerank_input_builder import (
    RerankInputBuilder,
)
from treeseg_vector_index_modular.vector_store_factory import VectorStoreFactory


DEFAULT_DATASET_PATH = PROJECT_DIR / "LPM_QA_DATASET" / "lpm_qa_labeled.csv"
DEFAULT_OUTPUT_DIR = HERE / "outputs"
DEFAULT_K_VALUES = [1, 3, 5]
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
INITIAL_TOP_K = 50
FINAL_TOP_N = 5
SUMMARY_TREE_TOP_DESCENDANT_LEAVES = 3
RANGE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$")
KNOWN_QUESTION_TYPES = {
    "definition",
    "mechanism",
    "classification",
    "cause_effect",
    "function",
    "process",
    "summary",
    "logistics",
    "course_specific",
    "course_specifics",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate transcript-only temporal retrieval on the LPM QA dataset."
    )
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the LPM QA CSV file.",
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
        help="Optional limit on the number of dataset rows to evaluate.",
    )
    parser.add_argument(
        "--k-values",
        default=",".join(str(value) for value in DEFAULT_K_VALUES),
        help="Comma-separated k values for Recall@k, Mean Temporal Distance@k, Max IoU@k, and nDCG@k.",
    )
    parser.add_argument(
        "--summary-tree-workers",
        default="auto",
        help="Summary-tree build workers. Use 'auto' or a positive integer.",
    )
    parser.add_argument(
        "--summary-tree-cache-dir",
        default=str(HERE / "summary_tree_cache"),
        help="Directory for summary-tree cache files. Set to an empty string to disable caching.",
    )
    parser.add_argument(
        "--rebuild-summary-tree-cache",
        action="store_true",
        help="Ignore any existing summary-tree cache files and rebuild them.",
    )
    return parser


def parse_k_values(raw_value: str) -> list[int]:
    values = []
    for part in (raw_value or "").split(","):
        text = part.strip()
        if not text:
            continue
        values.append(int(text))

    values = sorted(set(values))
    if not values:
        raise ValueError("Provide at least one k value.")
    if any(value <= 0 for value in values):
        raise ValueError("All k values must be positive integers.")
    if max(values) > FINAL_TOP_N:
        raise ValueError(
            f"k values cannot exceed {FINAL_TOP_N} because reranking always keeps {FINAL_TOP_N} hits."
        )
    return values


def resolve_summary_tree_workers(raw_value: str) -> int:
    value = (raw_value or "").strip().lower()
    if value in {"", "auto"}:
        cpu_count = os.cpu_count() or 1
        return min(2, max(1, cpu_count // 4))
    workers = int(value)
    if workers < 1:
        raise ValueError("--summary-tree-workers must be 'auto' or a positive integer.")
    return workers


def repair_dataset_row(row: dict[str, object]) -> dict[str, str]:
    fixed = {key: value for key, value in row.items() if key is not None}
    extras = row.get(None) or []

    question_type = str(fixed.get("question_type") or "").strip()
    answer_text = str(fixed.get("answer_text") or "").strip()
    extra_values = [str(value).strip() for value in extras if str(value).strip()]

    if extra_values and question_type and question_type not in KNOWN_QUESTION_TYPES:
        answer_text = f"{answer_text},{question_type}".strip()
        question_type = extra_values[0]
    elif extra_values and not question_type:
        question_type = extra_values[0]

    fixed["question"] = str(fixed.get("question") or "").strip()
    fixed["lecture_key"] = str(fixed.get("lecture_key") or "").strip()
    fixed["answer_timestamps"] = str(fixed.get("answer_timestamps") or "").strip()
    fixed["answer_text"] = answer_text
    fixed["question_type"] = question_type.strip()
    return fixed


def parse_answer_ranges(raw_value: str) -> list[tuple[float, float]]:
    text = (raw_value or "").strip()
    if not text:
        raise ValueError("answer_timestamps is empty")

    ranges = []
    for part in text.split("|"):
        piece = part.strip()
        if not piece:
            continue
        match = RANGE_RE.match(piece)
        if match is None:
            raise ValueError(f"Invalid timestamp range: {piece!r}")
        start = float(match.group(1))
        end = float(match.group(2))
        if end < start:
            raise ValueError(f"Timestamp end must be >= start: {piece!r}")
        ranges.append((start, end))

    if not ranges:
        raise ValueError("No valid timestamp ranges found")
    return ranges


def format_answer_ranges(answer_ranges: list[tuple[float, float]]) -> str:
    return " | ".join(f"{start:.1f}-{end:.1f}" for start, end in answer_ranges)


def load_examples(csv_path: Path, limit: int | None = None) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    examples = []
    skipped_rows = []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, raw_row in enumerate(reader, start=1):
            if limit is not None and len(examples) >= limit:
                break

            row = repair_dataset_row(raw_row)
            question = row["question"]
            lecture_key = row["lecture_key"]
            raw_timestamps = row["answer_timestamps"]

            if not question:
                skipped_rows.append(
                    {
                        "row_index": row_index,
                        "example_id": f"lpm_qa-{row_index:04d}",
                        "system": "",
                        "lecture_key": lecture_key,
                        "reason": "missing_question",
                        "question": question,
                        "answer_timestamps": raw_timestamps,
                    }
                )
                continue
            if not lecture_key:
                skipped_rows.append(
                    {
                        "row_index": row_index,
                        "example_id": f"lpm_qa-{row_index:04d}",
                        "system": "",
                        "lecture_key": lecture_key,
                        "reason": "missing_lecture_key",
                        "question": question,
                        "answer_timestamps": raw_timestamps,
                    }
                )
                continue

            try:
                answer_ranges = parse_answer_ranges(raw_timestamps)
            except ValueError as exc:
                skipped_rows.append(
                    {
                        "row_index": row_index,
                        "example_id": f"lpm_qa-{row_index:04d}",
                        "system": "",
                        "lecture_key": lecture_key,
                        "reason": f"invalid_answer_timestamps: {exc}",
                        "question": question,
                        "answer_timestamps": raw_timestamps,
                    }
                )
                continue

            examples.append(
                {
                    "row_index": row_index,
                    "example_id": f"lpm_qa-{row_index:04d}",
                    "question": question,
                    "lecture_key": lecture_key,
                    "answer_timestamps": raw_timestamps,
                    "answer_ranges": answer_ranges,
                    "answer_text": row["answer_text"],
                    "question_type": row["question_type"],
                }
            )

    return examples, skipped_rows


def load_target_lectures(lecture_keys: set[str]) -> tuple[list[object], dict[str, object]]:
    data_dir = PROJECT_DIR / "lpm_data"
    lectures = LectureCatalog.discover_lectures(data_dir=data_dir)
    lecture_by_key = {lecture.key: lecture for lecture in lectures if lecture.key in lecture_keys}
    target_lectures = [lecture_by_key[key] for key in sorted(lecture_by_key)]
    return target_lectures, lecture_by_key


def build_system_specs() -> list[dict[str, object]]:
    specs = [
        {"name": "treeseg__leaf", "kind": "leaf", "config": None},
        {"name": "treeseg__summary_tree", "kind": "summary_tree", "config": None},
    ]

    baseline_configs = build_default_baseline_configs(ocr_modes=["transcript_only"])
    for config in baseline_configs:
        config = replace(config, embedding_model=EMBEDDING_MODEL)
        specs.append(
            {
                "name": config.system_name,
                "kind": "baseline",
                "config": config,
            }
        )
    return specs


def build_tree_store(
    lectures: list[object],
    index_kind: str,
    args: argparse.Namespace,
):
    treeseg_config = LpmConfigBuilder.build_lpm_config(embedding_model=EMBEDDING_MODEL)
    build_options = None
    if index_kind == "summary_tree":
        cache_dir = args.summary_tree_cache_dir.strip() or None
        build_options = SummaryTreeBuildOptions(
            workers=resolve_summary_tree_workers(args.summary_tree_workers),
            cache_dir=cache_dir,
            rebuild_cache=args.rebuild_summary_tree_cache,
        )

    return VectorStoreFactory().build_vector_store(
        lectures=lectures,
        treeseg_config=treeseg_config,
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


def build_baseline_store(lectures: list[object], config):
    return BaselineStoreBuilder().build_store(
        lectures,
        config,
        build_global=False,
    )


def make_doc_id(lecture_key: str, hit: dict[str, object]) -> str:
    tree_path = hit.get("tree_path")
    if tree_path:
        return f"{lecture_key}::tree::{tree_path}"

    segment_id = hit.get("segment_id")
    if segment_id is not None:
        return f"{lecture_key}::seg::{int(segment_id)}"

    slide_index = hit.get("slide_index")
    if slide_index is not None:
        return f"{lecture_key}::slide::{int(slide_index)}"

    start = hit.get("start")
    end = hit.get("end")
    return f"{lecture_key}::span::{start}::{end}"


def search_hits(
    spec: dict[str, object],
    store,
    question: str,
    lecture_key: str,
    reranker,
) -> list[dict[str, object]]:
    if spec["kind"] == "summary_tree":
        query_embedding = store.encode_query(question)
        hits = store.search_with_embedding(
            query_embedding,
            top_k=INITIAL_TOP_K,
            lecture_key=lecture_key,
        )
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


def get_bounds(hit: dict[str, object]) -> tuple[float, float] | None:
    start = hit.get("start")
    end = hit.get("end")
    if start is None or end is None:
        return None
    return float(start), float(end)


def overlap_seconds(
    hit_start: float,
    hit_end: float,
    range_start: float,
    range_end: float,
) -> float:
    return max(0.0, min(hit_end, range_end) - max(hit_start, range_start))


def temporal_distance(
    hit_start: float,
    hit_end: float,
    range_start: float,
    range_end: float,
) -> float:
    if overlap_seconds(hit_start, hit_end, range_start, range_end) > 0.0:
        return 0.0
    if hit_end < range_start:
        return range_start - hit_end
    return hit_start - range_end


def interval_iou(
    hit_start: float,
    hit_end: float,
    range_start: float,
    range_end: float,
) -> float:
    overlap = overlap_seconds(hit_start, hit_end, range_start, range_end)
    if overlap <= 0.0:
        return 0.0
    hit_length = max(0.0, hit_end - hit_start)
    range_length = max(0.0, range_end - range_start)
    union = hit_length + range_length - overlap
    if union <= 0.0:
        return 0.0
    return overlap / union


def max_iou_against_ranges(
    hit_start: float,
    hit_end: float,
    answer_ranges: list[tuple[float, float]],
) -> float:
    return max(
        interval_iou(hit_start, hit_end, range_start, range_end)
        for range_start, range_end in answer_ranges
    )


def min_distance_to_ranges(
    hit_start: float,
    hit_end: float,
    answer_ranges: list[tuple[float, float]],
) -> float:
    return min(
        temporal_distance(hit_start, hit_end, range_start, range_end)
        for range_start, range_end in answer_ranges
    )


def build_qrels(
    query_id: str,
    lecture_key: str,
    candidates: list[dict[str, object]],
    answer_ranges: list[tuple[float, float]],
) -> list[Qrel]:
    qrels = []
    for candidate in candidates:
        bounds = get_bounds(candidate)
        if bounds is None:
            relevance = 0
        else:
            relevance = int(
                round(100.0 * max_iou_against_ranges(bounds[0], bounds[1], answer_ranges))
            )
        qrels.append(Qrel(query_id, make_doc_id(lecture_key, candidate), relevance))
    return qrels


def build_run(query_id: str, lecture_key: str, hits: list[dict[str, object]]) -> list[ScoredDoc]:
    run = []
    for hit in hits:
        score = hit.get("rerank_score", hit.get("score", 0.0))
        run.append(ScoredDoc(query_id, make_doc_id(lecture_key, hit), float(score)))
    return run


def compute_temporal_metrics(
    hits: list[dict[str, object]],
    answer_ranges: list[tuple[float, float]],
    candidates: list[dict[str, object]],
    lecture_key: str,
    query_id: str,
    ndcg_measures: dict[int, object],
    k_values: list[int],
) -> dict[str, float]:
    metrics = {}

    for k in k_values:
        subset = hits[:k]
        distances = []
        ious = []
        for hit in subset:
            bounds = get_bounds(hit)
            if bounds is None:
                continue
            distances.append(min_distance_to_ranges(bounds[0], bounds[1], answer_ranges))
            ious.append(max_iou_against_ranges(bounds[0], bounds[1], answer_ranges))

        metrics[f"recall@{k}"] = 1.0 if any(iou > 0.0 for iou in ious) else 0.0
        metrics[f"mean_temporal_distance@{k}"] = (
            sum(distances) / len(distances) if distances else float("nan")
        )
        metrics[f"max_iou@{k}"] = max(ious) if ious else 0.0

    qrels = build_qrels(query_id, lecture_key, candidates, answer_ranges)
    run = build_run(query_id, lecture_key, hits)
    ndcg_scores = calc_aggregate(list(ndcg_measures.values()), qrels, run)

    for k, measure in ndcg_measures.items():
        value = float(ndcg_scores.get(measure, 0.0))
        metrics[f"ndcg@{k}"] = 0.0 if math.isnan(value) else value

    return metrics


def metric_column_names(k_values: list[int]) -> list[str]:
    columns = []
    for prefix in ("recall", "mean_temporal_distance", "max_iou", "ndcg"):
        for k in k_values:
            columns.append(f"{prefix}@{k}")
    return columns


def mean_numeric(values: list[object]) -> float:
    cleaned = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            continue
        cleaned.append(numeric)
    if not cleaned:
        return float("nan")
    return sum(cleaned) / len(cleaned)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_approach_columns(spec: dict[str, object]) -> dict[str, object]:
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
        return row

    if spec["kind"] == "summary_tree":
        row["approach"] = "TreeSeg summary_tree"
        return row

    config = spec["config"]
    row["approach"] = (
        f"Baseline {config.chunk_strategy} "
        f"{config.chunk_size_tokens}tok overlap {config.overlap_percent}%"
    )
    row["baseline_chunk_strategy"] = config.chunk_strategy
    row["baseline_chunk_size_tokens"] = config.chunk_size_tokens
    row["baseline_overlap_percent"] = config.overlap_percent
    return row


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    k_values = parse_k_values(args.k_values)
    metric_columns = metric_column_names(k_values)
    ndcg_measures = {k: nDCG @ k for k in k_values}

    examples, skipped_rows = load_examples(dataset_path, limit=args.limit)
    if not examples:
        raise SystemExit("No valid dataset rows found.")

    lecture_keys = {example["lecture_key"] for example in examples}
    target_lectures, lecture_by_key = load_target_lectures(lecture_keys)
    if not target_lectures:
        raise SystemExit("No target lectures found in lpm_data.")

    filtered_examples = []
    for example in examples:
        if example["lecture_key"] not in lecture_by_key:
            skipped_rows.append(
                {
                    "row_index": example["row_index"],
                    "example_id": example["example_id"],
                    "system": "",
                    "lecture_key": example["lecture_key"],
                    "reason": "lecture_not_found",
                    "question": example["question"],
                    "answer_timestamps": example["answer_timestamps"],
                }
            )
            continue
        filtered_examples.append(example)

    if not filtered_examples:
        raise SystemExit("All dataset rows were skipped because their lectures were not found.")

    system_specs = build_system_specs()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset rows to evaluate: {len(filtered_examples)}")
    print(f"Dataset rows skipped before evaluation: {len(skipped_rows)}")
    print(f"Systems: {len(system_specs)}")
    print(f"k values: {k_values}")
    print(f"Max final reranked hits per query: {FINAL_TOP_N}")

    default_reranker = CrossEncoderReranker(RERANK_MODEL)
    summary_reranker = CrossEncoderReranker(
        RERANK_MODEL,
        input_builder=RerankInputBuilder.build_summary_tree_rerank_input,
    )

    summary_rows = []
    for spec in system_specs:
        system_name = spec["name"]
        system_kind = spec["kind"]

        print(f"Building {system_name}...")
        skipped_lectures = {}
        if system_kind == "leaf":
            store = build_tree_store(target_lectures, "leaf", args)
            reranker = default_reranker
        elif system_kind == "summary_tree":
            store = build_tree_store(target_lectures, "summary_tree", args)
            reranker = summary_reranker
        else:
            build_result = build_baseline_store(target_lectures, spec["config"])
            store = build_result.store
            skipped_lectures = build_result.skipped_lectures
            reranker = default_reranker

        system_metric_rows = []
        system_skips = 0

        for example in filtered_examples:
            lecture_key = example["lecture_key"]
            if lecture_key not in store.lecture_indices:
                skipped_rows.append(
                    {
                        "row_index": example["row_index"],
                        "example_id": example["example_id"],
                        "system": system_name,
                        "lecture_key": lecture_key,
                        "reason": skipped_lectures.get(lecture_key, "lecture_not_indexed"),
                        "question": example["question"],
                        "answer_timestamps": example["answer_timestamps"],
                    }
                )
                system_skips += 1
                continue

            hits = search_hits(
                spec=spec,
                store=store,
                question=example["question"],
                lecture_key=lecture_key,
                reranker=reranker,
            )
            candidates = store.lecture_indices[lecture_key]["segments"]
            metrics = compute_temporal_metrics(
                hits=hits,
                answer_ranges=example["answer_ranges"],
                candidates=candidates,
                lecture_key=lecture_key,
                query_id=example["example_id"],
                ndcg_measures=ndcg_measures,
                k_values=k_values,
            )

            system_metric_rows.append(metrics)

        summary_row = {
            **build_approach_columns(spec),
            "question_count": len(system_metric_rows),
            "skipped_count": system_skips,
        }
        for column in metric_columns:
            summary_row[column] = mean_numeric([row[column] for row in system_metric_rows])
        summary_rows.append(summary_row)
        print(
            f"{system_name}: evaluated {len(system_metric_rows)} rows, skipped {system_skips} rows"
        )

    retrieval_fieldnames = [
        "system",
        "kind",
        "approach",
        "retrieval_source",
        "embedding_model",
        "rerank_model",
        "initial_top_k",
        "final_top_n",
        "baseline_chunk_strategy",
        "baseline_chunk_size_tokens",
        "baseline_overlap_percent",
        "question_count",
        "skipped_count",
        *metric_columns,
    ]
    skipped_fieldnames = [
        "row_index",
        "example_id",
        "system",
        "lecture_key",
        "reason",
        "question",
        "answer_timestamps",
    ]

    write_csv(output_dir / "retrieval_evaluation.csv", summary_rows, retrieval_fieldnames)
    write_csv(output_dir / "skipped_rows.csv", skipped_rows, skipped_fieldnames)

    print(f"Wrote {output_dir / 'retrieval_evaluation.csv'}")
    print(f"Wrote {output_dir / 'skipped_rows.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
