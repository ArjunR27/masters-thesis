from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import exp
from pathlib import Path

import ollama

SCRIPT_DIR = Path(__file__).resolve().parent
MASTERS_THESIS_DIR = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(MASTERS_THESIS_DIR) not in sys.path:
    sys.path.insert(0, str(MASTERS_THESIS_DIR))

from baseline_rag_system.store_builder import BaselineStoreBuilder  # noqa: E402
from baseline_rag_system.types import (  # noqa: E402
    BaselineRagConfig,
    build_default_baseline_configs,
)
from retrieval_metrics import (  # noqa: E402
    DEFAULT_RETRIEVAL_TOLERANCE_SECONDS,
    compute_retrieval_metrics,
)
from treeseg_vector_index_modular.cross_encoder_reranker import (  # noqa: E402
    CrossEncoderReranker,
)
from treeseg_vector_index_modular.lecture_descriptor import LectureDescriptor  # noqa: E402
from treeseg_vector_index_modular.lecture_segment_builder import (  # noqa: E402
    SummaryTreeBuildOptions,
)
from treeseg_vector_index_modular.lpm_config_builder import LpmConfigBuilder  # noqa: E402
from treeseg_vector_index_modular.ollama_responder import OllamaResponder  # noqa: E402
from treeseg_vector_index_modular.rerank_input_builder import (  # noqa: E402
    RerankInputBuilder,
)
from treeseg_vector_index_modular.vector_store_factory import VectorStoreFactory  # noqa: E402

DATASET_PATH = SCRIPT_DIR / "storage" / "eduvid_data" / "real_world_test.csv"
VIDEOS_ROOT = SCRIPT_DIR / "storage" / "videos"
OUTPUT_DIR = SCRIPT_DIR / "storage" / "evaluation_outputs"
SUMMARY_TREE_CACHE_DIR = SCRIPT_DIR / "storage" / "summary_tree_cache"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"
TOP_K = 10
MAX_CONTEXT_HITS = 5
MAX_CONTEXT_CHARS = 8000
SUMMARY_TREE_TOP_DESCENDANT_LEAVES = 3

INSUFFICIENT_CONTEXT_RESPONSE = (
    "The retrieved lecture segments do not contain enough information to answer "
    "this question."
)

ASR_QUERY_SYSTEM_PROMPT = """You are an intelligent teaching assistant helping a student understand
material from a college-level lecture.

You will be given retrieved transcript evidence from the lecture. The evidence may include:
- High-level summary nodes that describe a larger section of the lecture
- Supporting transcript excerpts grounded in the lecture audio

Your job is to answer the student's question using ONLY the provided context.

Rules:
1. Base your answer strictly on the provided context.
2. If the answer is not directly stated but can be reasonably inferred, say that it is inferred.
3. If the context is insufficient, clearly say so instead of guessing.
4. Give a helpful college-level explanation, but stay concise.
5. Do not mention slides, OCR, or visual evidence."""

MULTIMODAL_QUERY_SYSTEM_PROMPT = """You are an intelligent teaching assistant helping a student understand
material from a college-level lecture.

You will be given retrieved evidence from the lecture. The evidence may include:
- Transcript excerpts grounded in the lecture audio
- Slide OCR text captured from lecture slides

Your job is to answer the student's question using ONLY the provided context.

Rules:
1. Base your answer strictly on the provided context.
2. If the answer is not directly stated but can be reasonably inferred, say that it is inferred.
3. If the context is insufficient, clearly say so instead of guessing.
4. Give a helpful college-level explanation, but stay concise.
5. You may use either transcript or slide evidence, but do not invent information not present in the retrieved context."""

TOKEN_RE = re.compile(r"\b\w+\b")


@dataclass(frozen=True)
class DatasetExample:
    example_id: str
    row_index: int
    url: str
    video_id: str
    question: str
    answer: str
    timestamp: str
    timestamp_seconds: int


@dataclass(frozen=True)
class EvaluationSystemSpec:
    name: str
    kind: str
    baseline_config: BaselineRagConfig | None = None

    @property
    def include_ocr(self) -> bool:
        return bool(self.baseline_config and self.baseline_config.attach_ocr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate EduVidQA answers for TreeSeg and optional baseline chunked "
            "RAG systems."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on the number of dataset rows to evaluate.",
    )
    parser.add_argument(
        "--leaf",
        action="store_true",
        help="Run only the TreeSeg leaf retriever. If no TreeSeg flag is set, both TreeSeg systems run.",
    )
    parser.add_argument(
        "--summary_tree",
        action="store_true",
        help="Run only the TreeSeg summary-tree retriever. If no TreeSeg flag is set, both TreeSeg systems run.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Include baseline chunked RAG systems.",
    )
    parser.add_argument(
        "--baseline-chunk-strategy",
        action="append",
        choices=["utterance_packed", "raw_token_window"],
        dest="baseline_chunk_strategies",
        help="Optional baseline chunk-strategy filter. Repeat to include multiple.",
    )
    parser.add_argument(
        "--baseline-chunk-size",
        action="append",
        type=int,
        choices=[128, 256, 512],
        dest="baseline_chunk_sizes",
        help="Optional baseline chunk-size filter. Repeat to include multiple.",
    )
    parser.add_argument(
        "--baseline-overlap-percent",
        action="append",
        type=int,
        choices=[0, 10],
        dest="baseline_overlap_percents",
        help="Optional baseline overlap-percent filter. Repeat to include multiple.",
    )
    parser.add_argument(
        "--baseline-ocr-mode",
        action="append",
        choices=["transcript_only", "combined_ocr"],
        dest="baseline_ocr_modes",
        help="Optional baseline OCR-mode filter. Repeat to include multiple.",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "").strip() or None,
        help=(
            "Optional Ollama API base URL. Defaults to local Ollama, or the "
            "OLLAMA_HOST environment variable if set."
        ),
    )
    parser.add_argument(
        "--summary-tree-workers",
        default="auto",
        help=(
            "Summary-tree build workers. Use 'auto' for a conservative default "
            "or pass a positive integer."
        ),
    )
    parser.add_argument(
        "--summary-tree-cache-dir",
        default=str(SUMMARY_TREE_CACHE_DIR),
        help=(
            "Directory for per-lecture summary-tree cache files. Set to an empty "
            "string to disable caching."
        ),
    )
    parser.add_argument(
        "--rebuild-summary-tree-cache",
        action="store_true",
        help="Ignore any existing summary-tree cache files and rebuild them.",
    )
    parser.add_argument(
        "--retrieval-tolerance-seconds",
        type=float,
        default=DEFAULT_RETRIEVAL_TOLERANCE_SECONDS,
        help="Tolerance window in seconds for relaxed timestamp-based retrieval metrics.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank retrieved results with a cross-encoder before QA and metric scoring.",
    )
    parser.add_argument(
        "--rerank-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model used when --rerank is set.",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=MAX_CONTEXT_HITS,
        help="Number of results kept after reranking.",
    )
    return parser


def build_system_specs(args: argparse.Namespace) -> list[EvaluationSystemSpec]:
    specs: list[EvaluationSystemSpec] = []

    tree_specs: list[EvaluationSystemSpec] = []
    if args.leaf:
        tree_specs.append(EvaluationSystemSpec(name="leaf", kind="leaf"))
    if args.summary_tree:
        tree_specs.append(EvaluationSystemSpec(name="summary_tree", kind="summary_tree"))
    if not tree_specs:
        tree_specs = [
            EvaluationSystemSpec(name="leaf", kind="leaf"),
            EvaluationSystemSpec(name="summary_tree", kind="summary_tree"),
        ]
    specs.extend(tree_specs)

    if args.baseline:
        baseline_configs = build_default_baseline_configs(
            chunk_strategies=args.baseline_chunk_strategies,
            chunk_sizes=args.baseline_chunk_sizes,
            overlap_percents=args.baseline_overlap_percents,
            ocr_modes=args.baseline_ocr_modes,
        )
        specs.extend(
            EvaluationSystemSpec(
                name=config.system_name,
                kind="baseline",
                baseline_config=config,
            )
            for config in baseline_configs
        )

    return specs


def parse_timestamp_to_seconds(raw_timestamp: str) -> int:
    text = raw_timestamp.strip()
    parts = text.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Unsupported timestamp format: {raw_timestamp!r}")
    numbers = [int(part) for part in parts]
    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds
    hours, minutes, seconds = numbers
    return hours * 3600 + minutes * 60 + seconds


def load_examples(csv_path: Path, limit: int | None = None) -> list[DatasetExample]:
    examples: list[DatasetExample] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader, start=1):
            if limit is not None and len(examples) >= limit:
                break
            examples.append(
                DatasetExample(
                    example_id=f"real_world_test-{row_index:06d}",
                    row_index=row_index,
                    url=(row.get("url") or "").strip(),
                    video_id=(row.get("id") or "").strip(),
                    question=(row.get("question") or "").strip(),
                    answer=(row.get("answer") or "").strip(),
                    timestamp=(row.get("timestamp") or "").strip(),
                    timestamp_seconds=parse_timestamp_to_seconds(
                        (row.get("timestamp") or "").strip()
                    ),
                )
            )
    return examples


def ordered_video_ids(examples: list[DatasetExample]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for example in examples:
        if example.video_id in seen:
            continue
        seen.add(example.video_id)
        ordered.append(example.video_id)
    return ordered


def build_lecture_descriptors(
    video_ids: list[str],
) -> tuple[list[LectureDescriptor], dict[str, LectureDescriptor], list[str]]:
    lectures: list[LectureDescriptor] = []
    lecture_by_video: dict[str, LectureDescriptor] = {}
    missing_videos: list[str] = []

    for video_id in video_ids:
        video_dir = VIDEOS_ROOT / video_id
        transcript_path = video_dir / f"{video_id}_transcripts.csv"
        if not transcript_path.exists():
            missing_videos.append(video_id)
            continue

        lecture = LectureDescriptor(
            speaker="eduvid",
            course_dir="real_world_test",
            meeting_id=video_id,
            video_id=video_id,
            transcripts_path=str(transcript_path),
            meeting_dir=str(video_dir),
        )
        lectures.append(lecture)
        lecture_by_video[video_id] = lecture

    return lectures, lecture_by_video, missing_videos


def resolve_summary_tree_workers(raw_value: str) -> int:
    value = (raw_value or "").strip().lower()
    if value in {"", "auto"}:
        cpu_count = os.cpu_count() or 1
        return min(2, max(1, cpu_count // 4))
    workers = int(value)
    if workers < 1:
        raise ValueError("--summary-tree-workers must be 'auto' or a positive integer.")
    return workers


def build_tree_store(
    lectures: list[LectureDescriptor],
    index_kind: str,
    summary_tree_build_options: SummaryTreeBuildOptions | None = None,
):
    config = LpmConfigBuilder.build_lpm_config(embedding_model=EMBEDDING_MODEL)
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
        summary_tree_build_options=summary_tree_build_options,
    )


def build_baseline_store(
    lectures: list[LectureDescriptor],
    config: BaselineRagConfig,
):
    return BaselineStoreBuilder().build_store(
        lectures,
        config,
        build_global=False,
    )


def build_reranker(args: argparse.Namespace, spec: EvaluationSystemSpec):
    if not args.rerank:
        return None
    input_builder = None
    if spec.kind == "summary_tree":
        input_builder = RerankInputBuilder.build_summary_tree_rerank_input
    return CrossEncoderReranker(
        args.rerank_model,
        input_builder=input_builder,
    )


def search_leaf_like(
    store,
    query: str,
    lecture_key: str,
    *,
    reranker=None,
    top_k: int = TOP_K,
    top_n: int = MAX_CONTEXT_HITS,
) -> list[dict[str, object]]:
    results = store.search(query, top_k=top_k, lecture_key=lecture_key)
    if reranker:
        return reranker.rerank(query, results, top_n=top_n)
    return results[: min(top_n, len(results))]


def search_summary_tree(
    store,
    query: str,
    lecture_key: str,
    *,
    reranker=None,
    top_k: int = TOP_K,
    top_n: int = MAX_CONTEXT_HITS,
) -> list[dict[str, object]]:
    query_embedding = store.encode_query(query)
    results = store.search_with_embedding(
        query_embedding, top_k=top_k, lecture_key=lecture_key
    )
    results = store.expand_summary_tree_results(
        query=query,
        results=results,
        lecture_key=lecture_key,
        top_descendant_leaves=SUMMARY_TREE_TOP_DESCENDANT_LEAVES,
        query_embedding=query_embedding,
    )
    if reranker:
        results = reranker.rerank(query, results, top_n=top_n)
    results = store.deduplicate_summary_tree_results(results)
    return results[: min(top_n, len(results))]


def retrieve_hits(
    spec: EvaluationSystemSpec,
    store,
    question: str,
    lecture_key: str,
    *,
    reranker=None,
    top_k: int = TOP_K,
    top_n: int = MAX_CONTEXT_HITS,
) -> list[dict[str, object]]:
    if spec.kind == "summary_tree":
        return search_summary_tree(
            store,
            question,
            lecture_key,
            reranker=reranker,
            top_k=top_k,
            top_n=top_n,
        )
    return search_leaf_like(
        store,
        question,
        lecture_key,
        reranker=reranker,
        top_k=top_k,
        top_n=top_n,
    )


def format_time_range(hit: dict[str, object]) -> str:
    start = hit.get("start")
    end = hit.get("end")
    if start is None or end is None:
        return ""
    return f"time={float(start):.2f}-{float(end):.2f}s"


def compact_text(text: str) -> str:
    return " ".join(text.split()).strip()


def build_leaf_block(
    hit: dict[str, object],
    rank: int,
    *,
    include_ocr: bool = False,
) -> str:
    parts = [f"[{rank}]"]
    segment_id = hit.get("segment_id")
    if segment_id is not None:
        parts.append(f"seg={segment_id}")
    time_text = format_time_range(hit)
    if time_text:
        parts.append(time_text)

    spoken, ocr = RerankInputBuilder.split_segment_text(str(hit.get("text") or ""))
    if not include_ocr:
        body = compact_text(spoken or str(hit.get("text") or ""))
        if not body:
            body = "<blank>"
        return "\n".join([" ".join(parts), f"Transcript:\n{body}"])

    spoken = compact_text(spoken)
    ocr = compact_text(ocr)
    if not spoken:
        spoken = "<blank>"
    if not ocr:
        ocr = "<blank>"
    return "\n".join(
        [
            " ".join(parts),
            f"Transcript:\n{spoken}",
            f"Slide OCR:\n{ocr}",
        ]
    )


def build_summary_tree_block(hit: dict[str, object], rank: int) -> str:
    parts = [f"[{rank}]"]
    if hit.get("is_leaf", True):
        parts.append("leaf")
    else:
        parts.append("summary-node")
    depth = hit.get("depth")
    if depth is not None:
        parts.append(f"depth={depth}")
    time_text = format_time_range(hit)
    if time_text:
        parts.append(time_text)

    if hit.get("is_leaf", True):
        body = compact_text(str(hit.get("text") or ""))
        if not body:
            body = "<blank>"
        return "\n".join([" ".join(parts), f"Transcript:\n{body}"])

    summary_text = compact_text(
        str(hit.get("summary_text") or hit.get("text") or "<blank>")
    )
    supporting_blocks: list[str] = []
    for leaf_rank, leaf in enumerate(hit.get("supporting_leaves") or [], start=1):
        leaf_parts = [f"Support {leaf_rank}"]
        segment_id = leaf.get("segment_id")
        if segment_id is not None:
            leaf_parts.append(f"seg={segment_id}")
        leaf_time = format_time_range(leaf)
        if leaf_time:
            leaf_parts.append(leaf_time)
        leaf_text = compact_text(str(leaf.get("text") or ""))
        if not leaf_text:
            leaf_text = "<blank>"
        supporting_blocks.append("\n".join([" ".join(leaf_parts), leaf_text]))

    if supporting_blocks:
        evidence = "\n\n".join(supporting_blocks)
    else:
        evidence = "<blank>"

    return "\n".join(
        [
            " ".join(parts),
            f"Summary:\n{summary_text}",
            f"Supporting transcript evidence:\n{evidence}",
        ]
    )


def build_context(
    results: list[dict[str, object]],
    index_kind: str,
    *,
    include_ocr: bool = False,
) -> str:
    if not results:
        return ""

    blocks: list[str] = []
    total_chars = 0
    for rank, hit in enumerate(results, start=1):
        if index_kind == "summary_tree":
            block = build_summary_tree_block(hit, rank)
        else:
            block = build_leaf_block(hit, rank, include_ocr=include_ocr)
        if not block:
            continue
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        total_chars += len(block) + 2
    return "\n\n".join(blocks).strip()


def generate_answer(
    client: ollama.Client,
    question: str,
    context: str,
    *,
    include_ocr: bool = False,
) -> str:
    if not context:
        return INSUFFICIENT_CONTEXT_RESPONSE
    system_prompt = (
        MULTIMODAL_QUERY_SYSTEM_PROMPT if include_ocr else ASR_QUERY_SYSTEM_PROMPT
    )
    return OllamaResponder.query_response(
        question,
        context,
        model=OLLAMA_MODEL,
        system_prompt=system_prompt,
        temperature=0.0,
        client=client,
    )


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def bleu1(reference: str, generated: str) -> float:
    ref_tokens = tokenize(reference)
    gen_tokens = tokenize(generated)
    if not ref_tokens or not gen_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    gen_counts = Counter(gen_tokens)
    overlap = sum(min(count, ref_counts[token]) for token, count in gen_counts.items())
    precision = overlap / len(gen_tokens)
    if precision == 0.0:
        return 0.0
    if len(gen_tokens) > len(ref_tokens):
        brevity_penalty = 1.0
    else:
        brevity_penalty = exp(1.0 - (len(ref_tokens) / len(gen_tokens)))
    return brevity_penalty * precision


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for idx_b, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[idx_b - 1] + 1)
            else:
                curr.append(max(prev[idx_b], curr[idx_b - 1]))
        prev = curr
    return prev[-1]


def rouge_l(reference: str, generated: str) -> float:
    ref_tokens = tokenize(reference)
    gen_tokens = tokenize(generated)
    if not ref_tokens or not gen_tokens:
        return 0.0
    lcs = lcs_length(ref_tokens, gen_tokens)
    precision = lcs / len(gen_tokens)
    recall = lcs / len(ref_tokens)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def meteor(reference: str, generated: str) -> float:
    ref_tokens = tokenize(reference)
    gen_tokens = tokenize(generated)
    if not ref_tokens or not gen_tokens:
        return 0.0

    positions_by_token: dict[str, list[int]] = defaultdict(list)
    for idx, token in enumerate(ref_tokens):
        positions_by_token[token].append(idx)

    used_positions: set[int] = set()
    matched_positions: list[int] = []
    matches = 0

    for token in gen_tokens:
        for position in positions_by_token.get(token, []):
            if position in used_positions:
                continue
            used_positions.add(position)
            matched_positions.append(position)
            matches += 1
            break

    if matches == 0:
        return 0.0

    precision = matches / len(gen_tokens)
    recall = matches / len(ref_tokens)
    f_mean = (10.0 * precision * recall) / (recall + 9.0 * precision)

    matched_positions.sort()
    chunks = 1
    for prev, curr in zip(matched_positions, matched_positions[1:]):
        if curr != prev + 1:
            chunks += 1

    penalty = 0.5 * ((chunks / matches) ** 3)
    return (1.0 - penalty) * f_mean


def compute_metrics(answer: str, generated: str) -> dict[str, float]:
    return {
        "bleu1": bleu1(answer, generated),
        "rouge_l": rouge_l(answer, generated),
        "meteor": meteor(answer, generated),
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "example_id",
        "row_index",
        "video_id",
        "timestamp",
        "timestamp_seconds",
        "system",
        "retrieval_hit_exact",
        "retrieval_mrr_exact",
        "retrieval_hit_relaxed",
        "retrieval_mrr_relaxed",
        "bleu1",
        "rouge_l",
        "meteor",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_metric(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _metric_delta_block(
    source_summary: dict[str, object],
    target_summary: dict[str, object],
    metric_names: list[str],
) -> dict[str, float]:
    return {
        metric: float(target_summary.get(metric, 0.0))
        - float(source_summary.get(metric, 0.0))
        for metric in metric_names
    }


def build_summary(
    examples: list[DatasetExample],
    skipped_examples: list[DatasetExample],
    metrics_rows: list[dict[str, object]],
    system_specs: list[EvaluationSystemSpec],
    system_skips: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "dataset_path": str(DATASET_PATH),
        "output_dir": str(OUTPUT_DIR),
        "requested_question_count": len(examples),
        "evaluated_question_count": len(examples) - len(skipped_examples),
        "skipped_question_count": len(skipped_examples),
        "skipped_example_ids": [example.example_id for example in skipped_examples],
        "systems": {},
        "comparison": {},
    }

    by_system: dict[str, list[dict[str, object]]] = {spec.name: [] for spec in system_specs}
    for row in metrics_rows:
        system_name = str(row["system"])
        by_system.setdefault(system_name, []).append(row)

    metric_names = [
        "retrieval_hit_exact",
        "retrieval_mrr_exact",
        "retrieval_hit_relaxed",
        "retrieval_mrr_relaxed",
        "bleu1",
        "rouge_l",
        "meteor",
    ]

    for spec in system_specs:
        rows = by_system.get(spec.name, [])
        skipped_rows = system_skips.get(spec.name, [])
        system_summary = {
            "question_count": len(rows),
            "skipped_question_count": len(skipped_rows),
            "skipped_example_ids": [row["example_id"] for row in skipped_rows],
            "skip_reasons": dict(
                Counter(str(row.get("reason") or "unknown") for row in skipped_rows)
            ),
            **{metric: mean_metric(rows, metric) for metric in metric_names},
        }
        if spec.baseline_config is not None:
            system_summary["config"] = spec.baseline_config.to_dict()
        summary["systems"][spec.name] = system_summary

    if "leaf" in summary["systems"] and "summary_tree" in summary["systems"]:
        summary["comparison"]["metric_deltas_summary_tree_minus_leaf"] = _metric_delta_block(
            summary["systems"]["leaf"],
            summary["systems"]["summary_tree"],
            metric_names,
        )

    baseline_deltas = {}
    for spec in system_specs:
        if spec.kind != "baseline":
            continue
        system_summary = summary["systems"][spec.name]
        comparison = {}
        if "leaf" in summary["systems"]:
            comparison["metric_deltas_vs_leaf"] = _metric_delta_block(
                summary["systems"]["leaf"],
                system_summary,
                metric_names,
            )
        if "summary_tree" in summary["systems"]:
            comparison["metric_deltas_vs_summary_tree"] = _metric_delta_block(
                summary["systems"]["summary_tree"],
                system_summary,
                metric_names,
            )
        baseline_deltas[spec.name] = comparison
    if baseline_deltas:
        summary["comparison"]["baseline_deltas"] = baseline_deltas

    return summary


def evaluate_system(
    spec: EvaluationSystemSpec,
    store,
    examples: list[DatasetExample],
    lecture_by_video: dict[str, LectureDescriptor],
    client: ollama.Client,
    *,
    reranker=None,
    skipped_lectures: dict[str, str] | None = None,
    retrieval_tolerance_seconds: float = DEFAULT_RETRIEVAL_TOLERANCE_SECONDS,
    result_top_n: int = MAX_CONTEXT_HITS,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    predictions: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    skipped_lectures = skipped_lectures or {}

    for index, example in enumerate(examples, start=1):
        lecture = lecture_by_video[example.video_id]
        if lecture.key not in store.lecture_indices:
            reason = skipped_lectures.get(lecture.key, "lecture_not_indexed")
            skipped_rows.append(
                {
                    "example_id": example.example_id,
                    "row_index": example.row_index,
                    "video_id": example.video_id,
                    "lecture_key": lecture.key,
                    "reason": reason,
                    "system": spec.name,
                }
            )
            print(
                f"[{spec.name}] {index}/{len(examples)} "
                f"{example.example_id} skipped ({reason})"
            )
            continue

        hits = retrieve_hits(
            spec,
            store,
            example.question,
            lecture.key,
            reranker=reranker,
            top_k=TOP_K,
            top_n=result_top_n,
        )
        hits = hits[: min(result_top_n, len(hits))]

        retrieval_metrics = compute_retrieval_metrics(
            hits,
            example.timestamp_seconds,
            tolerance_seconds=retrieval_tolerance_seconds,
        )
        context = build_context(
            hits,
            index_kind=spec.kind,
            include_ocr=spec.include_ocr,
        )
        generated = generate_answer(
            client,
            example.question,
            context,
            include_ocr=spec.include_ocr,
        )
        qa_metrics = compute_metrics(example.answer, generated)

        predictions.append(
            {
                "example_id": example.example_id,
                "row_index": example.row_index,
                "video_id": example.video_id,
                "lecture_key": lecture.key,
                "url": example.url,
                "timestamp": example.timestamp,
                "timestamp_seconds": example.timestamp_seconds,
                "question": example.question,
                "answer": example.answer,
                "generated": generated,
                "system": spec.name,
                "retrieved_hits": hits,
                "retrieval_metrics": retrieval_metrics,
                "config": spec.baseline_config.to_dict() if spec.baseline_config else None,
            }
        )

        metrics_rows.append(
            {
                "example_id": example.example_id,
                "row_index": example.row_index,
                "video_id": example.video_id,
                "timestamp": example.timestamp,
                "timestamp_seconds": example.timestamp_seconds,
                "system": spec.name,
                **retrieval_metrics,
                **qa_metrics,
            }
        )

        print(
            f"[{spec.name}] {index}/{len(examples)} "
            f"{example.example_id} complete"
        )

    return predictions, metrics_rows, skipped_rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    system_specs = build_system_specs(args)
    examples = load_examples(DATASET_PATH, limit=args.limit)
    if not examples:
        raise SystemExit("No dataset rows found to evaluate.")

    video_ids = ordered_video_ids(examples)
    lectures, lecture_by_video, missing_videos = build_lecture_descriptors(video_ids)

    if not lectures:
        raise SystemExit("No transcript bundles were found for the selected dataset rows.")

    skipped_examples = [example for example in examples if example.video_id in missing_videos]
    evaluated_examples = [
        example for example in examples if example.video_id in lecture_by_video
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dataset rows requested: {len(examples)}")
    print(f"Rows skipped (missing transcript): {len(skipped_examples)}")
    print(f"Videos indexed: {len(lectures)}")
    print(f"Systems selected: {len(system_specs)}")
    for spec in system_specs:
        print(f"  - {spec.name}")
    if args.ollama_host:
        print(f"Ollama host: {args.ollama_host}")
    print(f"Retrieval tolerance seconds: {args.retrieval_tolerance_seconds}")
    print(f"Rerank enabled: {args.rerank}")
    if args.rerank:
        print(f"Rerank model: {args.rerank_model}")
        print(f"Rerank top-n: {args.rerank_top_n}")

    summary_tree_build_options = None
    if any(spec.kind == "summary_tree" for spec in system_specs):
        resolved_summary_tree_workers = resolve_summary_tree_workers(
            args.summary_tree_workers
        )
        summary_tree_cache_dir = args.summary_tree_cache_dir.strip() or None
        summary_tree_build_options = SummaryTreeBuildOptions(
            workers=resolved_summary_tree_workers,
            cache_dir=summary_tree_cache_dir,
            rebuild_cache=args.rebuild_summary_tree_cache,
            ollama_host=args.ollama_host,
            cache_version="v1",
        )
        print(f"Summary-tree workers: {summary_tree_build_options.workers}")
        if summary_tree_build_options.workers > 2:
            print(
                "Warning: summary-tree worker counts above 2 are outside the "
                "conservative tested range and may reduce stability or throughput."
            )
        print(
            "Summary-tree cache dir: "
            f"{summary_tree_build_options.cache_dir or '<disabled>'}"
        )
        print(
            "Rebuild summary-tree cache: "
            f"{summary_tree_build_options.rebuild_cache}"
        )

    client = ollama.Client(host=args.ollama_host) if args.ollama_host else ollama.Client()

    all_metrics: list[dict[str, object]] = []
    all_predictions: dict[str, list[dict[str, object]]] = {}
    system_skips: dict[str, list[dict[str, object]]] = {}

    for spec in system_specs:
        reranker = build_reranker(args, spec)
        skipped_lectures = {}

        if spec.kind == "leaf":
            print("Building leaf index...")
            store = build_tree_store(lectures, index_kind="leaf")
        elif spec.kind == "summary_tree":
            print("Building summary-tree index...")
            store = build_tree_store(
                lectures,
                index_kind="summary_tree",
                summary_tree_build_options=summary_tree_build_options,
            )
        else:
            print(f"Building baseline index: {spec.name}")
            build_result = build_baseline_store(lectures, spec.baseline_config)
            store = build_result.store
            skipped_lectures = build_result.skipped_lectures

        print(f"Evaluating {spec.name}...")
        predictions, metrics_rows, skipped_rows = evaluate_system(
            spec,
            store,
            evaluated_examples,
            lecture_by_video,
            client,
            reranker=reranker,
            skipped_lectures=skipped_lectures,
            retrieval_tolerance_seconds=args.retrieval_tolerance_seconds,
            result_top_n=args.rerank_top_n if args.rerank else MAX_CONTEXT_HITS,
        )
        all_predictions[spec.name] = predictions
        system_skips[spec.name] = skipped_rows
        all_metrics.extend(metrics_rows)

    summary = build_summary(
        examples,
        skipped_examples,
        all_metrics,
        system_specs,
        system_skips,
    )

    for system_name, predictions in all_predictions.items():
        write_jsonl(OUTPUT_DIR / f"{system_name}_predictions.jsonl", predictions)
    write_metrics_csv(OUTPUT_DIR / "metrics_per_question.csv", all_metrics)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nEvaluation complete.")
    print(json.dumps(summary["systems"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
