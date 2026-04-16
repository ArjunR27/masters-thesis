"""LPM QA Custom Evaluation

Compares TreeSeg (leaf, summary_tree) and baseline RAG chunking strategies
on the LPM QA labeled dataset. Each system is evaluated in two variants:
  - ASR only: retrieval from transcript segments only
  - ASR + OCR: ASR retrieval combined with a separate per-slide OCR index

Ground-truth timestamps are time ranges (e.g. "1186.9-1194.6 | 1287.1-1334.5").
A retrieved segment counts as a hit if it temporally overlaps any GT range.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import ollama

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from baseline_rag_system.store_builder import BaselineStoreBuilder  # noqa: E402
from baseline_rag_system.types import (  # noqa: E402
    BaselineRagConfig,
    build_default_baseline_configs,
)
from lpm_qa_custom_eval.lpm_qa_dataset import LpmQaExample, load_lpm_qa_examples  # noqa: E402
from lpm_qa_custom_eval.range_retrieval_metrics import (  # noqa: E402
    DEFAULT_TOLERANCE_SECONDS,
    compute_range_retrieval_metrics,
)
from lpm_qa_custom_eval.token_retrieval_metrics import (  # noqa: E402
    compute_token_retrieval_metrics,
    extract_excerpt_tokens,
)
from treeseg_vector_index_modular.lecture_catalog import LectureCatalog  # noqa: E402
from treeseg_vector_index_modular.lecture_segment_builder import (  # noqa: E402
    SummaryTreeBuildOptions,
)
from treeseg_vector_index_modular.lpm_config_builder import LpmConfigBuilder  # noqa: E402
from treeseg_vector_index_modular.ollama_responder import OllamaResponder  # noqa: E402
from treeseg_vector_index_modular.rerank_input_builder import RerankInputBuilder  # noqa: E402
from treeseg_vector_index_modular.vector_store_factory import VectorStoreFactory  # noqa: E402

LPM_QA_PATH = PROJECT_DIR / "LPM_QA_DATASET" / "lpm_qa_labeled.csv"
LPM_DATA_DIR = PROJECT_DIR / "lpm_data"
OUTPUT_DIR = SCRIPT_DIR / "storage" / "outputs"
SUMMARY_TREE_CACHE_DIR = SCRIPT_DIR / "storage" / "summary_tree_cache"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"
TOP_K = 10
TOP_N = 5
MAX_CONTEXT_CHARS = 8000
SUMMARY_TREE_TOP_DESCENDANT_LEAVES = 3

INSUFFICIENT_CONTEXT_RESPONSE = (
    "The retrieved lecture segments do not contain enough information to answer this question."
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

import evaluate as hf_evaluate

_bleu_metric = hf_evaluate.load("bleu")
_rouge_metric = hf_evaluate.load("rouge")
_meteor_metric = hf_evaluate.load("meteor")


# ---------------------------------------------------------------------------
# System spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LpmSystemSpec:
    name: str
    kind: str  # "leaf" | "summary_tree" | "baseline"
    baseline_config: BaselineRagConfig | None = None
    use_ocr: bool = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LPM QA dataset across TreeSeg and baseline RAG systems."
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of QA examples to evaluate.")
    parser.add_argument("--leaf", action="store_true",
                        help="Include TreeSeg leaf system.")
    parser.add_argument("--summary-tree", action="store_true",
                        help="Include TreeSeg summary-tree system.")
    parser.add_argument("--baseline", action="store_true",
                        help="Include baseline chunking systems.")
    parser.add_argument("--no-ocr", action="store_true",
                        help="Skip +OCR variants (run ASR-only for all systems).")
    parser.add_argument("--no-asr-only", action="store_true",
                        help="Skip ASR-only variants (run +OCR only for all systems).")
    parser.add_argument("--baseline-chunk-strategy", action="append",
                        choices=["utterance_packed", "raw_token_window"],
                        dest="baseline_chunk_strategies")
    parser.add_argument("--baseline-chunk-size", action="append", type=int,
                        choices=[128, 256, 512], dest="baseline_chunk_sizes")
    parser.add_argument("--baseline-overlap-percent", action="append", type=int,
                        choices=[0, 10], dest="baseline_overlap_percents")
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "").strip() or None)
    parser.add_argument("--summary-tree-workers", default="auto")
    parser.add_argument("--summary-tree-cache-dir", default=str(SUMMARY_TREE_CACHE_DIR))
    parser.add_argument("--rebuild-summary-tree-cache", action="store_true")
    parser.add_argument("--retrieval-tolerance-seconds", type=float,
                        default=DEFAULT_TOLERANCE_SECONDS)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--rerank-top-n", type=int, default=TOP_N)
    parser.add_argument("--skip-qa", action="store_true",
                        help="Skip answer generation; compute retrieval metrics only.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser


def build_system_specs(args: argparse.Namespace) -> list[LpmSystemSpec]:
    run_leaf = args.leaf or args.summary_tree or args.baseline
    # If no system flags given, run everything
    run_all = not (args.leaf or args.summary_tree or args.baseline)

    include_asr_only = not args.no_asr_only
    include_ocr = not args.no_ocr

    specs: list[LpmSystemSpec] = []

    treeseg_kinds = []
    if run_all or args.leaf:
        treeseg_kinds.append("leaf")
    if run_all or args.summary_tree:
        treeseg_kinds.append("summary_tree")

    for kind in treeseg_kinds:
        if include_asr_only:
            specs.append(LpmSystemSpec(name=f"treeseg_{kind}", kind=kind, use_ocr=False))
        if include_ocr:
            specs.append(LpmSystemSpec(name=f"treeseg_{kind}+ocr", kind=kind, use_ocr=True))

    if run_all or args.baseline:
        baseline_configs = build_default_baseline_configs(
            chunk_strategies=args.baseline_chunk_strategies,
            chunk_sizes=args.baseline_chunk_sizes,
            overlap_percents=args.baseline_overlap_percents,
            ocr_modes=["transcript_only"],  # OCR is always handled separately
        )
        for config in baseline_configs:
            if include_asr_only:
                specs.append(LpmSystemSpec(
                    name=config.system_name,
                    kind="baseline",
                    baseline_config=config,
                    use_ocr=False,
                ))
            if include_ocr:
                specs.append(LpmSystemSpec(
                    name=f"{config.system_name}+ocr",
                    kind="baseline",
                    baseline_config=config,
                    use_ocr=True,
                ))

    return specs


# ---------------------------------------------------------------------------
# Store builders
# ---------------------------------------------------------------------------

def resolve_summary_tree_workers(raw_value: str) -> int:
    value = (raw_value or "").strip().lower()
    if value in {"", "auto"}:
        cpu_count = os.cpu_count() or 1
        return min(2, max(1, cpu_count // 4))
    workers = int(value)
    if workers < 1:
        raise ValueError("--summary-tree-workers must be 'auto' or a positive integer.")
    return workers


def build_treeseg_asr_store(lectures, index_kind: str, args: argparse.Namespace):
    config = LpmConfigBuilder.build_lpm_config(embedding_model=EMBEDDING_MODEL)
    cache_dir = args.summary_tree_cache_dir.strip() or None
    build_options = None
    if index_kind == "summary_tree":
        workers = resolve_summary_tree_workers(args.summary_tree_workers)
        build_options = SummaryTreeBuildOptions(
            workers=workers,
            cache_dir=cache_dir or None,
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


def build_baseline_asr_store(lectures, config: BaselineRagConfig):
    return BaselineStoreBuilder().build_store(lectures, config, build_global=False)


def build_shared_ocr_store(lectures):
    """Per-slide OCR index shared across all +OCR system variants."""
    return VectorStoreFactory().build_ocr_vector_store(
        lectures,
        embed_model=EMBEDDING_MODEL,
        normalize=True,
        build_global=False,
        ocr_min_conf=60.0,
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def search_treeseg_leaf(store, query: str, lecture_key: str, *, top_k: int, top_n: int) -> list[dict]:
    results = store.search(query, top_k=top_k, lecture_key=lecture_key)
    return results[:min(top_n, len(results))]


def search_treeseg_summary_tree(
    store, query: str, lecture_key: str, *, top_k: int, top_n: int
) -> list[dict]:
    query_embedding = store.encode_query(query)
    results = store.search_with_embedding(query_embedding, top_k=top_k, lecture_key=lecture_key)
    results = store.expand_summary_tree_results(
        query=query,
        results=results,
        lecture_key=lecture_key,
        top_descendant_leaves=SUMMARY_TREE_TOP_DESCENDANT_LEAVES,
        query_embedding=query_embedding,
    )
    results = store.deduplicate_summary_tree_results(results)
    return results[:min(top_n, len(results))]


def search_asr(
    spec: LpmSystemSpec,
    store,
    query: str,
    lecture_key: str,
    *,
    top_k: int,
    top_n: int,
) -> list[dict]:
    if spec.kind == "summary_tree":
        return search_treeseg_summary_tree(store, query, lecture_key, top_k=top_k, top_n=top_n)
    return search_treeseg_leaf(store, query, lecture_key, top_k=top_k, top_n=top_n)


def search_ocr(ocr_store, query: str, lecture_key: str, *, top_k: int, top_n: int) -> list[dict]:
    if lecture_key not in ocr_store.lecture_indices:
        return []
    results = ocr_store.search(query, top_k=top_k, lecture_key=lecture_key)
    return results[:min(top_n, len(results))]


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

def _compact(text: str) -> str:
    return " ".join(text.split()).strip()


def _format_time(hit: dict) -> str:
    start, end = hit.get("start"), hit.get("end")
    if start is None or end is None:
        return ""
    return f"time={float(start):.2f}-{float(end):.2f}s"


def build_asr_block(hit: dict, rank: int, kind: str) -> str:
    parts = [f"[{rank}]"]
    if kind == "summary_tree":
        parts.append("leaf" if hit.get("is_leaf", True) else "summary-node")
        depth = hit.get("depth")
        if depth is not None:
            parts.append(f"depth={depth}")
    t = _format_time(hit)
    if t:
        parts.append(t)

    if kind == "summary_tree" and not hit.get("is_leaf", True):
        summary = _compact(str(hit.get("summary_text") or hit.get("text") or "<blank>"))
        leaf_blocks = []
        for lr, leaf in enumerate(hit.get("supporting_leaves") or [], start=1):
            lparts = [f"Support {lr}"]
            lt = _format_time(leaf)
            if lt:
                lparts.append(lt)
            leaf_text = _compact(str(leaf.get("text") or ""))
            leaf_blocks.append("\n".join([" ".join(lparts), leaf_text or "<blank>"]))
        evidence = "\n\n".join(leaf_blocks) if leaf_blocks else "<blank>"
        return "\n".join([" ".join(parts), f"Summary:\n{summary}",
                          f"Supporting transcript evidence:\n{evidence}"])

    spoken, _ = RerankInputBuilder.split_segment_text(str(hit.get("text") or ""))
    body = _compact(spoken or str(hit.get("text") or "")) or "<blank>"
    return "\n".join([" ".join(parts), f"Transcript:\n{body}"])


def build_ocr_block(hit: dict, rank: int) -> str:
    parts = [f"[OCR {rank}]"]
    t = _format_time(hit)
    if t:
        parts.append(t)
    text = _compact(str(hit.get("text") or "")) or "<blank>"
    return "\n".join([" ".join(parts), f"Slide OCR:\n{text}"])


def build_context_string(
    asr_hits: list[dict],
    ocr_hits: list[dict],
    kind: str,
) -> str:
    blocks: list[str] = []
    total = 0

    for rank, hit in enumerate(asr_hits, start=1):
        block = build_asr_block(hit, rank, kind)
        if not block:
            continue
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        total += len(block) + 2

    for rank, hit in enumerate(ocr_hits, start=1):
        block = build_ocr_block(hit, rank)
        if not block:
            continue
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        total += len(block) + 2

    return "\n\n".join(blocks).strip()


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def generate_answer(
    client: ollama.Client,
    question: str,
    context: str,
    *,
    use_ocr: bool,
) -> str:
    if not context:
        return INSUFFICIENT_CONTEXT_RESPONSE
    system_prompt = MULTIMODAL_QUERY_SYSTEM_PROMPT if use_ocr else ASR_QUERY_SYSTEM_PROMPT
    return OllamaResponder.query_response(
        question,
        context,
        model=OLLAMA_MODEL,
        system_prompt=system_prompt,
        temperature=0.0,
        client=client,
    )


# ---------------------------------------------------------------------------
# QA metrics (same implementations as eduvid_evaluation)
# ---------------------------------------------------------------------------

def compute_qa_metrics(reference: str, generated: str) -> dict[str, float]:
    if not reference or not generated:
        return {"bleu1": 0.0, "rouge_l": 0.0, "meteor": 0.0}
    bleu_result = _bleu_metric.compute(
        predictions=[generated], references=[[reference]], max_order=1
    )
    rouge_result = _rouge_metric.compute(
        predictions=[generated], references=[reference]
    )
    meteor_result = _meteor_metric.compute(
        predictions=[generated], references=[reference]
    )
    return {
        "bleu1": float(bleu_result["bleu"]),
        "rouge_l": float(rouge_result["rougeL"]),
        "meteor": float(meteor_result["meteor"]),
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

METRIC_FIELDNAMES = [
    "system",
    "example_id",
    "lecture_key",
    "question_type",
    "question",
    "answer_ranges",
    "generated_answer",
    "retrieval_hit_exact",
    "retrieval_mrr_exact",
    "retrieval_hit_relaxed",
    "retrieval_mrr_relaxed",
    "token_iou",
    "token_precision",
    "token_recall",
    "token_precision_omega",
    "bleu1",
    "rouge_l",
    "meteor",
    "skipped",
    "skip_reason",
]


def write_metrics_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(
    system_specs: list[LpmSystemSpec],
    all_rows: list[dict],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numeric_metrics = [
        "retrieval_hit_exact", "retrieval_mrr_exact",
        "retrieval_hit_relaxed", "retrieval_mrr_relaxed",
        "token_iou", "token_precision", "token_recall", "token_precision_omega",
        "bleu1", "rouge_l", "meteor",
    ]
    systems_summary = {}
    for spec in system_specs:
        spec_rows = [r for r in all_rows if r["system"] == spec.name and not r.get("skipped")]
        if not spec_rows:
            continue
        agg = {m: sum(float(r.get(m) or 0) for r in spec_rows) / len(spec_rows)
               for m in numeric_metrics}
        systems_summary[spec.name] = {"n": len(spec_rows), **agg}

    # Per question-type breakdown
    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        if row.get("skipped"):
            continue
        qt = row.get("question_type") or "unknown"
        for m in numeric_metrics:
            by_type[row["system"]][f"{qt}__{m}"].append(float(row.get(m) or 0))
    question_type_summary = {
        sys_name: {k: sum(v) / len(v) for k, v in metrics.items()}
        for sys_name, metrics in by_type.items()
    }

    summary = {
        "total_examples": len({r["example_id"] for r in all_rows}),
        "systems": systems_summary,
        "question_type_breakdown": question_type_summary,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Segment count diagnostic
# ---------------------------------------------------------------------------

def print_segment_counts(store, label: str) -> None:
    rows = []
    total = 0
    for lecture_key, entry in store.lecture_indices.items():
        n = len(entry.get("segments", []))
        rows.append((lecture_key, n))
        total += n
    rows.sort(key=lambda x: x[0])
    print(f"  [segments] {label}")
    for lecture_key, n in rows:
        print(f"    {lecture_key}: {n} segments")
    if len(rows) > 1:
        print(f"    TOTAL: {total}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ollama_client = ollama.Client(host=args.ollama_host) if args.ollama_host else ollama.Client()

    # Load dataset
    examples = load_lpm_qa_examples(LPM_QA_PATH, limit=args.limit)
    if not examples:
        print("No examples loaded. Check LPM_QA_PATH.")
        return
    print(f"Loaded {len(examples)} QA examples.")

    # Discover lectures
    all_lectures = LectureCatalog.discover_lectures(data_dir=LPM_DATA_DIR)
    needed_keys = {ex.lecture_key for ex in examples}
    lectures = [lec for lec in all_lectures if lec.key in needed_keys]
    found_keys = {lec.key for lec in lectures}
    missing = needed_keys - found_keys
    if missing:
        print(f"WARNING: lectures not found in lpm_data: {sorted(missing)}")
    print(f"Using {len(lectures)} lecture(s): {sorted(found_keys)}")

    system_specs = build_system_specs(args)
    if not system_specs:
        print("No system specs selected. Use --leaf, --summary-tree, --baseline or omit flags to run all.")
        return
    print(f"Running {len(system_specs)} system(s): {[s.name for s in system_specs]}")

    # -----------------------------------------------------------------------
    # Build stores — one per unique (kind, baseline_config) pair
    # -----------------------------------------------------------------------
    asr_stores: dict[str, object] = {}  # key: spec.name -> store

    needs_ocr = any(s.use_ocr for s in system_specs)
    ocr_store = build_shared_ocr_store(lectures) if needs_ocr else None

    # TreeSeg stores: one per index kind
    for kind in ("leaf", "summary_tree"):
        if not any(s.kind == kind for s in system_specs):
            continue
        print(f"Building TreeSeg {kind} ASR store...")
        store = build_treeseg_asr_store(lectures, kind, args)
        print_segment_counts(store, label=f"treeseg_{kind}")
        # Assign to all specs of this kind
        for spec in system_specs:
            if spec.kind == kind:
                asr_stores[spec.name] = store

    # Baseline stores: one per unique baseline_config
    seen_configs: dict[str, object] = {}  # system_name (without +ocr) -> store
    for spec in system_specs:
        if spec.kind != "baseline" or spec.baseline_config is None:
            continue
        cfg_key = spec.baseline_config.system_name
        if cfg_key not in seen_configs:
            print(f"Building baseline ASR store: {cfg_key}")
            result = build_baseline_asr_store(lectures, spec.baseline_config)
            print_segment_counts(result.store, label=cfg_key)
            seen_configs[cfg_key] = result.store
        asr_stores[spec.name] = seen_configs[cfg_key]

    if needs_ocr and ocr_store is not None:
        print_segment_counts(ocr_store, label="ocr_slides")

    # -----------------------------------------------------------------------
    # Pre-cache excerpt tokens per (lecture_key, answer_ranges) — loaded once
    # from the transcript CSV, reused across all system evaluations.
    # -----------------------------------------------------------------------
    lecture_by_key = {lec.key: lec for lec in lectures}
    excerpt_token_cache: dict[str, set[str]] = {}  # key: example_id

    for ex in examples:
        if ex.lecture_key not in lecture_by_key:
            excerpt_token_cache[ex.example_id] = set()
            continue
        lec = lecture_by_key[ex.lecture_key]
        excerpt_token_cache[ex.example_id] = extract_excerpt_tokens(
            lec.transcripts_path, ex.answer_ranges
        )

    # -----------------------------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------------------------
    all_rows: list[dict] = []

    for spec in system_specs:
        print(f"\n=== Evaluating: {spec.name} ===")
        asr_store = asr_stores.get(spec.name)
        if asr_store is None:
            print(f"  [skip] No ASR store built for {spec.name}")
            continue

        for ex_idx, ex in enumerate(examples, start=1):
            # print(f"  [{ex_idx}/{len(examples)}] {ex.example_id} ({ex.lecture_key})", flush=True)
            if ex.lecture_key not in found_keys:
                all_rows.append({
                    "system": spec.name, "example_id": ex.example_id,
                    "lecture_key": ex.lecture_key, "question_type": ex.question_type,
                    "question": ex.question, "answer_ranges": str(ex.answer_ranges),
                    "skipped": True, "skip_reason": "lecture_not_found",
                })
                continue

            # Check ASR store has this lecture
            if ex.lecture_key not in asr_store.lecture_indices:
                all_rows.append({
                    "system": spec.name, "example_id": ex.example_id,
                    "lecture_key": ex.lecture_key, "question_type": ex.question_type,
                    "question": ex.question, "answer_ranges": str(ex.answer_ranges),
                    "skipped": True, "skip_reason": "lecture_not_in_asr_index",
                })
                continue

            # Retrieve
            asr_hits = search_asr(
                spec, asr_store, ex.question, ex.lecture_key,
                top_k=TOP_K, top_n=TOP_N,
            )
            ocr_hits = []
            if spec.use_ocr and ocr_store is not None:
                ocr_hits = search_ocr(ocr_store, ex.question, ex.lecture_key,
                                      top_k=TOP_K, top_n=TOP_N)

            # Retrieval metrics over pooled hits (ASR first, then OCR)
            all_hits = asr_hits + ocr_hits
            retrieval_metrics = compute_range_retrieval_metrics(
                all_hits,
                ex.answer_ranges,
                tolerance_seconds=args.retrieval_tolerance_seconds,
            )

            # Token-level metrics (density-normalised)
            excerpt_tokens = excerpt_token_cache.get(ex.example_id, set())
            all_lecture_segments = (
                asr_store.lecture_indices[ex.lecture_key].get("segments", [])
            )
            token_metrics = compute_token_retrieval_metrics(
                asr_hits,  # ASR hits only — token comparison against transcript
                excerpt_tokens,
                all_lecture_segments,
            )

            # Answer generation
            generated = INSUFFICIENT_CONTEXT_RESPONSE
            qa_metrics: dict[str, float] = {"bleu1": 0.0, "rouge_l": 0.0, "meteor": 0.0}
            if not args.skip_qa:
                context = build_context_string(asr_hits, ocr_hits, spec.kind)
                generated = generate_answer(
                    ollama_client, ex.question, context, use_ocr=spec.use_ocr
                )
                if ex.answer_text:
                    qa_metrics = compute_qa_metrics(ex.answer_text, generated)

            all_rows.append({
                "system": spec.name,
                "example_id": ex.example_id,
                "lecture_key": ex.lecture_key,
                "question_type": ex.question_type,
                "question": ex.question,
                "answer_ranges": str(ex.answer_ranges),
                "generated_answer": generated,
                **retrieval_metrics,
                **token_metrics,
                **qa_metrics,
                "skipped": False,
                "skip_reason": "",
            })

        n_valid = sum(1 for r in all_rows if r["system"] == spec.name and not r.get("skipped"))
        if n_valid:
            for metric in ["retrieval_hit_exact", "retrieval_hit_relaxed",
                           "retrieval_mrr_exact", "token_iou", "token_precision",
                           "token_recall",
                           "token_precision_omega", "bleu1", "rouge_l", "meteor"]:
                vals = [float(r.get(metric) or 0)
                        for r in all_rows if r["system"] == spec.name and not r.get("skipped")]
                print(f"  {metric}: {sum(vals)/len(vals):.4f}")

    # -----------------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------------
    metrics_path = output_dir / "metrics_per_question.csv"
    summary_path = output_dir / "summary.json"

    write_metrics_csv(all_rows, metrics_path)
    write_summary_json(system_specs, all_rows, summary_path)

    print(f"\nWrote per-question metrics: {metrics_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
