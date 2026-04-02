from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict
from math import exp

import ollama

SCRIPT_DIR = Path(__file__).resolve().parent
MASTERS_THESIS_DIR = SCRIPT_DIR.parent

if str(MASTERS_THESIS_DIR) not in sys.path:
    sys.path.insert(0, str(MASTERS_THESIS_DIR))

from treeseg_vector_index_modular.lecture_descriptor import LectureDescriptor  # noqa: E402
from treeseg_vector_index_modular.lpm_config_builder import LpmConfigBuilder  # noqa: E402
from treeseg_vector_index_modular.ollama_responder import OllamaResponder  # noqa: E402
from treeseg_vector_index_modular.vector_store_factory import VectorStoreFactory  # noqa: E402

DATASET_PATH = SCRIPT_DIR / "storage" / "eduvid_data" / "real_world_test.csv"
VIDEOS_ROOT = SCRIPT_DIR / "storage" / "videos"
OUTPUT_DIR = SCRIPT_DIR / "storage" / "evaluation_outputs"

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

TOKEN_RE = re.compile(r"\b\w+\b")
SCORE_RE = re.compile(r"Score:\s*(\d+)\s*/\s*(\d+)")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ASR-only EduVidQA real_world_test answers for the leaf and "
            "summary-tree retrievers."
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
        help="Run only the leaf retriever. If no retriever flag is set, both run.",
    )
    parser.add_argument(
        "--summary_tree",
        action="store_true",
        help="Run only the summary-tree retriever. If no retriever flag is set, both run.",
    )
    return parser


def selected_systems(args: argparse.Namespace) -> list[str]:
    systems: list[str] = []
    if args.leaf:
        systems.append("leaf")
    if args.summary_tree:
        systems.append("summary_tree")
    if not systems:
        return ["leaf", "summary_tree"]
    return systems


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


def build_store(lectures: list[LectureDescriptor], index_kind: str):
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
    )


def search_leaf(store, query: str, lecture_key: str) -> list[dict[str, object]]:
    return store.search(query, top_k=MAX_CONTEXT_HITS, lecture_key=lecture_key)


def search_summary_tree(store, query: str, lecture_key: str) -> list[dict[str, object]]:
    query_embedding = store.encode_query(query)
    results = store.search_with_embedding(
        query_embedding, top_k=TOP_K, lecture_key=lecture_key
    )
    results = store.expand_summary_tree_results(
        query=query,
        results=results,
        lecture_key=lecture_key,
        top_descendant_leaves=SUMMARY_TREE_TOP_DESCENDANT_LEAVES,
        query_embedding=query_embedding,
    )
    results = store.deduplicate_summary_tree_results(results)
    return results[:MAX_CONTEXT_HITS]


def format_time_range(hit: dict[str, object]) -> str:
    start = hit.get("start")
    end = hit.get("end")
    if start is None or end is None:
        return ""
    return f"time={float(start):.2f}-{float(end):.2f}s"


def compact_text(text: str) -> str:
    return " ".join(text.split()).strip()


def build_leaf_block(hit: dict[str, object], rank: int) -> str:
    parts = [f"[{rank}]"]
    segment_id = hit.get("segment_id")
    if segment_id is not None:
        parts.append(f"seg={segment_id}")
    time_text = format_time_range(hit)
    if time_text:
        parts.append(time_text)
    body = compact_text(str(hit.get("text") or ""))
    if not body:
        body = "<blank>"
    return "\n".join([" ".join(parts), f"Transcript:\n{body}"])


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


def build_context(results: list[dict[str, object]], index_kind: str) -> str:
    if not results:
        return ""

    blocks: list[str] = []
    total_chars = 0
    for rank, hit in enumerate(results, start=1):
        if index_kind == "summary_tree":
            block = build_summary_tree_block(hit, rank)
        else:
            block = build_leaf_block(hit, rank)
        if not block:
            continue
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        total_chars += len(block) + 2
    return "\n\n".join(blocks).strip()


def ask_ollama(client: ollama.Client, prompt: str, temperature: float = 0.0) -> str:
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content is not None:
            return str(content).strip()
    if isinstance(response, dict):
        message = response.get("message") or {}
        content = message.get("content")
        if content is not None:
            return str(content).strip()
        fallback = response.get("response")
        if fallback is not None:
            return str(fallback).strip()
    return str(response).strip()


def generate_answer(client: ollama.Client, question: str, context: str) -> str:
    if not context:
        return INSUFFICIENT_CONTEXT_RESPONSE
    return OllamaResponder.query_response(
        question,
        context,
        model=OLLAMA_MODEL,
        system_prompt=ASR_QUERY_SYSTEM_PROMPT,
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


def factqa_prompt(question: str, answer_1: str, answer_2: str) -> str:
    question = question.replace("\n", " ").strip()
    answer_1 = answer_1.replace("\n", " ").strip()
    answer_2 = answer_2.replace("\n", " ").strip()
    return (
        "Your job is to evaluate the similarity of different answers to a single "
        "question. You will be given a question from a specific computer science "
        "college course. You will also be given two possible answers to that "
        "question, and will have to evaluate the claims in one answer against the "
        "other.\n\n"
        "Steps:\n"
        "1. List all of the atomic claims made by Answer 1. Note that an answer "
        "saying that there is no information counts as a single claim.\n"
        "2. Tell me which of those claims are supported by Answer 2.\n"
        "3. Summarize the results using the template "
        "\"Score: <num supported claims>/<num total claims>\". Ensure that both "
        "numbers are integers.\n\n"
        f"Question: {question}\n"
        f"Answer 1: {answer_1}\n"
        f"Answer 2: {answer_2}"
    )


def extract_score(text: str) -> float | None:
    match = SCORE_RE.search(text)
    if not match:
        return None
    numerator = int(match.group(1))
    denominator = int(match.group(2))
    if denominator == 0:
        return None
    return numerator / denominator


def factqa(client: ollama.Client, question: str, answer: str, generated: str) -> tuple[float, float]:
    if not generated.strip():
        return 0.0, 0.0

    precision_prompt = factqa_prompt(question, answer, generated)
    recall_prompt = factqa_prompt(question, generated, answer)
    precision_response = ask_ollama(client, precision_prompt, temperature=0.0)
    recall_response = ask_ollama(client, recall_prompt, temperature=0.0)
    precision_score = extract_score(precision_response)
    recall_score = extract_score(recall_response)
    if precision_score is None or recall_score is None:
        return 0.0, 0.0
    return precision_score, recall_score


def compute_metrics(
    client: ollama.Client, question: str, answer: str, generated: str
) -> dict[str, float]:
    factqa_precision, factqa_recall = factqa(client, question, answer, generated)
    return {
        "bleu1": bleu1(answer, generated),
        "rouge_l": rouge_l(answer, generated),
        "meteor": meteor(answer, generated),
        "factqa_precision": factqa_precision,
        "factqa_recall": factqa_recall,
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
        "bleu1",
        "rouge_l",
        "meteor",
        "factqa_precision",
        "factqa_recall",
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


def build_summary(
    examples: list[DatasetExample],
    skipped_examples: list[DatasetExample],
    metrics_rows: list[dict[str, object]],
    systems: list[str],
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

    by_system: dict[str, list[dict[str, object]]] = {system: [] for system in systems}
    for row in metrics_rows:
        system_name = str(row["system"])
        by_system.setdefault(system_name, []).append(row)

    metric_names = [
        "bleu1",
        "rouge_l",
        "meteor",
        "factqa_precision",
        "factqa_recall",
    ]

    for system_name, rows in by_system.items():
        summary["systems"][system_name] = {
            "question_count": len(rows),
            **{metric: mean_metric(rows, metric) for metric in metric_names},
        }

    if "leaf" in summary["systems"] and "summary_tree" in summary["systems"]:
        leaf_summary = summary["systems"]["leaf"]
        summary_tree_summary = summary["systems"]["summary_tree"]
        summary["comparison"] = {
            "metric_deltas_summary_tree_minus_leaf": {
                metric: float(summary_tree_summary.get(metric, 0.0))
                - float(leaf_summary.get(metric, 0.0))
                for metric in metric_names
            }
        }

    return summary


def evaluate_system(
    system_name: str,
    store,
    examples: list[DatasetExample],
    lecture_by_video: dict[str, LectureDescriptor],
    client: ollama.Client,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    predictions: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []

    for index, example in enumerate(examples, start=1):
        lecture = lecture_by_video[example.video_id]
        if system_name == "summary_tree":
            hits = search_summary_tree(store, example.question, lecture.key)
        else:
            hits = search_leaf(store, example.question, lecture.key)

        context = build_context(hits, index_kind=system_name)
        generated = generate_answer(client, example.question, context)
        metrics = compute_metrics(client, example.question, example.answer, generated)

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
                "system": system_name,
                "retrieved_hits": hits,
            }
        )

        metrics_rows.append(
            {
                "example_id": example.example_id,
                "row_index": example.row_index,
                "video_id": example.video_id,
                "timestamp": example.timestamp,
                "timestamp_seconds": example.timestamp_seconds,
                "system": system_name,
                **metrics,
            }
        )

        print(
            f"[{system_name}] {index}/{len(examples)} "
            f"{example.example_id} complete"
        )

    return predictions, metrics_rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    systems = selected_systems(args)
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
    print(f"Systems selected: {', '.join(systems)}")

    client = ollama.Client()

    all_metrics: list[dict[str, object]] = []
    all_predictions: dict[str, list[dict[str, object]]] = {}

    if "leaf" in systems:
        print("Building leaf index...")
        leaf_store = build_store(lectures, index_kind="leaf")
        print("Evaluating leaf retriever...")
        leaf_predictions, leaf_metrics = evaluate_system(
            "leaf", leaf_store, evaluated_examples, lecture_by_video, client
        )
        all_predictions["leaf"] = leaf_predictions
        all_metrics.extend(leaf_metrics)

    if "summary_tree" in systems:
        print("Building summary-tree index...")
        summary_tree_store = build_store(lectures, index_kind="summary_tree")
        print("Evaluating summary-tree retriever...")
        summary_tree_predictions, summary_tree_metrics = evaluate_system(
            "summary_tree", summary_tree_store, evaluated_examples, lecture_by_video, client
        )
        all_predictions["summary_tree"] = summary_tree_predictions
        all_metrics.extend(summary_tree_metrics)

    summary = build_summary(examples, skipped_examples, all_metrics, systems)

    if "leaf" in all_predictions:
        write_jsonl(OUTPUT_DIR / "leaf_predictions.jsonl", all_predictions["leaf"])
    if "summary_tree" in all_predictions:
        write_jsonl(
            OUTPUT_DIR / "summary_tree_predictions.jsonl",
            all_predictions["summary_tree"],
        )
    write_metrics_csv(OUTPUT_DIR / "metrics_per_question.csv", all_metrics)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nEvaluation complete.")
    print(json.dumps(summary["systems"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
