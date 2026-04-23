from __future__ import annotations

import csv
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MASTERS_THESIS_DIR = SCRIPT_DIR.parent

if str(MASTERS_THESIS_DIR) not in sys.path:
    sys.path.insert(0, str(MASTERS_THESIS_DIR))

# Load .env from the project root
_dotenv_path = MASTERS_THESIS_DIR / ".env"
if _dotenv_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_dotenv_path)

from baseline_rag_system.store_builder import BaselineStoreBuilder  # noqa: E402
from baseline_rag_system.types import BaselineRagConfig  # noqa: E402
from treeseg_vector_index_modular.lecture_catalog import LectureCatalog  # noqa: E402
from treeseg_vector_index_modular.lecture_descriptor import LectureDescriptor  # noqa: E402
from treeseg_vector_index_modular.lpm_config_builder import LpmConfigBuilder  # noqa: E402
from treeseg_vector_index_modular.cross_encoder_reranker import CrossEncoderReranker  # noqa: E402
from treeseg_vector_index_modular.ollama_responder import OllamaResponder  # noqa: E402
from treeseg_vector_index_modular.rerank_input_builder import RerankInputBuilder  # noqa: E402
from treeseg_vector_index_modular.vector_store_factory import VectorStoreFactory  # noqa: E402

GENERATOR_BACKEND = "ollama"        # "ollama" | "openai"
GENERATOR_OLLAMA_MODEL = "llama3.2"
GENERATOR_OPENAI_MODEL = "gpt-5.4-nano"

RAGAS_JUDGE_MODEL = "gpt-5.4"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

LIMIT = 5 
TOP_K = 10 
TOP_N = 5 
MAX_CONTEXT_CHARS = 8000
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
SUMMARY_TREE_TOP_DESCENDANT_LEAVES = 3

DATASET_PATH = MASTERS_THESIS_DIR / "LPM_QA_DATASET" / "lpm_qa_labeled.csv"
LPM_DATA_DIR = MASTERS_THESIS_DIR / "lpm_data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

INSUFFICIENT_CONTEXT_RESPONSE = (
    "The retrieved lecture segments do not contain enough information to answer "
    "this question."
)

QUERY_SYSTEM_PROMPT = """You are an intelligent teaching assistant helping a student understand
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

# ─── Systems to evaluate ─────────────────────────────────────────────────────
# Add or remove entries here to control what gets benchmarked.
# For "baseline" systems, supply a BaselineRagConfig in "config".
SYSTEMS: list[dict] = [
    {"name": "treeseg_leaf", "kind": "leaf"},
    # {"name": "treeseg_summary_tree", "kind": "summary_tree"},
    {
        "name": "baseline_utterance_128_0ov",
        "kind": "baseline",
        "config": BaselineRagConfig(
            chunk_strategy="utterance_packed",
            chunk_size_tokens=128,
            overlap_percent=0,
            ocr_mode="transcript_only",
        ),
    },
    # {
    #     "name": "baseline_utterance_256_0ov",
    #     "kind": "baseline",
    #     "config": BaselineRagConfig(
    #         chunk_strategy="utterance_packed",
    #         chunk_size_tokens=256,
    #         overlap_percent=0,
    #         ocr_mode="transcript_only",
    #     ),
    # },
    # {
    #     "name": "baseline_token_256_0ov",
    #     "kind": "baseline",
    #     "config": BaselineRagConfig(
    #         chunk_strategy="raw_token_window",
    #         chunk_size_tokens=256,
    #         overlap_percent=0,
    #         ocr_mode="transcript_only",
    #     ),
    # },
    # Uncomment to add more baseline variants:
    # {
    #     "name": "baseline_utterance_512_10ov",
    #     "kind": "baseline",
    #     "config": BaselineRagConfig(
    #         chunk_strategy="utterance_packed",
    #         chunk_size_tokens=512,
    #         overlap_percent=10,
    #         ocr_mode="transcript_only",
    #     ),
    # },
]


def load_lpm_examples(csv_path: Path, limit: int | None = None) -> list[dict]:
    examples = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader, start=1):
            if limit is not None and len(examples) >= limit:
                break
            question = (row.get("question") or "").strip()
            answer = (row.get("answer_text") or "").strip()
            lecture_key = (row.get("lecture_key") or "").strip()
            if not question or not answer or not lecture_key:
                continue
            examples.append(
                {
                    "example_id": f"lpm-{row_index:05d}",
                    "question": question,
                    "answer": answer,
                    "lecture_key": lecture_key,
                }
            )
    return examples


def discover_lectures(lpm_data_dir: Path) -> dict[str, LectureDescriptor]:
    lectures = LectureCatalog.discover_lectures(lpm_data_dir)
    return {lec.key: lec for lec in lectures}


def build_store(system: dict, lectures: list[LectureDescriptor]):
    kind = system["kind"]
    if kind == "baseline":
        result = BaselineStoreBuilder().build_store(
            lectures, system["config"], build_global=False
        )
        if result.skipped_lectures:
            print(f"    Skipped lectures: {list(result.skipped_lectures.keys())}")
        return result.store
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
        index_kind=kind,
        summary_tree_build_options=None,
    )


def search_leaf_like(store, question: str, lecture_key: str, reranker) -> list[dict]:
    results = store.search(question, top_k=TOP_K, lecture_key=lecture_key)
    if reranker:
        return reranker.rerank(question, results, top_n=TOP_N)
    return results[: min(TOP_N, len(results))]


def search_summary_tree(store, question: str, lecture_key: str, reranker) -> list[dict]:
    query_embedding = store.encode_query(question)
    results = store.search_with_embedding(
        query_embedding, top_k=TOP_K, lecture_key=lecture_key
    )
    results = store.expand_summary_tree_results(
        query=question,
        results=results,
        lecture_key=lecture_key,
        top_descendant_leaves=SUMMARY_TREE_TOP_DESCENDANT_LEAVES,
        query_embedding=query_embedding,
    )
    if reranker:
        results = reranker.rerank(question, results, top_n=TOP_N)
    results = store.deduplicate_summary_tree_results(results)
    return results[: min(TOP_N, len(results))]


def retrieve_hits(system: dict, store, question: str, lecture_key: str, reranker) -> list[dict]:
    if system["kind"] == "summary_tree":
        return search_summary_tree(store, question, lecture_key, reranker)
    return search_leaf_like(store, question, lecture_key, reranker)


def compact_text(text: str) -> str:
    return " ".join(text.split()).strip()


def format_time_range(hit: dict) -> str:
    start = hit.get("start")
    end = hit.get("end")
    if start is None or end is None:
        return ""
    return f"time={float(start):.2f}-{float(end):.2f}s"


def build_leaf_block(hit: dict, rank: int) -> str:
    parts = [f"[{rank}]"]
    segment_id = hit.get("segment_id")
    if segment_id is not None:
        parts.append(f"seg={segment_id}")
    time_text = format_time_range(hit)
    if time_text:
        parts.append(time_text)
    spoken, _ = RerankInputBuilder.split_segment_text(str(hit.get("text") or ""))
    body = compact_text(spoken or str(hit.get("text") or ""))
    if not body:
        body = "<blank>"
    return "\n".join([" ".join(parts), f"Transcript:\n{body}"])


def build_summary_tree_block(hit: dict, rank: int) -> str:
    parts = [f"[{rank}]"]
    parts.append("leaf" if hit.get("is_leaf", True) else "summary-node")
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
    supporting_blocks = []
    for leaf_rank, leaf in enumerate(hit.get("supporting_leaves") or [], start=1):
        leaf_parts = [f"Support {leaf_rank}"]
        seg_id = leaf.get("segment_id")
        if seg_id is not None:
            leaf_parts.append(f"seg={seg_id}")
        leaf_time = format_time_range(leaf)
        if leaf_time:
            leaf_parts.append(leaf_time)
        leaf_text = compact_text(str(leaf.get("text") or ""))
        supporting_blocks.append(
            "\n".join([" ".join(leaf_parts), leaf_text or "<blank>"])
        )

    evidence = "\n\n".join(supporting_blocks) if supporting_blocks else "<blank>"
    return "\n".join(
        [
            " ".join(parts),
            f"Summary:\n{summary_text}",
            f"Supporting transcript evidence:\n{evidence}",
        ]
    )


def build_context(hits: list[dict], kind: str) -> str:
    if not hits:
        return ""
    blocks: list[str] = []
    total_chars = 0
    for rank, hit in enumerate(hits, start=1):
        block = (
            build_summary_tree_block(hit, rank)
            if kind == "summary_tree"
            else build_leaf_block(hit, rank)
        )
        if not block:
            continue
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        total_chars += len(block) + 2
    return "\n\n".join(blocks).strip()


def extract_context_texts(hits: list[dict], kind: str) -> list[str]:
    """Return a flat list of text strings for RAGAS retrieved_contexts."""
    texts: list[str] = []
    for hit in hits:
        if kind == "summary_tree" and not hit.get("is_leaf", True):
            summary = compact_text(
                str(hit.get("summary_text") or hit.get("text") or "")
            )
            if summary:
                texts.append(summary)
            for leaf in hit.get("supporting_leaves") or []:
                leaf_text = compact_text(str(leaf.get("text") or ""))
                if leaf_text:
                    texts.append(leaf_text)
        else:
            spoken, _ = RerankInputBuilder.split_segment_text(
                str(hit.get("text") or "")
            )
            body = compact_text(spoken or str(hit.get("text") or ""))
            if body:
                texts.append(body)
    return texts or ["<no context retrieved>"]


def generate_answer_ollama(question: str, context: str) -> str:
    if not context:
        return INSUFFICIENT_CONTEXT_RESPONSE
    return OllamaResponder.query_response(
        question,
        context,
        model=GENERATOR_OLLAMA_MODEL,
        system_prompt=QUERY_SYSTEM_PROMPT,
        temperature=0.0,
    )


def generate_answer_openai(question: str, context: str, client) -> str:
    if not context:
        return INSUFFICIENT_CONTEXT_RESPONSE
    user_content = (
        f"Retrieved lecture segments:\n"
        f"{'─' * 60}\n"
        f"{context}\n"
        f"{'─' * 60}\n\n"
        f"Student question: {question}"
    )
    response = client.chat.completions.create(
        model=GENERATOR_OPENAI_MODEL,
        messages=[
            {"role": "system", "content": QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def build_ragas_judge():
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(
        ChatOpenAI(model=RAGAS_JUDGE_MODEL, api_key=OPENAI_API_KEY)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    )
    return llm, embeddings


def build_reranker(kind: str) -> CrossEncoderReranker:
    input_builder = (
        RerankInputBuilder.build_summary_tree_rerank_input
        if kind == "summary_tree"
        else RerankInputBuilder.build_rerank_input
    )
    return CrossEncoderReranker(RERANKER_MODEL, input_builder=input_builder)


def run_system(
    system: dict,
    lectures_by_key: dict[str, LectureDescriptor],
    examples: list[dict],
    openai_client=None,
) -> list:
    from ragas import SingleTurnSample

    name = system["name"]
    kind = system["kind"]

    usable = [ex for ex in examples if ex["lecture_key"] in lectures_by_key]
    if not usable:
        print(f"  [skip] No usable examples for {name} — none of the lecture keys found in lpm_data/")
        return []

    unique_keys = {ex["lecture_key"] for ex in usable}
    lectures = [lectures_by_key[k] for k in unique_keys]
    print(f"  Building store ({len(lectures)} lecture(s))...")
    store = build_store(system, lectures)

    print(f"  Loading reranker ({RERANKER_MODEL})...")
    reranker = build_reranker(kind)

    samples: list[SingleTurnSample] = []
    for i, ex in enumerate(usable, start=1):
        print(f"  [{i}/{len(usable)}] {ex['example_id']}: {ex['question'][:70]}...")

        if ex["lecture_key"] not in store.lecture_indices:
            print(f"    [skip] lecture not indexed: {ex['lecture_key']}")
            continue

        hits = retrieve_hits(system, store, ex["question"], ex["lecture_key"], reranker)
        context = build_context(hits, kind)
        context_texts = extract_context_texts(hits, kind)

        if GENERATOR_BACKEND == "openai":
            generated = generate_answer_openai(ex["question"], context, openai_client)
        else:
            generated = generate_answer_ollama(ex["question"], context)

        samples.append(
            SingleTurnSample(
                user_input=ex["question"],
                response=generated,
                retrieved_contexts=context_texts,
                reference=ex["answer"],
            )
        )

    return samples


def save_results(df, system_name: str, scores: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    per_q_path = output_dir / f"{system_name}_per_question.csv"
    df.to_csv(per_q_path, index=False)
    print(f"  Saved: {per_q_path.name}")

    scores_path = output_dir / f"{system_name}_scores.json"
    scores_path.write_text(json.dumps(scores, indent=2))
    print(f"  Saved: {scores_path.name}")


def main():
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    generator_model = (
        GENERATOR_OLLAMA_MODEL if GENERATOR_BACKEND == "ollama" else GENERATOR_OPENAI_MODEL
    )
    print(f"Generator:   {GENERATOR_BACKEND} / {generator_model}")
    print(f"RAGAS judge: {RAGAS_JUDGE_MODEL}")
    print(f"Dataset:     {DATASET_PATH}")
    print(f"LPM data:    {LPM_DATA_DIR}")

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        sys.exit(1)
    if not LPM_DATA_DIR.exists():
        print(f"ERROR: lpm_data/ not found at {LPM_DATA_DIR}")
        sys.exit(1)
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY is not set — RAGAS judge calls will fail.")

    examples = load_lpm_examples(DATASET_PATH, LIMIT)
    print(f"\nLoaded {len(examples)} examples")

    lectures_by_key = discover_lectures(LPM_DATA_DIR)
    print(f"Discovered {len(lectures_by_key)} lectures in lpm_data/")

    openai_client = None
    if GENERATOR_BACKEND == "openai":
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)

    ragas_llm, ragas_embeddings = build_ragas_judge()
    metrics = [
        Faithfulness(),
        ContextPrecision(),
        ContextRecall(),
        AnswerCorrectness(),
        AnswerRelevancy(),
    ]

    all_scores: dict[str, dict] = {}

    for system in SYSTEMS:
        name = system["name"]
        print(f"\n{'=' * 60}")
        print(f"System: {name}")
        print(f"{'=' * 60}")

        samples = run_system(system, lectures_by_key, examples, openai_client)
        if not samples:
            print(f"  No samples — skipping RAGAS evaluation for {name}.")
            continue

        print(f"  Running RAGAS on {len(samples)} samples...")
        dataset = EvaluationDataset(samples=samples)
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        df = result.to_pandas()

        metric_cols = [
            "faithfulness",
            "context_precision",
            "context_recall",
            "answer_correctness",
            "answer_relevancy",
        ]
        scores = {
            col: float(df[col].mean())
            for col in metric_cols
            if col in df.columns
        }
        print(f"  Scores:")
        for metric, val in scores.items():
            print(f"    {metric}: {val:.4f}")

        save_results(df, name, scores, OUTPUT_DIR)
        all_scores[name] = scores

    summary_path = OUTPUT_DIR / "all_systems_summary.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_scores, indent=2))
    print(f"\nSummary saved to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
