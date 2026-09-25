from __future__ import annotations

import csv
import json
import os
import tempfile
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MASTERS_THESIS_DIR = SCRIPT_DIR.parent

if str(MASTERS_THESIS_DIR) not in sys.path:
    sys.path.insert(0, str(MASTERS_THESIS_DIR))

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
from treeseg_vector_index_modular.lecture_segment_builder import SummaryTreeBuildOptions  # noqa: E402
from treeseg_vector_index_modular.ollama_responder import OllamaResponder  # noqa: E402
from treeseg_vector_index_modular.rerank_input_builder import RerankInputBuilder  # noqa: E402
from treeseg_vector_index_modular.vector_store_factory import VectorStoreFactory  # noqa: E402

GENERATOR_BACKEND = "openai"        # "ollama" | "openai"
GENERATOR_OLLAMA_MODEL = "llama3.2"
GENERATOR_OPENAI_MODEL = "gpt-4o-mini"
SUMMARY_OPENAI_MODEL = "gpt-4o-mini"   # model used to build summary tree nodes

RAGAS_JUDGE_MODEL = "gpt-4.1-mini"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

LIMIT = 150
TOP_K = 50
TOP_N = 5
MAX_CONTEXT_CHARS = 8000
OCR_TOP_K = 10
OCR_TOP_N = 5
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
SUMMARY_TREE_TOP_DESCENDANT_LEAVES = 3

DATASET_PATH = MASTERS_THESIS_DIR / "LPM_QA_DATASET" / "lpm_qa_labeled.csv"
LPM_DATA_DIR = MASTERS_THESIS_DIR / "lpm_data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
SUMMARY_TREE_CACHE_DIR = OUTPUT_DIR / "summary_tree_cache"
RETRIEVAL_CHECKPOINT_DIR = OUTPUT_DIR / "retrieval_checkpoints"

INSUFFICIENT_CONTEXT_RESPONSE = (
    "The retrieved lecture segments do not contain enough information to answer "
    "this question."
)

QUERY_SYSTEM_PROMPT = """You are an intelligent teaching assistant helping a student understand
material from a college-level lecture.

You will be given retrieved evidence from the lecture. The evidence may include:
- High-level summary nodes that describe a larger section of the lecture
- Transcript excerpts grounded in the lecture audio
- Slide OCR text extracted from lecture slides shown at that moment

Your job is to answer the student's question using ONLY the provided context.

Rules:
1. Base your answer strictly on the provided context.
2. If the answer is not directly stated but can be reasonably inferred, say that it is inferred.
3. If the context is insufficient, clearly say so instead of guessing.
4. Give a helpful college-level explanation, but stay concise.
5. You may draw on both transcript and slide content when both are present."""

# ─── Systems to evaluate ─────────────────────────────────────────────────────
def _baseline(name: str, strategy: str, size: int, overlap: int) -> dict:
    return {
        "name": name,
        "kind": "baseline",
        "config": BaselineRagConfig(
            chunk_strategy=strategy,
            chunk_size_tokens=size,
            overlap_percent=overlap,
            ocr_mode="transcript_only",
        ),
    }


SYSTEMS: list[dict] = [
    {"name": "treeseg_leaf", "kind": "leaf"},
    {"name": "treeseg_summary_tree", "kind": "summary_tree"},
    # raw token-window baselines
    _baseline("baseline_raw_128_0ov",  "raw_token_window", 128, 0),
    _baseline("baseline_raw_128_10ov", "raw_token_window", 128, 10),
    _baseline("baseline_raw_256_0ov",  "raw_token_window", 256, 0),
    _baseline("baseline_raw_256_10ov", "raw_token_window", 256, 10),
    _baseline("baseline_raw_512_0ov",  "raw_token_window", 512, 0),
    _baseline("baseline_raw_512_10ov", "raw_token_window", 512, 10),
    # utterance-packed baselines
    _baseline("baseline_utt_128_0ov",  "utterance_packed", 128, 0),
    _baseline("baseline_utt_128_10ov", "utterance_packed", 128, 10),
    _baseline("baseline_utt_256_0ov",  "utterance_packed", 256, 0),
    _baseline("baseline_utt_256_10ov", "utterance_packed", 256, 10),
    _baseline("baseline_utt_512_0ov",  "utterance_packed", 512, 0),
    _baseline("baseline_utt_512_10ov", "utterance_packed", 512, 10),
]

# ─── Judge prompt templates ───────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """Context:
{context}

Generated answer:
{answer}

Your task:
1. Break the generated answer into individual atomic statements (one distinct fact or claim per statement, no pronouns).
2. For each statement, determine whether it can be directly inferred from the context: verdict 1 if supported, 0 if not.
3. Compute score = number of supported statements / total number of statements.
   If the answer has no statements, score = 0.0.

Respond only with JSON: {{"score": <float 0-1>}}"""

_CONTEXT_PRECISION_CHUNK_PROMPT = """Question: {question}

Answer: {answer}

Retrieved context chunk:
{chunk}

Given the question and the answer above, was this context chunk useful in arriving at the given answer?
Give verdict as 1 if useful and 0 if not.
Respond only with JSON: {{"relevant": <0 or 1>}}"""

_CONTEXT_RECALL_PROMPT = """Reference answer:
{reference}

Retrieved contexts:
{contexts}

Your task:
1. Identify each individual key claim or sentence in the reference answer.
2. For each claim, determine whether it is directly supported by the retrieved contexts: verdict 1 if supported, 0 if not.
3. Compute score = number of supported claims / total number of claims in the reference answer.
   If the reference has no claims, score = 0.0.

Respond only with JSON: {{"score": <float 0-1>}}"""

_ANSWER_CORRECTNESS_PROMPT = """Question: {question}

Reference answer:
{reference}

Generated answer:
{answer}

Your task:
1. Extract the key statements from both the reference answer and the generated answer.
2. Classify each statement into exactly one category:
   - TP (true positive): present in the generated answer AND directly supported by the reference answer
   - FP (false positive): present in the generated answer but NOT supported by the reference answer
   - FN (false negative): present in the reference answer but MISSING from the generated answer
3. Compute F1 score = 2*TP / (2*TP + FP + FN). If TP + FP + FN = 0, score = 1.0.

Respond only with JSON: {{"score": <float 0-1>}}"""

_ANSWER_RELEVANCY_PROMPT = """Answer:
{answer}

Generate 3 distinct questions that this answer could be responding to.
Also determine if the answer is noncommittal (evasive, vague, or ambiguous — e.g. "I don't know", "I'm not sure", "It depends"): noncommittal = 1 if so, 0 if the answer is substantive.

Respond only with JSON: {{"questions": ["...", "...", "..."], "noncommittal": <0 or 1>}}"""


# ─── Dataset loading ──────────────────────────────────────────────────────────

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


# ─── Store + retrieval helpers ────────────────────────────────────────────────

def _patch_summary_tree_to_openai(openai_client, openai_model: str) -> None:
    """Replace OllamaResponder.generate_summary with an OpenAI-backed version.

    This is done in-place on the class so that lecture_segment_builder.dfs(),
    which holds a reference to the same class, picks up the change.
    The patch persists for the lifetime of the process, which is fine since
    all summary_tree builds in this script want the same backend.
    """
    def _openai_generate_summary(
        text, is_leaf=True, model=None, temperature=0.2,
        keep_alive=None, client=None, host=None,
    ):
        # Ignore the `model` argument — dfs() passes "llama3.2" which must not
        # reach the OpenAI API. Always use the captured openai_model instead.
        if not text or not text.strip():
            return "Empty"
        system_prompt = (
            OllamaResponder.LEAF_SYSTEM_PROMPT if is_leaf
            else OllamaResponder.INTERNAL_SYSTEM_PROMPT
        )
        user_content = f"Transcript excerpt:\n{text}" if is_leaf else text
        resp = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=300,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    OllamaResponder.generate_summary = staticmethod(_openai_generate_summary)


def build_store(system: dict, lectures: list[LectureDescriptor], openai_client=None):
    kind = system["kind"]
    if kind == "baseline":
        result = BaselineStoreBuilder().build_store(
            lectures, system["config"], build_global=False
        )
        if result.skipped_lectures:
            print(f"    Skipped lectures: {list(result.skipped_lectures.keys())}")
        return result.store
    config = LpmConfigBuilder.build_lpm_config(embedding_model=EMBEDDING_MODEL)
    build_options = None
    if kind == "summary_tree":
        if GENERATOR_BACKEND == "openai" and openai_client is not None:
            print(f"  [summary_tree] Using OpenAI ({SUMMARY_OPENAI_MODEL}) for node summaries")
            _patch_summary_tree_to_openai(openai_client, SUMMARY_OPENAI_MODEL)
            cache_version = f"openai-{SUMMARY_OPENAI_MODEL}"
        else:
            cache_version = "v1"
        build_options = SummaryTreeBuildOptions(
            cache_dir=str(SUMMARY_TREE_CACHE_DIR),
            cache_version=cache_version,
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
        index_kind=kind,
        summary_tree_build_options=build_options,
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


def build_ocr_block(hit: dict, rank: int) -> str:
    parts = [f"[Slide {rank}]"]
    time_text = format_time_range(hit)
    if time_text:
        parts.append(time_text)
    body = compact_text(str(hit.get("text") or ""))
    if not body:
        return ""
    return "\n".join([" ".join(parts), body])


def build_ocr_context(ocr_hits: list[dict]) -> str:
    if not ocr_hits:
        return ""
    blocks = []
    for rank, hit in enumerate(ocr_hits, start=1):
        block = build_ocr_block(hit, rank)
        if block:
            blocks.append(block)
    return "\n\n".join(blocks).strip()


def extract_context_texts(hits: list[dict], kind: str) -> list[str]:
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


def build_reranker(kind: str) -> CrossEncoderReranker:
    input_builder = (
        RerankInputBuilder.build_summary_tree_rerank_input
        if kind == "summary_tree"
        else RerankInputBuilder.build_rerank_input
    )
    # Force CPU: MPS runs out of memory scoring 50 pairs while the embedding
    # model already occupies ~12 GB of MPS memory.
    return CrossEncoderReranker(RERANKER_MODEL, device="cpu", input_builder=input_builder)


def build_ocr_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(
        RERANKER_MODEL, device="cpu",
        input_builder=RerankInputBuilder.build_rerank_input_ocr,
    )


# ─── Phase A: retrieval ───────────────────────────────────────────────────────

# Returns list of (ex, context_string, context_texts_list) for each usable example.
RetrievalResult = tuple[dict, str, list[str]]


def _checkpoint_path(system_name: str) -> Path:
    return RETRIEVAL_CHECKPOINT_DIR / f"{system_name}.jsonl"


def _load_checkpoint(system_name: str) -> tuple[list[RetrievalResult], set[str]]:
    """Returns (results_so_far, set_of_completed_example_ids)."""
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
            ex = {k: record[k] for k in ("example_id", "question", "answer", "lecture_key")}
            results.append((ex, record["context"], record["context_texts"]))
            done_ids.add(record["example_id"])
    if done_ids:
        print(f"  [checkpoint] Resuming — {len(done_ids)} questions already done, loading from disk.")
    return results, done_ids


def _append_checkpoint(system_name: str, ex: dict, context: str, context_texts: list[str]) -> None:
    RETRIEVAL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    record = {**ex, "context": context, "context_texts": context_texts}
    with _checkpoint_path(system_name).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_retrieval(
    system: dict,
    lectures_by_key: dict[str, LectureDescriptor],
    examples: list[dict],
    openai_client=None,
) -> list[RetrievalResult]:
    name = system["name"]
    kind = system["kind"]

    usable = [ex for ex in examples if ex["lecture_key"] in lectures_by_key]
    if not usable:
        print(f"  [skip] No usable examples for {name} — no lecture keys found in lpm_data/")
        return []

    results, done_ids = _load_checkpoint(name)
    remaining = [ex for ex in usable if ex["example_id"] not in done_ids]

    if not remaining:
        print(f"  [checkpoint] All {len(results)} questions already done — skipping store build.")
        return results

    unique_keys = {ex["lecture_key"] for ex in remaining}
    lectures = [lectures_by_key[k] for k in unique_keys]
    print(f"  Building ASR store ({len(lectures)} lecture(s))...")
    store = build_store(system, lectures, openai_client=openai_client)

    print(f"  Building OCR store ({len(lectures)} lecture(s))...")
    ocr_store = VectorStoreFactory().build_ocr_vector_store(
        lectures, embed_model=EMBEDDING_MODEL, normalize=True, build_global=False,
    )

    print(f"  Loading rerankers ({RERANKER_MODEL})...")
    reranker = build_reranker(kind)
    ocr_reranker = build_ocr_reranker()

    total = len(usable)
    completed_so_far = len(done_ids)
    for ex in remaining:
        completed_so_far += 1
        print(f"  [retrieval {completed_so_far}/{total}] {ex['example_id']}: {ex['question'][:70]}...")
        if ex["lecture_key"] not in store.lecture_indices:
            print(f"    [skip] lecture not indexed: {ex['lecture_key']}")
            continue

        # Independent ASR retrieval
        asr_hits = retrieve_hits(system, store, ex["question"], ex["lecture_key"], reranker)

        # Independent OCR retrieval (only if the lecture has OCR slides indexed)
        ocr_hits: list[dict] = []
        if ex["lecture_key"] in ocr_store.lecture_indices:
            raw_ocr = ocr_store.search(ex["question"], top_k=OCR_TOP_K, lecture_key=ex["lecture_key"])
            ocr_hits = ocr_reranker.rerank(ex["question"], raw_ocr, top_n=OCR_TOP_N)

        # Build combined context: ASR section + OCR section
        asr_context = build_context(asr_hits, kind)
        ocr_context = build_ocr_context(ocr_hits)
        if asr_context and ocr_context:
            context = asr_context + "\n\n--- Slide Evidence ---\n\n" + ocr_context
        else:
            context = asr_context or ocr_context

        # context_texts for RAGAS judge: ASR texts + OCR texts
        context_texts = extract_context_texts(asr_hits, kind)
        if ocr_hits:
            context_texts += [compact_text(str(h.get("text") or "")) for h in ocr_hits if h.get("text")]

        results.append((ex, context, context_texts))
        _append_checkpoint(name, ex, context, context_texts)

    return results


# ─── Phase B: generation (Ollama fallback) ───────────────────────────────────

def _generate_ollama(ex: dict, context: str) -> str:
    if not context:
        return INSUFFICIENT_CONTEXT_RESPONSE
    return OllamaResponder.query_response(
        ex["question"],
        context,
        model=GENERATOR_OLLAMA_MODEL,
        system_prompt=QUERY_SYSTEM_PROMPT,
        temperature=0.0,
    )


def run_generation_ollama(
    retrieval_results: list[RetrievalResult],
) -> list[tuple[dict, str, list[str]]]:
    gen = []
    for ex, context, context_texts in retrieval_results:
        generated = _generate_ollama(ex, context)
        gen.append((ex, generated, context_texts))
    return gen


# ─── OpenAI Batch API helpers ─────────────────────────────────────────────────

def _make_chat_request(custom_id: str, model: str, system_prompt: str, user_content: str) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
        },
    }


def _submit_batch(requests: list[dict], client, label: str) -> str:
    """Write requests to a temp JSONL file, upload, submit batch. Returns batch_id."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".jsonl", delete=False) as f:
        for req in requests:
            f.write(json.dumps(req).encode() + b"\n")
        tmp_path = f.name

    try:
        print(f"  Uploading {label} batch file ({len(requests)} requests)...")
        with open(tmp_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Batch submitted: {batch.id}")
    return batch.id, file_obj.id


def _poll_batch(batch_id: str, client) -> str:
    """Poll until completed. Returns output_file_id."""
    delay = 10.0
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        if status == "completed":
            print(f"  Batch {batch_id} completed.")
            return batch.output_file_id
        if status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch {batch_id} ended with status: {status}")
        completed = batch.request_counts.completed if batch.request_counts else "?"
        total = batch.request_counts.total if batch.request_counts else "?"
        print(f"  Batch {batch_id} status: {status} ({completed}/{total}) — waiting {delay:.0f}s...")
        time.sleep(delay)
        delay = min(delay * 1.5, 60.0)


def _download_batch_results(output_file_id: str, input_file_id: str, client) -> dict[str, str]:
    """Download result JSONL, parse custom_id → response content. Deletes input file."""
    content = client.files.content(output_file_id).text
    results: dict[str, str] = {}
    for line in content.strip().split("\n"):
        if not line:
            continue
        obj = json.loads(line)
        custom_id = obj["custom_id"]
        try:
            results[custom_id] = (
                obj["response"]["body"]["choices"][0]["message"]["content"].strip()
            )
        except (KeyError, IndexError, TypeError):
            results[custom_id] = ""
    client.files.delete(input_file_id)
    return results


def _run_batch(requests: list[dict], client, label: str) -> dict[str, str]:
    """Full submit → poll → download cycle. Returns {custom_id: response_text}."""
    batch_id, input_file_id = _submit_batch(requests, client, label)
    output_file_id = _poll_batch(batch_id, client)
    return _download_batch_results(output_file_id, input_file_id, client)


# ─── Phase B: generation (OpenAI Batch) ──────────────────────────────────────

def generate_all_batch(
    all_retrieval: dict[str, list[RetrievalResult]],
    client,
) -> dict[str, list[tuple[dict, str, list[str]]]]:
    """Submit one batch for all systems' generation. Returns {system_name: [(ex, answer, ctx_texts)]}."""
    requests: list[dict] = []
    # Track (system_name, index, ex, context_texts) for result assembly
    order: list[tuple[str, int, dict, list[str]]] = []

    for sys_name, retrieval_results in all_retrieval.items():
        for i, (ex, context, context_texts) in enumerate(retrieval_results):
            if not context:
                continue  # no-context cases are handled via INSUFFICIENT_CONTEXT_RESPONSE
            user_content = (
                f"Retrieved lecture segments:\n"
                f"{'─' * 60}\n"
                f"{context}\n"
                f"{'─' * 60}\n\n"
                f"Student question: {ex['question']}"
            )
            requests.append(
                _make_chat_request(
                    custom_id=f"gen-{sys_name}-{i}",
                    model=GENERATOR_OPENAI_MODEL,
                    system_prompt=QUERY_SYSTEM_PROMPT,
                    user_content=user_content,
                )
            )
            order.append((sys_name, i, ex, context_texts))

    raw = _run_batch(requests, client, "generation") if requests else {}

    gen_results: dict[str, list[tuple[dict, str, list[str]]]] = {
        name: [] for name in all_retrieval
    }

    # Fill in batch answers; fall back to INSUFFICIENT_CONTEXT_RESPONSE for empty-context items
    batch_counters: dict[str, int] = {name: 0 for name in all_retrieval}
    for sys_name, retrieval_results in all_retrieval.items():
        for i, (ex, context, context_texts) in enumerate(retrieval_results):
            if not context:
                gen_results[sys_name].append((ex, INSUFFICIENT_CONTEXT_RESPONSE, context_texts))
            else:
                answer = raw.get(f"gen-{sys_name}-{i}", INSUFFICIENT_CONTEXT_RESPONSE)
                gen_results[sys_name].append((ex, answer, context_texts))

    return gen_results


# ─── Phase C: judging (OpenAI Batch) ─────────────────────────────────────────

def _parse_score(text: str, key: str, fallback: float = 0.0) -> float:
    try:
        obj = json.loads(text)
        return float(obj[key])
    except Exception:
        # Try extracting first float from text as fallback
        import re
        m = re.search(r"[-+]?\d*\.?\d+", text)
        return float(m.group()) if m else fallback


def judge_all_batch(
    all_gen_results: dict[str, list[tuple[dict, str, list[str]]]],
    client,
) -> dict[str, dict[str, list[float]]]:
    """
    Submit one batch for all 5 metrics across all systems.
    Returns {system_name: {metric_name: [per-question float scores]}}.
    """
    requests: list[dict] = []

    for sys_name, gen_results in all_gen_results.items():
        for i, (ex, generated, context_texts) in enumerate(gen_results):
            question = ex["question"]
            reference = ex["answer"]
            contexts_joined = "\n\n".join(context_texts)

            # Faithfulness: does the answer stay within the context?
            requests.append(
                _make_chat_request(
                    custom_id=f"judge-{sys_name}-{i}-faithfulness",
                    model=RAGAS_JUDGE_MODEL,
                    system_prompt="You are an expert evaluator. Follow the instructions exactly and respond only with JSON.",
                    user_content=_FAITHFULNESS_PROMPT.format(
                        context=contexts_joined, answer=generated
                    ),
                )
            )

            # Context Precision: one request per retrieved chunk
            for j, chunk in enumerate(context_texts):
                requests.append(
                    _make_chat_request(
                        custom_id=f"judge-{sys_name}-{i}-cp-{j}",
                        model=RAGAS_JUDGE_MODEL,
                        system_prompt="You are an expert evaluator. Follow the instructions exactly and respond only with JSON.",
                        user_content=_CONTEXT_PRECISION_CHUNK_PROMPT.format(
                            question=question, answer=generated, chunk=chunk
                        ),
                    )
                )

            # Context Recall
            requests.append(
                _make_chat_request(
                    custom_id=f"judge-{sys_name}-{i}-context_recall",
                    model=RAGAS_JUDGE_MODEL,
                    system_prompt="You are an expert evaluator. Follow the instructions exactly and respond only with JSON.",
                    user_content=_CONTEXT_RECALL_PROMPT.format(
                        reference=reference, contexts=contexts_joined
                    ),
                )
            )

            # Answer Correctness
            requests.append(
                _make_chat_request(
                    custom_id=f"judge-{sys_name}-{i}-answer_correctness",
                    model=RAGAS_JUDGE_MODEL,
                    system_prompt="You are an expert evaluator. Follow the instructions exactly and respond only with JSON.",
                    user_content=_ANSWER_CORRECTNESS_PROMPT.format(
                        question=question, reference=reference, answer=generated
                    ),
                )
            )

            # Answer Relevancy
            requests.append(
                _make_chat_request(
                    custom_id=f"judge-{sys_name}-{i}-answer_relevancy",
                    model=RAGAS_JUDGE_MODEL,
                    system_prompt="You are an expert evaluator. Follow the instructions exactly and respond only with JSON.",
                    user_content=_ANSWER_RELEVANCY_PROMPT.format(
                        answer=generated
                    ),
                )
            )

    raw = _run_batch(requests, client, "judging")

    per_system: dict[str, dict[str, list[float]]] = {}

    for sys_name, gen_results in all_gen_results.items():
        metrics: dict[str, list[float]] = {
            "faithfulness": [],
            "context_precision": [],
            "context_recall": [],
            "answer_correctness": [],
            "answer_relevancy": [],
        }

        for i, (ex, generated, context_texts) in enumerate(gen_results):
            # Faithfulness
            metrics["faithfulness"].append(
                _parse_score(raw.get(f"judge-{sys_name}-{i}-faithfulness", ""), "score")
            )

            # Context Precision: weighted precision@k over retrieved chunks
            relevance = []
            for j in range(len(context_texts)):
                text = raw.get(f"judge-{sys_name}-{i}-cp-{j}", "")
                relevance.append(_parse_score(text, "relevant", fallback=0.0))
            if any(r > 0 for r in relevance):
                n_rel = 0
                precision_sum = 0.0
                for k, r in enumerate(relevance):
                    if r > 0:
                        n_rel += 1
                        precision_sum += n_rel / (k + 1)
                cp = precision_sum / n_rel
            else:
                cp = 0.0
            metrics["context_precision"].append(cp)

            # Context Recall
            metrics["context_recall"].append(
                _parse_score(raw.get(f"judge-{sys_name}-{i}-context_recall", ""), "score")
            )

            # Answer Correctness
            metrics["answer_correctness"].append(
                _parse_score(raw.get(f"judge-{sys_name}-{i}-answer_correctness", ""), "score")
            )

            # Answer Relevancy — placeholder; filled in after embedding computation below
            metrics["answer_relevancy"].append(0.0)

        per_system[sys_name] = metrics

    # ── Answer Relevancy: embedding-based cosine similarity (mirrors RAGAS) ──
    # Parse synthetic questions + noncommittal flag from batch results
    ar_data: dict[tuple[str, int], tuple[str, list[str], int]] = {}
    for sys_name, gen_results in all_gen_results.items():
        for i, (ex, generated, _) in enumerate(gen_results):
            text = raw.get(f"judge-{sys_name}-{i}-answer_relevancy", "")
            try:
                obj = json.loads(text)
                synth_qs = [q for q in obj.get("questions", []) if isinstance(q, str) and q.strip()]
                noncommittal = int(obj.get("noncommittal", 0))
            except Exception:
                synth_qs = []
                noncommittal = 0
            ar_data[(sys_name, i)] = (ex["question"], synth_qs, noncommittal)

    # Batch-encode all original + synthetic questions with the retrieval embedding model
    from sentence_transformers import SentenceTransformer
    import numpy as _np
    print(f"  Computing answer relevancy embeddings ({EMBEDDING_MODEL})...")
    _embed = SentenceTransformer(EMBEDDING_MODEL)

    all_texts: list[str] = []
    text_keys: list[tuple] = []
    for (sys_name, i), (orig_q, synth_qs, _) in ar_data.items():
        all_texts.append(orig_q)
        text_keys.append((sys_name, i, "orig"))
        for j, sq in enumerate(synth_qs):
            all_texts.append(sq)
            text_keys.append((sys_name, i, f"synth_{j}"))

    if all_texts:
        embeddings = _embed.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)
        emb_map = {k: e for k, e in zip(text_keys, embeddings)}
    else:
        emb_map = {}

    for sys_name, gen_results in all_gen_results.items():
        for i in range(len(gen_results)):
            orig_q, synth_qs, noncommittal = ar_data.get((sys_name, i), ("", [], 0))
            if noncommittal or not synth_qs:
                score = 0.0
            else:
                orig_emb = emb_map.get((sys_name, i, "orig"))
                sims = []
                for j in range(len(synth_qs)):
                    synth_emb = emb_map.get((sys_name, i, f"synth_{j}"))
                    if orig_emb is not None and synth_emb is not None:
                        sims.append(float(_np.dot(orig_emb, synth_emb)))
                score = float(_np.mean(sims)) if sims else 0.0
            per_system[sys_name]["answer_relevancy"][i] = score

    return per_system


# ─── Output helpers ───────────────────────────────────────────────────────────

def _rouge_l(hypothesis: str, reference: str) -> float:
    """Sentence-level ROUGE-L (LCS-based F1)."""
    h = hypothesis.lower().split()
    r = reference.lower().split()
    if not h or not r:
        return 0.0
    # LCS length via DP
    m, n = len(r), len(h)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if r[i - 1] == h[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / n if n else 0.0
    recall = lcs / m if m else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _bleu_1(hypothesis: str, reference: str) -> float:
    """Unigram BLEU with brevity penalty."""
    h = hypothesis.lower().split()
    r = reference.lower().split()
    if not h:
        return 0.0
    ref_counts: dict[str, int] = {}
    for w in r:
        ref_counts[w] = ref_counts.get(w, 0) + 1
    clipped = 0
    for w in h:
        if ref_counts.get(w, 0) > 0:
            clipped += 1
            ref_counts[w] -= 1
    precision = clipped / len(h)
    bp = min(1.0, len(h) / len(r)) if r else 0.0
    return bp * precision


_LLM_METRIC_COLS = [
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_correctness",
    "answer_relevancy",
]


def save_results(
    system_name: str,
    gen_results: list[tuple[dict, str, list[str]]],
    per_q_metrics: dict[str, list[float]],
    output_dir: Path,
) -> dict[str, float]:
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (ex, generated, context_texts) in enumerate(gen_results):
        row = {
            "system": system_name,
            "example_id": ex.get("example_id", f"lpm-{i:05d}"),
            "lecture_key": ex.get("lecture_key", ""),
            "question": ex["question"],
            "reference_answer": ex["answer"],
            "generated_answer": generated,
            "rouge_l": _rouge_l(generated, ex["answer"]),
            "bleu_1": _bleu_1(generated, ex["answer"]),
            "insufficient_context": int(INSUFFICIENT_CONTEXT_RESPONSE in generated),
        }
        for metric in _LLM_METRIC_COLS:
            scores = per_q_metrics.get(metric, [])
            row[metric] = scores[i] if i < len(scores) else float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)

    # Per-question CSV (this system only, kept for per-system inspection)
    per_q_path = output_dir / f"{system_name}_per_question.csv"
    df.to_csv(per_q_path, index=False)
    print(f"  Saved: {per_q_path.name}")

    mean_scores = {
        col: float(df[col].mean())
        for col in _LLM_METRIC_COLS + ["rouge_l", "bleu_1"]
        if col in df.columns
    }
    return mean_scores, df


def _write_combined_csvs(all_per_q_dfs: dict[str, "pd.DataFrame"], all_scores: dict[str, dict], output_dir: Path) -> None:
    import pandas as pd

    # 1. per_question_all_systems.csv — flat, one row per (system × question)
    combined_df = pd.concat(list(all_per_q_dfs.values()), ignore_index=True)
    col_order = [
        "system", "example_id", "lecture_key", "question",
        "reference_answer", "generated_answer",
        "faithfulness", "context_precision", "context_recall",
        "answer_correctness", "answer_relevancy",
        "rouge_l", "bleu_1", "insufficient_context",
    ]
    col_order = [c for c in col_order if c in combined_df.columns]
    combined_df = combined_df[col_order]
    per_q_all_path = output_dir / "per_question_all_systems.csv"
    combined_df.to_csv(per_q_all_path, index=False)
    print(f"Saved: {per_q_all_path.name}")

    # 2. system_summary.csv — one row per system, means + stds
    summary_rows = []
    for system_name, df in all_per_q_dfs.items():
        row: dict = {"system": system_name, "n_questions": len(df)}
        for col in _LLM_METRIC_COLS + ["rouge_l", "bleu_1"]:
            if col in df.columns:
                row[f"{col}_mean"] = round(float(df[col].mean()), 4)
                row[f"{col}_std"] = round(float(df[col].std()), 4)
        if "insufficient_context" in df.columns:
            row["insufficient_context_rate"] = round(float(df["insufficient_context"].mean()), 4)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    system_summary_path = output_dir / "system_summary.csv"
    summary_df.to_csv(system_summary_path, index=False)
    print(f"Saved: {system_summary_path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    generator_model = (
        GENERATOR_OLLAMA_MODEL if GENERATOR_BACKEND == "ollama" else GENERATOR_OPENAI_MODEL
    )
    print(f"Generator:   {GENERATOR_BACKEND} / {generator_model}")
    print(f"Judge model: {RAGAS_JUDGE_MODEL}")
    print(f"Dataset:     {DATASET_PATH}")
    print(f"LPM data:    {LPM_DATA_DIR}")

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        sys.exit(1)
    if not LPM_DATA_DIR.exists():
        print(f"ERROR: lpm_data/ not found at {LPM_DATA_DIR}")
        sys.exit(1)
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY is not set — batch judge calls will fail.")

    examples = load_lpm_examples(DATASET_PATH, LIMIT)
    print(f"\nLoaded {len(examples)} examples")

    lectures_by_key = discover_lectures(LPM_DATA_DIR)
    print(f"Discovered {len(lectures_by_key)} lectures in lpm_data/")

    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # ── Phase A: retrieval (all systems) ────────────────────────────────────
    all_retrieval: dict[str, list[RetrievalResult]] = {}
    for system in SYSTEMS:
        name = system["name"]
        print(f"\n{'=' * 60}")
        print(f"System: {name}  [retrieval]")
        print(f"{'=' * 60}")
        retrieval_results = run_retrieval(system, lectures_by_key, examples, openai_client=openai_client)
        all_retrieval[name] = retrieval_results

    # Filter out systems with no retrieval results
    all_retrieval = {k: v for k, v in all_retrieval.items() if v}
    if not all_retrieval:
        print("No retrieval results for any system — exiting.")
        return

    # ── Phase B: generation (one batch for all systems) ──────────────────────
    print(f"\n{'=' * 60}")
    print("Phase B — Batch generation")
    print(f"{'=' * 60}")

    if GENERATOR_BACKEND == "openai":
        all_gen_results = generate_all_batch(all_retrieval, openai_client)
    else:
        all_gen_results = {
            name: run_generation_ollama(retrieval_results)
            for name, retrieval_results in all_retrieval.items()
        }

    # ── Phase C: judging (one batch for all systems) ─────────────────────────
    print(f"\n{'=' * 60}")
    print("Phase C — Batch judging")
    print(f"{'=' * 60}")

    all_judge_scores = judge_all_batch(all_gen_results, openai_client)

    # ── Save results ─────────────────────────────────────────────────────────
    all_scores: dict[str, dict] = {}
    all_per_q_dfs: dict = {}
    for name, gen_results in all_gen_results.items():
        print(f"\n{'=' * 60}")
        print(f"System: {name}  [saving results]")
        print(f"{'=' * 60}")
        per_q_metrics = all_judge_scores.get(name, {})
        scores, per_q_df = save_results(name, gen_results, per_q_metrics, OUTPUT_DIR)
        print(f"  Scores:")
        for metric, val in scores.items():
            print(f"    {metric}: {val:.4f}")
        all_scores[name] = scores
        all_per_q_dfs[name] = per_q_df

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "all_systems_summary.json"
    summary_path.write_text(json.dumps(all_scores, indent=2))
    print(f"\nSummary saved to {summary_path}")

    print(f"\n{'=' * 60}")
    print("Writing combined CSVs")
    print(f"{'=' * 60}")
    _write_combined_csvs(all_per_q_dfs, all_scores, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systems", nargs="+", metavar="NAME",
        help="Run only these system names (default: all). E.g. --systems treeseg_summary_tree"
    )
    args = parser.parse_args()
    if args.systems:
        known = {s["name"] for s in SYSTEMS}
        unknown = set(args.systems) - known
        if unknown:
            print(f"ERROR: unknown system(s): {unknown}")
            print(f"Known: {sorted(known)}")
            raise SystemExit(1)
        SYSTEMS[:] = [s for s in SYSTEMS if s["name"] in args.systems]
    main()
