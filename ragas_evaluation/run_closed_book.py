"""
Closed-book baseline: GPT answers questions without any retrieved context.

Run independently — does not require rerunning the full evaluation.

Usage:
    cd masters-thesis
    source /path/to/lpm_venv/bin/activate
    python ragas_evaluation/run_closed_book.py

Outputs:
    ragas_evaluation/outputs/no_retrieval_per_question.csv
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SCRIPT_DIR = Path(__file__).resolve().parent
MASTERS_DIR = SCRIPT_DIR.parent

_dotenv = MASTERS_DIR / ".env"
if _dotenv.exists():
    from dotenv import load_dotenv
    load_dotenv(_dotenv)

OPENAI_API_KEY     = os.environ.get("OPENAI_API_KEY", "")
GENERATOR_MODEL    = "gpt-4o-mini"
JUDGE_MODEL        = "gpt-4.1-mini"
EMBEDDING_MODEL    = "BAAI/bge-base-en-v1.5"
LIMIT              = 150

DATASET_PATH = MASTERS_DIR / "LPM_QA_DATASET" / "lpm_qa_labeled.csv"
OUTPUT_DIR   = SCRIPT_DIR / "outputs"
OUTPUT_CSV   = OUTPUT_DIR / "no_retrieval_per_question.csv"

SYSTEM_NAME = "no_retrieval"

CLOSED_BOOK_SYSTEM_PROMPT = (
    "You are an intelligent teaching assistant helping a student understand "
    "material from a college-level lecture. Answer the student's question based "
    "on your general knowledge. Be concise and accurate. "
    "If you genuinely do not know the answer, say so clearly."
)

# Judge prompts (answer_correctness + answer_relevancy only — no context needed)
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


# ─── Dataset ──────────────────────────────────────────────────────────────────

def load_examples() -> list[dict]:
    examples = []
    with DATASET_PATH.open("r", newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            if LIMIT and len(examples) >= LIMIT:
                break
            q = (row.get("question") or "").strip()
            a = (row.get("answer_text") or "").strip()
            lk = (row.get("lecture_key") or "").strip()
            if q and a and lk:
                examples.append({
                    "example_id": f"lpm-{i:05d}",
                    "question": q,
                    "answer": a,
                    "lecture_key": lk,
                })
    return examples


# ─── Batch API helpers ────────────────────────────────────────────────────────

def _make_request(custom_id: str, model: str, system: str, user: str) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": 0.0,
        },
    }


def _run_batch(requests: list[dict], client, label: str) -> dict[str, str]:
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".jsonl", delete=False) as f:
        for r in requests:
            f.write(json.dumps(r).encode() + b"\n")
        tmp = f.name

    try:
        print(f"  Uploading {label} ({len(requests)} requests)...")
        with open(tmp, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
    finally:
        Path(tmp).unlink(missing_ok=True)

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Batch submitted: {batch.id}")

    delay = 10.0
    while True:
        b = client.batches.retrieve(batch.id)
        if b.status == "completed":
            print(f"  Batch {batch.id} completed.")
            break
        if b.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch ended with status: {b.status}")
        done = b.request_counts.completed if b.request_counts else "?"
        total = b.request_counts.total if b.request_counts else "?"
        print(f"  {b.status} ({done}/{total}) — waiting {delay:.0f}s...")
        time.sleep(delay)
        delay = min(delay * 1.5, 60.0)

    content = client.files.content(b.output_file_id).text
    results: dict[str, str] = {}
    for line in content.strip().split("\n"):
        if not line:
            continue
        obj = json.loads(line)
        try:
            results[obj["custom_id"]] = (
                obj["response"]["body"]["choices"][0]["message"]["content"].strip()
            )
        except (KeyError, IndexError, TypeError):
            results[obj["custom_id"]] = ""
    client.files.delete(file_obj.id)
    return results


# ─── Scoring helpers ──────────────────────────────────────────────────────────

def _parse_score(text: str, key: str, fallback: float = 0.0) -> float:
    try:
        return float(json.loads(text)[key])
    except Exception:
        m = re.search(r"[-+]?\d*\.?\d+", text)
        return float(m.group()) if m else fallback


def _rouge_l(hyp: str, ref: str) -> float:
    h, r = hyp.lower().split(), ref.lower().split()
    if not h or not r:
        return 0.0
    m, n = len(r), len(h)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if r[i-1] == h[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs / n if n else 0.0
    rc = lcs / m if m else 0.0
    return 2 * p * rc / (p + rc) if p + rc else 0.0


def _bleu_1(hyp: str, ref: str) -> float:
    h, r = hyp.lower().split(), ref.lower().split()
    if not h:
        return 0.0
    counts = {}
    for w in r:
        counts[w] = counts.get(w, 0) + 1
    clipped = 0
    for w in h:
        if counts.get(w, 0) > 0:
            clipped += 1
            counts[w] -= 1
    bp = min(1.0, len(h) / len(r)) if r else 0.0
    return bp * (clipped / len(h))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Generator:  {GENERATOR_MODEL}")
    print(f"Judge:      {JUDGE_MODEL}")
    print(f"Embedding:  {EMBEDDING_MODEL}")
    print(f"Dataset:    {DATASET_PATH}")

    examples = load_examples()
    print(f"\nLoaded {len(examples)} examples")

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    # ── Phase 1: Generation (no context) ──────────────────────────────────────
    print("\n── Phase 1: Generation ──")
    gen_requests = [
        _make_request(
            custom_id=f"gen-{i}",
            model=GENERATOR_MODEL,
            system=CLOSED_BOOK_SYSTEM_PROMPT,
            user=f"Student question: {ex['question']}",
        )
        for i, ex in enumerate(examples)
    ]
    gen_raw = _run_batch(gen_requests, client, "generation")
    generated = [gen_raw.get(f"gen-{i}", "I don't know.") for i in range(len(examples))]

    # ── Phase 2: Judging (answer_correctness + answer_relevancy) ──────────────
    print("\n── Phase 2: Judging ──")
    judge_requests = []
    for i, (ex, ans) in enumerate(zip(examples, generated)):
        judge_requests.append(_make_request(
            custom_id=f"judge-{i}-answer_correctness",
            model=JUDGE_MODEL,
            system="You are an expert evaluator. Follow the instructions exactly and respond only with JSON.",
            user=_ANSWER_CORRECTNESS_PROMPT.format(
                question=ex["question"], reference=ex["answer"], answer=ans
            ),
        ))
        judge_requests.append(_make_request(
            custom_id=f"judge-{i}-answer_relevancy",
            model=JUDGE_MODEL,
            system="You are an expert evaluator. Follow the instructions exactly and respond only with JSON.",
            user=_ANSWER_RELEVANCY_PROMPT.format(answer=ans),
        ))

    judge_raw = _run_batch(judge_requests, client, "judging")

    # ── Answer relevancy: embedding-based cosine similarity ───────────────────
    print(f"\n── Phase 3: Answer relevancy embeddings ({EMBEDDING_MODEL}) ──")
    ar_data: list[tuple[str, list[str], int]] = []
    for i, ex in enumerate(examples):
        text = judge_raw.get(f"judge-{i}-answer_relevancy", "")
        try:
            obj = json.loads(text)
            synth_qs = [q for q in obj.get("questions", []) if isinstance(q, str) and q.strip()]
            noncommittal = int(obj.get("noncommittal", 0))
        except Exception:
            synth_qs = []
            noncommittal = 0
        ar_data.append((ex["question"], synth_qs, noncommittal))

    from sentence_transformers import SentenceTransformer
    import numpy as np
    embed = SentenceTransformer(EMBEDDING_MODEL)

    all_texts, text_keys = [], []
    for i, (orig_q, synth_qs, _) in enumerate(ar_data):
        all_texts.append(orig_q)
        text_keys.append((i, "orig"))
        for j, sq in enumerate(synth_qs):
            all_texts.append(sq)
            text_keys.append((i, f"synth_{j}"))

    embeddings = embed.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)
    emb_map = {k: e for k, e in zip(text_keys, embeddings)}

    answer_relevancy_scores = []
    for i, (_, synth_qs, noncommittal) in enumerate(ar_data):
        if noncommittal or not synth_qs:
            answer_relevancy_scores.append(0.0)
        else:
            orig_emb = emb_map.get((i, "orig"))
            sims = [
                float(np.dot(orig_emb, emb_map[(i, f"synth_{j}")]))
                for j in range(len(synth_qs))
                if (i, f"synth_{j}") in emb_map and orig_emb is not None
            ]
            answer_relevancy_scores.append(float(np.mean(sims)) if sims else 0.0)

    # ── Save results ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ex in enumerate(examples):
        ac = _parse_score(judge_raw.get(f"judge-{i}-answer_correctness", ""), "score")
        rows.append({
            "system":            SYSTEM_NAME,
            "example_id":        ex["example_id"],
            "lecture_key":       ex["lecture_key"],
            "question":          ex["question"],
            "reference_answer":  ex["answer"],
            "generated_answer":  generated[i],
            "faithfulness":      "",   # N/A — no context
            "context_precision": "",   # N/A — no context
            "context_recall":    "",   # N/A — no context
            "answer_correctness": ac,
            "answer_relevancy":  answer_relevancy_scores[i],
            "rouge_l":           _rouge_l(generated[i], ex["answer"]),
            "bleu_1":            _bleu_1(generated[i], ex["answer"]),
            "insufficient_context": 0,
        })

    fieldnames = [
        "system", "example_id", "lecture_key", "question",
        "reference_answer", "generated_answer",
        "faithfulness", "context_precision", "context_recall",
        "answer_correctness", "answer_relevancy",
        "rouge_l", "bleu_1", "insufficient_context",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {OUTPUT_CSV}")

    ac_scores  = [r["answer_correctness"] for r in rows]
    ar_scores  = [r["answer_relevancy"] for r in rows]
    rl_scores  = [r["rouge_l"] for r in rows]
    b1_scores  = [r["bleu_1"] for r in rows]
    print(f"\nClosed-book baseline results (n={len(rows)}):")
    print(f"  answer_correctness : {sum(ac_scores)/len(ac_scores):.4f}")
    print(f"  answer_relevancy   : {sum(ar_scores)/len(ar_scores):.4f}")
    print(f"  rouge_l            : {sum(rl_scores)/len(rl_scores):.4f}")
    print(f"  bleu_1             : {sum(b1_scores)/len(b1_scores):.4f}")


if __name__ == "__main__":
    main()
