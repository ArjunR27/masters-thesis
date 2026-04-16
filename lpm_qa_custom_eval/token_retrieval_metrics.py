"""Token-level retrieval metrics.

Definitions
-----------
te  : set of unique tokens in the ground-truth excerpt(s) for a query,
      extracted from the lecture transcript within the GT time ranges.
tr  : multiset of tokens across all retrieved chunks (duplicates across
      chunks are counted — this penalises redundant overlapping retrieval).
|te ∩ tr| : number of unique te tokens that appear in any retrieved chunk
            (each te token counted once regardless of how many chunks contain it).

Metrics
-------
IoU         = |te ∩ tr| / (|te| + |tr| - |te ∩ tr|)
Precision   = |te ∩ tr| / |tr|
Recall      = |te ∩ tr| / |te|
PrecisionΩ  = best-case precision: retrieve every segment in the index that
              contains any te token, then compute precision over that oracle set.
              Reports the upper bound on token efficiency for the system.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

TOKEN_RE = re.compile(r"\b\w+\b")

# Lazy import to avoid circular dependency at module level
_split_segment_text = None


def _get_splitter():
    global _split_segment_text
    if _split_segment_text is None:
        from treeseg_vector_index_modular.rerank_input_builder import RerankInputBuilder
        _split_segment_text = RerankInputBuilder.split_segment_text
    return _split_segment_text


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def extract_excerpt_tokens(
    transcripts_path: str,
    answer_ranges: tuple[tuple[float, float], ...],
) -> set[str]:
    """Return the set of unique lowercased tokens spoken within any GT time range.

    Reads the word-level transcript CSV (columns: Word, Start, End) and
    collects every word whose interval overlaps any GT range.
    """
    tokens: set[str] = set()
    try:
        with open(transcripts_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    w_start = float(row.get("Start") or 0.0)
                    w_end = float(row.get("End") or w_start)
                except (TypeError, ValueError):
                    continue
                for r_start, r_end in answer_ranges:
                    if w_start <= r_end and w_end >= r_start:
                        word = (row.get("Word") or "").strip()
                        if word:
                            tokens.update(tokenize(word))
                        break
    except (OSError, csv.Error):
        pass
    return tokens


def _spoken_tokens_from_hit(hit: dict) -> list[str]:
    """Extract spoken (non-OCR) tokens from a retrieved hit.

    For summary-tree internal nodes, uses supporting leaves' text.
    For all others, splits out the spoken portion before any [SLIDE] lines.
    Duplicates within a single hit are preserved (they contribute to |tr|).
    """
    splitter = _get_splitter()

    if not hit.get("is_leaf", True) and hit.get("supporting_leaves"):
        # Internal summary node — use raw leaf transcripts, not the LLM summary
        raw = " ".join(
            str(leaf.get("text") or "") for leaf in hit["supporting_leaves"]
        )
    else:
        raw = str(hit.get("text") or "")

    spoken, _ = splitter(raw)
    return tokenize(spoken if spoken else raw)


def compute_token_retrieval_metrics(
    hits: list[dict],
    excerpt_tokens: set[str],
    all_lecture_segments: list[dict],
) -> dict[str, float]:
    """Compute token-level IoU, Precision, Recall, and PrecisionΩ.

    Parameters
    ----------
    hits:
        Retrieved segments in rank order (top-N from the system under evaluation).
    excerpt_tokens:
        te — unique tokens from the GT transcript excerpt(s).
    all_lecture_segments:
        Every segment in the index for this lecture (used to compute PrecisionΩ).
    """
    zero = {"token_iou": 0.0, "token_precision": 0.0,
            "token_recall": 0.0, "token_precision_omega": 0.0}

    if not excerpt_tokens or not hits:
        return zero

    # Build tr as a flat token list (multiset — duplicates across chunks preserved)
    tr_tokens: list[str] = []
    for hit in hits:
        tr_tokens.extend(_spoken_tokens_from_hit(hit))

    tr_token_set = set(tr_tokens)
    intersection = excerpt_tokens & tr_token_set  # unique te tokens found in any hit

    te_size = len(excerpt_tokens)       # |te|
    tr_size = len(tr_tokens)            # |tr| — multiset count
    intersection_size = len(intersection)

    # IoU
    union_size = te_size + tr_size - intersection_size
    iou = intersection_size / union_size if union_size > 0 else 0.0

    # Precision and Recall
    precision = intersection_size / tr_size if tr_size > 0 else 0.0
    recall = intersection_size / te_size if te_size > 0 else 0.0

    # PrecisionΩ — oracle retrieval of all segments touching any te token
    omega_tr_tokens: list[str] = []
    for seg in all_lecture_segments:
        seg_tokens = _spoken_tokens_from_hit(seg)
        if any(t in excerpt_tokens for t in seg_tokens):
            omega_tr_tokens.extend(seg_tokens)

    if omega_tr_tokens:
        omega_tr_set = set(omega_tr_tokens)
        omega_intersection = excerpt_tokens & omega_tr_set
        precision_omega = len(omega_intersection) / len(omega_tr_tokens)
    else:
        precision_omega = 0.0

    return {
        "token_iou": iou,
        "token_precision": precision,
        "token_recall": recall,
        "token_precision_omega": precision_omega,
    }
