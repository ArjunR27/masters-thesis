from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from treeseg_vector_index_modular.lecture_descriptor import LectureDescriptor

from baseline_rag_system.chunk_builder import BaselineChunkBuilder
from baseline_rag_system.types import BaselineRagConfig
from eduvid_evaluation.retrieval_metrics import compute_retrieval_metrics


def make_lecture() -> LectureDescriptor:
    return LectureDescriptor(
        speaker="test",
        course_dir="course",
        meeting_id="01",
        video_id="vid",
        transcripts_path="unused.csv",
        meeting_dir="unused",
    )


def test_utterance_packed_without_overlap() -> None:
    lecture = make_lecture()
    config = BaselineRagConfig(
        chunk_strategy="utterance_packed",
        chunk_size_tokens=4,
        overlap_percent=0,
        ocr_mode="transcript_only",
    )
    utterances = [
        {"text": "a b", "start": 0.0, "end": 1.0},
        {"text": "c d", "start": 1.0, "end": 2.0},
        {"text": "e f", "start": 2.0, "end": 3.0},
    ]

    chunks = BaselineChunkBuilder.build_utterance_packed_chunks(
        lecture,
        utterances,
        config,
    )

    assert len(chunks) == 2, chunks
    assert chunks[0]["source_item_start"] == 0
    assert chunks[0]["source_item_end"] == 1
    assert chunks[1]["source_item_start"] == 2
    assert chunks[1]["source_item_end"] == 2
    assert chunks[0]["token_count"] <= config.chunk_size_tokens
    assert chunks[1]["token_count"] <= config.chunk_size_tokens


def test_utterance_packed_overlap_reuses_tail_items() -> None:
    lecture = make_lecture()
    config = BaselineRagConfig(
        chunk_strategy="utterance_packed",
        chunk_size_tokens=4,
        overlap_percent=50,
        ocr_mode="transcript_only",
    )
    utterances = [
        {"text": "a b", "start": 0.0, "end": 1.0},
        {"text": "c d", "start": 1.0, "end": 2.0},
        {"text": "e f", "start": 2.0, "end": 3.0},
    ]

    chunks = BaselineChunkBuilder.build_utterance_packed_chunks(
        lecture,
        utterances,
        config,
    )

    assert len(chunks) == 2, chunks
    assert chunks[0]["source_item_start"] == 0
    assert chunks[0]["source_item_end"] == 1
    assert chunks[1]["source_item_start"] == 1
    assert chunks[1]["source_item_end"] == 2


def test_raw_token_window_overlap_and_ocr_injection() -> None:
    lecture = make_lecture()
    config = BaselineRagConfig(
        chunk_strategy="raw_token_window",
        chunk_size_tokens=4,
        overlap_percent=50,
        ocr_mode="combined_ocr",
    )
    word_rows = [
        {"Word": "a", "Start": 0.0, "End": 0.9},
        {"Word": "b", "Start": 1.0, "End": 1.9},
        {"Word": "c", "Start": 2.0, "End": 2.9},
        {"Word": "d", "Start": 3.0, "End": 3.9},
        {"Word": "e", "Start": 4.0, "End": 4.9},
        {"Word": "f", "Start": 5.0, "End": 5.9},
    ]
    slide_entries = [
        {"slide_index": 0, "start": 0.0, "end": 2.5, "text": "intro slide"},
        {"slide_index": 1, "start": 2.5, "end": 5.5, "text": "second slide"},
    ]

    chunks = BaselineChunkBuilder.build_raw_token_window_chunks(
        lecture,
        word_rows,
        config,
        slide_entries=slide_entries,
    )

    assert len(chunks) == 2, chunks
    assert chunks[0]["source_item_start"] == 0
    assert chunks[0]["source_item_end"] == 3
    assert chunks[1]["source_item_start"] == 2
    assert chunks[1]["source_item_end"] == 5
    assert "[SLIDE] intro slide" in chunks[0]["text"]
    assert "[SLIDE] second slide" in chunks[1]["text"]


def test_retrieval_metrics_exact_and_relaxed() -> None:
    exact_metrics = compute_retrieval_metrics(
        [
            {"start": 0.0, "end": 4.0},
            {"start": 9.0, "end": 12.0},
        ],
        timestamp_seconds=10,
        tolerance_seconds=2.0,
    )
    assert exact_metrics["retrieval_hit_exact"] == 1.0
    assert exact_metrics["retrieval_mrr_exact"] == 0.5
    assert exact_metrics["retrieval_hit_relaxed"] == 1.0
    assert exact_metrics["retrieval_mrr_relaxed"] == 0.5

    relaxed_only_metrics = compute_retrieval_metrics(
        [
            {"start": 0.0, "end": 8.0},
            {"start": 20.0, "end": 30.0},
        ],
        timestamp_seconds=10,
        tolerance_seconds=2.0,
    )
    assert relaxed_only_metrics["retrieval_hit_exact"] == 0.0
    assert relaxed_only_metrics["retrieval_mrr_exact"] == 0.0
    assert relaxed_only_metrics["retrieval_hit_relaxed"] == 1.0
    assert relaxed_only_metrics["retrieval_mrr_relaxed"] == 1.0


def main() -> None:
    test_utterance_packed_without_overlap()
    test_utterance_packed_overlap_reuses_tail_items()
    test_raw_token_window_overlap_and_ocr_injection()
    test_retrieval_metrics_exact_and_relaxed()
    print("baseline_rag_system smoke tests passed")


if __name__ == "__main__":
    main()
