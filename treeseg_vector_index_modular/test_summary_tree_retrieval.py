import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import treeseg_vector_index_modular.lecture_segment_builder as lecture_segment_builder_module

from treeseg_vector_index_modular.cross_encoder_reranker import CrossEncoderReranker
from treeseg_vector_index_modular.device_resolver import DeviceResolver
from treeseg_vector_index_modular.lecture_catalog import LectureCatalog
from treeseg_vector_index_modular.lpm_config_builder import LpmConfigBuilder
from treeseg_vector_index_modular.rerank_input_builder import RerankInputBuilder
from treeseg_vector_index_modular.vector_store_factory import VectorStoreFactory
from treeseg_vector_index_modular.constants import PROJECT_DIR


class MockSummaryResponder:
    call_count = 0

    @classmethod
    def reset(cls):
        cls.call_count = 0

    @staticmethod
    def generate_summary(
        text,
        is_leaf=True,
        model="llama3.2",
        temperature=0.2,
        keep_alive=None,
        client=None,
        host=None,
    ):
        del model, temperature, keep_alive, client, host
        MockSummaryResponder.call_count += 1
        cleaned = " ".join((text or "").split())
        if not cleaned:
            cleaned = "<blank>"
        prefix = "Leaf summary:" if is_leaf else "Internal summary:"
        return f"{prefix} {cleaned[:220]}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Smoke-test summary-tree retrieval and expansion."
    )
    parser.add_argument(
        "--lecture",
        default=None,
        help="Lecture key or list index to process.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of initial retrieval hits to inspect.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Run the summary-tree reranker as part of the smoke test.",
    )
    parser.add_argument(
        "--mock-summaries",
        action="store_true",
        help="Bypass Ollama by monkeypatching a deterministic summary generator.",
    )
    return parser.parse_args(argv)


def choose_lecture(lectures, choice):
    if choice is None:
        return lectures[0]
    return next(
        lecture for lecture in lectures if lecture.key == LectureCatalog.resolve_lecture_choice(
            lectures, choice, allow_all=False
        )
    )


def derive_query(entry):
    internal_records = [
        segment
        for segment in entry["segments"]
        if segment.get("index_kind") == "summary_tree" and not segment.get("is_leaf", True)
    ]
    if not internal_records:
        raise RuntimeError("No internal summary-tree nodes found in the built index.")
    text = (internal_records[0].get("text") or "").strip()
    tokens = text.split()
    if not tokens:
        return "summary tree"
    return " ".join(tokens[: min(12, len(tokens))])


def main(argv=None):
    args = parse_args(argv)
    if args.mock_summaries:
        MockSummaryResponder.reset()
        lecture_segment_builder_module.OllamaResponder = MockSummaryResponder

    data_dir = PROJECT_DIR / "lpm_data"
    lectures = LectureCatalog.discover_lectures(data_dir=data_dir)
    if not lectures:
        raise SystemExit("No lectures found. Check the data directory.")

    lecture = choose_lecture(lectures, args.lecture)
    treeseg_config = LpmConfigBuilder.build_lpm_config(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    store = VectorStoreFactory().build_vector_store(
        [lecture],
        treeseg_config=treeseg_config,
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        normalize=True,
        build_global=False,
        max_gap_s="auto",
        lowercase=True,
        attach_ocr=True,
        include_ocr_in_treeseg=True,
        ocr_min_conf=60.0,
        ocr_per_slide=1,
        target_segments=None,
        index_kind="summary_tree",
    )

    entry = store.lecture_indices.get(lecture.key)
    if entry is None:
        raise SystemExit(f"No summary-tree index built for lecture {lecture.key}")

    internal_count = sum(1 for segment in entry["segments"] if not segment.get("is_leaf", True))
    leaf_count = sum(1 for segment in entry["segments"] if segment.get("is_leaf", False))
    if internal_count == 0 or leaf_count == 0:
        raise SystemExit(
            f"Expected both internal and leaf nodes, got internal={internal_count}, leaf={leaf_count}"
        )
    leaf_summaries = [
        segment
        for segment in entry["segments"]
        if segment.get("is_leaf", False) and segment.get("summary_text") is not None
    ]
    if leaf_summaries:
        raise SystemExit("Leaf summary-tree nodes should not have generated summaries.")
    if args.mock_summaries and MockSummaryResponder.call_count != internal_count:
        raise SystemExit(
            f"Expected {internal_count} summary calls, got {MockSummaryResponder.call_count}."
        )

    query = derive_query(entry)
    query_embedding = store.encode_query(query)
    raw_results = store.search_with_embedding(
        query_embedding, top_k=args.top_k, lecture_key=lecture.key
    )
    if not raw_results:
        raise SystemExit("Summary-tree retrieval returned no hits.")

    expanded_results = store.expand_summary_tree_results(
        query=query,
        results=raw_results,
        lecture_key=lecture.key,
        top_descendant_leaves=3,
        query_embedding=query_embedding,
    )
    internal_hits = [hit for hit in expanded_results if not hit.get("is_leaf", True)]
    if not internal_hits:
        raise SystemExit("Expected at least one internal summary-tree retrieval hit.")
    if not any(hit.get("supporting_leaves") for hit in internal_hits):
        raise SystemExit("Internal summary-tree hits did not expand to supporting leaves.")

    if args.rerank:
        reranker = CrossEncoderReranker(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=DeviceResolver.resolve_device(),
            input_builder=RerankInputBuilder.build_summary_tree_rerank_input,
        )
        reranked_results = reranker.rerank(query, expanded_results, top_n=5)
        if not reranked_results:
            raise SystemExit("Reranking returned no results.")
        expanded_results = reranked_results

    deduped_results = store.deduplicate_summary_tree_results(expanded_results)
    if not deduped_results:
        raise SystemExit("Deduplication removed all summary-tree results.")

    print(f"Lecture: {lecture.key}")
    print(f"Query: {query}")
    print(f"Flattened nodes: total={len(entry['segments'])} leaf={leaf_count} internal={internal_count}")
    print(f"Retrieved hits: raw={len(raw_results)} expanded={len(expanded_results)} deduped={len(deduped_results)}")
    top_internal = next(hit for hit in deduped_results if not hit.get("is_leaf", True))
    print(
        "Top internal hit:",
        top_internal.get("tree_path"),
        f"supporting_leaves={len(top_internal.get('supporting_leaves') or [])}",
    )
    print("Summary-tree retrieval smoke test passed.")


if __name__ == "__main__":
    raise SystemExit(main())
