from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import ollama

from treeseg_vector_index_modular import (
    ContextBuilder,
    CrossEncoderReranker,
    LectureCatalog,
    OllamaResponder,
    PROJECT_DIR,
    ResultFormatter,
)

if __package__ in {None, ""}:
    from baseline_rag_system.store_builder import BaselineStoreBuilder
    from baseline_rag_system.types import BaselineRagConfig
else:
    from .store_builder import BaselineStoreBuilder
    from .types import BaselineRagConfig


class BaselineRagCLI:
    def __init__(self, project_dir=PROJECT_DIR, store_builder=None):
        self.project_dir = Path(project_dir)
        self.store_builder = store_builder or BaselineStoreBuilder()

    @staticmethod
    def parse_args(argv=None):
        parser = argparse.ArgumentParser(
            description="Build a simple chunked baseline RAG index over lecture transcripts."
        )
        parser.add_argument(
            "--list-lectures",
            action="store_true",
            help="List available lectures and exit.",
        )
        parser.add_argument(
            "--lecture",
            default=None,
            help="Lecture key or number (use with --list-lectures to see options).",
        )
        parser.add_argument(
            "--query",
            default=None,
            help="Single query to run non-interactively.",
        )
        parser.add_argument(
            "--answer",
            action="store_true",
            help="Generate an answer from retrieved context for --query.",
        )
        parser.add_argument(
            "--chunk-strategy",
            choices=["utterance_packed", "raw_token_window"],
            default="utterance_packed",
            help="Chunking strategy for the baseline retriever.",
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            choices=[128, 256, 512],
            default=256,
            help="Chunk size in model tokens.",
        )
        parser.add_argument(
            "--overlap-percent",
            type=int,
            choices=[0, 10],
            default=0,
            help="Token overlap percent between adjacent chunks.",
        )
        parser.add_argument(
            "--ocr-mode",
            choices=["transcript_only", "combined_ocr"],
            default="transcript_only",
            help="Whether to use only transcripts or combine OCR with transcript chunks.",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=10,
            help="Number of retrieved chunks to show before any reranking trim.",
        )
        parser.add_argument(
            "--rerank",
            action="store_true",
            help="Rerank retrieved chunks with a cross-encoder.",
        )
        parser.add_argument(
            "--rerank-model",
            default="cross-encoder/ms-marco-MiniLM-L-6-v2",
            help="Cross-encoder model for reranking.",
        )
        parser.add_argument(
            "--rerank-top-n",
            type=int,
            default=5,
            help="Number of chunks to keep after reranking.",
        )
        parser.add_argument(
            "--answer-model",
            default="llama3.2",
            help="Ollama model used for answer generation.",
        )
        parser.add_argument(
            "--ollama-host",
            default=None,
            help="Optional Ollama host URL.",
        )
        parser.add_argument(
            "--data-dir",
            default=None,
            help="Optional override for the lecture data directory.",
        )
        return parser.parse_args(argv)

    @staticmethod
    def print_turn_separator():
        print("\n" + "=" * 72 + "\n")

    def _resolve_data_dir(self, args) -> Path:
        if args.data_dir:
            return Path(args.data_dir).expanduser().resolve()
        return self.project_dir / "lpm_data"

    @staticmethod
    def _select_lecture(lectures, lecture_arg):
        if lecture_arg:
            lecture_key = LectureCatalog.resolve_lecture_choice(
                lectures,
                lecture_arg,
                allow_all=False,
            )
            return lecture_key

        print("Available lectures:")
        print(LectureCatalog.format_lecture_list(lectures))
        print("Choose a lecture by number or key.")
        while True:
            choice = input("lecture> ").strip()
            if not choice:
                continue
            try:
                return LectureCatalog.resolve_lecture_choice(
                    lectures,
                    choice,
                    allow_all=False,
                )
            except ValueError as exc:
                print(str(exc))

    @staticmethod
    def _build_reranker(args):
        if not args.rerank:
            return None
        return CrossEncoderReranker(args.rerank_model)

    @staticmethod
    def _search(store, query, lecture_key, *, top_k, top_n, reranker=None):
        results = store.search(query, top_k=top_k, lecture_key=lecture_key)
        if reranker:
            return reranker.rerank(query, results, top_n=top_n)
        return results[: min(top_n, len(results))]

    def run(self, args=None):
        if args is None:
            args = self.parse_args()

        data_dir = self._resolve_data_dir(args)
        max_chars = 500
        lectures = LectureCatalog.discover_lectures(data_dir=data_dir)
        if not lectures:
            print("No lectures found. Check the data directory.")
            return

        if args.list_lectures:
            print(LectureCatalog.format_lecture_list(lectures))
            if args.query is None and args.lecture is None:
                return

        lecture_key = self._select_lecture(lectures, args.lecture)
        target_lectures = [lecture for lecture in lectures if lecture.key == lecture_key]

        config = BaselineRagConfig(
            chunk_strategy=args.chunk_strategy,
            chunk_size_tokens=args.chunk_size,
            overlap_percent=args.overlap_percent,
            ocr_mode=args.ocr_mode,
        )
        build_result = self.store_builder.build_store(
            target_lectures,
            config,
            build_global=False,
        )
        store = build_result.store
        if lecture_key not in store.lecture_indices:
            reason = build_result.skipped_lectures.get(lecture_key, "not_indexed")
            print(f"Unable to build baseline index for {lecture_key}: {reason}")
            return

        reranker = self._build_reranker(args)
        ollama_client = ollama.Client(host=args.ollama_host) if args.ollama_host else None

        def run_query(query: str) -> None:
            results = self._search(
                store,
                query,
                lecture_key,
                top_k=args.top_k,
                top_n=args.rerank_top_n,
                reranker=reranker,
            )
            ResultFormatter.print_results(results, max_chars=max_chars)
            if args.answer or args.query is None:
                context = ContextBuilder.build_context(
                    results,
                    include_ocr=config.attach_ocr,
                )
                response = OllamaResponder.query_response(
                    query,
                    context,
                    model=args.answer_model,
                    client=ollama_client,
                    host=args.ollama_host,
                )
                print(response)

        if args.query:
            query = args.query.strip()
            if not query:
                print("Empty query. Provide text for --query.")
                return
            run_query(query)
            return

        print(f"Selected lecture: {lecture_key}")
        print(
            "Baseline config: "
            f"strategy={config.chunk_strategy}, "
            f"chunk_size={config.chunk_size_tokens}, "
            f"overlap={config.overlap_percent}%, "
            f"ocr_mode={config.ocr_mode}"
        )
        print("Type a query to search (empty or 'exit' to quit).")
        while True:
            query = input("search> ").strip()
            if not query or query.lower() in {"exit", "quit", "q"}:
                break
            run_query(query)
            self.print_turn_separator()


def main():
    app = BaselineRagCLI(project_dir=PROJECT_DIR)
    app.run()


if __name__ == "__main__":
    main()
