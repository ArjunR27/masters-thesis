import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from treeseg_vector_index_modular.constants import PROJECT_DIR
from treeseg_vector_index_modular.context_builder import ContextBuilder
from treeseg_vector_index_modular.cross_encoder_reranker import CrossEncoderReranker
from treeseg_vector_index_modular.device_resolver import DeviceResolver
from treeseg_vector_index_modular.lecture_catalog import LectureCatalog
from treeseg_vector_index_modular.lpm_config_builder import LpmConfigBuilder
from treeseg_vector_index_modular.ollama_responder import OllamaResponder
from treeseg_vector_index_modular.rerank_input_builder import RerankInputBuilder
from treeseg_vector_index_modular.result_formatter import ResultFormatter
from treeseg_vector_index_modular.vector_store_factory import VectorStoreFactory


class TreeSegVectorIndexCLI:
    def __init__(self, project_dir=PROJECT_DIR, store_factory=None):
        self.project_dir = project_dir
        self.store_factory = store_factory or VectorStoreFactory()

    @staticmethod
    def parse_args(argv=None):
        parser = argparse.ArgumentParser(
            description="Build a TreeSeg-based vector index over LPM lectures."
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
            help="Single query to run (non-interactive).",
        )
        parser.add_argument(
            "--retrieval-mode",
            choices=["combined", "separate"],
            default="combined",
            help="Use combined (ASR+OCR) or separate (ASR-only + OCR-only) retrieval.",
        )
        parser.add_argument(
            "--rerank",
            action="store_true",
            help="Rerank retrieved segments with a cross-encoder.",
        )
        parser.add_argument(
            "--rerank-model",
            default="cross-encoder/ms-marco-MiniLM-L-6-v2",
            help="Cross-encoder model for reranking.",
        )
        parser.add_argument(
            "--ocr-rerank-model",
            default="BAAI/bge-reranker-v2-m3",
            help="Cross-encoder model for OCR-only reranking.",
        )
        parser.add_argument(
            "--rerank-top-n",
            type=int,
            default=5,
            help="Number of results to return after reranking.",
        )
        return parser.parse_args(argv)

    def run(self, args=None):
        if args is None:
            args = self.parse_args()

        data_dir = self.project_dir / "lpm_data"
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        top_k = 50
        top_n = args.rerank_top_n
        max_chars = 500

        lectures = LectureCatalog.discover_lectures(data_dir=data_dir)
        if not lectures:
            print("No lectures found. Check the data directory.")
            return

        if args.list_lectures:
            print(LectureCatalog.format_lecture_list(lectures))
            if args.query is None and args.lecture is None:
                return

        treeseg_config = LpmConfigBuilder.build_lpm_config(
            embedding_model=embedding_model,
        )

        if args.lecture:
            try:
                lecture_key = LectureCatalog.resolve_lecture_choice(
                    lectures, args.lecture, allow_all=False
                )
            except ValueError as exc:
                print(str(exc))
                return
            target_lectures = [lec for lec in lectures if lec.key == lecture_key]
        else:
            print("Available lectures:")
            print(LectureCatalog.format_lecture_list(lectures))
            print("Choose a lecture by number or key.")
            while True:
                choice = input("lecture> ").strip()
                if not choice:
                    continue
                try:
                    lecture_key = LectureCatalog.resolve_lecture_choice(
                        lectures, choice, allow_all=False
                    )
                    break
                except ValueError as exc:
                    print(str(exc))
                    continue
            target_lectures = [lec for lec in lectures if lec.key == lecture_key]

        retrieval_mode = args.retrieval_mode
        if retrieval_mode == "combined":
            store = self.store_factory.build_vector_store(
                target_lectures,
                treeseg_config=treeseg_config,
                embed_model=embedding_model,
                normalize=True,
                build_global=False,
                max_gap_s=0.8,
                lowercase=True,
                attach_ocr=True,
                ocr_min_conf=60.0,
                ocr_per_slide=1,
                target_segments=None,
            )
            reranker = None
            if args.rerank:
                reranker = CrossEncoderReranker(
                    args.rerank_model, device=DeviceResolver.resolve_device()
                )
        else:
            asr_store = self.store_factory.build_vector_store(
                target_lectures,
                treeseg_config=treeseg_config,
                embed_model=embedding_model,
                normalize=True,
                build_global=False,
                max_gap_s=0.8,
                lowercase=True,
                attach_ocr=True,
                include_ocr_in_treeseg=False,
                ocr_min_conf=60.0,
                ocr_per_slide=1,
                target_segments=None,
            )
            ocr_store = self.store_factory.build_ocr_vector_store(
                target_lectures,
                embed_model=embedding_model,
                normalize=True,
                build_global=False,
                ocr_min_conf=60.0,
            )

            asr_reranker = None
            ocr_reranker = None
            if args.rerank:
                asr_reranker = CrossEncoderReranker(
                    args.rerank_model, device=DeviceResolver.resolve_device()
                )
                ocr_reranker = CrossEncoderReranker(
                    args.ocr_rerank_model,
                    device=DeviceResolver.resolve_device(),
                    input_builder=RerankInputBuilder.build_rerank_input_ocr,
                )

        if args.query:
            if not args.query.strip():
                print("Empty query. Provide text for --query.")
                return
            if retrieval_mode == "combined":
                results = store.search(args.query, top_k=top_k, lecture_key=lecture_key)
                if reranker:
                    results = reranker.rerank(args.query, results, top_n=top_n)
                ResultFormatter.print_results(results, max_chars=max_chars)
            else:
                asr_results = asr_store.search(
                    args.query, top_k=top_k, lecture_key=lecture_key
                )
                if lecture_key not in ocr_store.lecture_indices:
                    ocr_results = []
                else:
                    ocr_results = ocr_store.search(
                        args.query, top_k=top_k, lecture_key=lecture_key
                    )
                if asr_reranker:
                    asr_results = asr_reranker.rerank(
                        args.query, asr_results, top_n=top_n
                    )
                if ocr_reranker:
                    ocr_results = ocr_reranker.rerank(
                        args.query, ocr_results, top_n=top_n
                    )
                print("ASR results:")
                ResultFormatter.print_results(asr_results, max_chars=max_chars)
                print("OCR results:")
                ResultFormatter.print_ocr_results(ocr_results, max_chars=max_chars)
            return

        print("Type a query to search (empty or 'exit' to quit).")
        while True:
            query = input("search> ").strip()
            if not query or query.lower() in {"exit", "quit", "q"}:
                break
            if retrieval_mode == "combined":
                results = store.search(query, top_k=top_k, lecture_key=lecture_key)
                if reranker:
                    results = reranker.rerank(query, results, top_n=top_n)
                context = ContextBuilder.build_context(results)
            else:
                asr_results = asr_store.search(
                    query, top_k=top_k, lecture_key=lecture_key
                )
                if lecture_key not in ocr_store.lecture_indices:
                    ocr_results = []
                else:
                    ocr_results = ocr_store.search(
                        query, top_k=top_k, lecture_key=lecture_key
                    )
                if asr_reranker:
                    asr_results = asr_reranker.rerank(
                        query, asr_results, top_n=top_n
                    )
                if ocr_reranker:
                    ocr_results = ocr_reranker.rerank(
                        query, ocr_results, top_n=top_n
                    )

                print("ASR results:")
                ResultFormatter.print_results(asr_results, max_chars=max_chars)
                print("OCR results:")
                ResultFormatter.print_ocr_results(ocr_results, max_chars=max_chars)
                context = ContextBuilder.build_separate_context(asr_results, ocr_results)

            response = OllamaResponder.query_response(query, context)
            print(response)

def main():
    app = TreeSegVectorIndexCLI(project_dir=PROJECT_DIR)
    app.run()


if __name__ == "__main__":
    main()
