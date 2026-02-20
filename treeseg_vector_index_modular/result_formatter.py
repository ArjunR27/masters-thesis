class ResultFormatter:
    @staticmethod
    def print_results(results, max_chars=500):
        for rank, hit in enumerate(results, start=1):
            if "rerank_score" in hit:
                header = (
                    f"{rank}. {hit['rerank_score']:.3f} (rerank) | "
                    f"{hit['lecture_key']} | seg {hit['segment_id']} "
                    f"({hit['start']}-{hit['end']}s)"
                )
            else:
                header = (
                    f"{rank}. {hit['score']:.3f} | {hit['lecture_key']} | seg "
                    f"{hit['segment_id']} ({hit['start']}-{hit['end']}s)"
                )
            print(header)
            text = hit["text"] or ""
            if text:
                for line in text.splitlines():
                    print(f"    {line}")
            print()

    @staticmethod
    def print_ocr_results(results, max_chars=500):
        for rank, hit in enumerate(results, start=1):
            slide_idx = hit.get("slide_index")
            if slide_idx is None:
                slide_idx = hit.get("segment_id")
            if "rerank_score" in hit:
                header = (
                    f"{rank}. {hit['rerank_score']:.3f} (rerank) | "
                    f"{hit['lecture_key']} | slide {slide_idx}"
                )
            else:
                header = (
                    f"{rank}. {hit['score']:.3f} | {hit['lecture_key']} | "
                    f"slide {slide_idx}"
                )
            print(header)
            text = hit.get("text") or ""
            if text:
                for line in text.splitlines():
                    print(f"    {line}")
            print()
