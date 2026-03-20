class ResultFormatter:
    @staticmethod
    def _format_score(hit):
        if "rerank_score" in hit:
            return f"{hit['rerank_score']:.3f} (rerank)"
        return f"{hit['score']:.3f}"

    @staticmethod
    def _print_summary_tree_result(rank, hit):
        score = ResultFormatter._format_score(hit)
        if hit.get("is_leaf", True):
            label = f"leaf seg {hit.get('segment_id')}"
        else:
            label = "summary node"

        parts = [f"{rank}. {score} | {hit['lecture_key']} | {label}"]
        tree_path = hit.get("tree_path")
        if tree_path:
            parts.append(f"path={tree_path}")
        depth = hit.get("depth")
        if depth is not None:
            parts.append(f"depth={depth}")
        start = hit.get("start")
        end = hit.get("end")
        if start is not None and end is not None:
            parts.append(f"({start}-{end}s)")
        print(" | ".join(parts))

        text = (hit.get("summary_text") if not hit.get("is_leaf", True) else hit.get("text")) or ""
        if text:
            for line in str(text).splitlines():
                print(f"    {line}")

        supporting_leaves = hit.get("supporting_leaves") or []
        for leaf in supporting_leaves:
            leaf_id = leaf.get("segment_id")
            start = leaf.get("start")
            end = leaf.get("end")
            leaf_header = f"    support seg {leaf_id}" if leaf_id is not None else "    support leaf"
            if start is not None and end is not None:
                leaf_header += f" ({start}-{end}s)"
            print(leaf_header)
            leaf_text = leaf.get("text") or ""
            for line in str(leaf_text).splitlines():
                print(f"        {line}")
        print()

    @staticmethod
    def print_results(results, max_chars=500):
        for rank, hit in enumerate(results, start=1):
            if hit.get("index_kind") == "summary_tree":
                ResultFormatter._print_summary_tree_result(rank, hit)
                continue

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
