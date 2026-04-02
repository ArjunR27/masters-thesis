from .constants import SLIDE_TOKEN
from .rerank_input_builder import RerankInputBuilder

# Builds the "context" for the LLM before getting passed in with the prompt
# formatting purposes
class ContextBuilder:
    @staticmethod
    def _format_context_header(hit, rank):
        parts = [f"[{rank}]"]
        score = hit.get("rerank_score", hit.get("score"))
        if isinstance(score, (int, float)):
            parts.append(f"score={score:.3f}")
        lecture_key = hit.get("lecture_key")
        if lecture_key:
            parts.append(str(lecture_key))
        if hit.get("index_kind") == "summary_tree":
            if hit.get("is_leaf", True):
                parts.append("leaf")
            else:
                parts.append("summary-node")
            depth = hit.get("depth")
            if depth is not None:
                parts.append(f"depth={depth}")
            tree_path = hit.get("tree_path")
            if tree_path:
                parts.append(f"path={tree_path}")
        slide_index = hit.get("slide_index")
        segment_id = hit.get("segment_id")
        if slide_index is not None:
            parts.append(f"slide={slide_index}")
        elif segment_id is not None:
            parts.append(f"seg={segment_id}")
        start = hit.get("start")
        end = hit.get("end")
        if start is not None and end is not None:
            parts.append(f"time={start}-{end}s")
        return " ".join(parts)

    @staticmethod
    def _build_leaf_context_block(hit, rank, include_ocr=True, slide_token=SLIDE_TOKEN):
        text = (hit.get("text") or "").strip()
        if not text:
            return ""
        spoken, ocr = RerankInputBuilder.split_segment_text(
            text, slide_token=slide_token
        )
        header = ContextBuilder._format_context_header(hit, rank)
        parts = [header, f"Spoken:\n{spoken if spoken else '<blank>'}"]
        if include_ocr:
            parts.append(f"Slide OCR:\n{ocr if ocr else '<blank>'}")
        return "\n".join(parts)

    @staticmethod
    def _build_summary_tree_block(hit, rank, slide_token=SLIDE_TOKEN):
        if hit.get("is_leaf", True):
            return ContextBuilder._build_leaf_context_block(
                hit,
                rank,
                include_ocr=True,
                slide_token=slide_token,
            )

        summary_text = (hit.get("summary_text") or hit.get("text") or "").strip()
        if not summary_text:
            summary_text = "<blank>"

        header = ContextBuilder._format_context_header(hit, rank)
        parts = [header, f"Summary Node:\n{summary_text}"]

        supporting_leaves = hit.get("supporting_leaves") or []
        if supporting_leaves:
            leaf_blocks = []
            for leaf_rank, leaf in enumerate(supporting_leaves, start=1):
                leaf_id = leaf.get("segment_id")
                start = leaf.get("start")
                end = leaf.get("end")
                leaf_header_parts = [f"Leaf {leaf_rank}"]
                if leaf_id is not None:
                    leaf_header_parts.append(f"seg={leaf_id}")
                if start is not None and end is not None:
                    leaf_header_parts.append(f"time={start}-{end}s")
                spoken, ocr = RerankInputBuilder.split_segment_text(
                    leaf.get("text", ""), slide_token=slide_token
                )
                leaf_parts = [
                    " ".join(leaf_header_parts),
                    f"Spoken:\n{spoken if spoken else '<blank>'}",
                    f"Slide OCR:\n{ocr if ocr else '<blank>'}",
                ]
                leaf_blocks.append("\n".join(leaf_parts))
            parts.append("Supporting Leaf Evidence:\n" + "\n\n".join(leaf_blocks))
        else:
            parts.append("Supporting Leaf Evidence:\n<blank>")
        return "\n".join(parts)

    @staticmethod
    def build_context(results, max_chars=8000, include_ocr=True, slide_token=SLIDE_TOKEN):
        if not results:
            return ""

        blocks = []
        total_chars = 0
        for rank, hit in enumerate(results, start=1):
            if hit.get("index_kind") == "summary_tree":
                block = ContextBuilder._build_summary_tree_block(
                    hit, rank, slide_token=slide_token
                )
            else:
                block = ContextBuilder._build_leaf_context_block(
                    hit,
                    rank,
                    include_ocr=include_ocr,
                    slide_token=slide_token,
                )
            if not block:
                continue

            if max_chars is not None and total_chars + len(block) > max_chars:
                break

            blocks.append(block)
            total_chars += len(block) + 2

        return "\n\n".join(blocks).strip()

    @staticmethod
    def build_ocr_context(results, max_chars=8000):
        if not results:
            return ""

        blocks = []
        total_chars = 0
        for rank, hit in enumerate(results, start=1):
            text = (hit.get("text") or "").strip()
            if not text:
                continue

            header = ContextBuilder._format_context_header(hit, rank)
            block = "\n".join([header, f"Slide OCR:\n{text}"])

            if max_chars is not None and total_chars + len(block) > max_chars:
                break

            blocks.append(block)
            total_chars += len(block) + 2

        return "\n\n".join(blocks).strip()

    @staticmethod
    def build_separate_context(asr_results, ocr_results, max_chars=8000):
        asr_context = ContextBuilder.build_context(
            asr_results, max_chars=max_chars, include_ocr=False
        )
        ocr_context = ContextBuilder.build_ocr_context(ocr_results, max_chars=max_chars)

        parts = []
        if asr_context:
            parts.append(f"ASR Context:\n{asr_context}")
        if ocr_context:
            parts.append(f"OCR Context:\n{ocr_context}")

        return "\n\n".join(parts).strip()
