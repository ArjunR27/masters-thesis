from .constants import SLIDE_TOKEN
from .rerank_input_builder import RerankInputBuilder


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
    def build_context(results, max_chars=8000, include_ocr=True, slide_token=SLIDE_TOKEN):
        if not results:
            return ""

        blocks = []
        total_chars = 0
        for rank, hit in enumerate(results, start=1):
            text = (hit.get("text") or "").strip()
            if not text:
                continue

            spoken, ocr = RerankInputBuilder.split_segment_text(text, slide_token=slide_token)
            header = ContextBuilder._format_context_header(hit, rank)
            parts = [header, f"Spoken:\n{spoken if spoken else '<blank>'}"]
            if include_ocr:
                parts.append(f"Slide OCR:\n{ocr if ocr else '<blank>'}")
            block = "\n".join(parts)

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
