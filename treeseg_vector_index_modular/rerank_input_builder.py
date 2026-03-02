from .constants import SLIDE_TOKEN


class RerankInputBuilder:
    @staticmethod
    def split_segment_text(text, slide_token=SLIDE_TOKEN):
        if not text:
            return "", ""
        spoken_lines = []
        ocr_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(slide_token):
                ocr_line = stripped[len(slide_token) :].lstrip(" :\t")
                if ocr_line:
                    ocr_lines.append(ocr_line)
            else:
                spoken_lines.append(stripped)
        return "\n".join(spoken_lines).strip(), "\n".join(ocr_lines).strip()

    @staticmethod
    def build_rerank_input(text, slide_token=SLIDE_TOKEN):
        spoken, ocr = RerankInputBuilder.split_segment_text(text, slide_token=slide_token)
        segment_block = spoken if spoken else "<blank>"
        parts = [f"Segment:\n{segment_block}"]
        if ocr:
            ocr_lines = [line.strip() for line in ocr.splitlines() if line.strip()]
            ocr_block = "\n".join(f"{slide_token} {line}" for line in ocr_lines)
            parts.append(f"{slide_token}\nSlide OCR:\n{ocr_block}")
        else:
            parts.append("Slide OCR:\n<blank>")
        return "\n\n".join(parts)

    @staticmethod
    def build_rerank_input_ocr(text, slide_token=SLIDE_TOKEN):
        ocr = (text or "").strip()
        if not ocr:
            ocr = "<blank>"
        return f"Slide OCR:\n{ocr}"
