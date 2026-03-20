from .constants import SLIDE_TOKEN


class RerankInputBuilder:
    @staticmethod
    def _extract_text(payload):
        if isinstance(payload, dict):
            return payload.get("text", "")
        return payload or ""

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
    def build_rerank_input(payload, slide_token=SLIDE_TOKEN):
        text = RerankInputBuilder._extract_text(payload)
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
    def build_summary_tree_rerank_input(payload, slide_token=SLIDE_TOKEN):
        if not isinstance(payload, dict):
            return RerankInputBuilder.build_rerank_input(payload, slide_token=slide_token)

        if payload.get("is_leaf", True):
            return RerankInputBuilder.build_rerank_input(payload, slide_token=slide_token)

        summary_text = (payload.get("summary_text") or payload.get("text") or "").strip()
        if not summary_text:
            summary_text = "<blank>"
        parts = [f"Summary Node:\n{summary_text}"]
        supporting_leaves = payload.get("supporting_leaves") or []
        if supporting_leaves:
            leaf_blocks = []
            for leaf in supporting_leaves:
                leaf_id = leaf.get("segment_id")
                label = f"Leaf {leaf_id}" if leaf_id is not None else (leaf.get("tree_path") or "Leaf")
                leaf_text = RerankInputBuilder._extract_text(leaf)
                leaf_block = RerankInputBuilder.build_rerank_input(
                    leaf_text, slide_token=slide_token
                )
                leaf_blocks.append(f"{label}\n{leaf_block}")
            parts.append("Supporting Leaf Evidence:\n" + "\n\n".join(leaf_blocks))
        else:
            parts.append("Supporting Leaf Evidence:\n<blank>")
        return "\n\n".join(parts)

    @staticmethod
    def build_rerank_input_ocr(payload, slide_token=SLIDE_TOKEN):
        ocr = RerankInputBuilder._extract_text(payload).strip()
        if not ocr:
            ocr = "<blank>"
        return f"Slide OCR:\n{ocr}"
