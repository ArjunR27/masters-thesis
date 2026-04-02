import os
import sys
import structlog

from utterances import (
    extract_utterances_from_transcript_file,
    load_slide_end_times,
    load_slide_ocr_texts,
)

from .constants import TREESEG_EXPLORATION

if str(TREESEG_EXPLORATION) not in sys.path:
    sys.path.insert(0, str(TREESEG_EXPLORATION))

from treeseg import TreeSeg
try:
    from .ollama_responder import OllamaResponder
except ModuleNotFoundError as exc:
    if exc.name == "ollama":
        OllamaResponder = None
    else:
        raise


logger = structlog.get_logger()


class LectureSegmentBuilder:
    @staticmethod
    def _compose_segment_text(segment_utts):
        text = "\n".join(utt.get("composite", "") for utt in segment_utts).strip()
        return text or "<blank>"

    @staticmethod
    def _encode_text(embedder, text, normalize_embeddings=True):
        embedding = embedder.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )
        return embedding

    @staticmethod
    def _build_internal_summary_input(left_content, right_content):
        sections = []
        if left_content:
            sections.append(f"Section A:\n{left_content}")
        if right_content:
            sections.append(f"Section B:\n{right_content}")
        return "\n\n".join(sections).strip() or "<blank>"

    @staticmethod
    def load_lecture_utterances(
        lecture,
        max_gap_s='auto',
        lowercase=True,
        attach_ocr=True,
        ocr_min_conf=60.0,
        ocr_per_slide=1,
    ):
        segments_path = None
        slides_dir = None
        if attach_ocr:
            segments_path = os.path.join(lecture.meeting_dir, "segments.txt")
            slides_dir = lecture.meeting_dir

        utterances = extract_utterances_from_transcript_file(
            csv_path=lecture.transcripts_path,
            max_gap_s=max_gap_s,
            lowercase=lowercase,
            segments_path=segments_path,
            slides_dir=slides_dir,
            ocr_min_conf=ocr_min_conf,
            ocr_per_slide=ocr_per_slide,
        )

        for i, utt in enumerate(utterances):
            utt.update(
                {
                    "meeting_id": lecture.meeting_id,
                    "video_id": lecture.video_id,
                    "speaker": lecture.speaker,
                    "course_dir": lecture.course_dir,
                    "idx": i,
                    "source": "transcripts",
                    "path": lecture.transcripts_path,
                }
            )
        return utterances

    @staticmethod
    def build_treeseg_entries(utterances, include_ocr=True, ocr_prefix="[SLIDE] "):
        entries = []
        for idx, utt in enumerate(utterances):
            entry = dict(utt)
            entry.setdefault("utterance_index", idx)

            spoken = (entry.get("text") or "").strip()
            ocr = (entry.get("ocr_text") or "").strip()

            if include_ocr and ocr:
                composite = spoken
                if composite:
                    composite = f"{composite}\n{ocr_prefix}{ocr}"
                else:
                    composite = f"{ocr_prefix}{ocr}"
            else:
                composite = spoken

            entry["composite"] = composite if composite else "<blank>"
            entries.append(entry)
        return entries

    @staticmethod
    def build_ocr_slide_entries(lecture, ocr_min_conf=60.0, line_sep="\n"):
        segments_path = os.path.join(lecture.meeting_dir, "segments.txt")
        slides_dir = lecture.meeting_dir
        if not os.path.exists(segments_path):
            return []

        end_times = load_slide_end_times(segments_path)
        if not end_times:
            return []

        ocr_by_slide = load_slide_ocr_texts(
            slides_dir, min_conf=ocr_min_conf, line_sep=line_sep
        )
        if not ocr_by_slide:
            return []

        entries = []
        for slide_idx, end_time in enumerate(end_times):
            ocr_text = ocr_by_slide.get(slide_idx)
            if not ocr_text:
                continue
            start_time = 0.0 if slide_idx == 0 else end_times[slide_idx - 1]
            entries.append(
                {
                    "segment_id": slide_idx + 1,
                    "slide_index": slide_idx,
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                    "text": ocr_text,
                    "lecture_key": lecture.key,
                    "speaker": lecture.speaker,
                    "course_dir": lecture.course_dir,
                    "meeting_id": lecture.meeting_id,
                    "video_id": lecture.video_id,
                    "modality": "ocr",
                }
            )
        return entries
    
    @staticmethod
    def dfs(node, entries, all_nodes, embedder, depth=0, normalize_embeddings=True):
        if OllamaResponder is None:
            raise RuntimeError(
                "Summary tree features require optional dependency 'ollama'."
            )
        if node is None:
            return ""

        node.depth = depth
    
        segment_utts = [entries[i] for i in node.segment]
        node.start = segment_utts[0].get("start")
        node.end = segment_utts[-1].get("end")
        node.utterance_start = node.segment[0]
        node.utterance_end = node.segment[-1]
        node.n_utterances = len(node.segment)
        node.raw_text = LectureSegmentBuilder._compose_segment_text(segment_utts)

        if node.left is None and node.right is None:
            node.is_leaf = True

            logger.info(
                "Processing leaf node",
                identifier=node.identifier,
                depth=depth,
                n_utterances=len(node.segment)
            )

            node.summary = None
            node.embedding = LectureSegmentBuilder._encode_text(
                embedder,
                node.raw_text,
                normalize_embeddings=normalize_embeddings,
            )
            node.descendant_leaf_tree_paths = [node.identifier]
            all_nodes.append(node)
            return node.raw_text
        
        else:
            node.is_leaf = False

            left_content = LectureSegmentBuilder.dfs(
                node.left,
                entries,
                all_nodes,
                embedder,
                depth + 1,
                normalize_embeddings=normalize_embeddings,
            )

            right_content = LectureSegmentBuilder.dfs(
                node.right,
                entries,
                all_nodes,
                embedder,
                depth + 1,
                normalize_embeddings=normalize_embeddings,
            )

            combined = LectureSegmentBuilder._build_internal_summary_input(
                left_content, right_content
            )

            logger.info(
                "Summarising internal node",
                identifier=node.identifier,
                depth=depth,
                n_utterances=len(node.segment)
            )

            node.summary = OllamaResponder.generate_summary(combined, is_leaf=False)

            node.embedding = LectureSegmentBuilder._encode_text(
                embedder,
                node.summary,
                normalize_embeddings=normalize_embeddings,
            )
            node.descendant_leaf_tree_paths = []
            for child in (node.left, node.right):
                child_leaf_paths = getattr(child, "descendant_leaf_tree_paths", [])
                node.descendant_leaf_tree_paths.extend(child_leaf_paths)

            all_nodes.append(node)

            return node.summary


    @staticmethod
    def build_summary_tree_for_lecture(
        lecture,
        utterances,
        treeseg_config,
        target_segments=None,
        include_ocr=False,
        normalize_embeddings=True,
    ):
        entries = LectureSegmentBuilder.build_treeseg_entries(
            utterances, include_ocr=include_ocr
        )

        if not entries:
            return None, []

        model = TreeSeg(configs=treeseg_config, entries=list(entries))
        k = float("inf") if target_segments is None else target_segments
        model.segment_meeting(K=k)

        root = model.root

        embedder = model.embedder

        all_nodes = []
        LectureSegmentBuilder.dfs(
            root,
            entries,
            all_nodes,
            embedder,
            depth=0,
            normalize_embeddings=normalize_embeddings,
        )

        logger.info(
            "Summary tree built",
            lecture_key=lecture.key,
            total_nodes=len(all_nodes),
            leaf_nodes=sum(1 for n in all_nodes if n.is_leaf)
        )

        return root, all_nodes

    @staticmethod
    def build_summary_tree_index_records_for_lecture(
        lecture,
        utterances,
        treeseg_config,
        target_segments=None,
        include_ocr=True,
        normalize_embeddings=True,
    ):
        _, all_nodes = LectureSegmentBuilder.build_summary_tree_for_lecture(
            lecture=lecture,
            utterances=utterances,
            treeseg_config=treeseg_config,
            target_segments=target_segments,
            include_ocr=include_ocr,
            normalize_embeddings=normalize_embeddings,
        )
        if not all_nodes:
            return []

        leaf_segment_ids = {}
        next_segment_id = 1
        for node in all_nodes:
            if not getattr(node, "is_leaf", False):
                continue
            leaf_segment_ids[node.identifier] = next_segment_id
            next_segment_id += 1

        records = []
        for node in all_nodes:
            node_id = getattr(node, "identifier", None)
            is_leaf = bool(getattr(node, "is_leaf", False))
            record = {
                "index_kind": "summary_tree",
                "node_id": node_id,
                "tree_path": node_id,
                "segment_id": leaf_segment_ids.get(node_id) if is_leaf else None,
                "is_leaf": is_leaf,
                "depth": getattr(node, "depth", None),
                "utterance_start": getattr(node, "utterance_start", None),
                "utterance_end": getattr(node, "utterance_end", None),
                "n_utterances": getattr(node, "n_utterances", 0),
                "start": getattr(node, "start", None),
                "end": getattr(node, "end", None),
                "text": getattr(node, "raw_text", None) if is_leaf else getattr(node, "summary", None),
                "summary_text": getattr(node, "summary", None),
                "raw_text": getattr(node, "raw_text", None),
                "descendant_leaf_tree_paths": list(
                    getattr(node, "descendant_leaf_tree_paths", [node_id] if is_leaf else [])
                ),
                "lecture_key": lecture.key,
                "speaker": lecture.speaker,
                "course_dir": lecture.course_dir,
                "meeting_id": lecture.meeting_id,
                "video_id": lecture.video_id,
                "embedding": getattr(node, "embedding", None),
            }
            records.append(record)
        return records

    @staticmethod
    def build_segments_for_lecture(
        lecture,
        utterances,
        treeseg_config,
        target_segments=None,
        include_ocr=True,
        ocr_prefix="[SLIDE] ",
    ):
        entries = LectureSegmentBuilder.build_treeseg_entries(
            utterances, include_ocr=include_ocr, ocr_prefix=ocr_prefix
        )
        if not entries:
            return []

        model = TreeSeg(configs=treeseg_config, entries=list(entries))
        k = float("inf") if target_segments is None else target_segments
        model.segment_meeting(K=k)

        segments = []
        for seg_idx, leaf in enumerate(model.leaves, start=1):
            indices = leaf.segment
            segment_utts = [entries[i] for i in indices]
            start_time = segment_utts[0].get("start")
            end_time = segment_utts[-1].get("end")
            text = "\n".join(utt.get("composite", "") for utt in segment_utts).strip()
            if not text:
                text = "<blank>"

            segments.append(
                {
                    "index_kind": "leaf",
                    "node_id": leaf.identifier,
                    "segment_id": seg_idx,
                    "tree_path": leaf.identifier,
                    "is_leaf": True,
                    "depth": None,
                    "utterance_start": indices[0],
                    "utterance_end": indices[-1],
                    "n_utterances": len(indices),
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "lecture_key": lecture.key,
                    "speaker": lecture.speaker,
                    "course_dir": lecture.course_dir,
                    "meeting_id": lecture.meeting_id,
                    "video_id": lecture.video_id,
                }
            )
        return segments
