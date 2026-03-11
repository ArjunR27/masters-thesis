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
    
    def dfs(node, entries, all_nodes, embedder, depth=0):
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

        if node.left is None and node.right is None:
            node.is_leaf = True

            raw_text = "\n".join(
                utt.get("composite", "") for utt in segment_utts
            ).strip() or "<blank>"

            logger.info(
                "Summarising leaf node",
                identifier=node.identifier,
                depth=depth,
                n_utterances=len(node.segment)
            )

            node.summary = OllamaResponder.generate_summary(raw_text)
            all_nodes.append(node)
            return node.summary
        
        else:
            node.is_leaf = False

            left_summary = LectureSegmentBuilder.dfs(
                node.left, entries, all_nodes, embedder, depth + 1
            )

            right_summary = LectureSegmentBuilder.dfs(
                node.right, entries, all_nodes, embedder, depth + 1
            )

            child_summaries = []

            if left_summary:
                child_summaries.append(f"Left child summary: \n {left_summary}")
            if right_summary:
                child_summaries.append(f"Right child summary: \n {right_summary}")
            
            combined = "\n\n".join(child_summaries).strip() or "<blank>"

            logger.info(
                "Summarising internal node",
                identifier=node.identifier,
                depth=depth,
                n_utterances=len(node.segment)
            )

            node.summary = OllamaResponder.generate_summary(combined, is_leaf=False)

            # Embed the summary for flattneed vector search
            node.embedding = embedder.encode(
                node.summary,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            all_nodes.append(node)

            return node.summary


    @staticmethod
    def build_summary_tree_for_lecture(
        lecture,
        utterances,
        treeseg_config,
        target_segments=None,
        include_ocr=False,
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
        LectureSegmentBuilder.dfs(root, entries, all_nodes, embedder, depth=0)

        logger.info(
            "Summary tree built",
            lecture_key=lecture.key,
            total_nodes=len(all_nodes),
            leaf_nodes=sum(1 for n in all_nodes if n.is_leaf)
        )

        return root, all_nodes


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
                    "segment_id": seg_idx,
                    "tree_path": leaf.identifier,
                    "is_leaf": True,
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
