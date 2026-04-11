from concurrent.futures import ThreadPoolExecutor, as_completed

import structlog

from .lecture_segment_builder import LectureSegmentBuilder, SummaryTreeBuildOptions
from .lpm_vector_index import LpmVectorIndex


class VectorStoreFactory:
    def __init__(self, logger=None, segment_builder=None):
        self.logger = logger or structlog.get_logger(__name__)
        self.segment_builder = segment_builder or LectureSegmentBuilder

    def build_vector_store(
        self,
        lectures,
        treeseg_config,
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        device=None,
        normalize=True,
        build_global=True,
        max_gap_s="auto",
        lowercase=True,
        attach_ocr=True,
        include_ocr_in_treeseg=None,
        ocr_min_conf=60.0,
        ocr_per_slide=1,
        target_segments=None,
        index_kind="leaf",
        summary_tree_build_options=None,
    ):
        store = LpmVectorIndex(
            model_name=embed_model,
            device=device,
            normalize=normalize,
            build_global=build_global,
        )

        if include_ocr_in_treeseg is None:
            include_ocr_in_treeseg = attach_ocr

        if index_kind == "summary_tree":
            build_options = summary_tree_build_options or SummaryTreeBuildOptions()
            self.logger.info(
                "Summary tree build configuration",
                workers=build_options.workers,
                cache_dir=build_options.cache_dir,
                rebuild_cache=build_options.rebuild_cache,
            )
            segments_by_lecture = self._build_summary_tree_segments(
                lectures=lectures,
                treeseg_config=treeseg_config,
                max_gap_s=max_gap_s,
                lowercase=lowercase,
                attach_ocr=attach_ocr,
                ocr_min_conf=ocr_min_conf,
                ocr_per_slide=ocr_per_slide,
                target_segments=target_segments,
                include_ocr_in_treeseg=include_ocr_in_treeseg,
                normalize=normalize,
                build_options=build_options,
            )

            for lecture in lectures:
                segments = segments_by_lecture.get(lecture.key) or []
                if not segments:
                    self.logger.warning("No segments built", lecture=lecture.key)
                    continue
                self.logger.info("Adding lecture to vector store", lecture=lecture.key)
                store.add_lecture(lecture, segments)
        else:
            for lecture in lectures:
                self.logger.info("Indexing lecture", lecture=lecture.key)
                utterances = self.segment_builder.load_lecture_utterances(
                    lecture,
                    max_gap_s=max_gap_s,
                    lowercase=lowercase,
                    attach_ocr=attach_ocr,
                    ocr_min_conf=ocr_min_conf,
                    ocr_per_slide=ocr_per_slide,
                )
                if not utterances:
                    self.logger.warning("No utterances found", lecture=lecture.key)
                    continue
                segments = self.segment_builder.build_segments_for_lecture(
                    lecture,
                    utterances,
                    treeseg_config=treeseg_config,
                    target_segments=target_segments,
                    include_ocr=include_ocr_in_treeseg,
                )
                if not segments:
                    self.logger.warning("No segments built", lecture=lecture.key)
                    continue
                store.add_lecture(lecture, segments)

        store.finalize()
        return store

    def _build_summary_tree_segments(
        self,
        *,
        lectures,
        treeseg_config,
        max_gap_s,
        lowercase,
        attach_ocr,
        ocr_min_conf,
        ocr_per_slide,
        target_segments,
        include_ocr_in_treeseg,
        normalize,
        build_options,
    ):
        shared_tree_embedder = None
        if build_options.workers > 1 and len(lectures) > 1:
            shared_tree_embedder = self.segment_builder.build_shared_summary_tree_embedder(
                treeseg_config,
                build_options=build_options,
            )

        if build_options.workers <= 1 or len(lectures) <= 1:
            return {
                lecture.key: self.segment_builder.build_or_load_summary_tree_index_records_for_lecture(
                    lecture,
                    treeseg_config=treeseg_config,
                    max_gap_s=max_gap_s,
                    lowercase=lowercase,
                    attach_ocr=attach_ocr,
                    ocr_min_conf=ocr_min_conf,
                    ocr_per_slide=ocr_per_slide,
                    target_segments=target_segments,
                    include_ocr=include_ocr_in_treeseg,
                    normalize_embeddings=normalize,
                    build_options=build_options,
                    tree_embedder=shared_tree_embedder,
                )
                for lecture in lectures
            }

        segments_by_lecture = {}
        with ThreadPoolExecutor(
            max_workers=build_options.workers,
            thread_name_prefix="summary-tree",
        ) as executor:
            future_to_lecture = {
                executor.submit(
                    self.segment_builder.build_or_load_summary_tree_index_records_for_lecture,
                    lecture,
                    treeseg_config=treeseg_config,
                    max_gap_s=max_gap_s,
                    lowercase=lowercase,
                    attach_ocr=attach_ocr,
                    ocr_min_conf=ocr_min_conf,
                    ocr_per_slide=ocr_per_slide,
                    target_segments=target_segments,
                    include_ocr=include_ocr_in_treeseg,
                    normalize_embeddings=normalize,
                    build_options=build_options,
                    tree_embedder=shared_tree_embedder,
                ): lecture
                for lecture in lectures
            }
            for future in as_completed(future_to_lecture):
                lecture = future_to_lecture[future]
                segments_by_lecture[lecture.key] = future.result()
        return segments_by_lecture

    def build_ocr_vector_store(
        self,
        lectures,
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        device=None,
        normalize=True,
        build_global=True,
        ocr_min_conf=60.0,
        ocr_line_sep="\n",
    ):
        store = LpmVectorIndex(
            model_name=embed_model,
            device=device,
            normalize=normalize,
            build_global=build_global,
        )

        for lecture in lectures:
            self.logger.info("Indexing OCR slides", lecture=lecture.key)
            slides = self.segment_builder.build_ocr_slide_entries(
                lecture, ocr_min_conf=ocr_min_conf, line_sep=ocr_line_sep
            )
            if not slides:
                self.logger.warning("No OCR slides found", lecture=lecture.key)
                continue
            store.add_lecture(lecture, slides)

        store.finalize()
        return store
