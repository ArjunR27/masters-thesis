import structlog

from .lecture_segment_builder import LectureSegmentBuilder
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
        max_gap_s=0.8,
        lowercase=True,
        attach_ocr=True,
        include_ocr_in_treeseg=None,
        ocr_min_conf=60.0,
        ocr_per_slide=1,
        target_segments=None,
    ):
        store = LpmVectorIndex(
            model_name=embed_model,
            device=device,
            normalize=normalize,
            build_global=build_global,
        )

        if include_ocr_in_treeseg is None:
            include_ocr_in_treeseg = attach_ocr

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
