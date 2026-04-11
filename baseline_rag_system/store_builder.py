from __future__ import annotations

import structlog

from treeseg_vector_index_modular.lpm_vector_index import LpmVectorIndex

from .chunk_builder import BaselineChunkBuilder
from .types import BaselineRagConfig, BaselineStoreBuildResult


class BaselineStoreBuilder:
    def __init__(self, logger=None, chunk_builder=None):
        self.logger = logger or structlog.get_logger(__name__)
        self.chunk_builder = chunk_builder or BaselineChunkBuilder

    def build_store(
        self,
        lectures,
        config: BaselineRagConfig,
        *,
        device=None,
        build_global=False,
    ) -> BaselineStoreBuildResult:
        store = LpmVectorIndex(
            model_name=config.embedding_model,
            device=device,
            normalize=config.normalize_embeddings,
            build_global=build_global,
        )

        skipped_lectures = {}
        for lecture in lectures:
            self.logger.info(
                "Building baseline lecture chunks",
                lecture=lecture.key,
                chunk_strategy=config.chunk_strategy,
                chunk_size_tokens=config.chunk_size_tokens,
                overlap_tokens=config.overlap_tokens,
                ocr_mode=config.ocr_mode,
            )
            result = self.chunk_builder.build_chunks_for_lecture(lecture, config)
            if result.skip_reason:
                skipped_lectures[lecture.key] = result.skip_reason
                self.logger.warning(
                    "Skipping baseline lecture",
                    lecture=lecture.key,
                    reason=result.skip_reason,
                )
                continue
            if not result.records:
                skipped_lectures[lecture.key] = "no_chunks"
                self.logger.warning("No baseline chunks built", lecture=lecture.key)
                continue
            store.add_lecture(lecture, result.records)

        store.finalize()
        return BaselineStoreBuildResult(
            store=store,
            skipped_lectures=skipped_lectures,
        )
