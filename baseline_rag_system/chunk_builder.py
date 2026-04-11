from __future__ import annotations

import os
from typing import Any

from utterances import iter_rows, load_slide_end_times, load_slide_ocr_texts

from treeseg_vector_index_modular.constants import SLIDE_TOKEN
from treeseg_vector_index_modular.lecture_segment_builder import LectureSegmentBuilder

from .tokenization import decode_tokens, encode_text
from .types import BaselineChunkBuildResult, BaselineChunkRecord, BaselineRagConfig


def _clean_text(text: str | None) -> str:
    return " ".join((text or "").split()).strip()


def _format_ocr_block(ocr_text: str) -> str:
    lines = []
    for raw_line in (ocr_text or "").splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            continue
        lines.append(f"{SLIDE_TOKEN} {line}")
    return "\n".join(lines).strip()


def _compose_chunk_text(spoken_text: str, ocr_text: str = "") -> str:
    spoken_text = _clean_text(spoken_text)
    formatted_ocr = _format_ocr_block(ocr_text)
    parts = []
    if spoken_text:
        parts.append(spoken_text)
    if formatted_ocr:
        parts.append(formatted_ocr)
    return "\n".join(parts).strip() or "<blank>"


def _unique_preserving_order(values: list[int]) -> tuple[int, ...]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _split_unit_if_needed(
    unit: dict[str, Any],
    *,
    chunk_size_tokens: int,
    model_name: str,
) -> list[dict[str, Any]]:
    token_ids = list(unit["token_ids"])
    if len(token_ids) <= chunk_size_tokens:
        return [unit]

    split_units = []
    for offset in range(0, len(token_ids), chunk_size_tokens):
        part_ids = token_ids[offset : offset + chunk_size_tokens]
        part_text = decode_tokens(model_name, part_ids) or unit["text"]
        split_units.append(
            {
                **unit,
                "text": part_text,
                "token_ids": part_ids,
                "token_count": len(part_ids),
            }
        )
    return split_units


def _build_chunk_payload(
    lecture,
    units: list[dict[str, Any]],
    config: BaselineRagConfig,
    *,
    chunk_id: int,
) -> dict[str, object]:
    chunk_text = "\n".join(unit["text"] for unit in units if unit["text"]).strip()
    if not chunk_text:
        chunk_text = "<blank>"

    slide_indices = []
    for unit in units:
        slide_indices.extend(unit.get("slide_indices", ()))

    record = BaselineChunkRecord(
        chunk_id=chunk_id,
        text=chunk_text,
        start=units[0].get("start"),
        end=units[-1].get("end"),
        lecture_key=lecture.key,
        speaker=lecture.speaker,
        course_dir=lecture.course_dir,
        meeting_id=lecture.meeting_id,
        video_id=lecture.video_id,
        chunk_strategy=config.chunk_strategy,
        chunk_size_tokens=config.chunk_size_tokens,
        overlap_tokens=config.overlap_tokens,
        ocr_mode=config.ocr_mode,
        token_count=sum(int(unit["token_count"]) for unit in units),
        source_item_start=units[0].get("source_item_start"),
        source_item_end=units[-1].get("source_item_end"),
        slide_indices=_unique_preserving_order(slide_indices),
    )
    return record.to_payload()


def pack_units_into_chunks(
    lecture,
    units: list[dict[str, Any]],
    config: BaselineRagConfig,
) -> list[dict[str, object]]:
    normalized_units = []
    for unit in units:
        normalized_units.extend(
            _split_unit_if_needed(
                unit,
                chunk_size_tokens=config.chunk_size_tokens,
                model_name=config.embedding_model,
            )
        )

    if not normalized_units:
        return []

    chunk_records = []
    start_idx = 0
    chunk_id = 1
    while start_idx < len(normalized_units):
        end_idx = start_idx
        token_total = 0
        selected_units = []
        while end_idx < len(normalized_units):
            candidate = normalized_units[end_idx]
            candidate_tokens = int(candidate["token_count"])
            if token_total > 0 and token_total + candidate_tokens > config.chunk_size_tokens:
                break
            selected_units.append(candidate)
            token_total += candidate_tokens
            end_idx += 1

        if not selected_units:
            selected_units.append(normalized_units[start_idx])
            end_idx = start_idx + 1

        chunk_records.append(
            _build_chunk_payload(
                lecture,
                selected_units,
                config,
                chunk_id=chunk_id,
            )
        )
        chunk_id += 1

        if end_idx >= len(normalized_units):
            break

        next_start = end_idx
        if config.overlap_tokens > 0:
            overlap_total = 0
            while next_start > start_idx and overlap_total < config.overlap_tokens:
                next_start -= 1
                overlap_total += int(normalized_units[next_start]["token_count"])
        if next_start <= start_idx:
            next_start = start_idx + 1
        start_idx = next_start

    return chunk_records


def build_slide_entries_for_lecture(
    lecture,
    *,
    ocr_min_conf: float,
    line_sep: str = "\n",
) -> list[dict[str, object]]:
    segments_path = os.path.join(lecture.meeting_dir, "segments.txt")
    slides_dir = lecture.meeting_dir
    if not os.path.exists(segments_path):
        return []

    end_times = load_slide_end_times(segments_path)
    if not end_times:
        return []

    ocr_by_slide = load_slide_ocr_texts(
        slides_dir,
        min_conf=ocr_min_conf,
        line_sep=line_sep,
    )
    if not ocr_by_slide:
        return []

    slide_entries = []
    for slide_idx, end_time in enumerate(end_times):
        slide_text = ocr_by_slide.get(slide_idx)
        if not slide_text:
            continue
        start_time = 0.0 if slide_idx == 0 else end_times[slide_idx - 1]
        slide_entries.append(
            {
                "slide_index": slide_idx,
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "text": slide_text,
            }
        )
    return slide_entries


def _collect_overlapping_slide_entries(
    slide_entries: list[dict[str, object]],
    *,
    start: float | None,
    end: float | None,
) -> list[dict[str, object]]:
    if start is None or end is None:
        return []
    matches = []
    for entry in slide_entries:
        slide_start = float(entry["start"])
        slide_end = float(entry["end"])
        if slide_start <= end and slide_end >= start:
            matches.append(entry)
    return matches


class BaselineChunkBuilder:
    @staticmethod
    def build_chunks_for_lecture(
        lecture,
        config: BaselineRagConfig,
    ) -> BaselineChunkBuildResult:
        slide_entries = []
        if config.attach_ocr:
            slide_entries = build_slide_entries_for_lecture(
                lecture,
                ocr_min_conf=config.ocr_min_conf,
            )
            if not slide_entries:
                return BaselineChunkBuildResult(
                    records=[],
                    skip_reason="missing_ocr_assets",
                )

        if config.chunk_strategy == "utterance_packed":
            records = BaselineChunkBuilder.build_utterance_packed_chunks_for_lecture(
                lecture,
                config,
            )
        else:
            records = BaselineChunkBuilder.build_raw_token_window_chunks_for_lecture(
                lecture,
                config,
                slide_entries=slide_entries,
            )
        return BaselineChunkBuildResult(records=records, skip_reason=None)

    @staticmethod
    def build_utterance_packed_chunks_for_lecture(
        lecture,
        config: BaselineRagConfig,
    ) -> list[dict[str, object]]:
        utterances = LectureSegmentBuilder.load_lecture_utterances(
            lecture,
            max_gap_s=config.max_gap_s,
            lowercase=config.lowercase,
            attach_ocr=config.attach_ocr,
            ocr_min_conf=config.ocr_min_conf,
            ocr_per_slide=config.ocr_per_slide,
        )
        return BaselineChunkBuilder.build_utterance_packed_chunks(
            lecture,
            utterances,
            config,
        )

    @staticmethod
    def build_utterance_packed_chunks(
        lecture,
        utterances: list[dict[str, object]],
        config: BaselineRagConfig,
    ) -> list[dict[str, object]]:
        units = []
        for idx, utterance in enumerate(utterances):
            spoken_text = _clean_text(str(utterance.get("text") or ""))
            ocr_text = str(utterance.get("ocr_text") or "") if config.attach_ocr else ""
            text = _compose_chunk_text(spoken_text, ocr_text)
            token_ids = encode_text(config.embedding_model, text)
            if not token_ids:
                continue
            slide_indices = []
            if config.attach_ocr and utterance.get("ocr_text") and utterance.get("slide_index") is not None:
                slide_indices.append(int(utterance["slide_index"]))
            units.append(
                {
                    "text": text,
                    "token_ids": token_ids,
                    "token_count": len(token_ids),
                    "start": utterance.get("start"),
                    "end": utterance.get("end"),
                    "source_item_start": idx,
                    "source_item_end": idx,
                    "slide_indices": tuple(slide_indices),
                }
            )
        return pack_units_into_chunks(lecture, units, config)

    @staticmethod
    def build_raw_token_window_chunks_for_lecture(
        lecture,
        config: BaselineRagConfig,
        *,
        slide_entries: list[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        word_rows = list(iter_rows(lecture.transcripts_path))
        return BaselineChunkBuilder.build_raw_token_window_chunks(
            lecture,
            word_rows,
            config,
            slide_entries=slide_entries or [],
        )

    @staticmethod
    def build_raw_token_window_chunks(
        lecture,
        word_rows: list[dict[str, object]],
        config: BaselineRagConfig,
        *,
        slide_entries: list[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        words = []
        all_token_ids = []
        token_to_word_index = []

        for row in word_rows:
            text = _clean_text(str(row.get("Word") or ""))
            if not text:
                continue
            if config.lowercase:
                text = text.lower()
            try:
                start = float(row.get("Start") or 0.0)
            except (TypeError, ValueError):
                start = 0.0
            try:
                end = float(row.get("End") or start)
            except (TypeError, ValueError):
                end = start
            token_ids = encode_text(config.embedding_model, text)
            if not token_ids:
                continue
            word_index = len(words)
            words.append({"text": text, "start": start, "end": end})
            all_token_ids.extend(token_ids)
            token_to_word_index.extend([word_index] * len(token_ids))

        if not all_token_ids:
            return []

        chunk_records = []
        step = max(1, config.chunk_size_tokens - config.overlap_tokens)
        chunk_id = 1
        for window_start in range(0, len(all_token_ids), step):
            window_end = min(window_start + config.chunk_size_tokens, len(all_token_ids))
            token_ids = all_token_ids[window_start:window_end]
            if not token_ids:
                break

            first_word_idx = token_to_word_index[window_start]
            last_word_idx = token_to_word_index[window_end - 1]
            start = float(words[first_word_idx]["start"])
            end = float(words[last_word_idx]["end"])
            spoken_text = decode_tokens(config.embedding_model, token_ids)

            chunk_slide_entries = []
            if config.attach_ocr:
                chunk_slide_entries = _collect_overlapping_slide_entries(
                    slide_entries or [],
                    start=start,
                    end=end,
                )
            ocr_text = "\n".join(
                str(entry["text"])
                for entry in chunk_slide_entries
                if entry.get("text")
            )
            chunk_text = _compose_chunk_text(spoken_text, ocr_text)
            slide_indices = tuple(
                int(entry["slide_index"])
                for entry in chunk_slide_entries
                if entry.get("slide_index") is not None
            )

            chunk_records.append(
                BaselineChunkRecord(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    start=round(start, 3),
                    end=round(end, 3),
                    lecture_key=lecture.key,
                    speaker=lecture.speaker,
                    course_dir=lecture.course_dir,
                    meeting_id=lecture.meeting_id,
                    video_id=lecture.video_id,
                    chunk_strategy=config.chunk_strategy,
                    chunk_size_tokens=config.chunk_size_tokens,
                    overlap_tokens=config.overlap_tokens,
                    ocr_mode=config.ocr_mode,
                    token_count=len(token_ids),
                    source_item_start=first_word_idx,
                    source_item_end=last_word_idx,
                    slide_indices=slide_indices,
                ).to_payload()
            )
            chunk_id += 1

            if window_end >= len(all_token_ids):
                break

        return chunk_records
