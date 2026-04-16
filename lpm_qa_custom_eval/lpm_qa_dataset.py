from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")


def _parse_answer_ranges(raw: str) -> tuple[tuple[float, float], ...]:
    """Parse 'start-end | start-end ...' into a tuple of (float, float) pairs."""
    ranges = []
    for match in _RANGE_RE.finditer(raw):
        start = float(match.group(1))
        end = float(match.group(2))
        if end >= start:
            ranges.append((start, end))
    return tuple(ranges)


@dataclass(frozen=True)
class LpmQaExample:
    example_id: str
    question: str
    lecture_key: str
    answer_ranges: tuple[tuple[float, float], ...]
    answer_text: str
    question_type: str


def load_lpm_qa_examples(
    csv_path: Path,
    limit: int | None = None,
) -> list[LpmQaExample]:
    examples: list[LpmQaExample] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader, start=1):
            if limit is not None and len(examples) >= limit:
                break
            raw_ts = (row.get("answer_timestamps") or "").strip()
            ranges = _parse_answer_ranges(raw_ts)
            if not ranges:
                continue
            examples.append(
                LpmQaExample(
                    example_id=f"lpm_qa-{row_index:04d}",
                    question=(row.get("question") or "").strip(),
                    lecture_key=(row.get("lecture_key") or "").strip(),
                    answer_ranges=ranges,
                    answer_text=(row.get("answer_text") or "").strip(),
                    question_type=(row.get("question_type") or "").strip(),
                )
            )
    return examples
