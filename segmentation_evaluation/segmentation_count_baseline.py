from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "segment_count_baseline_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))

import matplotlib
import numpy as np
from collections import defaultdict

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
RETRIEVER_EVAL_DIR = HERE.parent
PROJECT_DIR = HERE.parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from treeseg_vector_index_modular.lecture_catalog import LectureCatalog
from treeseg_vector_index_modular.lecture_segment_builder import LectureSegmentBuilder
from treeseg_vector_index_modular.lpm_config_builder import LpmConfigBuilder


DEFAULT_OUT_DIR = RETRIEVER_EVAL_DIR / "segment_count_baseline"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Natural-K TreeSeg vs slide-count baseline. "
            "Counts TreeSeg segments and compares against segments.txt slide counts."
        )
    )
    parser.add_argument("--speaker", default=None, help="Filter speaker (optional).")
    parser.add_argument("--course-dir", default=None, help="Filter course dir (optional).")
    parser.add_argument(
        "--meeting-ids",
        default=None,
        help="Comma-separated meeting ids (optional), e.g. 01,02,03.",
    )
    parser.add_argument(
        "--max-lectures",
        type=int,
        default=None,
        help="Limit number of discovered lectures (optional).",
    )
    parser.add_argument(
        "--no-lowercase",
        action="store_true",
        help="Disable lowercasing of transcript text.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF embedding model for TreeSeg.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Force embedding device (default: auto-resolve).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    return parser.parse_args()


def parse_meeting_ids(raw):
    if not raw:
        return None
    values = [part.strip() for part in str(raw).split(",")]
    values = [v for v in values if v]
    return values or None


def parse_numeric_slide_boundaries(segments_path: Path):
    values = []
    invalid_lines = 0

    with segments_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                values.append(float(raw))
            except ValueError:
                invalid_lines += 1

    return values, invalid_lines


def slugify(value: str):
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "unknown"


def evaluate_lecture(
    lecture,
    treeseg_config,
    lowercase: bool,
):
    segments_path = Path(lecture.meeting_dir) / "segments.txt"
    if not segments_path.exists():
        return None, {"lecture_key": lecture.key, "reason": "missing_segments_txt"}

    slide_end_times, invalid_lines = parse_numeric_slide_boundaries(segments_path)
    if not slide_end_times:
        return None, {"lecture_key": lecture.key, "reason": "no_numeric_slide_boundaries"}

    utterances = LectureSegmentBuilder.load_lecture_utterances(
        lecture=lecture,
        lowercase=lowercase,
        attach_ocr=False,
    )
    if not utterances:
        return None, {"lecture_key": lecture.key, "reason": "no_utterances"}

    segments = LectureSegmentBuilder.build_segments_for_lecture(
        lecture,
        utterances,
        treeseg_config=treeseg_config,
        target_segments=None,
        include_ocr=False,
    )
    if not segments:
        return None, {"lecture_key": lecture.key, "reason": "treeseg_returned_no_segments"}

    n_slides = len(slide_end_times)
    n_treeseg_segments = len(segments)
    diff = n_treeseg_segments - n_slides
    abs_diff = abs(diff)
    ratio = (n_treeseg_segments / float(n_slides)) if n_slides else np.nan
    max_gap_s_used = np.nan
    raw_gap = utterances[0].get("max_gap_s") if utterances else None
    if raw_gap is not None:
        try:
            max_gap_s_used = float(raw_gap)
        except (TypeError, ValueError):
            max_gap_s_used = np.nan

    row = {
        "lecture_key": lecture.key,
        "speaker": lecture.speaker,
        "course_dir": lecture.course_dir,
        "meeting_id": lecture.meeting_id,
        "n_slides": int(n_slides),
        "n_treeseg_segments": int(n_treeseg_segments),
        "diff": int(diff),
        "abs_diff": int(abs_diff),
        "ratio": float(ratio) if not np.isnan(ratio) else np.nan,
        "max_gap_s_used": float(max_gap_s_used) if not np.isnan(max_gap_s_used) else np.nan,
        "invalid_segments_lines": int(invalid_lines),
    }
    return row, None


def mean_or_nan(values):
    if not values:
        return float("nan")
    return float(statistics.mean(values))


def median_or_nan(values):
    if not values:
        return float("nan")
    return float(statistics.median(values))


def build_summary(rows, total_candidates: int, skipped_rows):
    n_slides_values = [r["n_slides"] for r in rows]
    n_treeseg_values = [r["n_treeseg_segments"] for r in rows]
    diff_values = [r["diff"] for r in rows]
    abs_diff_values = [r["abs_diff"] for r in rows]
    ratio_values = [r["ratio"] for r in rows if not np.isnan(r["ratio"])]
    max_gap_values = [r["max_gap_s_used"] for r in rows if not np.isnan(r["max_gap_s_used"])]
    ratio_count = len(ratio_values)

    low_count = sum(1 for r in ratio_values if r < 0.8)
    mid_count = sum(1 for r in ratio_values if 0.8 <= r <= 1.2)
    high_count = sum(1 for r in ratio_values if r > 1.2)

    def pct(count):
        if ratio_count == 0:
            return 0.0
        return (100.0 * count) / float(ratio_count)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_candidates": int(total_candidates),
        "evaluated_lectures": int(len(rows)),
        "skipped_lectures": int(len(skipped_rows)),
        "stats": {
            "n_slides_mean": mean_or_nan(n_slides_values),
            "n_slides_median": median_or_nan(n_slides_values),
            "n_treeseg_segments_mean": mean_or_nan(n_treeseg_values),
            "n_treeseg_segments_median": median_or_nan(n_treeseg_values),
            "diff_mean": mean_or_nan(diff_values),
            "diff_median": median_or_nan(diff_values),
            "abs_diff_mean": mean_or_nan(abs_diff_values),
            "abs_diff_median": median_or_nan(abs_diff_values),
            "ratio_mean": mean_or_nan(ratio_values),
            "ratio_median": median_or_nan(ratio_values),
            "max_gap_s_used_mean": mean_or_nan(max_gap_values),
            "max_gap_s_used_median": median_or_nan(max_gap_values),
        },
        "ratio_bands": {
            "ratio_lt_0_8": {"count": low_count, "percent": pct(low_count)},
            "ratio_0_8_to_1_2": {"count": mid_count, "percent": pct(mid_count)},
            "ratio_gt_1_2": {"count": high_count, "percent": pct(high_count)},
        },
        "skip_reasons": {},
    }

    for row in skipped_rows:
        reason = row.get("reason", "unknown")
        summary["skip_reasons"][reason] = summary["skip_reasons"].get(reason, 0) + 1

    return summary


def plot_grouped_bars(rows, out_path: Path, title: str):
    fig_width = min(36, max(14, int(0.24 * len(rows)) + 10))
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    x = np.arange(len(rows))
    width = 0.42

    lecture_keys = [r["lecture_key"] for r in rows]
    slide_counts = [r["n_slides"] for r in rows]
    treeseg_counts = [r["n_treeseg_segments"] for r in rows]

    ax.bar(x - width / 2, slide_counts, width=width, label="Slides", color="#4c78a8")
    ax.bar(
        x + width / 2,
        treeseg_counts,
        width=width,
        label="TreeSeg (Natural K)",
        color="#f58518",
    )

    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.legend()

    if len(rows) <= 60:
        ax.set_xticks(x)
        ax.set_xticklabels(lecture_keys, rotation=90, fontsize=7)
    else:
        step = max(1, len(rows) // 25)
        ticks = np.arange(0, len(rows), step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([lecture_keys[i] for i in ticks], rotation=60, fontsize=7)
        ax.set_xlabel("Lecture (subset of labels shown due to density)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def maybe_plot_by_course(rows, out_dir: Path):
    if len(rows) <= 60:
        return

    per_course_dir = out_dir / "by_course"
    per_course_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["course_dir"]].append(row)

    for course_dir in sorted(grouped):
        ordered = sorted(
            grouped[course_dir],
            key=lambda row: (row["n_slides"], row["lecture_key"]),
        )
        if not ordered:
            continue
        chart_path = per_course_dir / f"slides_vs_treeseg_{slugify(course_dir)}.png"
        plot_grouped_bars(
            ordered,
            chart_path,
            f"Slides vs TreeSeg Segments (Natural K) - {course_dir}",
        )


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lowercase = not args.no_lowercase
    meeting_ids = parse_meeting_ids(args.meeting_ids)

    lectures = LectureCatalog.discover_lectures(
        data_dir=PROJECT_DIR / "lpm_data",
        speaker=args.speaker,
        course_dir=args.course_dir,
        meeting_ids=meeting_ids,
    )
    lectures = sorted(lectures, key=lambda lec: lec.key)
    total_candidates = len(lectures)

    if args.max_lectures is not None:
        lectures = lectures[: max(0, args.max_lectures)]

    treeseg_config = LpmConfigBuilder.build_lpm_config(
        embedding_model=args.embedding_model,
        device=args.device,
    )

    rows = []
    skipped = []
    for idx, lecture in enumerate(lectures, start=1):
        try:
            row, skip = evaluate_lecture(
                lecture=lecture,
                treeseg_config=treeseg_config,
                lowercase=lowercase,
            )
        except Exception as exc:
            row = None
            skip = {
                "lecture_key": lecture.key,
                "reason": f"exception:{type(exc).__name__}",
                "detail": str(exc),
            }

        if row is not None:
            rows.append(row)
            print(
                f"[{idx}/{len(lectures)}] OK {lecture.key} | "
                f"slides={row['n_slides']} treeseg={row['n_treeseg_segments']} "
                f"max_gap_s={row['max_gap_s_used']:.3f}"
            )
        else:
            skipped.append(skip)
            print(
                f"[{idx}/{len(lectures)}] SKIP {lecture.key} | "
                f"reason={skip.get('reason', 'unknown')}"
            )

    if not rows:
        raise RuntimeError("No lectures were successfully evaluated.")

    rows = sorted(rows, key=lambda row: (row["n_slides"], row["lecture_key"]))

    csv_path = out_dir / "lecture_counts.csv"
    fieldnames = [
        "lecture_key",
        "speaker",
        "course_dir",
        "meeting_id",
        "n_slides",
        "n_treeseg_segments",
        "diff",
        "abs_diff",
        "ratio",
        "max_gap_s_used",
        "invalid_segments_lines",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = build_summary(rows, total_candidates=total_candidates, skipped_rows=skipped)
    summary_path = out_dir / "summary_stats.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_path = out_dir / "slides_vs_treeseg_bar.png"
    plot_grouped_bars(
        rows,
        plot_path,
        "Natural-K TreeSeg Segments vs Slide Count per Lecture",
    )
    maybe_plot_by_course(rows, out_dir)

    skipped_path = out_dir / "skipped_lectures.json"
    skipped_path.write_text(json.dumps(skipped, indent=2), encoding="utf-8")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {skipped_path}")
    print(
        f"Evaluated {len(rows)} lecture(s), skipped {len(skipped)} "
        f"out of {len(lectures)} processed lecture(s)."
    )


if __name__ == "__main__":
    main()
