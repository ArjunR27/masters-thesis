"""Evaluate TreeSeg segmentation quality using WinDiff and Pk metrics.

Reference boundaries are derived from slide transitions in segments.txt.
Baselines (random, fixed-window) use n_slides as K, independent of TreeSeg.
Lower WinDiff and Pk scores are better.
"""

import argparse
import bisect
import random
import sys
from pathlib import Path

import numpy as np
from tabulate import tabulate

HERE = Path(__file__).resolve()
PROJECT_DIR = HERE.parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(PROJECT_DIR / "treeseg_exploration") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "treeseg_exploration"))

from treeseg_vector_index_modular.lecture_catalog import LectureCatalog
from treeseg_vector_index_modular.lecture_segment_builder import LectureSegmentBuilder
from treeseg_vector_index_modular.lpm_config_builder import LpmConfigBuilder
from treeseg.treeseg import TreeSeg
from utterances import load_slide_end_times


# ---------------------------------------------------------------------------
# WinDiff and Pk (from treeseg_exploration/treeseg_evaluation.py)
# ---------------------------------------------------------------------------

def window_diff(ref, hyp, k=None):
    """WindowDiff metric. Lower is better."""
    if len(ref) != len(hyp):
        raise ValueError("ref and hyp must have the same length")
    ref = list(map(int, ref))
    hyp = list(map(int, hyp))
    if k is None:
        k = int(round(len(ref) / (sum(ref) + 1) / 2.0))
    k = max(1, min(k, len(ref) - 1))
    ref_cum = np.cumsum([0] + ref)
    hyp_cum = np.cumsum([0] + hyp)
    errors = sum(
        1 for i in range(len(ref) - k + 1)
        if (ref_cum[i + k] - ref_cum[i]) != (hyp_cum[i + k] - hyp_cum[i])
    )
    return errors / (len(ref) - k + 1)


def pk_metric(ref, hyp, k=None):
    """Pk metric. Lower is better."""
    if len(ref) != len(hyp):
        raise ValueError("ref and hyp must have the same length")
    ref = list(map(int, ref))
    hyp = list(map(int, hyp))
    if k is None:
        k = int(round(len(ref) / (sum(ref) + 1) / 2.0))
    k = max(1, min(k, len(ref) - 1))
    ref_cum = np.cumsum([0] + ref)
    hyp_cum = np.cumsum([0] + hyp)
    errors = sum(
        1 for i in range(len(ref) - k + 1)
        if ((ref_cum[i + k] - ref_cum[i]) == 0) != ((hyp_cum[i + k] - hyp_cum[i]) == 0)
    )
    return errors / (len(ref) - k + 1)


# ---------------------------------------------------------------------------
# Reference sequence from slide transitions
# ---------------------------------------------------------------------------

def load_slide_end_times_for_lecture(lecture):
    """Load slide end times from segments.txt. Returns list of floats or None."""
    path = Path(lecture.meeting_dir) / "segments.txt"
    if not path.exists():
        return None
    times = load_slide_end_times(str(path))
    return times if times else None


def build_reference_sequence(entries, slide_end_times):
    """Binary boundary sequence derived from slide transitions.

    boundary[i] = 1 if utterance i and i+1 belong to different slides, else 0.
    Last position is always 0 (no boundary after the final utterance).
    """
    # Assign each utterance to a slide by its start timestamp
    slide_assignments = []
    for entry in entries:
        t = entry.get("start", 0.0)
        idx = bisect.bisect_left(slide_end_times, t)
        idx = min(idx, len(slide_end_times) - 1)
        slide_assignments.append(idx)

    n = len(slide_assignments)
    ref = [0] * n
    for i in range(n - 1):
        if slide_assignments[i] != slide_assignments[i + 1]:
            ref[i] = 1
    return ref


# ---------------------------------------------------------------------------
# Predicted sequences
# ---------------------------------------------------------------------------

def build_treeseg_sequence(n_utterances, segment_groups):
    """Binary boundary sequence from TreeSeg leaf segments.

    Boundary placed at the last utterance index of each segment except the last.
    """
    hyp = [0] * n_utterances
    for group in segment_groups[:-1]:
        if group:
            hyp[group[-1]] = 1
    return hyp


def build_random_sequence(n_utterances, n_boundaries, rng):
    """Randomly place n_boundaries boundaries across n_utterances positions."""
    seq = [0] * n_utterances
    if n_boundaries <= 0 or n_utterances < 2:
        return seq
    positions = sorted(rng.sample(range(n_utterances - 1), min(n_boundaries, n_utterances - 1)))
    for p in positions:
        seq[p] = 1
    return seq


def build_fixed_window_sequence(n_utterances, n_boundaries):
    """Evenly spaced boundaries across utterances."""
    seq = [0] * n_utterances
    if n_boundaries <= 0 or n_utterances < 2:
        return seq
    step = n_utterances / (n_boundaries + 1)
    for i in range(1, n_boundaries + 1):
        pos = min(int(round(i * step)) - 1, n_utterances - 2)
        seq[pos] = 1
    return seq


# ---------------------------------------------------------------------------
# Per-lecture evaluation
# ---------------------------------------------------------------------------

def get_treeseg_segmentation(lecture, treeseg_config):
    utterances = LectureSegmentBuilder.load_lecture_utterances(lecture)
    entries = LectureSegmentBuilder.build_treeseg_entries(utterances, include_ocr=False)
    if not entries:
        return [], []
    model = TreeSeg(configs=treeseg_config, entries=list(entries))
    model.segment_meeting(K=float("inf"))
    segment_groups = [leaf.segment for leaf in model.leaves]
    return entries, segment_groups


def evaluate_lecture(lecture, treeseg_config, n_random_trials):
    """Run WinDiff/Pk evaluation for a single lecture. Returns None if skipped."""
    print(f"  Segmenting {lecture.key} ...")

    slide_end_times = load_slide_end_times_for_lecture(lecture)
    if slide_end_times is None or len(slide_end_times) < 2:
        print(f"  [skip] {lecture.key}: segments.txt missing or fewer than 2 slides")
        return None

    entries, segment_groups = get_treeseg_segmentation(lecture, treeseg_config)
    if len(entries) < 2 or not segment_groups:
        print(f"  [skip] {lecture.key}: too few utterances or no segments")
        return None

    n_utterances = len(entries)
    n_slides = len(slide_end_times)          # baseline K (independent of TreeSeg)
    n_boundaries = n_slides - 1              # number of boundaries for baselines

    ref = build_reference_sequence(entries, slide_end_times)
    n_ref_boundaries = sum(ref)

    if n_ref_boundaries == 0:
        print(f"  [skip] {lecture.key}: no slide transitions map to utterance boundaries")
        return None

    # TreeSeg prediction
    treeseg_hyp = build_treeseg_sequence(n_utterances, segment_groups)
    treeseg_wd = window_diff(ref, treeseg_hyp)
    treeseg_pk = pk_metric(ref, treeseg_hyp)

    # Random baseline
    rng = random.Random(42)
    rand_wds, rand_pks = [], []
    for _ in range(n_random_trials):
        hyp = build_random_sequence(n_utterances, n_boundaries, rng)
        rand_wds.append(window_diff(ref, hyp))
        rand_pks.append(pk_metric(ref, hyp))

    # Fixed-window baseline
    fw_hyp = build_fixed_window_sequence(n_utterances, n_boundaries)
    fw_wd = window_diff(ref, fw_hyp)
    fw_pk = pk_metric(ref, fw_hyp)

    n_treeseg_boundaries = sum(treeseg_hyp)
    print(
        f"  Done: {n_utterances} utt, ref_boundaries={n_ref_boundaries}, "
        f"treeseg_boundaries={n_treeseg_boundaries} | "
        f"WD={treeseg_wd:.4f} Pk={treeseg_pk:.4f}"
    )

    return {
        "lecture_key": lecture.key,
        "n_utterances": n_utterances,
        "n_ref_boundaries": n_ref_boundaries,
        "n_treeseg_boundaries": n_treeseg_boundaries,
        "treeseg": {"wd": treeseg_wd, "pk": treeseg_pk},
        "random": {
            "wd_mean": float(np.mean(rand_wds)),
            "wd_std": float(np.std(rand_wds)),
            "pk_mean": float(np.mean(rand_pks)),
            "pk_std": float(np.std(rand_pks)),
        },
        "fixed_window": {"wd": fw_wd, "pk": fw_pk},
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(lecture_results):
    valid = [r for r in lecture_results if r is not None]
    if not valid:
        return None
    return {
        "n_lectures": len(valid),
        "treeseg": {
            "wd": float(np.mean([r["treeseg"]["wd"] for r in valid])),
            "pk": float(np.mean([r["treeseg"]["pk"] for r in valid])),
        },
        "random": {
            "wd_mean": float(np.mean([r["random"]["wd_mean"] for r in valid])),
            "wd_std": float(np.mean([r["random"]["wd_std"] for r in valid])),
            "pk_mean": float(np.mean([r["random"]["pk_mean"] for r in valid])),
            "pk_std": float(np.mean([r["random"]["pk_std"] for r in valid])),
        },
        "fixed_window": {
            "wd": float(np.mean([r["fixed_window"]["wd"] for r in valid])),
            "pk": float(np.mean([r["fixed_window"]["pk"] for r in valid])),
        },
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def fmt(v):
    return f"{v:.4f}" if v is not None and not np.isnan(v) else "nan"


def print_results(lecture_results, aggregated, n_random_trials):
    valid = [r for r in lecture_results if r is not None]

    rows = [
        [
            r["lecture_key"],
            r["n_utterances"],
            r["n_ref_boundaries"],
            r["n_treeseg_boundaries"],
            fmt(r["treeseg"]["wd"]),
            fmt(r["treeseg"]["pk"]),
        ]
        for r in valid
    ]
    print("\n=== Per-Lecture TreeSeg Scores ===")
    print(
        tabulate(
            rows,
            headers=["Lecture", "#Utt", "#Ref Boundaries", "#TreeSeg Boundaries", "WinDiff", "Pk"],
            tablefmt="grid",
        )
    )

    if aggregated is None:
        print("\nNo valid lectures to aggregate.")
        return

    ts = aggregated["treeseg"]
    rnd = aggregated["random"]
    fw = aggregated["fixed_window"]

    summary_rows = [
        ["TreeSeg", fmt(ts["wd"]), fmt(ts["pk"])],
        [
            f"Random (mean±std, n={n_random_trials})",
            f"{fmt(rnd['wd_mean'])}±{fmt(rnd['wd_std'])}",
            f"{fmt(rnd['pk_mean'])}±{fmt(rnd['pk_std'])}",
        ],
        ["Fixed-Window", fmt(fw["wd"]), fmt(fw["pk"])],
    ]
    print(f"\n=== Aggregate Summary ({aggregated['n_lectures']} lectures) ===")
    print(tabulate(summary_rows, headers=["Method", "WinDiff", "Pk"], tablefmt="grid"))
    print("Note: Lower is better for both WinDiff and Pk.")
    print("Reference boundaries = slide transitions from segments.txt.")
    print(f"Baseline K = n_slides (independent of TreeSeg's segment count).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TreeSeg segmentation quality via WinDiff and Pk against slide transitions."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_DIR / "lpm_data",
        help="Path to lpm_data/ directory (default: %(default)s)",
    )
    parser.add_argument("--speaker", default=None, help="Filter to one speaker")
    parser.add_argument("--course-dir", default=None, help="Filter to one course directory")
    parser.add_argument("--meeting-ids", nargs="+", default=None, help="One or more meeting IDs")
    parser.add_argument(
        "--n-random-trials",
        type=int,
        default=100,
        help="Number of random baseline trials per lecture (default: 100)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    treeseg_config = LpmConfigBuilder.build_lpm_config()

    lectures = LectureCatalog.discover_lectures(
        data_dir=args.data_dir,
        speaker=args.speaker,
        course_dir=args.course_dir,
        meeting_ids=args.meeting_ids,
    )

    if not lectures:
        print("No lectures found. Check --data-dir and filter arguments.")
        return

    print(f"Found {len(lectures)} lecture(s). Running evaluation...\n")

    lecture_results = []
    for i, lecture in enumerate(lectures, start=1):
        print(f"[{i}/{len(lectures)}] {lecture.key}")
        result = evaluate_lecture(lecture, treeseg_config, args.n_random_trials)
        lecture_results.append(result)

    aggregated = aggregate_results(lecture_results)
    print_results(lecture_results, aggregated, args.n_random_trials)


if __name__ == "__main__":
    main()
