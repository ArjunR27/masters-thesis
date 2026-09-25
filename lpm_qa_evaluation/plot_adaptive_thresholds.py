from __future__ import annotations

import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utterances import estimate_max_gap_s_from_rows, iter_rows

FIXED_THRESHOLD_S = 0.8


def compute_per_lecture_thresholds() -> list[float]:
    transcript_paths = sorted(PROJECT_DIR.glob("lpm_data/*/*/*/*_transcripts.csv"))
    thresholds: list[float] = []
    for path in transcript_paths:
        rows = list(iter_rows(path))
        if rows:
            thresholds.append(estimate_max_gap_s_from_rows(rows))
    return thresholds


def plot_adaptive_threshold_distribution(thresholds: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(thresholds, bins=20, color="steelblue", edgecolor="white")

    ax.axvline(
        FIXED_THRESHOLD_S,
        color="firebrick",
        linestyle="--",
        linewidth=2,
        label=f"Fixed threshold ({FIXED_THRESHOLD_S}s)",
    )

    ax.set_xlabel("Per-Lecture Adaptive Threshold (seconds)", fontsize=12)
    ax.set_ylabel("Number of Lectures", fontsize=12)
    ax.set_title(
        f"Distribution of Adaptive Pause Thresholds Across {len(thresholds)} Lectures",
        fontsize=13,
    )
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = HERE / "outputs" / "adaptive_threshold_distribution.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


def main() -> None:
    thresholds = compute_per_lecture_thresholds()
    if not thresholds:
        print("No transcripts found under lpm_data/ — nothing to plot.")
        return

    print(f"n={len(thresholds)}")
    print(f"min={min(thresholds):.2f}s  max={max(thresholds):.2f}s")
    print(f"mean={sum(thresholds) / len(thresholds):.2f}s")
    print(f"median={statistics.median(thresholds):.2f}s")
    print(f"stdev={statistics.stdev(thresholds):.2f}s")

    plot_adaptive_threshold_distribution(thresholds)


if __name__ == "__main__":
    main()
