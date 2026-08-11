"""
Heatmap of answer_correctness by domain × system.

Usage:
    python ragas_evaluation/analysis/plot_domain_heatmap.py

Outputs:
    ragas_evaluation/analysis/outputs/domain_heatmap.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BREAKDOWN = HERE / "outputs" / "per_domain_breakdown.csv"
OUT = HERE / "outputs" / "domain_heatmap.png"

SYSTEM_ORDER = [
    "treeseg_leaf", "treeseg_summary_tree",
    "baseline_utt_128_0ov", "baseline_utt_128_10ov",
    "baseline_utt_256_0ov", "baseline_utt_256_10ov",
    "baseline_utt_512_0ov", "baseline_utt_512_10ov",
    "baseline_raw_128_0ov", "baseline_raw_128_10ov",
    "baseline_raw_256_0ov", "baseline_raw_256_10ov",
    "baseline_raw_512_0ov", "baseline_raw_512_10ov",
]

DOMAIN_LABELS = {
    "anat-1":  "Anatomy",
    "bio-3":   "Biology",
    "dental":  "Dental",
    "ml-1":    "Mach. Learning",
    "psy-1":   "Psych. (intro)",
    "psy-2":   "Psych. (dev.)",
}


def short_name(s: str) -> str:
    return (
        s.replace("treeseg_leaf", "T-Leaf")
         .replace("treeseg_summary_tree", "T-SumTree")
         .replace("baseline_raw_", "Raw-")
         .replace("baseline_utt_", "Utt-")
         .replace("_0ov", "-0ov")
         .replace("_10ov", "-10ov")
    )


def main() -> None:
    df = pd.read_csv(BREAKDOWN)

    pivot = df.pivot(index="domain", columns="system", values="answer_correctness_mean")
    pivot = pivot.rename(index=DOMAIN_LABELS)

    # Reorder columns
    col_order = [s for s in SYSTEM_ORDER if s in pivot.columns]
    pivot = pivot[col_order]
    col_labels = [short_name(c) for c in col_order]

    # Sort rows by mean score descending
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    matrix = pivot.values
    row_labels = pivot.index.tolist()

    fig, ax = plt.subplots(figsize=(14, 4.5))

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            text_color = "black" if 0.55 < val < 0.85 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=text_color, fontweight="bold")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title("Answer Correctness by Subject Domain × System",
                 fontsize=11, fontweight="bold", pad=10)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Answer Correctness")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
