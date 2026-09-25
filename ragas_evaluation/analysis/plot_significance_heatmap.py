"""
Statistical significance heatmap (Wilcoxon p-values) for answer_correctness.

Usage:
    python ragas_evaluation/analysis/plot_significance_heatmap.py

Outputs:
    ragas_evaluation/analysis/outputs/significance_heatmap.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIG = HERE / "outputs" / "significance_matrix.csv"
OUT = HERE / "outputs" / "significance_heatmap.png"

SYSTEM_ORDER = [
    "treeseg_leaf", "treeseg_summary_tree",
    "baseline_utt_128_10ov", "baseline_utt_128_0ov",
    "baseline_utt_256_10ov", "baseline_utt_256_0ov",
    "baseline_utt_512_0ov", "baseline_utt_512_10ov",
    "baseline_raw_128_10ov", "baseline_raw_128_0ov",
    "baseline_raw_256_10ov", "baseline_raw_256_0ov",
    "baseline_raw_512_0ov", "baseline_raw_512_10ov",
]


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
    sig = pd.read_csv(SIG)
    ac = sig[sig["metric"] == "answer_correctness"].copy()

    systems = SYSTEM_ORDER
    n = len(systems)
    idx = {s: i for i, s in enumerate(systems)}

    # Build symmetric p-value matrix; NaN on diagonal
    matrix = np.full((n, n), np.nan)
    for _, row in ac.iterrows():
        a, b = row["system_a"], row["system_b"]
        if a in idx and b in idx:
            p = row["p_value"]
            matrix[idx[a], idx[b]] = p
            matrix[idx[b], idx[a]] = p

    labels = [short_name(s) for s in systems]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Color: significant (p<0.05) = teal, not significant = light grey, diagonal = white
    cmap = mcolors.ListedColormap(["#e8e8e8", "#1baf7a"])
    sig_matrix = np.where(np.isnan(matrix), np.nan, (matrix < 0.05).astype(float))

    masked = np.ma.masked_invalid(sig_matrix)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Annotate with p-values
    for i in range(n):
        for j in range(n):
            if i == j or np.isnan(matrix[i, j]):
                continue
            p = matrix[i, j]
            text_color = "white" if p < 0.05 else "#555"
            ax.text(j, i, f"{p:.3f}", ha="center", va="center",
                    fontsize=6, color=text_color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Pairwise Statistical Significance — Answer Correctness\n"
                 "(Wilcoxon signed-rank, n=150; green = p < 0.05)",
                 fontsize=11, fontweight="bold", pad=10)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
