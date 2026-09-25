"""
Scatter plots: retrieval quality vs answer correctness.

Usage:
    python ragas_evaluation/analysis/plot_retrieval_vs_answer.py

Outputs:
    ragas_evaluation/analysis/outputs/retrieval_vs_answer.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
UNIFIED = HERE / "outputs" / "unified_table.csv"
OUT = HERE / "outputs" / "retrieval_vs_answer.png"


def short_name(s: str) -> str:
    return (
        s.replace("treeseg_leaf", "T-Leaf")
         .replace("treeseg_summary_tree", "T-SumTree")
         .replace("baseline_raw_", "Raw-")
         .replace("baseline_utt_", "Utt-")
         .replace("_0ov", "-0ov")
         .replace("_10ov", "-10ov")
    )


def system_color(s: str) -> str:
    if s.startswith("treeseg"):
        return "#2a78d6"   # blue
    if s.startswith("baseline_raw"):
        return "#e34948"   # red
    return "#1baf7a"       # green


def scatter_panel(ax, df, x_col, x_label, y_col, y_label):
    x = df[x_col].values
    y = df[y_col].values

    for _, row in df.iterrows():
        ax.scatter(row[x_col], row[y_col],
                   color=system_color(row["system"]), s=70, zorder=3)
        ax.annotate(
            short_name(row["system"]),
            (row[x_col], row[y_col]),
            textcoords="offset points", xytext=(5, 3),
            fontsize=7, color="#333",
        )

    # Trendline
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#888", linewidth=1,
            linestyle="--", zorder=2)
    ax.text(0.05, 0.95, f"r = {r:+.3f}  (p = {p:.3f})",
            transform=ax.transAxes, fontsize=8, va="top", color="#555")

    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)


def main() -> None:
    df = pd.read_csv(UNIFIED).dropna(subset=["ndcg@5", "recall@5", "answer_correctness"])

    # Legend patches
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="#2a78d6", label="TreeSeg"),
        Patch(color="#e34948", label="Raw baseline"),
        Patch(color="#1baf7a", label="Utt baseline"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Retrieval Quality vs Answer Correctness", fontsize=13,
                 fontweight="bold", y=1.01)

    scatter_panel(axes[0], df, "ndcg@5", "nDCG@5 (retrieval ranking quality)",
                  "answer_correctness", "Answer Correctness")
    axes[0].set_title("nDCG@5 vs Answer Correctness", fontsize=10, fontweight="semibold")

    scatter_panel(axes[1], df, "recall@5", "Recall@5 (retrieval coverage)",
                  "answer_correctness", "Answer Correctness")
    axes[1].set_title("Recall@5 vs Answer Correctness", fontsize=10, fontweight="semibold")

    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
