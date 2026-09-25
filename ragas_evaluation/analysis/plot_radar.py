"""
Radar chart comparing TreeSeg-Leaf, TreeSeg-SumTree, and best baseline.

Usage:
    python ragas_evaluation/analysis/plot_radar.py

Outputs:
    ragas_evaluation/analysis/outputs/radar.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SUMMARY = HERE.parent.parent / "ragas_evaluation/outputs/system_summary.csv"
OUT = HERE / "outputs" / "radar.png"

METRICS = [
    ("faithfulness_mean",       "Faithfulness"),
    ("context_precision_mean",  "Context\nPrecision"),
    ("context_recall_mean",     "Context\nRecall"),
    ("answer_correctness_mean", "Answer\nCorrectness"),
    ("answer_relevancy_mean",   "Answer\nRelevancy"),
]

# (system_name, display_label, color, linewidth, alpha)
SYSTEMS = [
    ("treeseg_leaf",          "T-Leaf",        "#1f77b4", 2.5, 1.0),
    ("treeseg_summary_tree",  "T-SumTree",     "#d62728", 2.5, 1.0),
    ("baseline_raw_128_0ov",  "Raw-128-0ov",   "#ff7f0e", 1.5, 0.85),
    ("baseline_raw_128_10ov", "Raw-128-10ov",  "#ffbb78", 1.5, 0.85),
    ("baseline_raw_256_0ov",  "Raw-256-0ov",   "#8c564b", 1.5, 0.85),
    ("baseline_raw_256_10ov", "Raw-256-10ov",  "#c49c94", 1.5, 0.85),
    ("baseline_raw_512_0ov",  "Raw-512-0ov",   "#e377c2", 1.5, 0.85),
    ("baseline_raw_512_10ov", "Raw-512-10ov",  "#f7b6d2", 1.5, 0.85),
    ("baseline_utt_128_0ov",  "Utt-128-0ov",   "#2ca02c", 1.5, 0.85),
    ("baseline_utt_128_10ov", "Utt-128-10ov",  "#98df8a", 1.5, 0.85),
    ("baseline_utt_256_0ov",  "Utt-256-0ov",   "#9467bd", 1.5, 0.85),
    ("baseline_utt_256_10ov", "Utt-256-10ov",  "#c5b0d5", 1.5, 0.85),
    ("baseline_utt_512_0ov",  "Utt-512-0ov",   "#17becf", 1.5, 0.85),
    ("baseline_utt_512_10ov", "Utt-512-10ov",  "#9edae5", 1.5, 0.85),
]


def main() -> None:
    df = pd.read_csv(SUMMARY).set_index("system")

    metric_cols = [m for m, _ in METRICS]
    metric_labels = [label for _, label in METRICS]
    n = len(metric_cols)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for sys_name, display, color, lw, alpha in SYSTEMS:
        if sys_name not in df.index:
            continue
        values = [float(df.loc[sys_name, col]) for col in metric_cols]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=lw, linestyle="-",
                alpha=alpha, label=display)
        ax.fill(angles, values, color=color, alpha=0.03)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="#888")
    ax.grid(color="#ccc", linewidth=0.5)

    ax.set_title("Multi-metric Comparison — All 14 Systems",
                 fontsize=11, fontweight="bold", pad=20)

    ax.legend(loc="upper right", bbox_to_anchor=(1.55, 1.15),
              fontsize=7.5, frameon=True, ncol=1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
