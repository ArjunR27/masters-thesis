"""
Plot retrieval metrics per system from unified_table.csv.

Usage:
    python ragas_evaluation/analysis/plot_retrieval_metrics.py

Outputs:
    ragas_evaluation/analysis/outputs/retrieval_metrics.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

HERE = Path(__file__).resolve().parent
UNIFIED = HERE / "outputs" / "unified_table.csv"
OUT = HERE / "outputs" / "retrieval_metrics.png"

METRICS = [
    ("recall@5",                   "Recall@5",                    True,  "#2a78d6"),  # blue
    ("ndcg@5",                     "nDCG@5",                      True,  "#e34948"),  # red
    ("max_iou@5",                  "Max IoU@5",                   True,  "#eda100"),  # orange
    ("mean_temporal_distance@5",   "Mean Temporal Distance@5 (s)", False, "#1baf7a"),  # light green
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
    df = pd.read_csv(UNIFIED)
    df["label"] = df["system"].apply(short_name)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Retrieval Metrics by System", fontsize=13, fontweight="bold", y=0.98)

    for ax, (col, title, higher_better, color) in zip(axes.flat, METRICS):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        sub = df[["label", col]].dropna().sort_values(col, ascending=not higher_better)

        bars = ax.barh(sub["label"], sub[col], color=color, height=0.55, zorder=3)

        # value labels inside/outside bars
        x_max = sub[col].max()
        for bar in bars:
            w = bar.get_width()
            offset = x_max * 0.01
            ha = "left"
            ax.text(
                w + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}",
                va="center",
                ha=ha,
                fontsize=8,
                color="#444",
            )

        ax.set_title(title, fontsize=10, fontweight="semibold", pad=6)
        ax.set_xlabel("Score", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)

        # extend x-axis slightly so value labels don't get clipped
        ax.set_xlim(0, x_max * 1.18)

        direction = "↑ higher is better" if higher_better else "↓ lower is better"
        ax.annotate(
            direction,
            xy=(1, 0), xycoords="axes fraction",
            fontsize=7, color="#888",
            ha="right", va="bottom",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
