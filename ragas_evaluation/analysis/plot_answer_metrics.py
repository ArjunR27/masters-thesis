"""
Plot LLM-as-a-judge answer generation metrics per system from unified_table.csv.

Usage:
    python ragas_evaluation/analysis/plot_answer_metrics.py

Outputs:
    ragas_evaluation/analysis/outputs/answer_metrics.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

HERE = Path(__file__).resolve().parent
UNIFIED = HERE / "outputs" / "unified_table.csv"
OUT = HERE / "outputs" / "answer_metrics.png"

METRICS = [
    ("faithfulness",       "Faithfulness",       "#2a78d6"),  # blue
    ("context_precision",  "Context Precision",  "#e34948"),  # red
    ("context_recall",     "Context Recall",     "#eda100"),  # orange
    ("answer_correctness", "Answer Correctness", "#1baf7a"),  # green
    ("answer_relevancy",   "Answer Relevancy",   "#4a3aa7"),  # violet
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

    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    fig.suptitle("LLM-as-a-Judge Answer Generation Metrics by System", fontsize=13, fontweight="bold", y=0.99)

    for ax, (col, title, color) in zip(axes.flat, METRICS):
        sub = df[["label", col]].dropna().sort_values(col, ascending=False)

        bars = ax.barh(sub["label"], sub[col], color=color, height=0.55, zorder=3)

        x_max = sub[col].max()
        for bar in bars:
            w = bar.get_width()
            ax.text(
                w + x_max * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}",
                va="center", ha="left", fontsize=8, color="#444",
            )

        ax.set_title(title, fontsize=10, fontweight="semibold", pad=6)
        ax.set_xlabel("Score", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(0, x_max * 1.18)
        ax.annotate("↑ higher is better", xy=(1, 0), xycoords="axes fraction",
                    fontsize=7, color="#888", ha="right", va="bottom")

    # hide unused 6th subplot
    axes.flat[-1].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
