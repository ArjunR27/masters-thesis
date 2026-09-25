"""
Plot LLM-as-a-judge answer generation metrics per system for the EduVidQA
evaluation, from eduvid_evaluation/outputs/eduvid_ragas/system_summary.csv.

Same style as plot_answer_metrics.py (TinyLPM-QA), minus the closed-book
baseline overlay, which EduVidQA does not have.

Usage:
    python ragas_evaluation/analysis/plot_eduvid_answer_metrics.py

Outputs:
    ragas_evaluation/analysis/outputs/eduvid_answer_metrics.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

HERE = Path(__file__).resolve().parent
MASTERS = HERE.parent.parent
SUMMARY = MASTERS / "ragas_evaluation/eduvid_evaluation/outputs/eduvid_ragas/system_summary.csv"
OUT = HERE / "outputs" / "eduvid_answer_metrics.png"

METRICS = [
    ("faithfulness_mean",       "Faithfulness",       "#2a78d6"),  # blue
    ("context_precision_mean",  "Context Precision",  "#e34948"),  # red
    ("context_recall_mean",     "Context Recall",     "#eda100"),  # orange
    ("answer_correctness_mean", "Answer Correctness", "#1baf7a"),  # green
    ("answer_relevancy_mean",   "Answer Relevancy",   "#4a3aa7"),  # violet
]


def short_name(s: str) -> str:
    return (
        s.replace("treeseg__leaf", "TreeSeg-Leaf")
         .replace("treeseg__summary_tree", "TreeSeg-SumTree")
         .replace("baseline__raw_token_window__", "Raw-")
         .replace("baseline__utterance_packed__", "Utt-")
         .replace("__transcript_only", "")
         .replace("tok__ov0", "-0ov")
         .replace("tok__ov10", "-10ov")
    )


def main() -> None:
    df = pd.read_csv(SUMMARY)
    df["label"] = df["system"].apply(short_name)

    title = "LLM-as-a-Judge Answer Generation Metrics by System (EduVidQA)"

    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)

    for ax, (col, title_panel, color) in zip(axes.flat, METRICS):
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

        ax.set_title(title_panel, fontsize=10, fontweight="semibold", pad=6)
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
