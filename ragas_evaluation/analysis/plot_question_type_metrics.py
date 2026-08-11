"""
Plot LLM-judge metrics averaged across all systems, broken down by question type.

Usage:
    python ragas_evaluation/analysis/plot_question_type_metrics.py

Outputs:
    ragas_evaluation/analysis/outputs/question_type_metrics.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

HERE = Path(__file__).resolve().parent
AVERAGED = HERE / "outputs" / "per_question_type_averaged.csv"
OUT = HERE / "outputs" / "question_type_metrics.png"

METRICS = [
    ("faithfulness",       "Faithfulness",       "#2a78d6"),
    ("context_precision",  "Context Precision",  "#e34948"),
    ("context_recall",     "Context Recall",     "#eda100"),
    ("answer_correctness", "Answer Correctness", "#1baf7a"),
    ("answer_relevancy",   "Answer Relevancy",   "#4a3aa7"),
]

TYPE_LABELS = {
    "definition":     "Definition",
    "cause_effect":   "Cause & Effect",
    "mechanism":      "Mechanism",
    "summary":        "Summary",
    "classification": "Classification",
    "process":        "Process",
    "function":       "Function",
    "logistics":      "Logistics",
    "course_specific": "Course-Specific",
}


def main() -> None:
    df = pd.read_csv(AVERAGED)
    df["label"] = df["question_type"].map(TYPE_LABELS).fillna(df["question_type"])

    # Fixed order across all panels: sorted by answer_correctness descending
    fixed_order = (
        df[["label", "answer_correctness"]]
        .dropna()
        .sort_values("answer_correctness", ascending=True)  # ascending so top bar = best in horizontal chart
        ["label"]
        .tolist()
    )

    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    fig.suptitle(
        "Answer Metrics by Question Type\n(mean across all 14 systems)",
        fontsize=13, fontweight="bold", y=0.99,
    )

    for ax, (col, title, color) in zip(axes.flat, METRICS):
        sub = df[["label", col]].dropna()
        sub = sub.set_index("label").reindex(fixed_order).reset_index().dropna()

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
        ax.set_xlabel("Score (mean across all systems)", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(0, x_max * 1.18)
        ax.annotate("↑ higher is better", xy=(1, 0), xycoords="axes fraction",
                    fontsize=7, color="#888", ha="right", va="bottom")

    axes.flat[-1].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
