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
MASTERS = HERE.parent.parent
AVERAGED = HERE / "outputs" / "per_question_type_averaged.csv"
NO_RETRIEVAL_CSV = MASTERS / "ragas_evaluation/outputs/no_retrieval_per_question.csv"
DATASET = MASTERS / "LPM_QA_DATASET/lpm_qa_labeled.csv"
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


def _load_closed_book_by_type() -> dict[str, dict[str, float]]:
    """Returns {question_type: {metric: mean}} if no_retrieval CSV exists, else {}."""
    if not NO_RETRIEVAL_CSV.exists():
        return {}
    nr = pd.read_csv(NO_RETRIEVAL_CSV)
    dataset = pd.read_csv(DATASET)[["question", "question_type"]].drop_duplicates("question")
    dataset["question_type"] = dataset["question_type"].str.strip()
    nr = nr.merge(dataset, on="question", how="left")
    nr["label"] = nr["question_type"].map(TYPE_LABELS).fillna(nr["question_type"])
    result = {}
    for label, grp in nr.groupby("label"):
        result[label] = {}
        for col in ("answer_correctness", "answer_relevancy"):
            vals = pd.to_numeric(grp[col], errors="coerce").dropna()
            if len(vals):
                result[label][col] = float(vals.mean())
    return result


def main() -> None:
    df = pd.read_csv(AVERAGED)
    df["label"] = df["question_type"].map(TYPE_LABELS).fillna(df["question_type"])

    closed_book = _load_closed_book_by_type()
    has_closed_book = bool(closed_book)

    # Fixed order across all panels: sorted by answer_correctness descending
    fixed_order = (
        df[["label", "answer_correctness"]]
        .dropna()
        .sort_values("answer_correctness", ascending=True)
        ["label"]
        .tolist()
    )

    title_suffix = "\n(mean across all 14 systems)"
    if has_closed_book:
        title_suffix = "\n(mean across all 14 systems; grey = closed-book baseline)"

    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    fig.suptitle(
        f"Answer Metrics by Question Type{title_suffix}",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # Only answer_correctness and answer_relevancy have closed-book values
    closed_book_metrics = {"answer_correctness", "answer_relevancy"}
    CB_COLOR = "#999999"

    for ax, (col, title, color) in zip(axes.flat, METRICS):
        sub = df[["label", col]].dropna()
        sub = sub.set_index("label").reindex(fixed_order).reset_index().dropna()
        y_pos = list(range(len(sub)))
        labels_ordered = sub["label"].tolist()

        do_grouped = has_closed_book and col in closed_book_metrics
        bar_h = 0.35 if do_grouped else 0.55
        offset = bar_h / 2 if do_grouped else 0

        # Main bars (shifted up when grouped)
        bars = ax.barh(
            [y + offset for y in y_pos], sub[col],
            height=bar_h, color=color, zorder=3,
            label="14-system mean" if do_grouped else None,
        )

        # Closed-book bars (shifted down)
        cb_bars = None
        cb_vals_list = []
        if do_grouped:
            cb_vals_list = [closed_book.get(lbl, {}).get(col) or 0.0 for lbl in labels_ordered]
            cb_bars = ax.barh(
                [y - offset for y in y_pos], cb_vals_list,
                height=bar_h, color=CB_COLOR, zorder=3,
                label="No retrieval",
            )

        x_max = sub[col].max()
        if do_grouped and cb_vals_list:
            x_max = max(x_max, max(cb_vals_list))

        for bar in bars:
            w = bar.get_width()
            ax.text(w + x_max * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}", va="center", ha="left", fontsize=8, color="#444")

        if cb_bars is not None:
            for bar, val in zip(cb_bars, cb_vals_list):
                ax.text(val + x_max * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", ha="left", fontsize=7.5, color="#555")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_ordered, fontsize=8)

        if do_grouped:
            ax.legend(fontsize=7.5, loc="lower right")

        ax.set_title(title, fontsize=10, fontweight="semibold", pad=6)
        ax.set_xlabel("Score (mean across all systems)", fontsize=8)
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
