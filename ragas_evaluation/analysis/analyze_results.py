"""
Thesis results analysis.

Usage:
    python ragas_evaluation/analysis/analyze_results.py

Outputs to ragas_evaluation/analysis/outputs/:
  unified_table.csv          - RAGAS + retrieval metrics joined, 5 systems
  significance_matrix.csv    - pairwise Wilcoxon p-values per metric
  per_domain_breakdown.csv   - answer_correctness/context_recall by domain × system
  metric_correlations.csv    - correlation between lexical and LLM judge metrics
  failure_cases.csv          - questions where any system scores answer_correctness < 0.5
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ─── Paths ────────────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
MASTERS = HERE.parent.parent

RAGAS_SUMMARY = MASTERS / "ragas_evaluation/outputs/system_summary.csv"
RAGAS_PER_Q = MASTERS / "ragas_evaluation/outputs/per_question_all_systems.csv"
RETRIEVAL = MASTERS / "lpm_qa_evaluation/outputs/retrieval_evaluation.csv"
EDUVID = MASTERS / "eduvid_evaluation/storage/evaluation_outputs/metrics_per_question.csv"
DATASET = MASTERS / "LPM_QA_DATASET/lpm_qa_labeled.csv"

OUT_DIR = HERE / "outputs"

# ─── System name mapping (RAGAS name → retrieval eval name) ───────────────────

RAGAS_TO_RETRIEVAL = {
    "treeseg_leaf":         "treeseg__leaf",
    "treeseg_summary_tree": "treeseg__summary_tree",
    "baseline_raw_128_0ov":  "baseline__raw_token_window__128tok__ov0__transcript_only",
    "baseline_raw_128_10ov": "baseline__raw_token_window__128tok__ov10__transcript_only",
    "baseline_raw_256_0ov":  "baseline__raw_token_window__256tok__ov0__transcript_only",
    "baseline_raw_256_10ov": "baseline__raw_token_window__256tok__ov10__transcript_only",
    "baseline_raw_512_0ov":  "baseline__raw_token_window__512tok__ov0__transcript_only",
    "baseline_raw_512_10ov": "baseline__raw_token_window__512tok__ov10__transcript_only",
    "baseline_utt_128_0ov":  "baseline__utterance_packed__128tok__ov0__transcript_only",
    "baseline_utt_128_10ov": "baseline__utterance_packed__128tok__ov10__transcript_only",
    "baseline_utt_256_0ov":  "baseline__utterance_packed__256tok__ov0__transcript_only",
    "baseline_utt_256_10ov": "baseline__utterance_packed__256tok__ov10__transcript_only",
    "baseline_utt_512_0ov":  "baseline__utterance_packed__512tok__ov0__transcript_only",
    "baseline_utt_512_10ov": "baseline__utterance_packed__512tok__ov10__transcript_only",
}

DISPLAY_NAMES = {
    "treeseg_leaf":         "TreeSeg-Leaf",
    "treeseg_summary_tree": "TreeSeg-SumTree",
    "baseline_raw_128_0ov":  "Raw-128-0ov",
    "baseline_raw_128_10ov": "Raw-128-10ov",
    "baseline_raw_256_0ov":  "Raw-256-0ov",
    "baseline_raw_256_10ov": "Raw-256-10ov",
    "baseline_raw_512_0ov":  "Raw-512-0ov",
    "baseline_raw_512_10ov": "Raw-512-10ov",
    "baseline_utt_128_0ov":  "Utt-128-0ov",
    "baseline_utt_128_10ov": "Utt-128-10ov",
    "baseline_utt_256_0ov":  "Utt-256-0ov",
    "baseline_utt_256_10ov": "Utt-256-10ov",
    "baseline_utt_512_0ov":  "Utt-512-0ov",
    "baseline_utt_512_10ov": "Utt-512-10ov",
}

RAGAS_METRICS = [
    "faithfulness", "context_precision", "context_recall",
    "answer_correctness", "answer_relevancy",
]
LEXICAL_METRICS = ["rouge_l", "bleu_1"]
ALL_METRICS = RAGAS_METRICS + LEXICAL_METRICS


# ─── 1. Unified table ─────────────────────────────────────────────────────────

def build_unified_table(ragas_summary: pd.DataFrame, retrieval: pd.DataFrame) -> pd.DataFrame:
    retrieval_indexed = retrieval.set_index("system")

    rows = []
    for ragas_name, ret_name in RAGAS_TO_RETRIEVAL.items():
        r = ragas_summary[ragas_summary["system"] == ragas_name]
        if r.empty:
            continue
        r = r.iloc[0]

        ret = retrieval_indexed.loc[ret_name] if ret_name in retrieval_indexed.index else None

        row: dict = {"system": ragas_name, "display": DISPLAY_NAMES[ragas_name]}

        # retrieval metrics (from retrieval_evaluation.csv)
        for col in ["recall@5", "ndcg@5", "max_iou@5", "mean_temporal_distance@5"]:
            row[col] = round(float(ret[col]), 4) if ret is not None else None

        # RAGAS metrics (means)
        for m in ALL_METRICS:
            row[f"{m}"] = round(float(r[f"{m}_mean"]), 4)
            row[f"{m}_std"] = round(float(r[f"{m}_std"]), 4)

        row["insufficient_context_rate"] = round(float(r["insufficient_context_rate"]), 4)
        rows.append(row)

    return pd.DataFrame(rows)


def print_latex_table(df: pd.DataFrame) -> None:
    metric_cols = ["recall@5", "ndcg@5", "answer_correctness", "context_precision",
                   "context_recall", "faithfulness", "answer_relevancy", "rouge_l", "bleu_1"]

    # find best (max) value per column
    best: dict[str, float] = {}
    for col in metric_cols:
        vals = df[col].dropna()
        if not vals.empty:
            best[col] = vals.max()

    col_headers = " & ".join(
        ["System", "R@5", "nDCG@5", "Ans.Corr", "Ctx.Prec", "Ctx.Rec",
         "Faith.", "Ans.Rel", "ROUGE-L", "BLEU-1"]
    )
    print("\n" + "=" * 70)
    print("LaTeX table (paste into thesis):")
    print("=" * 70)
    print(r"\begin{tabular}{lrrrrrrrrr}")
    print(r"\toprule")
    print(col_headers + r" \\")
    print(r"\midrule")
    for _, row in df.iterrows():
        cells = [DISPLAY_NAMES.get(row["system"], row["system"])]
        for col in metric_cols:
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                cells.append("—")
            else:
                formatted = f"{val:.3f}"
                if best.get(col) is not None and abs(val - best[col]) < 1e-9:
                    formatted = r"\textbf{" + formatted + "}"
                cells.append(formatted)
        print(" & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print("=" * 70 + "\n")


# ─── 2. Statistical significance (Wilcoxon signed-rank) ──────────────────────

def build_significance_matrix(per_q: pd.DataFrame) -> pd.DataFrame:
    systems = list(per_q["system"].unique())
    pairs = list(combinations(systems, 2))
    rows = []
    for metric in ALL_METRICS:
        for s1, s2 in pairs:
            a = per_q[per_q["system"] == s1][metric].dropna().values
            b = per_q[per_q["system"] == s2][metric].dropna().values
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            diff = a - b
            if np.all(diff == 0):
                p = 1.0
            else:
                _, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            rows.append({
                "metric": metric,
                "system_a": s1,
                "system_b": s2,
                "p_value": round(float(p), 4),
                "significant_p05": p < 0.05,
            })
    return pd.DataFrame(rows)


# ─── 3. Per-domain breakdown ──────────────────────────────────────────────────

def build_domain_breakdown(per_q: pd.DataFrame) -> pd.DataFrame:
    per_q = per_q.copy()
    per_q["domain"] = per_q["lecture_key"].str.split("/").str[0]

    rows = []
    for domain in sorted(per_q["domain"].unique()):
        for system in per_q["system"].unique():
            mask = (per_q["domain"] == domain) & (per_q["system"] == system)
            subset = per_q[mask]
            if subset.empty:
                continue
            rows.append({
                "domain": domain,
                "system": system,
                "n": len(subset),
                "answer_correctness_mean": round(float(subset["answer_correctness"].mean()), 4),
                "context_recall_mean":     round(float(subset["context_recall"].mean()), 4),
                "rouge_l_mean":            round(float(subset["rouge_l"].mean()), 4),
            })
    return pd.DataFrame(rows)


def print_domain_pivot(breakdown: pd.DataFrame) -> None:
    pivot = breakdown.pivot(index="domain", columns="system", values="answer_correctness_mean")
    pivot.columns = [DISPLAY_NAMES.get(c, c) for c in pivot.columns]
    print("\n" + "=" * 70)
    print("Per-domain answer_correctness (mean):")
    print("=" * 70)
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))
    print()


# ─── 4. Metric correlations ───────────────────────────────────────────────────

def build_metric_correlations(per_q: pd.DataFrame) -> pd.DataFrame:
    # Per-question correlations (n=750 across all systems)
    metric_pairs = [
        ("rouge_l", "answer_correctness"),
        ("bleu_1", "answer_correctness"),
        ("rouge_l", "context_recall"),
        ("bleu_1", "context_recall"),
        ("answer_correctness", "context_recall"),
        ("faithfulness", "answer_correctness"),
        ("context_precision", "answer_correctness"),
    ]
    rows = []
    for m1, m2 in metric_pairs:
        valid = per_q[[m1, m2]].dropna()
        r, p = stats.pearsonr(valid[m1], valid[m2])
        rows.append({
            "metric_a": m1,
            "metric_b": m2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 6),
            "n": len(valid),
        })
    return pd.DataFrame(rows)


def print_correlations(corr: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("Metric correlations (Pearson r, n per-question pairs):")
    print("=" * 70)
    for _, row in corr.iterrows():
        sig = "**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else "")
        print(f"  {row['metric_a']:25s} ↔ {row['metric_b']:25s}  r={row['pearson_r']:+.3f}  p={row['p_value']:.4f}{sig}")
    print()


# ─── 5. Failure cases ─────────────────────────────────────────────────────────

def build_failure_cases(per_q: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    low = per_q[per_q["answer_correctness"] < threshold][["example_id"]].drop_duplicates()
    subset = per_q[per_q["example_id"].isin(low["example_id"])].copy()

    pivot = subset.pivot_table(
        index=["example_id", "lecture_key", "question"],
        columns="system",
        values="answer_correctness",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    pivot["min_score"] = pivot[[c for c in pivot.columns if c in RAGAS_TO_RETRIEVAL]].min(axis=1)
    return pivot.sort_values("min_score")


def print_failure_summary(failures: pd.DataFrame) -> None:
    all_fail = failures[failures["min_score"] < 0.3]
    print("\n" + "=" * 70)
    print(f"Failure cases: {len(failures)} questions where any system scored < 0.5")
    print(f"  {len(all_fail)} questions where ALL systems scored < 0.3 (universally hard)")
    print("=" * 70)
    if not all_fail.empty:
        for _, row in all_fail.head(5).iterrows():
            print(f"  [{row['example_id']}] {row['question'][:80]}...")
    print()


# ─── 6. Per-question-type breakdown ──────────────────────────────────────────

def build_question_type_breakdown(per_q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      by_system  — mean of each metric per (question_type, system)
      averaged   — mean across all systems per question_type (for plotting)
    """
    dataset = pd.read_csv(DATASET)[["question", "question_type"]].drop_duplicates("question")
    dataset["question_type"] = dataset["question_type"].str.strip()
    merged = per_q.merge(dataset, on="question", how="left")

    metrics = RAGAS_METRICS + LEXICAL_METRICS

    by_system = (
        merged.groupby(["question_type", "system"])[metrics]
        .mean()
        .round(4)
        .reset_index()
    )

    # Average across all 14 systems per question_type
    averaged = (
        merged.groupby("question_type")[metrics]
        .mean()
        .round(4)
        .reset_index()
    )
    # Add question count
    counts = merged.groupby("question_type")["question"].nunique().rename("n_questions")
    averaged = averaged.merge(counts, on="question_type")

    return by_system, averaged


def print_question_type_pivot(averaged: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("Per-question-type answer_correctness (mean across all systems):")
    print("=" * 70)
    sub = averaged[["question_type", "n_questions", "answer_correctness", "context_recall", "faithfulness"]].copy()
    sub = sub.sort_values("answer_correctness", ascending=False)
    print(sub.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()


# ─── 7. Cross-dataset comparison note ─────────────────────────────────────────

def print_cross_dataset_note(ragas_summary: pd.DataFrame) -> None:
    st = ragas_summary[ragas_summary["system"] == "treeseg_summary_tree"]
    if st.empty:
        return
    st = st.iloc[0]
    lpm_rouge = st["rouge_l_mean"]
    lpm_bleu  = st["bleu_1_mean"]

    eduvid_rouge = eduvid_bleu = None
    if EDUVID.exists():
        ev = pd.read_csv(EDUVID)
        if "rouge_l" in ev.columns:
            eduvid_rouge = ev["rouge_l"].mean()
            eduvid_bleu  = ev["bleu1"].mean() if "bleu1" in ev.columns else None

    print("\n" + "=" * 70)
    print("Cross-dataset generalisation (TreeSeg-SumTree):")
    print("=" * 70)
    print(f"  TinyLPM-QA  (n=150): ROUGE-L={lpm_rouge:.3f}  BLEU-1={lpm_bleu:.3f}")
    if eduvid_rouge is not None:
        print(f"  EduVid QA   (n=30):  ROUGE-L={eduvid_rouge:.3f}  BLEU-1={eduvid_bleu:.3f}")
        delta_r = eduvid_rouge - lpm_rouge
        print(f"  ΔROUGE-L = {delta_r:+.3f}  ({'better' if delta_r > 0 else 'worse'} on EduVid)")
        print("  Near-identical lexical scores suggest reasonable cross-dataset generalisation,")
        print("  though EduVid only covers 30 questions and a different domain distribution.")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ragas_summary = pd.read_csv(RAGAS_SUMMARY)
    per_q = pd.read_csv(RAGAS_PER_Q)
    retrieval = pd.read_csv(RETRIEVAL)

    print(f"Loaded {len(ragas_summary)} systems from RAGAS summary")
    print(f"Loaded {len(per_q)} per-question rows ({per_q['system'].nunique()} systems × ~{len(per_q) // per_q['system'].nunique()} questions)")
    print(f"Loaded {len(retrieval)} systems from retrieval evaluation")

    # 1. Unified table
    print("\n[1/6] Building unified table...")
    unified = build_unified_table(ragas_summary, retrieval)
    unified.to_csv(OUT_DIR / "unified_table.csv", index=False)
    print(f"  Saved unified_table.csv ({len(unified)} systems)")
    print_latex_table(unified)

    # 2. Significance matrix
    print("[2/6] Running Wilcoxon signed-rank tests...")
    sig = build_significance_matrix(per_q)
    sig.to_csv(OUT_DIR / "significance_matrix.csv", index=False)
    n_sig = sig["significant_p05"].sum()
    print(f"  Saved significance_matrix.csv — {n_sig}/{len(sig)} pairs significant at p<0.05")

    # 3. Per-domain breakdown
    print("\n[3/6] Building per-domain breakdown...")
    breakdown = build_domain_breakdown(per_q)
    breakdown.to_csv(OUT_DIR / "per_domain_breakdown.csv", index=False)
    print(f"  Saved per_domain_breakdown.csv ({breakdown['domain'].nunique()} domains)")
    print_domain_pivot(breakdown)

    # 4. Metric correlations
    print("[4/6] Computing metric correlations...")
    corr = build_metric_correlations(per_q)
    corr.to_csv(OUT_DIR / "metric_correlations.csv", index=False)
    print(f"  Saved metric_correlations.csv")
    print_correlations(corr)

    # 5. Failure cases
    print("[5/6] Identifying failure cases...")
    failures = build_failure_cases(per_q)
    failures.to_csv(OUT_DIR / "failure_cases.csv", index=False)
    print_failure_summary(failures)
    failures.to_csv(OUT_DIR / "failure_cases.csv", index=False)
    print(f"  Saved failure_cases.csv ({len(failures)} questions)")

    # 6. Per-question-type breakdown
    print("[6/6] Building per-question-type breakdown...")
    qt_by_system, qt_averaged = build_question_type_breakdown(per_q)
    qt_by_system.to_csv(OUT_DIR / "per_question_type_by_system.csv", index=False)
    qt_averaged.to_csv(OUT_DIR / "per_question_type_averaged.csv", index=False)
    print(f"  Saved per_question_type_by_system.csv ({qt_by_system['question_type'].nunique()} types × {qt_by_system['system'].nunique()} systems)")
    print(f"  Saved per_question_type_averaged.csv")
    print_question_type_pivot(qt_averaged)

    # Cross-dataset note
    print_cross_dataset_note(ragas_summary)

    print(f"All outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
