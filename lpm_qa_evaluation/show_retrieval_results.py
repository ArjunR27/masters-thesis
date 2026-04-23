from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from tabulate import tabulate


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_PATH = HERE / "outputs" / "retrieval_evaluation.csv"
DEFAULT_SORT_BY = "ndcg@5"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretty-print retrieval evaluation results from retrieval_evaluation.csv."
    )
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_RESULTS_PATH),
        help="Path to retrieval_evaluation.csv.",
    )
    parser.add_argument(
        "--sort-by",
        default=DEFAULT_SORT_BY,
        help="Metric column to sort by.",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order instead of descending.",
    )
    return parser


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str) -> float:
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def metric_columns(fieldnames: list[str]) -> list[str]:
    return [
        name
        for name in fieldnames
        if any(name.startswith(prefix) for prefix in ("recall@", "mean_temporal_distance@", "max_iou@", "ndcg@"))
    ]


def sort_rows(rows: list[dict[str, str]], sort_by: str, ascending: bool) -> list[dict[str, str]]:
    def sort_key(row: dict[str, str]) -> tuple[int, float]:
        value = parse_float(row.get(sort_by, ""))
        if math.isnan(value):
            return (1, 0.0)
        return (0, value)

    return sorted(rows, key=sort_key, reverse=not ascending)


def format_number(value: str, digits: int = 3) -> str:
    numeric = parse_float(value)
    if math.isnan(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def build_overview_table(rows: list[dict[str, str]]) -> list[list[str]]:
    table = []
    for row in rows:
        table.append(
            [
                row["approach"],
                row["embedding_model"],
                row["rerank_model"],
                row["question_count"],
                format_number(row.get("recall@5", "")),
                format_number(row.get("mean_temporal_distance@5", "")),
                format_number(row.get("max_iou@5", "")),
                format_number(row.get("ndcg@5", "")),
            ]
        )
    return table


def build_full_metrics_table(rows: list[dict[str, str]], metrics: list[str]) -> list[list[str]]:
    table = []
    for row in rows:
        metric_values = [format_number(row.get(metric, "")) for metric in metrics]
        table.append([row["approach"], *metric_values])
    return table


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"Results CSV not found: {csv_path}")

    rows = read_rows(csv_path)
    if not rows:
        raise SystemExit("Results CSV is empty.")

    fieldnames = list(rows[0].keys())
    metrics = metric_columns(fieldnames)
    if args.sort_by not in fieldnames:
        raise SystemExit(f"Unknown sort column: {args.sort_by}")

    rows = sort_rows(rows, args.sort_by, args.ascending)

    print(f"Results file: {csv_path}")
    print(f"Sorted by: {args.sort_by} ({'ascending' if args.ascending else 'descending'})")
    print()
    print("Overview")
    print(
        tabulate(
            build_overview_table(rows),
            headers=[
                "Approach",
                "Embedding",
                "Reranker",
                "Qs",
                "Recall@5",
                "MeanDist@5",
                "MaxIoU@5",
                "nDCG@5",
            ],
            tablefmt="github",
        )
    )
    print()
    print("All Metrics")
    print(
        tabulate(
            build_full_metrics_table(rows, metrics),
            headers=["Approach", *metrics],
            tablefmt="github",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
