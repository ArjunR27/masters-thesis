from __future__ import annotations

DEFAULT_TOLERANCE_SECONDS = 15.0


def _ranges_overlap(
    hit_start: float,
    hit_end: float,
    range_start: float,
    range_end: float,
) -> bool:
    return hit_start <= range_end and hit_end >= range_start


def hit_covers_any_range(
    hit: dict,
    answer_ranges: tuple[tuple[float, float], ...],
) -> bool:
    start = hit.get("start")
    end = hit.get("end")
    if start is None or end is None:
        return False
    return any(
        _ranges_overlap(float(start), float(end), r_start, r_end)
        for r_start, r_end in answer_ranges
    )


def hit_covers_any_range_relaxed(
    hit: dict,
    answer_ranges: tuple[tuple[float, float], ...],
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
) -> bool:
    start = hit.get("start")
    end = hit.get("end")
    if start is None or end is None:
        return False
    return any(
        _ranges_overlap(
            float(start) - tolerance_seconds,
            float(end) + tolerance_seconds,
            r_start,
            r_end,
        )
        for r_start, r_end in answer_ranges
    )


def compute_range_retrieval_metrics(
    hits: list[dict],
    answer_ranges: tuple[tuple[float, float], ...],
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
) -> dict[str, float]:
    """
    Compute retrieval metrics against a set of ground-truth time ranges.

    A hit "covers" a range if their intervals overlap (exact) or overlap
    when the hit is expanded by ±tolerance_seconds (relaxed).
    """
    exact_hit = 1.0 if any(hit_covers_any_range(h, answer_ranges) for h in hits) else 0.0
    relaxed_hit = 1.0 if any(
        hit_covers_any_range_relaxed(h, answer_ranges, tolerance_seconds) for h in hits
    ) else 0.0

    mrr_exact = 0.0
    mrr_relaxed = 0.0
    for rank, hit in enumerate(hits, start=1):
        if mrr_exact == 0.0 and hit_covers_any_range(hit, answer_ranges):
            mrr_exact = 1.0 / rank
        if mrr_relaxed == 0.0 and hit_covers_any_range_relaxed(
            hit, answer_ranges, tolerance_seconds
        ):
            mrr_relaxed = 1.0 / rank
        if mrr_exact > 0.0 and mrr_relaxed > 0.0:
            break

    return {
        "retrieval_hit_exact": exact_hit,
        "retrieval_mrr_exact": mrr_exact,
        "retrieval_hit_relaxed": relaxed_hit,
        "retrieval_mrr_relaxed": mrr_relaxed,
    }
