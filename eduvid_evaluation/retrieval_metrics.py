from __future__ import annotations


DEFAULT_RETRIEVAL_TOLERANCE_SECONDS = 15.0


def hit_covers_timestamp(hit: dict[str, object], timestamp_seconds: int | float) -> bool:
    start = hit.get("start")
    end = hit.get("end")
    if start is None or end is None:
        return False
    return float(start) <= float(timestamp_seconds) <= float(end)


def hit_within_tolerance(
    hit: dict[str, object],
    timestamp_seconds: int | float,
    tolerance_seconds: float = DEFAULT_RETRIEVAL_TOLERANCE_SECONDS,
) -> bool:
    start = hit.get("start")
    end = hit.get("end")
    if start is None or end is None:
        return False
    return (
        float(start) - float(tolerance_seconds)
        <= float(timestamp_seconds)
        <= float(end) + float(tolerance_seconds)
    )


def _mrr_from_hits(
    hits: list[dict[str, object]],
    predicate,
) -> float:
    for rank, hit in enumerate(hits, start=1):
        if predicate(hit):
            return 1.0 / float(rank)
    return 0.0


def compute_retrieval_metrics(
    hits: list[dict[str, object]],
    timestamp_seconds: int | float,
    tolerance_seconds: float = DEFAULT_RETRIEVAL_TOLERANCE_SECONDS,
) -> dict[str, float]:
    exact_hit = 1.0 if any(
        hit_covers_timestamp(hit, timestamp_seconds) for hit in hits
    ) else 0.0
    relaxed_hit = 1.0 if any(
        hit_within_tolerance(hit, timestamp_seconds, tolerance_seconds=tolerance_seconds)
        for hit in hits
    ) else 0.0
    return {
        "retrieval_hit_exact": exact_hit,
        "retrieval_mrr_exact": _mrr_from_hits(
            hits,
            lambda hit: hit_covers_timestamp(hit, timestamp_seconds),
        ),
        "retrieval_hit_relaxed": relaxed_hit,
        "retrieval_mrr_relaxed": _mrr_from_hits(
            hits,
            lambda hit: hit_within_tolerance(
                hit,
                timestamp_seconds,
                tolerance_seconds=tolerance_seconds,
            ),
        ),
    }
