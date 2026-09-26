"""Evaluation metrics.

question_micro
    Unweighted mean of per-question scores. A question with mean@32 rollouts
    contributes the mean of those rollouts, then counts as one question.

benchmark_macro
    Unweighted mean of per-benchmark question_micro scores.

question_weighted
    Mean of per-benchmark question_micro scores weighted by question count.
    On a single pooled set this equals question_micro.

historical_union, HCG, gain, damage, and turnover live in analysis/.
Switch counts below are relative to Train-Prior, which is the selector
diagnostic. They are not the iteration-to-iteration Gain/Damage tables.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np


def question_scores(records: list[dict], selected_correct: np.ndarray) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for rec, y in zip(records, selected_correct):
        buckets[rec["split_group_id"]].append(float(y))
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def question_micro(records: list[dict], selected_correct: np.ndarray) -> float:
    q = question_scores(records, selected_correct)
    if not q:
        return float("nan")
    return float(np.mean(list(q.values())))


def official_score(records: list[dict], selected_correct: np.ndarray) -> float:
    """Alias used by the training loop. Same number as question_micro."""
    return question_micro(records, selected_correct)


def per_benchmark(records: list[dict], selected_correct: np.ndarray) -> dict[str, dict]:
    by_b: dict[str, list[int]] = defaultdict(list)
    rec_b: dict[str, list[dict]] = defaultdict(list)
    for i, r in enumerate(records):
        by_b[r["benchmark"]].append(i)
        rec_b[r["benchmark"]].append(r)
    y = np.asarray(selected_correct)
    out = {}
    for b, idx in by_b.items():
        qmap = question_scores(rec_b[b], y[idx])
        out[b] = {
            "n_questions": len(qmap),
            "n_rows": len(idx),
            "question_micro": float(np.mean(list(qmap.values()))) if qmap else float("nan"),
            "metric_type": rec_b[b][0]["metric_type"],
        }
    return out


def benchmark_macro(per_bench: dict[str, dict]) -> float:
    if not per_bench:
        return float("nan")
    return float(np.mean([v["question_micro"] for v in per_bench.values()]))


def question_weighted(per_bench: dict[str, dict]) -> float:
    num = 0.0
    den = 0.0
    for v in per_bench.values():
        num += v["question_micro"] * v["n_questions"]
        den += v["n_questions"]
    return float(num / den) if den else float("nan")


def checkpoint_official_scores(records: list[dict]) -> list[float]:
    k = len(records[0]["correct"])
    out = []
    for j in range(k):
        y = np.asarray([r["correct"][j] for r in records], dtype=float)
        out.append(official_score(records, y))
    return out


def prior_switch_stats(
    records: list[dict],
    selected: np.ndarray,
    prior: int,
    selected_correct: np.ndarray,
) -> dict:
    """Switches relative to Train-Prior. Not iteration-to-iteration turnover."""
    n = len(records)
    prior_correct = np.asarray([r["correct"][prior] for r in records], dtype=int)
    sel = np.asarray(selected, dtype=int)
    y = np.asarray(selected_correct, dtype=int)
    switched = sel != int(prior)
    recovered = (prior_correct == 0) & (y == 1)
    damaged = (prior_correct == 1) & (y == 0)
    rec_n = int(recovered.sum())
    dam_n = int(damaged.sum())
    k = len(records[0]["correct"]) if records else 0
    dist = {f"iter{j+1}": int((sel == j).sum()) for j in range(k)}
    return {
        "n_rows": n,
        "switch_count": int(switched.sum()),
        "switch_rate": float(switched.mean()) if n else float("nan"),
        "recovered_vs_prior": rec_n,
        "damaged_vs_prior": dam_n,
        "net_vs_prior": rec_n - dam_n,
        "selection_distribution": dist,
    }


def oracle_gap_closed(score_sel: float, score_prior: float, score_union: float) -> Optional[float]:
    denom = score_union - score_prior
    if denom <= 0:
        return None
    return float((score_sel - score_prior) / denom)
