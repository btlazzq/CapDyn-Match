"""Score fusion, Train-Prior, and the margin gate.

Fusion, applied only inside CapDyn-Match:

    s_k = alpha * z(representation_score_k) + (1 - alpha) * z(capability_agreement_score_k)

z is a row-wise z-score across the K historical candidates of one question.
Zero-variance rows become zeros. eps is added only in the denominator of
rows whose population standard deviation is positive.

The margin gate selects the unique highest score when top1 - top2 >= tau.
Otherwise it returns Train-Prior. Ties for the highest score always fall back.
The comparison is strict greater-than for uniqueness, and >= for the margin.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

EPS = 1e-8


def row_zscore(v: np.ndarray, eps: float = EPS) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True, ddof=0)
    out = np.zeros_like(x, dtype=np.float64)
    nz = std.squeeze(axis=1) > 0
    if np.any(nz):
        out[nz] = (x[nz] - mean[nz]) / (std[nz] + eps)
    if np.any(~np.isfinite(out)):
        out[~np.isfinite(out)] = 0.0
    return out


def fuse_scores(s_rep: np.ndarray, s_agree: np.ndarray, alpha: float) -> np.ndarray:
    fused = float(alpha) * row_zscore(s_rep) + (1.0 - float(alpha)) * row_zscore(s_agree)
    return np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)


def train_prior_index(official_scores: Sequence[float]) -> tuple[int, dict]:
    """Iteration with the highest training-split accuracy.

    Ties select the latest tied iteration (highest index). That is Iter3 when
    K = 3 and Iter6 when K = 6. The rule uses only the scores passed in, which
    the caller must compute on the training split.
    """
    s = np.asarray(list(official_scores), dtype=np.float64)
    if s.ndim != 1 or s.shape[0] < 2:
        raise ValueError("need one training score per historical iteration")
    m = float(np.max(s))
    winners = np.flatnonzero(np.isclose(s, m, rtol=0.0, atol=1e-15))
    chosen = int(winners[-1])
    return chosen, {
        "scores": s.tolist(),
        "tied": bool(len(winners) > 1),
        "winners": winners.tolist(),
        "chosen": chosen,
    }


def pick(scores: np.ndarray, prior: int, margin: float) -> tuple[int, dict]:
    """Unique top1 with gap >= margin, else Train-Prior. Do not use argmax."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if s.shape[0] < 2:
        raise ValueError("need at least 2 candidate scores")
    if not np.all(np.isfinite(s)):
        return int(prior), {
            "selected": int(prior),
            "used_fallback": True,
            "reason": "non_finite",
            "score_gap": 0.0,
            "best": None,
            "second": None,
        }
    order = np.argsort(-s, kind="mergesort")
    best = int(order[0])
    second = int(order[1])
    gap = float(s[best] - s[second])
    unique_max = bool(s[best] > s[second])
    if unique_max and gap >= float(margin):
        return best, {
            "selected": best,
            "used_fallback": False,
            "reason": "unique_max",
            "score_gap": gap,
            "best": best,
            "second": second,
        }
    reason = "tie" if not unique_max else "margin"
    return int(prior), {
        "selected": int(prior),
        "used_fallback": True,
        "reason": reason,
        "score_gap": gap,
        "best": best,
        "second": second,
    }


def select_batch(
    scores: np.ndarray, prior: int, margin: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    n = scores.shape[0]
    selected = np.empty(n, dtype=int)
    fallback = np.empty(n, dtype=bool)
    gaps = np.empty(n, dtype=np.float64)
    reasons: list[str] = []
    for i in range(n):
        k, info = pick(scores[i], prior=prior, margin=margin)
        selected[i] = k
        fallback[i] = info["used_fallback"]
        gaps[i] = info["score_gap"]
        reasons.append(info["reason"])
    return selected, fallback, gaps, reasons


def gather_correct(correct: np.ndarray, selected: np.ndarray) -> np.ndarray:
    c = np.asarray(correct, dtype=int)
    k = np.asarray(selected, dtype=int)
    return c[np.arange(c.shape[0]), k]


def historical_union_correct(correct: np.ndarray) -> np.ndarray:
    """Oracle coverage: 1 if any historical iteration is correct. Not a selector."""
    return (np.asarray(correct, dtype=int).max(axis=1) > 0).astype(int)
