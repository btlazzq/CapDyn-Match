"""Question-grouped nested splits.

The split unit is split_group_id (benchmark + question). Every historical
candidate is a column of that row, so candidates cannot fall into different
folds. mean@32 rollouts share split_group_id and therefore share a fold.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def unique_split_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    seen = set()
    for r in records:
        sid = r["split_group_id"]
        if sid in seen:
            continue
        seen.add(sid)
        rows.append(
            {
                "split_group_id": sid,
                "benchmark": r["benchmark"],
                "question_id": r["question_id"],
                "metric_type": r["metric_type"],
            }
        )
    return pd.DataFrame(rows).sort_values(["benchmark", "question_id"]).reset_index(drop=True)


def assign_stratified_grouped_folds(split_df: pd.DataFrame, n_splits: int, seed: int) -> np.ndarray:
    """Shuffle within each benchmark, then assign k % n_splits."""
    rng = np.random.RandomState(seed)
    folds = np.full(len(split_df), -1, dtype=int)
    for _, idx in split_df.groupby("benchmark").groups.items():
        idx = np.array(list(idx))
        order = rng.permutation(len(idx))
        idx = idx[order]
        for k, i in enumerate(idx):
            folds[i] = k % n_splits
    if np.any(folds < 0):
        raise RuntimeError("unassigned split groups")
    return folds


def outer_folds_for_seed(split_df: pd.DataFrame, n_outer: int, seed: int) -> pd.DataFrame:
    out = split_df.copy()
    out["outer_fold"] = assign_stratified_grouped_folds(out, n_outer, seed)
    return out


def inner_folds_for_outer_train(
    outer_train_split: pd.DataFrame, n_inner: int, seed: int, outer_fold: int
) -> pd.DataFrame:
    inner = outer_train_split.copy().reset_index(drop=True)
    inner_seed = int(seed) * 1000 + int(outer_fold)
    inner["inner_fold"] = assign_stratified_grouped_folds(inner, n_inner, inner_seed)
    return inner


def expand_groups_to_indices(records: list[dict], split_ids: Iterable[str]) -> np.ndarray:
    want = set(split_ids)
    idx = [i for i, r in enumerate(records) if r["split_group_id"] in want]
    return np.asarray(idx, dtype=int)


def assert_no_question_leak(train_ids: Iterable[str], val_ids: Iterable[str], name: str) -> None:
    leak = set(train_ids) & set(val_ids)
    if leak:
        raise RuntimeError(f"{name} question leak: {len(leak)} split_group_id overlap")


def assert_mean32_intact(records: list[dict], fold_by_split: dict[str, int]) -> None:
    found: dict[str, set[int]] = {}
    for r in records:
        if r.get("metric_type") != "mean@32":
            continue
        found.setdefault(r["split_group_id"], set()).add(int(fold_by_split[r["split_group_id"]]))
    bad = {k: v for k, v in found.items() if len(v) != 1}
    if bad:
        raise RuntimeError(f"mean@32 rollouts crossed folds: {list(bad)[:5]}")
