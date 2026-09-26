"""Question-level paired bootstrap.

Resamples questions with replacement. The interval is the 2.5 and 97.5
percentiles of the mean paired difference. Default size and seed match the
nested-CV runs: 10000 replicates, seed 20260912.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_BOOT = 10_000
BOOTSTRAP_SEED = 20260912


def paired_bootstrap(
    question_df: pd.DataFrame,
    method_col: str,
    baseline_col: str,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    y = question_df[method_col].to_numpy(dtype=np.float64)
    b = question_df[baseline_col].to_numpy(dtype=np.float64)
    delta = y - b
    n = len(delta)
    if n == 0:
        return {
            "n_questions": 0,
            "mean_delta": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "p_value": float("nan"),
        }
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    boot = delta[idx].mean(axis=1)
    n_pos = int((boot > 0).sum())
    n_neg = int((boot < 0).sum())
    p = 2.0 * min((n_pos + 1) / (n_boot + 1), (n_neg + 1) / (n_boot + 1))
    return {
        "n_questions": n,
        "mean_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "p_value": float(min(1.0, p)),
        "n_boot": n_boot,
        "seed": seed,
    }


def question_frame(records: list[dict], columns: dict[str, np.ndarray]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "split_group_id": [r["split_group_id"] for r in records],
            "benchmark": [r["benchmark"] for r in records],
        }
    )
    for name, arr in columns.items():
        df[name] = np.asarray(arr, dtype=float)
    agg = {name: "mean" for name in columns}
    agg["benchmark"] = "first"
    return df.groupby("split_group_id", as_index=False).agg(agg)
