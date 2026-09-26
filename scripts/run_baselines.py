"""Evaluate individual iterations, query-only, CapAgree-only,
Representation-only, and CapDyn-Match on the same nested folds.

This entry calls the shared nested-CV runner. It does not fit a second selector.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_main import filter_records, jsonable  # noqa: E402
from capdyn.config import load_config  # noqa: E402
from evaluation.nested_cv import run_nested_cv  # noqa: E402

METHODS = [
    "query_only",
    "capagree_only",
    "representation_only",
    "capdyn_match",
    "train_prior",
    "historical_union",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--domain", required=True, choices=["math", "general", "code"])
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--hidden", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    records = filter_records(cfg, args)
    hidden = np.load(args.hidden)
    result = run_nested_cv(records, hidden, cfg)
    k = int(cfg["n_iterations"])
    names = [f"always_iter{j+1}" for j in range(k)] + METHODS
    by_seed: dict[int, dict[str, list]] = {}
    for fold in result["folds"]:
        by_seed.setdefault(fold["seed"], {name: [] for name in names})
        weight = fold["n_outer_test_questions"]
        for name in names:
            by_seed[fold["seed"]][name].append((fold["methods"][name]["question_micro"], weight))
    summary = {}
    for name in names:
        seed_scores = []
        for seed, cols in sorted(by_seed.items()):
            pairs = cols[name]
            w = np.array([p[1] for p in pairs], dtype=float)
            s = np.array([p[0] for p in pairs], dtype=float)
            seed_scores.append(float(np.average(s, weights=w)))
        arr = np.asarray(seed_scores, dtype=float)
        summary[name] = {
            "question_micro_mean": float(arr.mean()),
            "question_micro_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "per_seed": seed_scores,
        }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baselines.json").write_text(
        json.dumps(jsonable({"domain": args.domain, "methods": summary}), indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out / 'baselines.json'}")


if __name__ == "__main__":
    main()
