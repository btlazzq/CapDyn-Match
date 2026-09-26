"""Run CapDyn-Match nested cross-validation.

    python scripts/run_main.py \
        --config configs/rzero_4b.yaml \
        --domain math \
        --candidates /path/to/candidates.jsonl \
        --hidden ./outputs/representations.npy \
        --output ./outputs/main
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capdyn.config import load_config  # noqa: E402
from capdyn.records import load_records  # noqa: E402
from evaluation.nested_cv import run_nested_cv  # noqa: E402


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def filter_records(cfg, args):
    domains = cfg.get("domains") or {}
    spec = domains.get(args.domain) or {}
    benches = spec.get("benchmarks")
    mean32 = cfg.get("mean32_benchmarks") or []
    exact = set()
    for name, block in domains.items():
        if name != "math":
            exact.update(block.get("benchmarks") or [])
    exact.update(cfg.get("exact_agreement_benchmarks") or [])
    return load_records(
        args.candidates,
        n_iterations=int(cfg["n_iterations"]),
        mean32_benchmarks=mean32,
        exact_agreement_benchmarks=exact,
        benchmarks=benches,
        domain=args.domain,
    )


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
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    preds = []
    public_folds = []
    for fold in result["folds"]:
        preds.extend(fold.pop("predictions"))
        fold.pop("search_rows", None)
        fold.pop("fitted_questions", None)
        public_folds.append(fold)
    pd.DataFrame(preds).to_csv(out / "predictions.csv", index=False)
    payload = {"domain": args.domain, "trajectory": cfg.get("trajectory"), "folds": public_folds}
    (out / "nested_cv.json").write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    print(f"wrote {out / 'nested_cv.json'}")


if __name__ == "__main__":
    main()
