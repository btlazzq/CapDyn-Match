"""Within-domain leave-one-benchmark-out evaluation.

    python scripts/run_lodo.py \
        --config configs/rzero_4b.yaml \
        --domain math \
        --candidates /path/to/candidates.jsonl \
        --hidden ./outputs/representations.npy \
        --output ./outputs/lodo
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
from evaluation.lodo import run_lodo  # noqa: E402
from scripts.run_main import filter_records, jsonable  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--domain", required=True, choices=["math", "general", "code"])
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--hidden", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    records = filter_records(cfg, args)
    hidden = np.load(args.hidden)
    result = run_lodo(records, hidden, cfg, seed=args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    preds = []
    public = []
    for block in result["held_out"]:
        preds.extend(block.pop("predictions"))
        public.append(block)
    pd.DataFrame(preds).to_csv(out / "predictions.csv", index=False)
    payload = {
        "domain": args.domain,
        "protocol": result["protocol"],
        "seed": result["seed"],
        "held_out_labels_used_for_fitting": False,
        "held_out": public,
    }
    (out / "lodo.json").write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    print(f"wrote {out / 'lodo.json'}")


if __name__ == "__main__":
    main()
