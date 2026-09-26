"""Protocol checks that do not need a model checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capdyn.agreement import agreement_features, n_agreement_features
from capdyn.selector import fuse_scores, pick, train_prior_index
from evaluation.lodo import run_lodo
from evaluation.nested_cv import run_nested_cv


def test_margin_and_prior():
    chosen, info = train_prior_index([0.2, 0.2, 0.2])
    assert chosen == 2 and info["tied"]
    chosen6, _ = train_prior_index([0.4, 0.4, 0.1, 0.4, 0.2, 0.4])
    assert chosen6 == 5
    sel, meta = pick(np.array([0.1, 0.9, 0.8]), prior=2, margin=0.05)
    assert sel == 1 and meta["used_fallback"] is False
    sel, meta = pick(np.array([0.1, 0.9, 0.8]), prior=2, margin=0.2)
    assert sel == 2 and meta["reason"] == "margin"
    sel, meta = pick(np.array([0.5, 0.5, 0.1]), prior=0, margin=0.0)
    assert sel == 0 and meta["reason"] == "tie"


def test_agreement_width():
    feats = agreement_features(["a", "a", "b"], use_math_verify=False)
    assert len(feats) == 7
    assert len(agreement_features(["a"] * 6, use_math_verify=False)) == n_agreement_features(6)
    # Non-transitive-style hand labels are not used: a==b and a==c with strings is transitive.
    assert feats[0] == 0.0 and feats[1] == 1.0


def test_fusion_shape():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(4, 3))
    fused = fuse_scores(s, s[::-1], 0.5)
    assert fused.shape == (4, 3)


def _synthetic(k=3, n_per_bench=6):
    records = []
    benches = ["gsm8k", "math", "amc"] if k else ["gsm8k"]
    # two benchmarks for nested CV; three when LODO needs them
    return records


def _make_records(benchmarks, n_each, k, rollouts=1, mean32=()):
    records = []
    rng = np.random.default_rng(0)
    for b_i, bench in enumerate(benchmarks):
        n_roll = rollouts if bench in mean32 else 1
        for q in range(n_each):
            base = rng.integers(0, 2, size=k)
            for roll in range(n_roll):
                correct = base.copy()
                if roll:
                    correct = rng.integers(0, 2, size=k)
                records.append(
                    {
                        "benchmark": bench,
                        "question_id": f"{bench}-{q}",
                        "rollout_id": roll,
                        "split_group_id": f"{bench}::{bench}-{q}",
                        "question": f"question {bench} number {q} unique",
                        "domain": "math",
                        "metric_type": "mean@32" if bench in mean32 else "greedy@1",
                        "agreement_mode": "exact",
                        "chat_mode": "math_system",
                        "answers": [f"ans-{bench}-{q}-{j}-{int(correct[j])}" for j in range(k)],
                        "correct": correct.tolist(),
                        "responses": [],
                        "gold": None,
                    }
                )
    hidden = rng.normal(size=(len(records), k, 16)).astype(np.float32)
    return records, hidden


def test_nested_cv_boundaries():
    records, hidden = _make_records(["gsm8k", "math"], 6, 3, rollouts=2, mean32={"gsm8k"})
    cfg = {
        "alphas": [0.25, 0.5],
        "margins": [0.0, 0.1],
        "svd_components": 4,
        "tfidf_max_features": 100,
        "tfidf_ngram": [1, 1],
        "lr_max_iter": 50,
        "n_outer": 2,
        "n_inner": 2,
        "seeds": [0],
    }
    result = run_nested_cv(records, hidden, cfg)
    assert len(result["folds"]) == 2
    for fold in result["folds"]:
        assert "capdyn_match" in fold["methods"]
        assert "majority_vote" not in fold["methods"]
        assert "vanilla_linear" not in fold["methods"]
        # Query-only and Representation-only are scored with the CapDyn margin.
        assert fold["margin"] in (0.0, 0.1)
        questions = {r["question"] for r in records}
        assert set(fold["fitted_questions"]).issubset(questions)
    # mean@32 groups were not split: every fold's predictions for one question share outer_fold
    groups = {}
    for fold in result["folds"]:
        for row in fold["predictions"]:
            groups.setdefault(row["split_group_id"], set()).add(row["outer_fold"])
    assert all(len(v) == 1 for v in groups.values())


def test_lodo_excludes_held_out():
    records, hidden = _make_records(["gsm8k", "math", "minerva"], 5, 3)
    cfg = {
        "alphas": [0.5],
        "margins": [0.0],
        "svd_components": 4,
        "tfidf_max_features": 50,
        "tfidf_ngram": [1, 1],
        "lr_max_iter": 40,
        "seeds": [0],
    }
    result = run_lodo(records, hidden, cfg, seed=0)
    held = {block["held_out"] for block in result["held_out"]}
    assert held == {"gsm8k", "math", "minerva"}
    for block in result["held_out"]:
        assert block["held_out"] not in block["train_benchmarks"]
        assert block["held_out_labels_used_for_fitting"] is False


if __name__ == "__main__":
    test_margin_and_prior()
    test_agreement_width()
    test_fusion_shape()
    test_nested_cv_boundaries()
    test_lodo_excludes_held_out()
    print("ok")
