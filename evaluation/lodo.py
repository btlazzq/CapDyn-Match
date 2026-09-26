"""Leave-one-benchmark-out inside one domain.

For held-out benchmark D:
  training benchmarks = the other benchmarks in the same domain
  test benchmark = D

TF-IDF, SVD, both probes, alpha, tau, and Train-Prior are fit on the training
benchmarks only. Inner validation is leave-one-training-benchmark-out. The
mean across those inner benchmarks is unweighted. Held-out correctness is used
only when the finished selector is scored.
"""

from __future__ import annotations

import numpy as np

from capdyn.features import QueryRouter
from capdyn.records import correct_matrix, subset
from capdyn.selector import fuse_scores, gather_correct, select_batch, train_prior_index
from evaluation.metrics import checkpoint_official_scores, official_score, prior_switch_stats
from evaluation.nested_cv import (
    _hyper_from_config,
    _pack_methods,
    _prediction_rows,
    _summarize,
    choose_hyperparams,
    fit_agreement,
    fit_representation,
)


class LeakageError(RuntimeError):
    pass


def _ids(records, key):
    return {str(r[key]) for r in records}


def assert_held_out_clean(train_records, test_records, held_out: str, fitted_questions: set[str] | None = None) -> None:
    train_b = {r["benchmark"] for r in train_records}
    test_b = {r["benchmark"] for r in test_records}
    if held_out in train_b:
        raise LeakageError(f"held-out benchmark {held_out} is in the training pool")
    if test_b != {held_out}:
        raise LeakageError(f"test benchmarks {sorted(test_b)} are not exactly {{{held_out}}}")
    for key in ("split_group_id", "question_id"):
        overlap = _ids(train_records, key) & _ids(test_records, key)
        if overlap:
            raise LeakageError(f"{key} overlaps the held-out split ({len(overlap)})")
    train_q = {r["question"] for r in train_records}
    test_q = {r["question"] for r in test_records}
    if train_q & test_q:
        raise LeakageError("question text overlaps the held-out benchmark")
    if fitted_questions is not None and fitted_questions & test_q:
        raise LeakageError("a feature transform was fit on held-out question text")


def run_lodo_one(
    records: list[dict],
    hidden: np.ndarray,
    held_out: str,
    cfg: dict,
    seed: int,
) -> dict:
    hp = _hyper_from_config(cfg)
    train_i = [i for i, r in enumerate(records) if r["benchmark"] != held_out]
    test_i = [i for i, r in enumerate(records) if r["benchmark"] == held_out]
    if not train_i or not test_i:
        raise RuntimeError(f"empty train or test split for held-out {held_out}")
    train_records = subset(records, train_i)
    test_records = subset(records, test_i)
    train_h = hidden[train_i]
    test_h = hidden[test_i]
    assert_held_out_clean(train_records, test_records, held_out)
    train_benchmarks = sorted({r["benchmark"] for r in train_records})
    if len(train_benchmarks) < 2:
        raise RuntimeError("domain-internal LODO needs at least two training benchmarks")

    buckets = {}
    for b in train_benchmarks:
        idx = [i for i, r in enumerate(train_records) if r["benchmark"] == b]
        buckets[b] = (subset(train_records, idx), train_h[idx])

    search_rows = []
    for val_b in train_benchmarks:
        inner_tr_i = [i for i, r in enumerate(train_records) if r["benchmark"] != val_b]
        inner_tr = subset(train_records, inner_tr_i)
        inner_h = train_h[inner_tr_i]
        inner_val, inner_val_h = buckets[val_b]
        if any(r["benchmark"] == held_out for r in inner_tr):
            raise LeakageError("held-out benchmark entered an inner-training split")
        rep = fit_representation(inner_tr, inner_h, seed, hp)
        agree = fit_agreement(inner_tr, seed, hp, include_agreement=True)
        assert_held_out_clean(inner_tr, test_records, held_out, agree.train_questions)
        s_rep = rep.score_groups(inner_val_h)
        s_agree = agree.predict_proba(inner_val)
        prior, _ = train_prior_index(checkpoint_official_scores(inner_tr))
        for alpha in hp["alphas"]:
            fused = fuse_scores(s_rep, s_agree, alpha)
            for margin in hp["margins"]:
                sel, _, _, _ = select_batch(fused, prior, margin)
                y = gather_correct(correct_matrix(inner_val), sel)
                stats = prior_switch_stats(inner_val, sel, prior, y)
                search_rows.append(
                    {
                        "val_benchmark": val_b,
                        "alpha": float(alpha),
                        "margin": float(margin),
                        "val_score": official_score(inner_val, y),
                        "damaged_vs_prior": stats["damaged_vs_prior"],
                        "prior": int(prior),
                    }
                )

    chosen = choose_hyperparams(search_rows)
    alpha, margin = chosen["alpha"], chosen["margin"]
    rep = fit_representation(train_records, train_h, seed, hp)
    agree = fit_agreement(train_records, seed, hp, include_agreement=True)
    query = QueryRouter(
        svd_components=hp["svd_components"],
        tfidf_max_features=hp["tfidf_max_features"],
        tfidf_ngram=hp["tfidf_ngram"],
        lr_max_iter=hp["lr_max_iter"],
    )
    query.fit(train_records, random_state=seed)
    assert_held_out_clean(train_records, test_records, held_out, agree.train_questions)
    assert_held_out_clean(train_records, test_records, held_out, query.scorer.train_questions)
    prior, prior_info = train_prior_index(checkpoint_official_scores(train_records))
    packed = _pack_methods(
        test_records,
        rep.score_groups(test_h),
        agree.predict_proba(test_records),
        query.score(test_records),
        prior,
        alpha,
        margin,
    )
    methods = {name: _summarize(name, payload) for name, payload in packed.items() if not name.startswith("_")}
    return {
        "held_out": held_out,
        "seed": int(seed),
        "alpha": alpha,
        "margin": margin,
        "train_prior": int(prior),
        "train_prior_info": prior_info,
        "train_benchmarks": train_benchmarks,
        "inner_choice": chosen,
        "methods": methods,
        "predictions": _prediction_rows(test_records, packed, seed, -1, prior, alpha, margin),
        "held_out_labels_used_for_fitting": False,
    }


def run_lodo(records: list[dict], hidden: np.ndarray, cfg: dict, seed: int | None = None) -> dict:
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.shape[0] != len(records):
        raise ValueError("hidden rows do not match records")
    seed = int(cfg.get("seeds", [0])[0] if seed is None else seed)
    benchmarks = sorted({r["benchmark"] for r in records})
    results = [run_lodo_one(records, hidden, b, cfg, seed) for b in benchmarks]
    return {"seed": seed, "held_out": results, "protocol": "within-domain leave-one-benchmark-out"}
