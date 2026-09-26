"""Nested cross-validation.

Outer fold
  outer training questions
    inner folds: fit scaler, SVD, TF-IDF, probes; select alpha and tau
  refit those components on the full outer training split
  evaluate on the outer test fold

Query-only, CapAgree-only, and Representation-only reuse the tau (margin)
selected for CapDyn-Match. They do not run a separate margin search.
Train-Prior is the training-split accuracy argmax and is recomputed on the
split that is currently legal: inner-train during search, outer-train at test.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from capdyn.features import CapabilityAgreementScorer, QueryRouter
from capdyn.records import correct_matrix, subset
from capdyn.selector import (
    fuse_scores,
    gather_correct,
    historical_union_correct,
    select_batch,
    train_prior_index,
)
from capdyn.verifier import RepresentationProbe
from evaluation.metrics import (
    benchmark_macro,
    checkpoint_official_scores,
    official_score,
    oracle_gap_closed,
    per_benchmark,
    prior_switch_stats,
    question_micro,
    question_weighted,
)
from evaluation.splits import (
    assert_mean32_intact,
    assert_no_question_leak,
    expand_groups_to_indices,
    inner_folds_for_outer_train,
    outer_folds_for_seed,
    unique_split_table,
)


def _hyper_from_config(cfg: dict) -> dict:
    return {
        "alphas": tuple(cfg.get("alphas", (0.25, 0.50, 0.75))),
        "margins": tuple(cfg.get("margins", (0.0, 0.05, 0.10))),
        "svd_components": int(cfg.get("svd_components", 128)),
        "tfidf_max_features": int(cfg.get("tfidf_max_features", 30000)),
        "tfidf_ngram": tuple(cfg.get("tfidf_ngram", (1, 2))),
        "lr_max_iter": int(cfg.get("lr_max_iter", 400)),
    }


def fit_representation(records, hidden, random_state: int, hp: dict) -> RepresentationProbe:
    h = np.asarray(hidden, dtype=np.float32)
    n, k, d = h.shape
    y = correct_matrix(records).reshape(-1)
    probe = RepresentationProbe(svd_components=hp["svd_components"], lr_max_iter=hp["lr_max_iter"])
    probe.fit(h.reshape(n * k, d), y, random_state=random_state)
    return probe


def fit_agreement(records, random_state: int, hp: dict, include_agreement: bool) -> CapabilityAgreementScorer:
    scorer = CapabilityAgreementScorer(
        include_agreement=include_agreement,
        svd_components=hp["svd_components"],
        tfidf_max_features=hp["tfidf_max_features"],
        tfidf_ngram=hp["tfidf_ngram"],
        lr_max_iter=hp["lr_max_iter"],
    )
    scorer.fit(records, random_state=random_state)
    return scorer


def choose_hyperparams(rows: list[dict]) -> dict:
    """Max mean inner official score; then less prior-damage, larger margin, alpha nearer 0.5."""
    if not rows:
        raise RuntimeError("empty inner search")
    df = pd.DataFrame(rows)
    agg = df.groupby(["alpha", "margin"], as_index=False).agg(
        score=("val_score", "mean"),
        damaged=("damaged_vs_prior", "sum"),
    )
    agg["alpha_dist"] = (agg["alpha"] - 0.5).abs()
    agg = agg.sort_values(
        by=["score", "damaged", "margin", "alpha_dist"],
        ascending=[False, True, False, True],
    )
    best = agg.iloc[0]
    return {
        "alpha": float(best["alpha"]),
        "margin": float(best["margin"]),
        "inner_mean_score": float(best["score"]),
        "inner_damaged_vs_prior": int(best["damaged"]),
    }


def _pack_methods(
    records: list[dict],
    s_rep: np.ndarray,
    s_agree: np.ndarray,
    s_query: np.ndarray,
    prior: int,
    alpha: float,
    margin: float,
) -> dict:
    fused = fuse_scores(s_rep, s_agree, alpha)
    cmat = correct_matrix(records)
    n = len(records)
    cap_sel, cap_fb, cap_gap, cap_reason = select_batch(fused, prior, margin)
    rep_sel, rep_fb, _, _ = select_batch(s_rep, prior, margin)
    agree_sel, agree_fb, _, _ = select_batch(s_agree, prior, margin)
    query_sel, query_fb, _, _ = select_batch(s_query, prior, margin)
    fused_argmax, _, _, _ = select_batch(fused, prior, 0.0)
    methods = {
        "train_prior": np.full(n, int(prior), dtype=int),
        "query_only": query_sel,
        "capagree_only": agree_sel,
        "representation_only": rep_sel,
        "fused_argmax": fused_argmax,
        "capdyn_match": cap_sel,
    }
    for j in range(cmat.shape[1]):
        methods[f"always_iter{j+1}"] = np.full(n, j, dtype=int)
    packed = {}
    fallbacks = {
        "capdyn_match": cap_fb,
        "representation_only": rep_fb,
        "capagree_only": agree_fb,
        "query_only": query_fb,
    }
    for name, sel in methods.items():
        y = gather_correct(cmat, sel)
        packed[name] = {
            "selected": sel,
            "correct": y,
            "question_micro": question_micro(records, y),
            "per_benchmark": per_benchmark(records, y),
            "fallback_rate": float(np.mean(fallbacks[name])) if name in fallbacks else 0.0,
            **prior_switch_stats(records, sel, prior, y),
        }
    union = historical_union_correct(cmat)
    packed["historical_union"] = {
        "selected": None,
        "correct": union,
        "question_micro": question_micro(records, union),
        "per_benchmark": per_benchmark(records, union),
        "fallback_rate": float("nan"),
    }
    packed["_scores"] = {
        "representation": s_rep,
        "capagree": s_agree,
        "query": s_query,
        "fused": fused,
        "gap": cap_gap,
        "reason": cap_reason,
        "fallback": cap_fb,
    }
    return packed


def _summarize(name: str, packed_method: dict) -> dict:
    per_b = packed_method["per_benchmark"]
    return {
        "question_micro": packed_method["question_micro"],
        "benchmark_macro": benchmark_macro(per_b),
        "question_weighted": question_weighted(per_b),
        "per_benchmark": per_b,
        "fallback_rate": packed_method.get("fallback_rate"),
        "recovered_vs_prior": packed_method.get("recovered_vs_prior"),
        "damaged_vs_prior": packed_method.get("damaged_vs_prior"),
    }


def run_outer_fold(
    records: list[dict],
    hidden: np.ndarray,
    split_df: pd.DataFrame,
    seed: int,
    outer_fold: int,
    n_inner: int,
    hp: dict,
) -> dict:
    outer_test_ids = split_df.loc[split_df["outer_fold"] == outer_fold, "split_group_id"].tolist()
    outer_train_ids = split_df.loc[split_df["outer_fold"] != outer_fold, "split_group_id"].tolist()
    assert_no_question_leak(outer_train_ids, outer_test_ids, f"seed{seed}-outer{outer_fold}")
    train_idx = expand_groups_to_indices(records, outer_train_ids)
    test_idx = expand_groups_to_indices(records, outer_test_ids)
    train_records = subset(records, train_idx)
    test_records = subset(records, test_idx)
    train_hidden = hidden[train_idx]
    test_hidden = hidden[test_idx]

    inner_split = inner_folds_for_outer_train(
        split_df.loc[split_df["outer_fold"] != outer_fold].copy(),
        n_inner=n_inner,
        seed=seed,
        outer_fold=outer_fold,
    )
    search_rows = []
    for inner_fold in range(n_inner):
        inner_val_ids = inner_split.loc[inner_split["inner_fold"] == inner_fold, "split_group_id"].tolist()
        inner_train_ids = inner_split.loc[inner_split["inner_fold"] != inner_fold, "split_group_id"].tolist()
        assert_no_question_leak(inner_train_ids, inner_val_ids, f"seed{seed}-outer{outer_fold}-inner{inner_fold}")
        if set(inner_val_ids) & set(outer_test_ids):
            raise RuntimeError("inner validation leaked into the outer test fold")
        it_idx = expand_groups_to_indices(records, inner_train_ids)
        iv_idx = expand_groups_to_indices(records, inner_val_ids)
        it_rec, iv_rec = subset(records, it_idx), subset(records, iv_idx)
        rep = fit_representation(it_rec, hidden[it_idx], seed, hp)
        agree = fit_agreement(it_rec, seed, hp, include_agreement=True)
        s_rep = rep.score_groups(hidden[iv_idx])
        s_agree = agree.predict_proba(iv_rec)
        prior, _ = train_prior_index(checkpoint_official_scores(it_rec))
        val_questions = {r["question"] for r in iv_rec}
        if agree.train_questions and val_questions & agree.train_questions:
            raise RuntimeError("CapAgree TF-IDF saw inner-validation question text")
        for alpha in hp["alphas"]:
            fused = fuse_scores(s_rep, s_agree, alpha)
            for margin in hp["margins"]:
                sel, _, _, _ = select_batch(fused, prior, margin)
                y = gather_correct(correct_matrix(iv_rec), sel)
                stats = prior_switch_stats(iv_rec, sel, prior, y)
                search_rows.append(
                    {
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "alpha": float(alpha),
                        "margin": float(margin),
                        "val_score": official_score(iv_rec, y),
                        "damaged_vs_prior": stats["damaged_vs_prior"],
                        "prior": int(prior),
                    }
                )

    chosen = choose_hyperparams(search_rows)
    alpha, margin = chosen["alpha"], chosen["margin"]
    rep = fit_representation(train_records, train_hidden, seed, hp)
    agree = fit_agreement(train_records, seed, hp, include_agreement=True)
    query = QueryRouter(
        svd_components=hp["svd_components"],
        tfidf_max_features=hp["tfidf_max_features"],
        tfidf_ngram=hp["tfidf_ngram"],
        lr_max_iter=hp["lr_max_iter"],
    )
    query.fit(train_records, random_state=seed)
    test_questions = {r["question"] for r in test_records}
    for scorer in (agree, query.scorer):
        if scorer.train_questions and test_questions & scorer.train_questions:
            raise RuntimeError("query features were fit on outer-test question text")
    s_rep = rep.score_groups(test_hidden)
    s_agree = agree.predict_proba(test_records)
    s_query = query.score(test_records)
    prior, prior_info = train_prior_index(checkpoint_official_scores(train_records))
    packed = _pack_methods(
        test_records,
        s_rep,
        s_agree,
        s_query,
        prior,
        alpha,
        margin,
    )
    methods = {}
    for name, payload in packed.items():
        if name.startswith("_"):
            continue
        methods[name] = _summarize(name, payload)
    cap = methods["capdyn_match"]["question_micro"]
    pri = methods["train_prior"]["question_micro"]
    uni = methods["historical_union"]["question_micro"]
    return {
        "seed": seed,
        "outer_fold": outer_fold,
        "alpha": alpha,
        "margin": margin,
        "train_prior": int(prior),
        "train_prior_info": prior_info,
        "inner_choice": chosen,
        "n_outer_train_questions": len(set(outer_train_ids)),
        "n_outer_test_questions": len(set(outer_test_ids)),
        "fitted_questions": sorted(agree.train_questions or []),
        "methods": methods,
        "oracle_gap_closed": oracle_gap_closed(cap, pri, uni),
        "predictions": _prediction_rows(test_records, packed, seed, outer_fold, prior, alpha, margin),
        "search_rows": search_rows,
    }


def _prediction_rows(records, packed, seed, outer_fold, prior, alpha, margin) -> list[dict]:
    sel = packed["capdyn_match"]["selected"]
    y = packed["capdyn_match"]["correct"]
    scores = packed["_scores"]
    rows = []
    k = correct_matrix(records).shape[1]
    for i, r in enumerate(records):
        row = {
            "seed": seed,
            "outer_fold": outer_fold,
            "benchmark": r["benchmark"],
            "question_id": r["question_id"],
            "rollout_id": r["rollout_id"],
            "split_group_id": r["split_group_id"],
            "alpha": alpha,
            "margin": margin,
            "train_prior": int(prior),
            "selected_iteration": int(sel[i]) + 1,
            "selected_correct": int(y[i]),
            "used_fallback": bool(scores["fallback"][i]),
            "fallback_reason": scores["reason"][i],
            "score_gap": float(scores["gap"][i]),
        }
        for j in range(k):
            row[f"correct_iter{j+1}"] = int(r["correct"][j])
            row[f"representation_score_iter{j+1}"] = float(scores["representation"][i, j])
            row[f"capagree_score_iter{j+1}"] = float(scores["capagree"][i, j])
            row[f"fused_score_iter{j+1}"] = float(scores["fused"][i, j])
        rows.append(row)
    return rows


def run_nested_cv(
    records: list[dict],
    hidden: np.ndarray,
    cfg: dict,
    *,
    seeds: Optional[list[int]] = None,
) -> dict:
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.shape[0] != len(records) or hidden.shape[1] != len(records[0]["correct"]):
        raise ValueError(
            f"hidden shape {hidden.shape} does not match records "
            f"({len(records)}, {len(records[0]['correct'])})"
        )
    hp = _hyper_from_config(cfg)
    n_outer = int(cfg.get("n_outer", 5))
    n_inner = int(cfg.get("n_inner", 5))
    seeds = list(seeds if seeds is not None else cfg.get("seeds", (0, 1, 2)))
    split_base = unique_split_table(records)
    folds = []
    for seed in seeds:
        split_df = outer_folds_for_seed(split_base, n_outer, int(seed))
        fold_map = dict(zip(split_df["split_group_id"], split_df["outer_fold"].astype(int)))
        assert_mean32_intact(records, fold_map)
        for outer in range(n_outer):
            folds.append(run_outer_fold(records, hidden, split_df, int(seed), outer, n_inner, hp))
    return {"folds": folds, "n_outer": n_outer, "n_inner": n_inner, "seeds": seeds}
