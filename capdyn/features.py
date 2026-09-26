"""Query TF-IDF/SVD features and per-iteration CapAgree correctness probes.

The query transform and every logistic regression are fit on the records
passed to fit(). Agreement features describe the candidate set; they do not
include gold labels. Each historical iteration has its own class-balanced
logistic regression on the shared feature vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize as l2_normalize

from capdyn.agreement import agreement_matrix, n_agreement_features
from capdyn.verifier import LR_MAX_ITER, SVD_COMPONENTS, safe_n_components

TFIDF_MAX_FEATURES = 30_000
TFIDF_NGRAM = (1, 2)


@dataclass
class CapabilityAgreementScorer:
    include_agreement: bool = True
    svd_components: int = SVD_COMPONENTS
    tfidf_max_features: int = TFIDF_MAX_FEATURES
    tfidf_ngram: tuple[int, int] = TFIDF_NGRAM
    lr_max_iter: int = LR_MAX_ITER
    tfidf: Optional[TfidfVectorizer] = None
    svd: Optional[TruncatedSVD] = None
    clfs: list = field(default_factory=list)
    n_components: Optional[int] = None
    constant_prob: Optional[np.ndarray] = None
    class_counts: dict = field(default_factory=dict)
    fitted_n_samples: int = 0
    random_state: int = 0
    train_questions: Optional[set[str]] = None
    n_candidates: int = 0

    def _question_features(self, questions: list[str], fit: bool) -> np.ndarray:
        if fit:
            self.tfidf = TfidfVectorizer(
                ngram_range=tuple(self.tfidf_ngram),
                max_features=self.tfidf_max_features,
            )
            x = self.tfidf.fit_transform(questions)
            n_comp = safe_n_components(x.shape[0], x.shape[1], self.svd_components)
            self.n_components = n_comp
            if n_comp is not None and n_comp >= 1:
                self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
                z = self.svd.fit_transform(x)
            else:
                self.svd = None
                z = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        else:
            if self.tfidf is None:
                raise RuntimeError("query TF-IDF is not fit")
            x = self.tfidf.transform(questions)
            if self.svd is not None:
                z = self.svd.transform(x)
            else:
                z = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        z = np.asarray(z, dtype=np.float64)
        z = l2_normalize(z, norm="l2")
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_x(self, records: list[dict], fit: bool) -> np.ndarray:
        questions = [r["question"] for r in records]
        qz = self._question_features(questions, fit=fit)
        if not self.include_agreement:
            return qz
        agr = agreement_matrix(records)
        expected = n_agreement_features(self.n_candidates)
        if agr.shape[1] != expected:
            raise ValueError(f"agreement width {agr.shape[1]} != {expected}")
        return np.concatenate([qz, agr], axis=1)

    def fit(self, records: list[dict], random_state: int = 0) -> "CapabilityAgreementScorer":
        self.random_state = int(random_state)
        self.fitted_n_samples = len(records)
        self.train_questions = {r["question"] for r in records}
        y = np.asarray([r["correct"] for r in records], dtype=int)
        if y.ndim != 2 or y.shape[1] < 2:
            raise ValueError("labels must be (N, K) with K >= 2")
        self.n_candidates = int(y.shape[1])
        self.class_counts = {
            f"i{k+1}": {0: int((y[:, k] == 0).sum()), 1: int((y[:, k] == 1).sum())}
            for k in range(self.n_candidates)
        }
        x = self._build_x(records, fit=True)
        constants = np.full(self.n_candidates, np.nan, dtype=np.float64)
        self.clfs = [None] * self.n_candidates
        for k in range(self.n_candidates):
            uniq = np.unique(y[:, k])
            if len(uniq) < 2:
                constants[k] = float(uniq[0])
                continue
            clf = LogisticRegression(
                max_iter=self.lr_max_iter,
                class_weight="balanced",
                solver="liblinear",
                random_state=self.random_state,
            )
            clf.fit(x, y[:, k])
            self.clfs[k] = clf
        self.constant_prob = constants
        return self

    def predict_proba(self, records: list[dict]) -> np.ndarray:
        if self.n_candidates < 2:
            raise RuntimeError("CapabilityAgreementScorer is not fit")
        n = len(records)
        for record in records:
            if len(record["correct"]) != self.n_candidates or len(record["answers"]) != self.n_candidates:
                raise ValueError("record candidate count does not match the fitted scorer")
        x = self._build_x(records, fit=False)
        out = np.zeros((n, self.n_candidates), dtype=np.float64)
        for k in range(self.n_candidates):
            if self.constant_prob is not None and np.isfinite(self.constant_prob[k]):
                out[:, k] = float(self.constant_prob[k])
                continue
            clf = self.clfs[k]
            if clf is None:
                raise RuntimeError(f"CapAgree probe for iteration {k+1} is not fit")
            p = clf.predict_proba(x)
            classes = list(clf.classes_)
            out[:, k] = p[:, classes.index(1)] if 1 in classes else 0.0
        return out


class QueryRouter:
    """Question TF-IDF/SVD only. No agreement features and no hidden states.

    The margin passed to select() is the CapDyn-Match tau for that fold.
    """

    def __init__(self, **scorer_kwargs):
        self.scorer = CapabilityAgreementScorer(include_agreement=False, **scorer_kwargs)

    def fit(self, records: list[dict], random_state: int = 0) -> "QueryRouter":
        self.scorer.fit(records, random_state=random_state)
        return self

    def score(self, records: list[dict]):
        return self.scorer.predict_proba(records)
