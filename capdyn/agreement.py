"""Answer normalization and cross-iteration agreement features.

One feature layout is used for every trajectory and every candidate count K >= 2.

Layout (length K + C(K, 2) + 1):
  - one-hot of the number of distinct answers, where distinctness is the
    number of connected components under answer equality (1 .. K);
  - pairwise equality in index order (i, j) with i < j;
  - majority component size divided by K.

For K = 3 the length is 7. Pairwise columns are (0,1), (0,2), (1,2).
The earlier three-iteration runner emitted pairs in the order (0,1), (1,2), (0,2)
and assigned a unique-count of 2 whenever equality was not a single transitive
chain through pairs (0,1) and (1,2). A linear probe is invariant to that column
permutation when equality is transitive. It is not invariant when equality is
non-transitive. This module follows the single N-candidate definition above.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

NO_BOXED = "<NO_BOXED>"
ANSWER_PATTERN_BOXED = r"(?i)\\boxed\s*{([^\n]+)}"


def extract_boxed_answer(response: str | None) -> Optional[str]:
    if response is None:
        return None
    try:
        match = re.search(ANSWER_PATTERN_BOXED, response)
    except Exception:
        return None
    if match is None:
        return None
    return match.group(1)


def normalize_extracted_answer(extracted: Optional[str]) -> str:
    if extracted is None:
        return NO_BOXED
    text = str(extracted).strip()
    if not text or text.lower() == "none":
        return NO_BOXED
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1].strip()
    text = re.sub(r"\s+", " ", text)
    if text.endswith("."):
        text = text[:-1].strip()
    return text if text else NO_BOXED


def extract_and_normalize(response: str | None) -> str:
    return normalize_extracted_answer(extract_boxed_answer(response))


def _looks_like_code_or_long(text: str) -> bool:
    if text is None:
        return False
    if "\n" in text or len(text) > 200:
        return True
    if "def " in text or "class " in text or "import " in text:
        return True
    return False


def answers_equal(a: str, b: str, use_math_verify: Optional[bool] = None) -> bool:
    """Equality on normalized answers.

    Math may use math_verify. Knowledge and code pass use_math_verify=False
    and compare strings only. Correctness labels are never consulted.
    """
    if a == b:
        return True
    if a == NO_BOXED or b == NO_BOXED:
        return False
    if use_math_verify is False:
        return False
    if use_math_verify is None and (_looks_like_code_or_long(a) or _looks_like_code_or_long(b)):
        return False
    try:
        from math_verify import parse, verify

        return bool(verify(parse(a), parse(b)))
    except Exception:
        return False


def _use_math_verify(record: dict | None, override: Optional[bool] = None) -> Optional[bool]:
    if override is not None:
        return override
    if record is None:
        return None
    if record.get("agreement_mode") == "exact":
        return False
    return None


def agreement_features(
    answers: Sequence[str],
    use_math_verify: Optional[bool] = None,
) -> list[float]:
    """Cross-iteration agreement vector. See module docstring."""
    k = len(answers)
    if k < 2:
        raise ValueError("need at least 2 historical answers")
    eq = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            e = float(answers_equal(answers[i], answers[j], use_math_verify=use_math_verify))
            eq[i][j] = eq[j][i] = e
    parent = list(range(k))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(k):
        for j in range(i + 1, k):
            if eq[i][j]:
                union(i, j)
    from collections import Counter

    sizes = Counter(find(i) for i in range(k))
    unique_count = len(sizes)
    majority_count = max(sizes.values()) if sizes else 0
    feats = [float(unique_count == u) for u in range(1, k + 1)]
    feats.extend(eq[i][j] for i in range(k) for j in range(i + 1, k))
    feats.append(majority_count / float(k))
    return feats


def agreement_matrix(records: list[dict], answer_key: str = "answers") -> "object":
    import numpy as np

    rows = []
    for record in records:
        rows.append(
            agreement_features(
                record[answer_key],
                use_math_verify=_use_math_verify(record),
            )
        )
    return np.asarray(rows, dtype=np.float64)


def n_agreement_features(k: int) -> int:
    return k + k * (k - 1) // 2 + 1
