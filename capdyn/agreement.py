"""Answer normalization and CapAgree features g(Y).

g(Y) has four values, in this order, for every K >= 2:
output diversity, modal-answer share, output entropy, average pairwise consistency.
Distinct answers are connected components under answer equality.
Entropy is the natural-log Shannon entropy of that component distribution.
"""

from __future__ import annotations

import math
import re
from collections import Counter
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
    k = len(answers)
    if k < 2:
        raise ValueError("need at least 2 historical answers")
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

    pair_hits = 0
    pair_total = 0
    for i in range(k):
        for j in range(i + 1, k):
            pair_total += 1
            if answers_equal(answers[i], answers[j], use_math_verify=use_math_verify):
                pair_hits += 1
                union(i, j)
    counts = list(Counter(find(i) for i in range(k)).values())
    diversity = len(counts) / float(k)
    modal_share = max(counts) / float(k)
    entropy = 0.0
    for count in counts:
        p = count / float(k)
        entropy -= p * math.log(p)
    pairwise = pair_hits / float(pair_total)
    return [diversity, modal_share, entropy, pairwise]


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
    if k < 2:
        raise ValueError("need at least 2 historical answers")
    return 4
