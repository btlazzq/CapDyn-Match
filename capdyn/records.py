"""Candidate-response schema.

Long-form JSONL, one row per (question, rollout, iteration):

    question_id, benchmark, query, iteration, response, final_answer, correctness

Optional: rollout_id, domain, gold, agreement_mode, chat_mode, metric_type.

iteration is 1-based (1 = first historical checkpoint). rollout_id defaults to 0.
Rows that share benchmark + question_id stay in the same fold. mean@32 rollouts
share that id and must not be split across folds.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from capdyn.agreement import extract_and_normalize, normalize_extracted_answer


def _read_rows(path: Path) -> list[dict]:
    text_path = Path(path)
    rows: list[dict] = []
    if text_path.suffix == ".json":
        payload = json.loads(text_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("rows") or payload.get("data")
        if not isinstance(payload, list):
            raise ValueError("JSON candidate file must be a list")
        rows = payload
    else:
        with text_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _iteration_index(value: Any, zero_based: bool) -> int:
    if isinstance(value, str) and value.startswith("i") and value[1:].isdigit():
        return int(value[1:]) - 1
    n = int(value)
    return n if zero_based else n - 1


def load_records(
    path: str | Path,
    *,
    n_iterations: int | None = None,
    mean32_benchmarks: Iterable[str] = (),
    exact_agreement_benchmarks: Iterable[str] = (),
    domain: str | None = None,
    benchmarks: Iterable[str] | None = None,
) -> list[dict]:
    raw = _read_rows(Path(path))
    if not raw:
        raise ValueError(f"no rows in {path}")
    mean32 = set(mean32_benchmarks)
    exact = set(exact_agreement_benchmarks)
    bench_filter = set(benchmarks) if benchmarks is not None else None

    if "answers" in raw[0] and "correct" in raw[0]:
        records = [_from_grouped(row, mean32, exact) for row in raw]
    else:
        records = _from_long(raw, n_iterations, mean32, exact)

    if bench_filter is not None:
        records = [r for r in records if r["benchmark"] in bench_filter]
    if domain is not None:
        records = [r for r in records if r.get("domain") in (None, domain)]
    if not records:
        raise ValueError("no records left after domain/benchmark filters")
    _validate(records, n_iterations)
    return records


def _from_grouped(row: dict, mean32: set[str], exact: set[str]) -> dict:
    benchmark = str(row["benchmark"])
    question_id = str(row["question_id"])
    rollout_id = int(row.get("rollout_id", 0))
    answers = [normalize_extracted_answer(None if a is None else str(a)) for a in row["answers"]]
    correct = [int(c) for c in row["correct"]]
    return _pack(row, benchmark, question_id, rollout_id, answers, correct, mean32, exact)


def _from_long(
    raw: list[dict],
    n_iterations: int | None,
    mean32: set[str],
    exact: set[str],
) -> list[dict]:
    iter_values = []
    for row in raw:
        iter_values.append(row.get("iteration"))
    numeric = []
    for v in iter_values:
        if isinstance(v, str) and v.startswith("i"):
            numeric.append(int(v[1:]))
        else:
            numeric.append(int(v))
    zero_based = 0 in numeric
    groups: dict[tuple, list] = defaultdict(list)
    for row in raw:
        key = (
            str(row["benchmark"]),
            str(row["question_id"]),
            int(row.get("rollout_id", 0)),
        )
        groups[key].append(row)
    records = []
    for (benchmark, question_id, rollout_id), items in groups.items():
        items = sorted(items, key=lambda r: _iteration_index(r["iteration"], zero_based))
        idxs = [_iteration_index(r["iteration"], zero_based) for r in items]
        if idxs != list(range(len(items))):
            raise ValueError(
                f"{benchmark}/{question_id}/rollout {rollout_id} iterations are not 0..K-1, got {idxs}"
            )
        if n_iterations is not None and len(items) != n_iterations:
            raise ValueError(
                f"{benchmark}/{question_id} has {len(items)} iterations, config expects {n_iterations}"
            )
        answers = []
        correct = []
        responses = []
        for item in items:
            final = item.get("final_answer")
            if final is None or str(final).strip() == "":
                answers.append(extract_and_normalize(item.get("response")))
            else:
                answers.append(normalize_extracted_answer(str(final)))
            if "correctness" not in item and "correct" not in item:
                raise ValueError(f"{benchmark}/{question_id} is missing correctness")
            correct.append(int(item.get("correctness", item.get("correct"))))
            responses.append(item.get("response") or "")
        base = dict(items[0])
        base["responses"] = responses
        records.append(_pack(base, benchmark, question_id, rollout_id, answers, correct, mean32, exact))
    records.sort(key=lambda r: (r["benchmark"], r["question_id"], r["rollout_id"]))
    return records


def _pack(row, benchmark, question_id, rollout_id, answers, correct, mean32, exact) -> dict:
    if len(answers) != len(correct):
        raise ValueError("answers and correctness differ in length")
    metric = row.get("metric_type")
    if not metric:
        metric = "mean@32" if benchmark in mean32 else "greedy@1"
    mode = row.get("agreement_mode")
    if not mode:
        mode = "exact" if benchmark in exact else "math"
    split_group_id = row.get("split_group_id") or f"{benchmark}::{question_id}"
    return {
        "benchmark": benchmark,
        "question_id": question_id,
        "rollout_id": int(rollout_id),
        "split_group_id": str(split_group_id),
        "question": row.get("query") or row.get("question") or "",
        "domain": row.get("domain"),
        "metric_type": metric,
        "agreement_mode": "exact" if mode == "exact" else "math",
        "chat_mode": row.get("chat_mode") or "math_system",
        "answers": list(answers),
        "correct": [int(c) for c in correct],
        "responses": list(row.get("responses") or []),
        "gold": row.get("gold"),
        "encoder_user": row.get("encoder_user"),
        "encoder_prefix": row.get("encoder_prefix"),
        "task_prompt": row.get("task_prompt"),
    }


def _validate(records: list[dict], n_iterations: int | None) -> None:
    k = len(records[0]["correct"])
    if n_iterations is not None and k != n_iterations:
        raise ValueError(f"records have K={k}, config n_iterations={n_iterations}")
    for r in records:
        if len(r["correct"]) != k or len(r["answers"]) != k:
            raise ValueError(f"inconsistent K at {r['split_group_id']}")
        if any(c not in (0, 1) for c in r["correct"]):
            raise ValueError(f"correctness must be 0 or 1 at {r['split_group_id']}")


def correct_matrix(records: list[dict]):
    import numpy as np

    return np.asarray([r["correct"] for r in records], dtype=int)


def subset(records: list[dict], idx) -> list[dict]:
    return [records[i] for i in idx]
