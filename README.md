# CapDyn-Match

## Overview

CapDyn-Match performs response-level historical capability recovery for self-evolving language model agents by selecting among responses from frozen historical iterations.

The same selector is used for a three-iteration trajectory and a six-iteration trajectory. Agent0, R-Zero 4B, and R-Zero 1.7B differ only by config: candidate count, reference encoder, and hidden layer.

CapAgree `g(Y)` is output diversity, modal-answer share, output entropy, and average pairwise consistency.

## Repository Structure

- `capdyn/` implements the method: frozen query-response representations, CapAgree features, the shared correctness probe, score fusion, Train-Prior, the margin gate, and the query-only router.
- `evaluation/` contains nested cross-validation, within-domain leave-one-benchmark-out, metrics, and the paired bootstrap. Representation-only and CapAgree-only are selector modes in this loop.
- `scripts/` contains the entry points, including parallel candidate generation and parallel representation extraction.
- `configs/` holds one file per trajectory.

## Installation

```bash
pip install -r requirements.txt
```

Optional, only if you generate candidates or compare math answers:

```bash
pip install vllm math-verify
```

`math-verify` is used when `agreement_mode` is `math`. Knowledge and code benchmarks use exact string equality. If `math-verify` is not installed, unequal math strings are treated as different answers.

## Data Preparation

Candidate responses are a JSONL file with one row per question, rollout, and iteration. `iteration` is 1-based.

Required fields:

- `question_id`
- `benchmark`
- `query`
- `iteration`
- `response`
- `final_answer`
- `correctness` (0 or 1)

Optional: `rollout_id` (default 0), `domain`, `gold`, `chat_mode`.

Rows that share `benchmark` and `question_id` stay in the same fold. For mean@32 benchmarks (`amc`, `aime2024`, `aime2025`), all 32 rollouts use the same question id and different `rollout_id` values. CapDyn-Match averages those rollouts inside the question.

`chat_mode` is `math_system`, `user_only`, `evalplus_prefill`, or `lcb_codeqwen`. Code and general benchmarks should be marked so agreement does not call the math equality checker. The loaders do that for the benchmark names listed in the configs.

A two-row schema example is in `examples/candidates.jsonl`. This repository does not include benchmark datasets.

## Candidate Responses

No model weights are included. Generate responses yourself, or pass responses that were already generated.

Historical iterations run at the same time when several GPUs are given. Extra GPUs split the prompt list.

```bash
python scripts/generate_candidates.py \
    --config configs/rzero_4b.yaml \
    --dataset /path/to/questions.jsonl \
    --model_path /path/to/iter1 \
    --model_path /path/to/iter2 \
    --model_path /path/to/iter3 \
    --gpus 0,1,2 \
    --output ./outputs/candidates.jsonl
```

Use `configs/agent0_4b.yaml` or `configs/rzero_1.7b_6iter.yaml` for the other trajectories. The six-iteration config expects six `--model_path` values, in iteration order.

Sampling defaults to temperature 0 and one sequence. mean@32 sampling has to match the evaluation that produced the official correctness labels. Set `generation.n` and `generation.temperature` in the config. The script does not grade code or multiple-choice execution. Pass `--score math_verify` only when the question file contains a `gold` field and the benchmark is a math task. Otherwise fill `correctness` from the official evaluator before training.

## Representation Extraction

Every historical response is encoded by one frozen reference model. The 4B configs use iteration 2 and layer 23. The six-iteration config uses iteration 4 and layer 18. The vector is the hidden state at the last non-padding token.

Shards run in parallel, one process per GPU.

```bash
python scripts/extract_representations.py \
    --config configs/agent0_4b.yaml \
    --model_path /path/to/model \
    --input_path /path/to/candidates \
    --output_path ./outputs/representations.npy \
    --gpus 0,1,2,3
```

Pass `--domain math`, `general`, or `code` to extract one domain. The output array is aligned with the candidate file after the same filtering and sort: `(N, K, D)`.

## Main Experiment

```bash
python scripts/run_main.py \
    --config configs/rzero_4b.yaml \
    --domain math \
    --candidates /path/to/candidates.jsonl \
    --hidden ./outputs/representations.npy \
    --output ./outputs/main
```

Domains are fit separately. Math, general, and code are three runs.

## Nested Cross-Validation

`scripts/run_main.py` is the nested CV entry. For each outer fold the inner loop fits TF-IDF, StandardScaler, Truncated SVD, and the logistic probes on inner-training questions only, then selects `alpha` and `tau`. Those components are refit on the full outer training split and applied to the outer test fold.

Frozen encoder states can be cached across folds. They do not depend on correctness labels. The cache must be produced by the reference encoder above, not by a different model per iteration.

Query-only, CapAgree-only, and Representation-only reuse the `tau` chosen for CapDyn-Match. They do not search a separate margin.

## Leave-One-Dataset-Out Evaluation

LODO is within one domain. Holding out a math benchmark trains on the other math benchmarks only.

```bash
python scripts/run_lodo.py \
    --config configs/rzero_4b.yaml \
    --domain math \
    --candidates /path/to/candidates.jsonl \
    --hidden ./outputs/representations.npy \
    --output ./outputs/lodo
```

Held-out correctness is used only for the final score. Inner validation is leave-one-training-benchmark-out, and those benchmark scores are averaged without weighting by size.

## Baselines and Ablations

```bash
python scripts/run_baselines.py \
    --config configs/rzero_4b.yaml \
    --domain math \
    --candidates /path/to/candidates.jsonl \
    --hidden ./outputs/representations.npy \
    --output ./outputs/baselines
```

The table includes:

- each historical iteration, including always-latest
- Query-only Router
- CapAgree-only
- Representation-only
- CapDyn-Match
- Train-Prior
- Historical Union, which is an oracle upper bound and not a trained selector

Representation-only uses the representation score, the margin gate, and Train-Prior. It does not use CapAgree.

## Six-Iteration Evaluation

`scripts/run_six_iteration.py` calls the same nested CV as `scripts/run_main.py` and defaults the config to `configs/rzero_1.7b_6iter.yaml`. There is no second selector implementation. Set `n_iterations` in the config to run any `K >= 2`.

```bash
python scripts/run_six_iteration.py \
    --domain math \
    --candidates /path/to/candidates.jsonl \
    --hidden ./outputs/representations.npy \
    --output ./outputs/six_iter
```

## Upstream Agent0 and R-Zero

- Agent0 `f775b5101e62fe92976831adf4a21a38fcc0a767`
- R-Zero `5699329d018d79535b7910abdedf5a6eebf355fd`

Dependency pins are in `third_party/`.

## Notes

- No model weights are included.
- No private datasets are included.
- Paths shown in this file are placeholders.
- Nested CV and LODO fit every learned transform on the training split only.
