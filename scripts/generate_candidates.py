"""Generate historical candidate responses in parallel.

Each historical iteration is a separate frozen model. Iterations run at the
same time when more than one GPU is listed. Extra GPUs split the prompt list
for a given iteration. The reference encoder is not used here.

Example:

    python scripts/generate_candidates.py \
        --config configs/rzero_4b.yaml \
        --dataset /path/to/questions.jsonl \
        --model_path /path/to/iter1 \
        --model_path /path/to/iter2 \
        --model_path /path/to/iter3 \
        --gpus 0,1,2 \
        --output ./outputs/candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capdyn.agreement import extract_and_normalize, normalize_extracted_answer  # noqa: E402
from capdyn.config import load_config  # noqa: E402
from capdyn.representations import MATH_SYSTEM_PROMPT, build_prompt  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def maybe_correct(response: str, gold, mode: str) -> int | None:
    if gold is None or mode in ("", "none"):
        return None
    pred = extract_and_normalize(response)
    gold_n = normalize_extracted_answer(str(gold))
    if mode == "exact":
        return int(pred == gold_n)
    if mode == "math_verify":
        try:
            from math_verify import parse, verify

            return int(bool(verify(parse(gold_n), parse(pred))))
        except Exception:
            return int(pred == gold_n)
    raise ValueError(f"unknown score mode {mode}")


def worker(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = load_config(args.config)
    gen = cfg.get("generation") or {}
    system_prompt = cfg.get("system_prompt", MATH_SYSTEM_PROMPT)
    rows = read_jsonl(Path(args.input))
    model_path = args.model_path[0] if isinstance(args.model_path, list) else args.model_path
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = [
        build_prompt(
            tokenizer,
            row.get("query") or row["question"],
            chat_mode=row.get("chat_mode") or "math_system",
            extra=row,
            system_prompt=system_prompt,
        )
        for row in rows
    ]
    temperature = float(gen.get("temperature", 0.0))
    top_p = float(gen.get("top_p", 1.0))
    max_new = int(gen.get("max_new_tokens", 4096))
    n = int(gen.get("n", 1))
    backend = args.backend
    texts: list[list[str]] = []
    if backend == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(model=model_path, tensor_parallel_size=1, dtype=cfg.get("dtype", "bfloat16"), trust_remote_code=True)
        sampling = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new, n=n)
        outputs = llm.generate(prompts, sampling)
        for out in outputs:
            texts.append([c.text for c in out.outputs])
    else:
        import torch
        from transformers import AutoModelForCausalLM

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.get("dtype", "bfloat16")]
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            gen_kwargs = dict(
                max_new_tokens=max_new,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=temperature > 0,
            )
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            with torch.inference_mode():
                gen_ids = model.generate(**enc, **gen_kwargs)
            prompt_len = enc["input_ids"].shape[1]
            decoded = []
            for seq in gen_ids:
                decoded.append(tokenizer.decode(seq[prompt_len:], skip_special_tokens=True))
            texts.append(decoded)
    score_mode = args.score
    out_rows = []
    iteration = int(args.iteration)
    for row, samples in zip(rows, texts):
        for s, response in enumerate(samples):
            rollout = int(row.get("rollout_id", s if n > 1 else 0))
            if n > 1:
                rollout = s
            correctness = maybe_correct(response, row.get("gold"), score_mode)
            item = {
                "question_id": row["question_id"],
                "benchmark": row["benchmark"],
                "domain": row.get("domain"),
                "query": row.get("query") or row.get("question"),
                "rollout_id": rollout,
                "iteration": iteration,
                "response": response,
                "final_answer": extract_and_normalize(response),
                "gold": row.get("gold"),
                "chat_mode": row.get("chat_mode") or "math_system",
            }
            if correctness is not None:
                item["correctness"] = correctness
            out_rows.append(item)
    write_jsonl(Path(args.output), out_rows)


def launch(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    n_iter = int(cfg["n_iterations"])
    if len(args.model_path) != n_iter:
        raise SystemExit(f"config expects {n_iter} models, got {len(args.model_path)}")
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("pass at least one GPU id, or cpu")
    rows = read_jsonl(Path(args.dataset))
    n_data = max(1, len(gpus) // n_iter)
    work = Path(args.output).parent / (Path(args.output).stem + "_shards")
    work.mkdir(parents=True, exist_ok=True)
    procs = []
    shard_outputs = []
    job = 0
    for iteration, model in enumerate(args.model_path, start=1):
        for shard in range(n_data):
            shard_rows = [row for i, row in enumerate(rows) if i % n_data == shard]
            in_path = work / f"iter{iteration}_shard{shard}_in.jsonl"
            out_path = work / f"iter{iteration}_shard{shard}_out.jsonl"
            write_jsonl(in_path, shard_rows)
            gpu = gpus[job % len(gpus)]
            job += 1
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--config",
                args.config,
                "--model_path",
                model,
                "--iteration",
                str(iteration),
                "--gpu",
                gpu,
                "--input",
                str(in_path),
                "--output",
                str(out_path),
                "--backend",
                args.backend,
                "--score",
                args.score,
            ]
            procs.append((subprocess.Popen(cmd), out_path))
            shard_outputs.append(out_path)
    failed = False
    for proc, out_path in procs:
        code = proc.wait()
        if code != 0:
            failed = True
            print(f"worker failed ({code}): {out_path}", file=sys.stderr)
    if failed:
        raise SystemExit(1)
    merged = []
    for path in shard_outputs:
        merged.extend(read_jsonl(path))
    merged.sort(key=lambda r: (r["benchmark"], str(r["question_id"]), int(r["rollout_id"]), int(r["iteration"])))
    write_jsonl(Path(args.output), merged)
    print(f"wrote {len(merged)} rows to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--model_path", action="append", default=[])
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--output")
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--score", choices=["none", "exact", "math_verify"], default="none")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        if not args.dataset or not args.output or not args.model_path:
            raise SystemExit("--dataset, --output, and one --model_path per iteration are required")
        launch(args)


if __name__ == "__main__":
    main()
