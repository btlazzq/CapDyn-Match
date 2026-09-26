"""Extract shared frozen-encoder representations.

All historical responses for a question use the reference encoder in the
config (not the iteration that produced the response). Shards run in parallel,
one process per GPU.

    python scripts/extract_representations.py \
        --config configs/rzero_4b.yaml \
        --model_path /path/to/reference-encoder \
        --input_path /path/to/candidates.jsonl \
        --output_path ./outputs/representations.npy \
        --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capdyn.config import load_config  # noqa: E402
from capdyn.records import load_records  # noqa: E402
from capdyn.representations import MATH_SYSTEM_PROMPT, build_encoder_text, encode_texts, load_encoder  # noqa: E402


def domain_benchmarks(cfg: dict, domain: str | None) -> tuple[list[str] | None, set[str]]:
    domains = cfg.get("domains") or {}
    mean32 = set(cfg.get("mean32_benchmarks") or [])
    if domain is None:
        benches = []
        for spec in domains.values():
            benches.extend(spec.get("benchmarks") or [])
        return (benches or None), mean32
    spec = domains.get(domain) or {}
    return spec.get("benchmarks"), mean32


def exact_benchmarks(cfg: dict) -> set[str]:
    found = set()
    for name, spec in (cfg.get("domains") or {}).items():
        if name != "math":
            found.update(spec.get("benchmarks") or [])
    found.update(cfg.get("exact_agreement_benchmarks") or [])
    return found


def worker(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = load_config(args.config)
    benches, mean32 = domain_benchmarks(cfg, args.domain)
    records = load_records(
        args.input_path,
        n_iterations=int(cfg["n_iterations"]),
        mean32_benchmarks=mean32,
        exact_agreement_benchmarks=exact_benchmarks(cfg),
        benchmarks=benches,
        domain=args.domain,
    )
    shard_records = [r for i, r in enumerate(records) if i % args.n_shards == args.shard]
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_encoder(args.model_path, cfg.get("dtype", "bfloat16"), device)
    layer = int(cfg["layer_index"])
    n_layers = int(getattr(model.config, "num_hidden_layers"))
    hidden_size = int(getattr(model.config, "hidden_size"))
    if n_layers < layer + 1:
        raise RuntimeError(f"model has {n_layers} layers, config asks for layer {layer}")
    expected = cfg.get("hidden_size")
    if expected is not None and int(expected) != hidden_size:
        raise RuntimeError(f"config hidden_size {expected} != model hidden_size {hidden_size}")
    system_prompt = cfg.get("system_prompt", MATH_SYSTEM_PROMPT)
    max_length = int(cfg.get("max_length", 8192))
    batch_size = int(cfg.get("batch_size", 2))
    k = int(cfg["n_iterations"])
    out = np.zeros((len(shard_records), k, hidden_size), dtype=np.float32)
    texts = []
    index = []
    for row_i, record in enumerate(shard_records):
        responses = record["responses"]
        if len(responses) != k:
            raise RuntimeError(
                f"{record['question_id']} has no raw responses. "
                "Representation extraction needs the response field, not only final_answer."
            )
        for cand, response in enumerate(responses):
            texts.append(
                build_encoder_text(
                    tokenizer,
                    record.get("encoder_user") or record["question"],
                    response,
                    chat_mode=record.get("chat_mode") or "math_system",
                    extra=record,
                    system_prompt=system_prompt,
                )
            )
            index.append((row_i, cand))
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        vecs = encode_texts(model, tokenizer, chunk, device, max_length, layer)
        for vec, (row_i, cand) in zip(vecs, index[start : start + batch_size]):
            out[row_i, cand] = vec
    np.save(args.shard_output, out)
    meta = {
        "shard": args.shard,
        "n_shards": args.n_shards,
        "n_rows": len(shard_records),
        "hidden_size": hidden_size,
        "layer_index": layer,
        "row_ids": [
            {"benchmark": r["benchmark"], "question_id": r["question_id"], "rollout_id": r["rollout_id"]}
            for r in shard_records
        ],
    }
    Path(args.shard_meta).write_text(json.dumps(meta), encoding="utf-8")


def launch(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n_shards = len(gpus)
    work = Path(args.output_path).parent / (Path(args.output_path).stem + "_shards")
    work.mkdir(parents=True, exist_ok=True)
    procs = []
    metas = []
    arrays = []
    for shard, gpu in enumerate(gpus):
        shard_out = work / f"shard_{shard}.npy"
        shard_meta = work / f"shard_{shard}.json"
        arrays.append(shard_out)
        metas.append(shard_meta)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--config",
            args.config,
            "--model_path",
            args.model_path,
            "--input_path",
            args.input_path,
            "--gpu",
            gpu,
            "--shard",
            str(shard),
            "--n_shards",
            str(n_shards),
            "--shard_output",
            str(shard_out),
            "--shard_meta",
            str(shard_meta),
        ]
        if args.domain:
            cmd.extend(["--domain", args.domain])
        procs.append(subprocess.Popen(cmd))
    if any(p.wait() != 0 for p in procs):
        raise SystemExit("a representation shard failed")
    benches, mean32 = domain_benchmarks(cfg, args.domain)
    records = load_records(
        args.input_path,
        n_iterations=int(cfg["n_iterations"]),
        mean32_benchmarks=mean32,
        exact_agreement_benchmarks=exact_benchmarks(cfg),
        benchmarks=benches,
        domain=args.domain,
    )
    hidden_size = json.loads(Path(metas[0]).read_text(encoding="utf-8"))["hidden_size"]
    k = int(cfg["n_iterations"])
    hidden = np.zeros((len(records), k, hidden_size), dtype=np.float32)
    filled = np.zeros(len(records), dtype=bool)
    for shard, meta_path in enumerate(metas):
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        arr = np.load(arrays[shard])
        cursor = [i for i in range(len(records)) if i % n_shards == shard]
        if len(cursor) != arr.shape[0]:
            raise RuntimeError("shard row count does not match the filtered records")
        for local, global_i in enumerate(cursor):
            ident = meta["row_ids"][local]
            rec = records[global_i]
            if (
                ident["benchmark"] != rec["benchmark"]
                or str(ident["question_id"]) != str(rec["question_id"])
                or int(ident["rollout_id"]) != int(rec["rollout_id"])
            ):
                raise RuntimeError("shard row order does not match the candidate file")
            hidden[global_i] = arr[local]
            filled[global_i] = True
    if not filled.all():
        raise RuntimeError("some candidate rows did not receive a representation")
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, hidden)
    print(f"wrote {hidden.shape} to {args.output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path")
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--domain")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=1)
    parser.add_argument("--shard_output")
    parser.add_argument("--shard_meta")
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        if not args.model_path or not args.input_path or not args.output_path:
            raise SystemExit("--model_path, --input_path, and --output_path are required")
        launch(args)


if __name__ == "__main__":
    main()
