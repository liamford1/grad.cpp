#!/usr/bin/env python3
"""PyTorch baseline for the BENCHMARKS.md head-to-head.

Builds the same model configs as the C++ presets (`small` mirrors what
`./build/grad bench` times; `medium`/`modern` mirror the training
presets) in idiomatic PyTorch, and times training the same way: warmup,
repeated timed windows, and median steps/s and tokens/s.

The point is a fair fight, so the PyTorch side is written the way a
competent PyTorch user would write it: fused QKV projection,
F.scaled_dot_product_attention, optional torch.compile and autocast,
not a transliteration of the C++ internals. Architecture, parameter
count, optimizer settings, dropout placement, and loss match the C++
implementation exactly. The model itself is in pytorch_model.py.

Take numbers on an idle machine (no training run in the background,
the C++ numbers in BENCHMARKS.md are measured that way too):

  .venv/bin/python benchmarks/pytorch_baseline.py --preset small  --device cpu
  .venv/bin/python benchmarks/pytorch_baseline.py --preset small  --device mps
  .venv/bin/python benchmarks/pytorch_baseline.py --preset medium --device mps
  .venv/bin/python benchmarks/pytorch_baseline.py --preset modern --device mps --amp bf16

`--dry-run` just builds the model and prints the parameter count, for
checking parity against the C++ side (22.0M / 69.8M / 69.0M).
"""

import argparse
import datetime
import json
import platform
import resource
import statistics
import sys
import time

# torch is imported in main(), not here, so this table can be read without
# it: CI's check_presets.py compares it against `grad presets --json`.
#
# Mirrors the model fields of the C++ presets (src/cli/presets.cpp):
# vocab, d_model, layers, heads, max_len, seq_length, micro-batch,
# grad_accum, dropout, arch. Dropout is per preset there (medium/modern
# train under one epoch and use 0), and mask generation is a real share of
# step time, so it has to match here.
PRESETS = {
    "fast":        dict(vocab=500,   d=128, layers=2, heads=4,  max_len=1024, seq=64,  batch=4, accum=1, dropout=0.1, modern=False),
    "fast-modern": dict(vocab=500,   d=128, layers=2, heads=4,  max_len=1024, seq=64,  batch=4, accum=1, dropout=0.1, modern=True),
    "small":       dict(vocab=5000,  d=512, layers=6, heads=8,  max_len=1024, seq=96,  batch=8, accum=1, dropout=0.1, modern=False),
    "medium":      dict(vocab=16000, d=768, layers=8, heads=12, max_len=1024, seq=256, batch=8, accum=4, dropout=0.0, modern=False),
    "modern":      dict(vocab=16000, d=768, layers=8, heads=12, max_len=1024, seq=256, batch=8, accum=4, dropout=0.0, modern=True),
}

GRAD_CLIP = 5.0
LR = 3e-4  # Adam, betas (0.9, 0.999), eps 1e-8, wd 0, same as C++ bench mode


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1 << 20) if sys.platform == "darwin" else rss / 1024


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", choices=PRESETS, default="small")
    ap.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    ap.add_argument("--steps", type=int, default=20,
                    help="timed optimizer steps (default 20, like C++ bench)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--trials", type=int, default=5,
                    help="independent timed windows; report their median")
    ap.add_argument("--json", type=str, default=None,
                    help="write machine-readable benchmark results")
    ap.add_argument("--amp", choices=["bf16", "fp16"], default=None,
                    help="autocast dtype (C++ side is fp32; this is 'PyTorch at its best')")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads")
    ap.add_argument("--dry-run", action="store_true", help="build model, print params, exit")
    args = ap.parse_args()
    if args.steps < 1 or args.warmup < 0 or args.trials < 1:
        ap.error("steps/trials must be positive and warmup non-negative")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from pytorch_model import GPT

    cfg = PRESETS[args.preset]
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(42)

    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        print("MPS not available on this machine", file=sys.stderr)
        return 1

    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"preset={args.preset} device={device.type} params={n_params / 1e6:.1f}M "
          f"dropout={cfg['dropout']} "
          f"torch={torch.__version__} threads={torch.get_num_threads()}"
          + (f" amp={args.amp}" if args.amp else "")
          + (" compile" if args.compile else ""))
    if args.dry_run:
        return 0

    if args.compile:
        model = torch.compile(model)

    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=0.0)

    # Random tokens: content doesn't affect throughput, and a fixed pool
    # avoids timing the RNG. (The C++ bench feeds real Shakespeare
    # windows; the compute per step is identical.)
    B, T, V = cfg["batch"], cfg["seq"], cfg["vocab"]
    pool = [(torch.randint(V, (B, T), device=device),
             torch.randint(V, (B, T), device=device)) for _ in range(8)]

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)

    def micro_step(i: int) -> None:
        x, y = pool[i % len(pool)]
        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
        loss.backward()

    def opt_step(step: int) -> None:
        for i in range(cfg["accum"]):
            micro_step(step * cfg["accum"] + i)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        opt.zero_grad(set_to_none=True)

    def sync() -> None:
        if device.type == "mps":
            torch.mps.synchronize()

    model.train()
    for s in range(args.warmup):
        opt_step(s)
    sync()

    steps_per_s_trials = []
    tokens_per_s_trials = []
    print(f"protocol={args.warmup} warmup, {args.trials} trials x {args.steps} steps")
    for trial in range(args.trials):
        sync()
        start = time.perf_counter()
        for s in range(args.steps):
            opt_step(args.warmup + trial * args.steps + s)
        sync()
        elapsed = time.perf_counter() - start
        steps_per_s = args.steps / elapsed
        tokens_per_s = steps_per_s * cfg["accum"] * B * T
        steps_per_s_trials.append(steps_per_s)
        tokens_per_s_trials.append(tokens_per_s)
        print(f"  trial {trial + 1}: {elapsed:.3f}s, {steps_per_s:.2f} steps/s, "
              f"{tokens_per_s:.0f} tok/s")

    median_steps = statistics.median(steps_per_s_trials)
    median_tokens = statistics.median(tokens_per_s_trials)
    rss_mb = peak_rss_mb()
    mps_mb = (torch.mps.driver_allocated_memory() / (1 << 20)
              if device.type == "mps" else None)
    print(f"  median: {median_steps:.2f} steps/s, {median_tokens:.0f} tok/s")
    print(f"  peak RSS {rss_mb:.0f} MB"
          + (f", MPS driver {mps_mb:.0f} MB" if mps_mb is not None else ""))

    if args.json:
        result = {
            "schema_version": 1,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "engine": "pytorch",
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "system": platform.platform(),
            "device": device.type,
            "amp": args.amp,
            "compiled": args.compile,
            "threads": torch.get_num_threads(),
            "parameters": n_params,
            "config": cfg,
            "protocol": {
                "warmup_steps": args.warmup,
                "steps_per_trial": args.steps,
                "trials": args.trials,
            },
            "training_steps_per_second": steps_per_s_trials,
            "training_tokens_per_second": tokens_per_s_trials,
            "median_training_steps_per_second": median_steps,
            "median_training_tokens_per_second": median_tokens,
            "peak_rss_mb": rss_mb,
            "mps_driver_mb": mps_mb,
        }
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump(result, out, indent=2)
            out.write("\n")
        print(f"benchmark JSON: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
