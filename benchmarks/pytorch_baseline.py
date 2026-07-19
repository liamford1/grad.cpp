#!/usr/bin/env python3
"""PyTorch baseline for the BENCHMARKS.md head-to-head.

Builds the same model configs as the C++ presets (`small` mirrors what
`./build/grad bench` times; `medium`/`modern` mirror the training
presets) in idiomatic PyTorch, and times training the same way: a few
warmup steps, then a timed window, reporting steps/s and tokens/s.

The point is a fair fight, so the PyTorch side is written the way a
competent PyTorch user would write it — fused QKV projection,
F.scaled_dot_product_attention, optional torch.compile and autocast —
not a transliteration of the C++ internals. Architecture, parameter
count, optimizer settings, dropout placement, and loss match the C++
implementation exactly.

Take numbers on an idle machine (no training run in the background —
the C++ numbers in BENCHMARKS.md are measured that way too):

  .venv/bin/python benchmarks/pytorch_baseline.py --preset small  --device cpu
  .venv/bin/python benchmarks/pytorch_baseline.py --preset small  --device mps
  .venv/bin/python benchmarks/pytorch_baseline.py --preset medium --device mps
  .venv/bin/python benchmarks/pytorch_baseline.py --preset modern --device mps --amp bf16

`--dry-run` just builds the model and prints the parameter count, for
checking parity against the C++ side (22.0M / 69.8M / 69.0M).
"""

import argparse
import math
import resource
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Mirrors kPresets in src/main.cpp: vocab, d_model, layers, heads,
# max_len, seq_length, micro-batch, grad_accum, arch.
PRESETS = {
    "fast":        dict(vocab=500,   d=128, layers=2, heads=4,  max_len=1024, seq=64,  batch=4, accum=1, modern=False),
    "fast-modern": dict(vocab=500,   d=128, layers=2, heads=4,  max_len=1024, seq=64,  batch=4, accum=1, modern=True),
    "small":       dict(vocab=5000,  d=512, layers=6, heads=8,  max_len=1024, seq=96,  batch=8, accum=1, modern=False),
    "medium":      dict(vocab=16000, d=768, layers=8, heads=12, max_len=1024, seq=256, batch=8, accum=4, modern=False),
    "modern":      dict(vocab=16000, d=768, layers=8, heads=12, max_len=1024, seq=256, batch=8, accum=4, modern=True),
}

DROPOUT = 0.1
GRAD_CLIP = 5.0
LR = 3e-4  # Adam, betas (0.9, 0.999), eps 1e-8, wd 0 — same as C++ bench mode


def swiglu_hidden(d_model: int) -> int:
    """8d/3 rounded up to a multiple of 64 (matches feedforward.h)."""
    return ((8 * d_model // 3) + 63) // 64 * 64


class Rope(nn.Module):
    """Rotary embeddings, theta 10000 (matches multihead_attention.cpp)."""

    def __init__(self, head_size: int, max_len: int):
        super().__init__()
        half = head_size // 2
        inv_freq = 10000.0 ** (-2.0 * torch.arange(half, dtype=torch.float32) / head_size)
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                    # (max_len, half)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # x: (B, H, T, hs)
        T = x.size(-2)
        cos, sin = self.cos[:T], self.sin[:T]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, d: int, heads: int, max_len: int, rope: bool):
        super().__init__()
        assert d % heads == 0
        self.heads = heads
        # The C++ model keeps Q/K/V biases in both archs; one fused
        # projection here is the idiomatic-PyTorch equivalent of its
        # three flat sgemms.
        self.qkv = nn.Linear(d, 3 * d, bias=True)
        self.proj = nn.Linear(d, d, bias=True)
        self.drop = nn.Dropout(DROPOUT)
        self.rope = Rope(d // heads, max_len) if rope else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.heads, -1).transpose(1, 2)
        k = k.view(B, T, self.heads, -1).transpose(1, 2)
        v = v.view(B, T, self.heads, -1).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=DROPOUT if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.drop(self.proj(out))


class FeedForward(nn.Module):
    """GPT-2 style: W2(gelu(W1 x)); modern: bias-free SwiGLU.

    Dropout after the activation and after the down-projection, matching
    feedforward.cpp in both archs.
    """

    def __init__(self, d: int, modern: bool):
        super().__init__()
        self.modern = modern
        if modern:
            h = swiglu_hidden(d)
            self.gate = nn.Linear(d, h, bias=False)
            self.up = nn.Linear(d, h, bias=False)
            self.down = nn.Linear(h, d, bias=False)
        else:
            self.up = nn.Linear(d, 4 * d, bias=True)
            self.down = nn.Linear(4 * d, d, bias=True)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.modern:
            h = F.silu(self.gate(x)) * self.up(x)
        else:
            h = F.gelu(self.up(x), approximate="tanh")
        return self.drop(self.down(self.drop(h)))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, max_len: int, modern: bool):
        super().__init__()
        norm = (lambda: nn.RMSNorm(d, eps=1e-5)) if modern else (lambda: nn.LayerNorm(d, eps=1e-5))
        self.norm1, self.norm2 = norm(), norm()
        self.attn = CausalSelfAttention(d, heads, max_len, rope=modern)
        self.ffn = FeedForward(d, modern)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d, modern = cfg["d"], cfg["modern"]
        self.modern = modern
        self.wte = nn.Embedding(cfg["vocab"], d)
        self.wpe = None if modern else nn.Embedding(cfg["max_len"], d)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList(
            Block(d, cfg["heads"], cfg["max_len"], modern) for _ in range(cfg["layers"]))
        self.final_norm = nn.RMSNorm(d, eps=1e-5) if modern else nn.LayerNorm(d, eps=1e-5)
        # Weight tying: logits = final_norm(x) @ wte^T, same as gpt_model.cpp.

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(idx.size(1), device=idx.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return F.linear(self.final_norm(x), self.wte.weight)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)  # bytes on macOS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", choices=PRESETS, default="small")
    ap.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    ap.add_argument("--steps", type=int, default=20,
                    help="timed optimizer steps (default 20, like C++ bench)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--amp", choices=["bf16", "fp16"], default=None,
                    help="autocast dtype (C++ side is fp32; this is 'PyTorch at its best')")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads")
    ap.add_argument("--dry-run", action="store_true", help="build model, print params, exit")
    args = ap.parse_args()

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

    start = time.perf_counter()
    for s in range(args.steps):
        opt_step(args.warmup + s)
    sync()
    elapsed = time.perf_counter() - start

    steps_per_s = args.steps / elapsed
    tokens_per_s = steps_per_s * cfg["accum"] * B * T
    print(f"{args.steps} optimizer steps ({cfg['accum']} micro-batch each) in {elapsed:.2f}s")
    print(f"  {steps_per_s:.2f} steps/s")
    print(f"  {tokens_per_s:.0f} tokens/s")
    print(f"  peak RSS {peak_rss_mb():.0f} MB"
          + (f", MPS driver {torch.mps.driver_allocated_memory() / (1 << 20):.0f} MB"
             if device.type == "mps" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
