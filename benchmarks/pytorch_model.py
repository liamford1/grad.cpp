"""The PyTorch side of the BENCHMARKS.md head-to-head: the grad.cpp model
written as idiomatic PyTorch.

Architecture, parameter count, dropout placement, and weight tying match
the C++ implementation exactly; the internals are what a competent
PyTorch user would write (fused QKV projection,
F.scaled_dot_product_attention), not a transliteration of the C++.
Kept apart from pytorch_baseline.py so that script's preset table can be
imported without torch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_dropout(p: float) -> nn.Module:
    """Identity at p == 0, matching Variable::dropout's early return."""
    return nn.Dropout(p) if p > 0 else nn.Identity()


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
    def __init__(self, d: int, heads: int, max_len: int, rope: bool, dropout: float):
        super().__init__()
        assert d % heads == 0
        self.heads = heads
        self.dropout = dropout
        # The C++ model keeps Q/K/V biases in both archs; one fused
        # projection here is the idiomatic-PyTorch equivalent of its
        # three flat sgemms.
        self.qkv = nn.Linear(d, 3 * d, bias=True)
        self.proj = nn.Linear(d, d, bias=True)
        self.drop = make_dropout(dropout)
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
            dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.drop(self.proj(out))


class FeedForward(nn.Module):
    """GPT-2 style: W2(gelu(W1 x)); modern: bias-free SwiGLU.

    Dropout after the activation and after the down-projection, matching
    feedforward.cpp in both archs.
    """

    def __init__(self, d: int, modern: bool, dropout: float):
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
        self.drop = make_dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.modern:
            h = F.silu(self.gate(x)) * self.up(x)
        else:
            h = F.gelu(self.up(x), approximate="tanh")
        return self.drop(self.down(self.drop(h)))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, max_len: int, modern: bool, dropout: float):
        super().__init__()
        norm = (lambda: nn.RMSNorm(d, eps=1e-5)) if modern else (lambda: nn.LayerNorm(d, eps=1e-5))
        self.norm1, self.norm2 = norm(), norm()
        self.attn = CausalSelfAttention(d, heads, max_len, rope=modern, dropout=dropout)
        self.ffn = FeedForward(d, modern, dropout)

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
        self.drop = make_dropout(cfg["dropout"])
        self.blocks = nn.ModuleList(
            Block(d, cfg["heads"], cfg["max_len"], modern, cfg["dropout"])
            for _ in range(cfg["layers"]))
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
