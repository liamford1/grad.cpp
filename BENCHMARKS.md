# Benchmark Log

One row per optimization iteration, so the performance story is readable at a glance. Newest last.

**Machine:** Apple M2 Pro (arm64), macOS. Release build (`-O3 -march=native`), Apple Accelerate BLAS, single process.

**Workload:** `./build/transformer bench 20`
- *Training:* full model config — d_model 512, 6 layers, 8 heads, seq 96, batch 8, vocab 5000 (~22M params). 3 warmup steps, then 20 timed steps.
- *Generation:* 64 tokens sampled at temperature 0.8 from a short prompt, same model config.

| # | Date | Change | Train steps/s | Train tok/s | Gen tok/s | Peak RSS |
|---|------------|-----------------------------------------------|------:|------:|------:|---------:|
| 0 | 2026-07-18 | Baseline: clean CPU implementation after removing the broken CUDA port (BLAS matmul, everything else naive single-threaded loops) | 1.2 | 934 | 86.7 | 1695 MB |
| 1 | 2026-07-18 | Batched 3D attention (was: batch flattened into one long sequence) + backward passes rewritten as single BLAS accumulations + embedding-gradient fix | 2.2 | 1668 | 87.6 | 1600 MB |
| 2 | 2026-07-18 | Vectorized elementwise math: GELU/softmax through SIMD `vvtanhf`/`vvexpf`, xorshift dropout masks, `sdot`/`sscal` grad clipping | 3.5 | 2707 | 126.7 | 1488 MB |
| 3 | 2026-07-18 | KV-cache incremental decoding (`InferenceSession`): generation no longer re-runs the full prefix per token | 3.5 | 2707 | 385.1 | 1488 MB |
| 4 | 2026-07-18 | Thread pool (`parallel.h`): GELU, dropout, softmax, LayerNorm, Adam across all 10 cores | 3.4 | 2618 | — | 1450 MB |
| 5 | 2026-07-18 | Batched-attention restructure: parallel per-(batch, head) with cached softmax, flat-sgemm projections, zero physical transposes | 6.1 | 4679 | — | 1184 MB |
| 6 | 2026-07-18 | Fast paths in add backward: in-place residual accumulation, row-major bias column sums | 6.7 | 5174 | 375.5 | 1457 MB |
| 7 | 2026-07-18 | Metal (MPS) backend for large matmuls, zero-copy via unified memory. **No change at 22M scale — by measurement and then by design** (see note); kicks in automatically at ~10 GFLOPs/op | 6.7 | 5174 | 375.5 | 1457 MB |

## Notes

**#7 — Metal backend: an honest null result at this scale.** The GPU path works (MPS matmuls verified against CPU BLAS across every transpose/beta combination) and costs nothing: tensors ≥256KB are page-aligned so `MTLBuffer` wraps them zero-copy through Apple Silicon's unified memory, and every call can fall back to CPU. But **measurement said no**: routing the 22M model's matmuls to the GPU made training *slower* (4.5 vs 4.6 steps/s under identical conditions). A per-shape sweep found the crossover:

| GEMM shape (M×N×K) | context | CPU GF/s | GPU GF/s | GPU/CPU |
|---|---|---:|---:|---:|
| 768×512×512 | 22M model, QKV proj | 1251 | 599 | 0.48× |
| 768×5000×512 | 22M model, logits | 1557 | 1817 | 1.17× |
| 4096×3072×768 | 60M model, FFN | 1991 | 2963 | 1.49× |
| 4096×16000×768 | 60M model, logits | 2200 | 3950 | 1.80× |
| 8192×4096×1024 | 120M model, FFN | 2212 | 3881 | 1.75× |

(CPU numbers depressed ~30% by a concurrent training run; the true crossover is if anything higher.) Apple's AMX units are simply excellent at small-to-medium GEMMs, and synchronous MPS dispatch can't amortize below ~10 GFLOPs per op. So the default threshold is set there: the current model runs entirely on CPU (hence identical numbers above), and the backend engages automatically as model/batch/context grow — exactly the scales where training needs it. `TRANSFORMER_METAL=0` disables it; `TRANSFORMER_METAL_THRESHOLD` tunes the crossover.

The same instinct that would have shipped this as a "GPU acceleration 🚀" bullet point without measuring is how the repo got its dead CUDA port. The infrastructure earns its keep at the next model size up; at this one, the honest number is a tie.

**#0 — Baseline.** Starting point after restoring the pre-CUDA implementation and fixing the generation memory leak (which alone brought generation peak memory down from >5GB). Generation has no KV cache yet, so its tok/s decays quadratically as context grows — 86.7 tok/s is measured at short context and is flattering. Peak RSS of 1.7GB for a 22M-param model points at per-op allocation churn in the autograd graph.

**#4–6 — Multithreading round (1.9× training).** An instructive sequence:
- **#4 was flat** — a lesson in profiling before optimizing. The thread pool parallelized GELU/dropout/softmax/LayerNorm/Adam across all 10 cores, but those had already been shrunk by the #2 vectorization; the profile (taken *before* #2, not re-taken) had gone stale. What actually dominated by then was the attention block itself.
- **#5 restructured attention**: the 3D path now runs its 64 per-(batch, head) units in parallel with per-task scratch, caches the softmax output (and attention-dropout masks) from forward so backward recomputes nothing — which also makes the attention-dropout gradient exact where it previously ignored the mask — and does all four projections and their gradients as single flat `(batch*seq, d)` sgemms with transpose folded in. Seven physical `transpose()` materializations went away.
- **#6** gave the add op's backward fast paths for the two shapes training actually uses: same-shape residual adds accumulate in place via SIMD, and row-vector biases reduce with cache-friendly row-major column sums (the generic path walked columns strided).

Net effect of the day: **1.2 → 6.7 steps/s (5.6×)**. The remaining profile is mostly BLAS itself — the next big lever is a different device (Metal), not more CPU tuning.

**#3 — KV cache (3× generation at short context, asymptotically much more).** Generation used to rebuild the whole autograd graph over the full prefix for every new token — O(context²) forward work per token, so tok/s degraded as text grew. `InferenceSession` decodes incrementally: per-layer K/V projections are cached, each new token attends against the cache, and everything runs on raw float buffers with zero graph bookkeeping and zero per-token allocation (scratch buffers are reused). Greedy output over 150 tokens is byte-identical to the old path. The benchmark number (64 tokens from a short prompt) understates the win: per-token cost is now nearly flat in context length instead of linear, so at context 1000 the gap is ~15×. Generating two 150-token samples from the trained checkpoint now takes ~1s total including model load.

**#2 — Vectorized elementwise math (1.6× training, 1.4× generation).** After #1, profiling showed scalar `tanhf` at ~24% of step time (GELU forward + backward over the 768×2048 FFN activations), with dropout, softmax `expf`, and gradient clipping close behind. Changes: GELU and softmax/log-softmax route their transcendentals through Accelerate's vForce (`vvtanhf`/`vvexpf`, with portable scalar fallbacks); dropout masks come from a thread-local xorshift128+ generator compared in the integer domain instead of a freshly-seeded `mt19937` + `uniform_real_distribution` per call; grad-norm clipping uses `sdot`/`sscal` (the scalar float reduction couldn't auto-vectorize without `-ffast-math`); backward passes accumulate into gradient buffers directly instead of building temporaries and `add_inplace`-ing them.

**#1 — Correct batching + BLAS backwards (1.8× training).** Three related changes, found by profiling:
- Training used to flatten the batch into a single 768-token sequence, so attention was one 768×768 matrix — tokens attended across sequence boundaries (a modeling bug) and attention cost 8× more than the correct 8×96×96. Training now feeds proper (batch, seq) 3D inputs.
- The 3D paths turned out to be pathologically slow: the weight-tying backward was a quadruple loop doing ~2B scalar accessor calls per step. It and `Variable::matmul`'s backward are now single `sgemm` calls with transpose flags and beta=1 in-place accumulation — no materialized transposes, no temporaries.
- Profiling exposed that the embedding table never received gradients from the lookup side (grad tracking was keyed off the token IDs, which never require grad) — it only learned through the tied output projection. Fixed, and guarded by a new full-model gradient-check test (`test_model_gradients`), which compares analytical vs numerical gradients across every component through the batched 3D path.
