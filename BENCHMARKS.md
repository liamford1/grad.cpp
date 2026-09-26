# Benchmark Log

One row per optimization iteration, so the performance story is readable at a glance. Newest last.

**Machine:** Apple M2 Pro (arm64, 10-core CPU), macOS, for rounds #0-11 and the PyTorch head-to-head. The 70M long-run measurement below is from an Apple M3 Pro (12-core CPU). Release build (`-O3 -march=native`), Apple Accelerate BLAS, single process.

**Historical workload (#0-11):** `./build/grad bench 20`
- *Training:* full model config: d_model 512, 6 layers, 8 heads, seq 96, batch 8, vocab 5000 (~22M params). 3 warmup steps, then one 20-step timed window.
- *Generation:* 64 tokens sampled at temperature 0.8 from a short prompt, same model config.

The v0.1 benchmark protocol uses repeated windows and machine-readable provenance:

```bash
./build/grad bench --steps 20 --warmup 3 --trials 5 --json grad-benchmark.json
.venv/bin/python benchmarks/pytorch_baseline.py \
  --preset small --device cpu --steps 20 --warmup 3 --trials 5 \
  --json pytorch-cpu-benchmark.json
```

The Metal switches the entries below mention (`TRANSFORMER_METAL`, `TRANSFORMER_METAL_THRESHOLD`, `TRANSFORMER_METAL_FP16`) have since been renamed `GRAD_METAL`, `GRAD_METAL_THRESHOLD` and `GRAD_METAL_FP16`, and `TRANSFORMER_THREADS` is now `GRAD_THREADS`. The entries keep the names in effect when they were measured, and the old names still work as a fallback.

Run benchmarks on an otherwise idle machine. The JSON includes every trial, the median, exact model configuration, git/build/compiler/system metadata, backend availability, and true peak RSS.

| # | Date | Change | Train steps/s | Train tok/s | Gen tok/s | Reported RSS* |
|---|------------|-----------------------------------------------|------:|------:|------:|---------:|
| 0 | 2026-07-18 | Baseline: clean CPU implementation after removing the broken CUDA port (BLAS matmul, everything else naive single-threaded loops) | 1.2 | 934 | 86.7 | 1695 MB |
| 1 | 2026-07-18 | Batched 3D attention (was: batch flattened into one long sequence) + backward passes rewritten as single BLAS accumulations + embedding-gradient fix | 2.2 | 1668 | 87.6 | 1600 MB |
| 2 | 2026-07-18 | Vectorized elementwise math: GELU/softmax through SIMD `vvtanhf`/`vvexpf`, xorshift dropout masks, `sdot`/`sscal` grad clipping | 3.5 | 2707 | 126.7 | 1488 MB |
| 3 | 2026-07-18 | KV-cache incremental decoding (`InferenceSession`): generation no longer re-runs the full prefix per token | 3.5 | 2707 | 385.1 | 1488 MB |
| 4 | 2026-07-18 | Thread pool (`parallel.h`): GELU, dropout, softmax, LayerNorm, Adam across all 10 cores | 3.4 | 2618 | n/a | 1450 MB |
| 5 | 2026-07-18 | Batched-attention restructure: parallel per-(batch, head) with cached softmax, flat-sgemm projections, zero physical transposes | 6.1 | 4679 | n/a | 1184 MB |
| 6 | 2026-07-18 | Fast paths in add backward: in-place residual accumulation, row-major bias column sums | 6.7 | 5174 | 375.5 | 1457 MB |
| 7 | 2026-07-18 | Metal (MPS) backend for large matmuls, zero-copy via unified memory. **No change at 22M scale, by measurement and then by design** (see note); kicks in automatically at ~10 GFLOPs/op | 6.7 | 5174 | 375.5 | 1457 MB |
| 8 | 2026-07-18 | Memory round after the first 70M run took the machine down (see note): ARC for the Metal backend (was leaking 7 ObjC objects per GPU matmul), constant-memory DataLoader (was a 1.4GB index permutation), medium preset batch 16 → 8 | 6.6 | 5063 | 381.4 | 1304 MB |
| 9 | 2026-07-19 | Lazy gradient allocation + graph retirement during backward: forward holds only activations, and each node's tensors are freed the moment its backward fn has run (see note). Medium-preset peak 5.7 → 4.1GB | 7.4 | 5667 | 366.9 | 1096 MB |
| 10 | 2026-07-19 | Overhead round, driven by a live-training profile (see note): 4-lane dropout RNG, strided per-head attention (gather/scatter copies eliminated), uninitialized allocation for fully-written outputs, move semantics for op results (was: one full activation copy per op, three for the logits) | 7.9 | 6074 | 358.9 | 1085 MB |
| 11 | 2026-07-19 | fp16 GPU operands (fp32 accumulate), opt-in via `TRANSFORMER_METAL_FP16=1`. **Another honest null at this scale** (see note): modern-preset training identical at 0.28 it/s, and lowering the GPU threshold to feed it more matmuls ran 14% *slower* | 7.9 | 6074 | n/a | 1085 MB |

\* Historical rows recorded resident memory at the end of the benchmark, not a high-water mark. The v0.1 JSON protocol reports `getrusage` peak RSS; the old values remain here unchanged as historical observations.

## Long-run measurement: 70M medium preset (2026-09)

Not an optimization round, and on different hardware (M3 Pro), so not comparable row-for-row with the table above. It is the throughput a real run sustained: the `medium` preset (69.8M parameters, d768 L8, seq 256, 8 × 4 accumulation) trained for 40,000 steps on TinyStories. The median optimizer step took 3.23s, or **2,540 tokens/s** (10th to 90th percentile 3.20 to 3.44s), over 37.1 hours in four resumed sessions. End-of-step RSS stayed between 1.8 and 2.3GB with no drift. Details and the full per-step CSV are in the [run report](docs/runs/2026-09-tinystories-70m/README.md).

## Head-to-head: PyTorch 2.14 on M3 Pro (2026-09-25)

Both engines measured under the repeated-trial protocol on an idle M3 Pro, fp32, with the same configs, optimizer settings and loss (PyTorch side: [`benchmarks/pytorch_baseline.py`](benchmarks/pytorch_baseline.py), idiomatic fused QKV and `scaled_dot_product_attention`). Raw JSON and method: [`benchmarks/results/2026-09-25-m3pro/`](benchmarks/results/2026-09-25-m3pro/).

| training config | grad.cpp (CPU) | PyTorch (CPU) | PyTorch (MPS GPU) |
|---|---:|---:|---:|
| 22M `small` · d512 L6 · seq 96 · batch 8 | **6,560 tok/s** | 4,370 | 9,464 |
| 70M `medium` (GPT-2 block) · d768 L8 · seq 256 · 8×4 | **2,542 tok/s** | 2,458 | 6,794 |
| 69M `modern` (RMSNorm/RoPE/SwiGLU) · same shape | **2,602 tok/s** | 2,443 | 6,008 |

Readings:

- **On CPU, grad.cpp's lead shrinks with scale: 1.50× at 22M, 1.03 to 1.07× at 70M.** At 22M per-op overhead is a large share of the step, and that is where a specialized engine wins. At 70M the step is dominated by large GEMMs, and both engines hand those to the same Accelerate/AMX BLAS. Parity there is the expected result, not a regression.
- **The GPU gap widens with scale: PyTorch MPS is 1.44× faster at 22M and 2.3 to 2.7× faster at 70M.** grad.cpp sends only matmuls above ~10 GFLOP to the GPU, one synchronous dispatch at a time (#7, #11). PyTorch keeps the whole step on the device. Closing this gap takes a device-resident Metal execution path, not more CPU tuning.
- These records replace the 70M comparison withdrawn below.

## Head-to-head: PyTorch (2026-07-19, M2 Pro, superseded)

Same machine, same session, runs interleaved minutes apart. The PyTorch side is [`benchmarks/pytorch_baseline.py`](benchmarks/pytorch_baseline.py): the identical 22.0M-parameter config, optimizer settings, dropout placement, and loss, but written as idiomatic PyTorch with fused QKV and `scaled_dot_product_attention`. Training throughput is fp32.

| training config | grad.cpp (CPU) | PyTorch (CPU) | PyTorch (MPS GPU) |
|---|---:|---:|---:|
| 22M · d512 L6 · seq 96 · batch 8 | **5,418 tok/s** | 3,685 | 8,112 |

Readings:

- **grad.cpp beats PyTorch CPU by 1.5× on this config.** Specialization and lower per-op overhead are useful here; this does not imply general PyTorch API or workload superiority.
- **PyTorch MPS leads by 1.5×.** The current Metal backend dispatches synchronously while the CPU waits. Persistent device execution and fused kernels are the next architectural work.
- Conditions: the 22M number re-measured at 7.1 steps/s vs optimization round #11's 7.9 due to ambient variance; both sides of the head-to-head ran interleaved under the same conditions.

### Audit correction: withdrawn 70M comparison

The first version of this section reported 3,650 tok/s for the 70M GPT-2 config and 3,490 tok/s for the 69M modern config. A v0.1 source audit found that the cited modern CSV window (steps 1850-1925) has a median around **2,344 tok/s**, consistent with the 0.28 optimizer-step/s observation in note #11, not 3,490 tok/s. The 70M rows and scale-dependent speedup claim are withdrawn until both engines are rerun from the new repeated-trial JSON protocol. Component-level memory and kernel experiments below remain historical engineering notes, not head-to-head framework claims.

## Notes

**#11. fp16 GPU matmuls: correct, and not (yet) worth it.** The backend can now convert operands to fp16 on-device (a compute kernel fills persistent private scratch inside the same command buffer) and run the MPS matmul at half the operand bandwidth with an fp32 result matrix, so all accumulation - including beta=1 gradient accumulation - stays full precision. Verified against CPU BLAS across every transpose/beta case; the worst error is ~0.6x the fp16 input-rounding bound, i.e. exactly the rounding and nothing else. But measured on the modern preset: **0.28 it/s with it, 0.28 without** (only the 50-GFLOP logits matmuls cross the GPU threshold, and their bandwidth win just covers the conversion cost), and dropping the threshold to 5 GFLOP to route the FFN matmuls through it ran **0.24 it/s - 14% slower**. Synchronous dispatch with the CPU idle during GPU work still loses to AMX at these sizes, halved bandwidth or not. Default off (`TRANSFORMER_METAL_FP16=1` to enable); like the backend itself in #7, it should earn its keep at the next model size up, or when dispatch overlaps CPU work.

**#10. Overhead round: +7% at bench scale, ~2-4% at medium scale, and a lesson about profiles.** `sample` on a live medium training step showed BLAS at only ~34% of thread-summed CPU time; the rest was overhead: dropout mask generation ~14% (one xorshift step per element, ~550M draws per optimizer step), memmove ~12% (attention head gather/scatter plus a deep copy of every op result into its Variable, with the logits tensor being copied three times per micro-batch), memset ~6% (zero-fill on allocation of outputs that are immediately overwritten), and elementwise/norm loops for the rest. All four are fixed: the RNG now yields four 16-bit lanes per draw (rate quantized to 1/65536ths, so 0.1 becomes 0.100006), the per-head attention matmuls take strided views (`lda = d_model`) so no head is ever copied, fully-written outputs allocate via `Tensor::uninitialized`, and op results move into their Variables. Every change is math-identical; gradient checks and both-arch inference parity pass unchanged.

The honest outcome: bench (22M, overhead-heavy) gained 7%, but medium wall-clock only ~2-4% - the removed work ran spread across pool threads, while medium's critical path is BLAS plus memory bandwidth. Thread-summed CPU shares overestimate wall-clock impact for parallelized overhead. The remaining lever that attacks the actual critical path is fp16 GPU matmuls (halved bandwidth on the dominant term).

**#9. Lazy grads + backward retirement (28% off medium's peak, 12% faster training).** Two coupled changes to the autograd core:
- **Gradients allocate on first write** (`ensureGrad()`), not at Variable construction. The forward pass used to allocate and zero a grad tensor for every intermediate: for medium, ~1.75GB of memsets per micro-batch that backward would mostly overwrite anyway. A node whose grad is never touched now means "no gradient flowed here" and its backward fn returns early (this also skips the dead recompute in the 2D attention path, whose composed matmul/add nodes are shadowed by the module's custom backward).
- **`backward()` retires each node as its backward fn completes**: data and grad freed, closure caches (softmax outputs, dropout masks) released. Reverse-topological order guarantees every consumer of a node's data has already run, and a node's grad is only ever read by its own backward fn, so the live set during backward is the frontier, not the whole graph. Parameters and inputs are leaves (no backward fn) and are never touched; the root keeps its tensors because the trainer reads the loss value after backward.

Measured on the medium warm-start probe: peak phys footprint 5.7 → 4.1GB, troughs at ~1.9GB, loss trajectory identical (gradient checks pass across every component). The freed headroom bought a test of micro-batch 16 × accum 2 (same effective batch 32, FFN matmuls at 19 GFLOP, above the Metal crossover): **measurement said no again**: 32% slower per token and 7.0GB peak, with or without the GPU (`TRANSFORMER_METAL=0` gave the same 0.17 it/s), so the bigger working set loses more to cache pressure than the GPU wins back. Medium stays at 8 × 4.

**#8. Memory round: why the first medium (70M, TinyStories) run crashed a 16GB machine.** The bench numbers above barely move because the 22M config was never the problem. At medium scale (batch 16, seq 256, d_model 768, 8 layers, vocab 16k), measured `phys_footprint` blew past **10GB within the first training step**. The autograd graph holds data *plus an eagerly-zeroed grad* for every intermediate, and the logits/log-softmax chain alone is ~1GB, on top of a **1.4GB** DataLoader index permutation (181M mapped windows × 8 bytes) and ~1.1GB of params+Adam state. macOS ran out of application memory and the machine went down. (`ps` RSS showed only ~3GB at the time; the gap is GPU-wrapped and compressed pages, so watch `footprint`, not RSS.) Three fixes:
- **Metal backend now compiles with ARC.** `sgemm` alloc/new-creates seven Objective-C objects per call (3 `MTLBuffer`, 3 `MPSMatrix`, 1 MPS kernel); without `-fobjc-arc` those are +1 references that `@autoreleasepool` never touches, so they leaked on every large matmul, ~50/step at medium scale, including no-copy buffers holding GPU mappings onto freed tensor pages. Measured slow (malloc recycles the pages) but unbounded, and gone under ARC (MPS-vs-BLAS test still passes; forced-GPU bench unchanged at 6.4 steps/s vs 4.8 CPU-only).
- **DataLoader stopped materializing the epoch permutation.** Shuffled loading now draws window indices uniformly per row (with replacement); a 40K-step run consumes <2% of the mapped corpus's epoch, where the two are statistically indistinguishable. Saves 1.4GB resident and ~15s of startup shuffle.
- **medium is batch 8, 40K steps** (same token budget, same checkpoint/eval wall-clock cadence). Measured footprint over a 4-minute run: sawtooth between 2.45GB (between steps) and a flat **5.4GB peak**, no drift, 0.90 it/s, about 12h for the full run. The next real memory lever, if batch 16 is ever wanted on 16GB, is retiring graph nodes' tensors as backward passes them instead of holding everything until `release_graph()`.

**#7. Metal backend: an honest null result at this scale.** The GPU path works (MPS matmuls verified against CPU BLAS across every transpose/beta combination) and costs nothing: tensors ≥256KB are page-aligned so `MTLBuffer` wraps them zero-copy through Apple Silicon's unified memory, and every call can fall back to CPU. But **measurement said no**: routing the 22M model's matmuls to the GPU made training *slower* (4.5 vs 4.6 steps/s under identical conditions). A per-shape sweep found the crossover:

| GEMM shape (M×N×K) | context | CPU GF/s | GPU GF/s | GPU/CPU |
|---|---|---:|---:|---:|
| 768×512×512 | 22M model, QKV proj | 1251 | 599 | 0.48× |
| 768×5000×512 | 22M model, logits | 1557 | 1817 | 1.17× |
| 4096×3072×768 | 60M model, FFN | 1991 | 2963 | 1.49× |
| 4096×16000×768 | 60M model, logits | 2200 | 3950 | 1.80× |
| 8192×4096×1024 | 120M model, FFN | 2212 | 3881 | 1.75× |

(CPU numbers depressed ~30% by a concurrent training run; the true crossover is if anything higher.) Apple's AMX units are simply excellent at small-to-medium GEMMs, and synchronous MPS dispatch can't amortize below ~10 GFLOPs per op. So the default threshold is set there: the current model runs entirely on CPU (hence identical numbers above), and the backend engages automatically as model/batch/context grow, exactly the scales where training needs it. `TRANSFORMER_METAL=0` disables it; `TRANSFORMER_METAL_THRESHOLD` tunes the crossover.

The same instinct that would have shipped this as a "GPU acceleration 🚀" bullet point without measuring is how the repo got its dead CUDA port. The infrastructure earns its keep at the next model size up; at this one, the honest number is a tie.

**#0. Baseline.** Starting point after restoring the pre-CUDA implementation and fixing the generation memory leak (which alone brought generation peak memory down from >5GB). Generation has no KV cache yet, so its tok/s decays quadratically as context grows, and 86.7 tok/s is measured at short context and is flattering. Peak RSS of 1.7GB for a 22M-param model points at per-op allocation churn in the autograd graph.

**#4-6. Multithreading round (1.9× training).** An instructive sequence:
- **#4 was flat**, a lesson in profiling before optimizing. The thread pool parallelized GELU/dropout/softmax/LayerNorm/Adam across all 10 cores, but those had already been shrunk by the #2 vectorization; the profile (taken *before* #2, not re-taken) had gone stale. What actually dominated by then was the attention block itself.
- **#5 restructured attention**: the 3D path now runs its 64 per-(batch, head) units in parallel with per-task scratch, caches the softmax output (and attention-dropout masks) from forward so backward recomputes nothing, which also makes the attention-dropout gradient exact where it previously ignored the mask, and does all four projections and their gradients as single flat `(batch*seq, d)` sgemms with transpose folded in. Seven physical `transpose()` materializations went away.
- **#6** gave the add op's backward fast paths for the two shapes training actually uses: same-shape residual adds accumulate in place via SIMD, and row-vector biases reduce with cache-friendly row-major column sums (the generic path walked columns strided).

Net effect of the day: **1.2 → 6.7 steps/s (5.6×)**. The remaining profile is mostly BLAS itself, so the next big lever is a different device (Metal), not more CPU tuning.

**#3. KV cache (3× generation at short context, asymptotically much more).** Generation used to rebuild the whole autograd graph over the full prefix for every new token, which is O(context²) forward work per token, so tok/s degraded as text grew. `InferenceSession` decodes incrementally: per-layer K/V projections are cached, each new token attends against the cache, and everything runs on raw float buffers with zero graph bookkeeping and zero per-token allocation (scratch buffers are reused). Greedy output over 150 tokens is byte-identical to the old path. The benchmark number (64 tokens from a short prompt) understates the win: per-token cost is now nearly flat in context length instead of linear, so at context 1000 the gap is ~15×. Generating two 150-token samples from the trained checkpoint now takes ~1s total including model load.

**#2. Vectorized elementwise math (1.6× training, 1.4× generation).** After #1, profiling showed scalar `tanhf` at ~24% of step time (GELU forward + backward over the 768×2048 FFN activations), with dropout, softmax `expf`, and gradient clipping close behind. Changes: GELU and softmax/log-softmax route their transcendentals through Accelerate's vForce (`vvtanhf`/`vvexpf`, with portable scalar fallbacks); dropout masks come from a thread-local xorshift128+ generator compared in the integer domain instead of a freshly-seeded `mt19937` + `uniform_real_distribution` per call; grad-norm clipping uses `sdot`/`sscal` (the scalar float reduction couldn't auto-vectorize without `-ffast-math`); backward passes accumulate into gradient buffers directly instead of building temporaries and `add_inplace`-ing them.

**#1. Correct batching + BLAS backwards (1.8× training).** Three related changes, found by profiling:
- Training used to flatten the batch into a single 768-token sequence, so attention was one 768×768 matrix: tokens attended across sequence boundaries (a modeling bug) and attention cost 8× more than the correct 8×96×96. Training now feeds proper (batch, seq) 3D inputs.
- The 3D paths turned out to be pathologically slow: the weight-tying backward was a quadruple loop doing ~2B scalar accessor calls per step. It and `Variable::matmul`'s backward are now single `sgemm` calls with transpose flags and beta=1 in-place accumulation, with no materialized transposes and no temporaries.
- Profiling exposed that the embedding table never received gradients from the lookup side (grad tracking was keyed off the token IDs, which never require grad), so it only learned through the tied output projection. Fixed, and guarded by a new full-model gradient-check test (`test_model_gradients`), which compares analytical vs numerical gradients across every component through the batched 3D path.
