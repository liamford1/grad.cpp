# Metal-resident execution

Status: implemented (`--device metal`), tested for correctness against the
CPU, soak-tested on long runs (see Correctness), and measured on an idle M3
Pro (see Measurements below): 14,760 tok/s at 22M and 7,282 / 7,509 tok/s
at 70M, faster than PyTorch MPS at every size tested. CPU mode, the
default, is unchanged bit for bit.

## Why

Today the GPU sees only GEMMs above ~10 GFLOP, through MPS, one synchronous
command buffer per call (BENCHMARKS.md #7, #11). Everything else runs on the
CPU, and the CPU waits for every GPU call. Below 10 GFLOP a synchronous
dispatch loses to AMX, so at 22M nothing reaches the GPU at all, and at 70M
only the logits matmuls do. PyTorch MPS keeps the whole step on the device
and is 1.44x faster at 22M and 2.3 to 2.7x faster at 70M (2026-09-25
head-to-head). The gap is the execution model, not a kernel: a 70M
micro-batch is ~850 GFLOP of GEMM plus ~1,000 small ops, and the small ops
only pay on the GPU if the CPU never waits for them one at a time.

## Model: the device is an executor, not an address space

Apple Silicon memory is unified, so a tensor never moves. In Metal mode every
tensor's storage is a shared `MTLBuffer`, the CPU and the GPU address the
same bytes, and "running on the GPU" means only that an op's work is
*encoded* into a per-process command stream instead of executed on the
calling thread. The CPU encodes a whole forward, backward and AdamW step
while the GPU drains the stream behind it; the two meet only when the CPU
needs a value.

**Mode.** `grad::set_device(Device::Metal)` switches the process, and
`grad train|train-fast|bench|eval --device metal` (or `GRAD_DEVICE=metal`)
does it before anything is allocated. The mode decides two things: new
tensors come from the Metal allocator, and Variable ops, the fused module
ops (attention, norms, embeddings, the logits projection) and the optimizer
encode GPU work. The default is CPU, where nothing changes, including the
existing threshold GEMM offload. Tensor-level methods (`Tensor::add`,
`matmul`, ...) and KV-cached generation stay on the CPU in both modes: they
are cold paths or latency-bound single-token work, and the fence below keeps
them correct on Metal-mode tensors.

**Stream.** One `MTLCommandQueue`, one open command buffer and one open
compute encoder. Kernels use serial dispatch, so each sees its predecessor's
writes without explicit barriers; MPS GEMMs end the compute encoder, encode
into the same command buffer, and hazard tracking orders them against
neighbours. The buffer is committed every `GRAD_METAL_COMMIT` dispatches
(default 32) without waiting, so the GPU starts on the head of the step while
the CPU encodes the rest. At ~5 us of CPU per encoded dispatch, 32
dispatches is ~0.2 ms, the longest the GPU can sit idle after a sync; commit
overhead (~10-30 us) is then under 10% of encoding time.

**Sync model.** A CPU access to a tensor that the GPU may still be using must
wait. The granularity chosen is a *global fence gated by a per-tensor bit*:
each Tensor carries `device_visible_`, set the first time its storage is
handed to the stream (`Tensor::device_data()`, the only way GPU code obtains
a pointer), and every CPU accessor (`raw()`, `values()`, `getValue`,
`setValue`, copies, in-place ops) runs

    if (device_visible_) [[unlikely]] metal::fence();   // commit + wait if work is pending

*Why not a plain global fence:* the data loader writes each micro-batch's
token ids and the embedding validates them on the CPU while the previous
micro-batch's backward is still in flight. Those tensors were never given to
the GPU, so under a plain global fence they would drain the stream twice per
micro-batch for nothing; with the bit they cost nothing. *Why not per-tensor
fences* (waiting only for the command buffer that last touched a tensor): in
a training step the CPU reads GPU results at exactly two points, the losses
and the gradient norm, both at the end of the step when nothing else is in
flight to overlap with, so tracking last-writer sequence numbers per tensor
would buy nothing measurable while adding bookkeeping to every encode. The
design leaves room for it: the bit becomes a sequence number.

*Cost.* In CPU mode the bit is never set: one byte load and a not-taken
branch per accessor call. Accessors are called per op, not per element (the
one per-element caller, `compute_grad_norm`, now hoists `raw()` out of its
loop, same summation order), so this is below measurement noise and changes
no arithmetic. In Metal mode a fence that finds work pending costs the time
to drain the stream; the stream counts these (`metal::stream_stats()`).
Measured on the test models: **one per optimizer step** in the trainer (it
reads the micro-batch losses and the gradient norm together after encoding
AdamW; `MetalTrainingParity` asserts it), none inside a `grad bench` trial
(steps pipeline and the trial ends with one sync, as the PyTorch baseline
does), and one per evaluation batch. Pointers returned by `raw()` stay valid
for CPU use only until the tensor is next handed to the GPU; the
gradient-check tests, which poke parameters through such pointers, do so
between syncs.

**Memory lifetime.** Freeing a tensor whose storage a queued kernel will
still read would be a use-after-free, and backward retires activations as
the wave passes (#9). The Metal allocator therefore never returns a block to
its free list while work that might reference it is outstanding: a release
is tagged with the open command buffer's sequence number and the block
becomes reusable when that buffer completes (completion handler) or at the
next sync. Blocks are page-rounded shared `MTLBuffer`s cached by size, so
steady-state steps allocate nothing from the OS and pointer-to-buffer lookup
is a map search (~50 ns) rather than a `newBufferWithBytesNoCopy` wrap per
use (the page-mapping cost of wrapping ~1,500 operands per step would be
tens of milliseconds). A fresh block is by construction not referenced by
pending work, so the CPU may fill it without a fence; small zero-filled
tensors (<= 64 KB) are cleared on the CPU for that reason, larger ones with a
fill kernel so the encoding thread does not spend its time in memset.

**Run-ahead bound.** Deferred reuse makes memory grow with the CPU's lead:
if nothing reads a result, the CPU can encode several micro-batches while
the GPU works on the first, and every activation they release waits for its
command buffer. A commit therefore waits for the oldest buffer while more
than `GRAD_METAL_MAX_IN_FLIGHT` (default 16) are outstanding: at 32
dispatches each, about half of a 70M micro-batch (~1,000 dispatches) of lead,
which keeps the GPU fed and bounds the extra memory.

**Errors.** A command buffer failure is found when the stream drains
completed buffers (at a sync, or while throttling) and is rethrown by the
next fence or sync, on the thread that fences. Nothing
falls back to the CPU after work is queued: the queued work may have
consumed its inputs (beta = 1 accumulation), as in the existing sgemm.

## Kernels

GEMMs with whole-tensor operands (projections, FFN, logits, all backward
weight and input gradients) are `MPSMatrixMultiplication` objects cached by
shape and encoded into the stream. Attention's per-(batch, head) products
have head operands that are column slices of `(B*S, d)` matrices (row pitch
`d`, offset `h*hs`), which MPS cannot batch; they use one batched strided
GEMM kernel (simdgroup 8x8 matrices, 32x32 tiles) that indexes batch
`z = b*H + h` with two stride levels, so no head is copied, mirroring the CPU
path. The remaining kernels are one Metal source: fill/copy/add/mul/scale
and accumulating variants, bias add and broadcast adds, column sums, GELU,
SiLU, LayerNorm and RMSNorm forward and backward, causal attention softmax
forward and backward, generic softmax, log-softmax, NLL, a fused
cross-entropy, embedding gather and scatter, RoPE, dropout, AdamW, and the
gradient-norm reduction and clip scaling.

**Compilation.** The kernels live in `src/transformer/metal_kernels.metal`,
are embedded into the library as a byte array at configure time (a string
literal would hit -Wpedantic's 65,536-character limit), and are
compiled once per process with `newLibraryWithSource` in safe math mode
with precise math functions (two settings from macOS 15; see the known
failure under Correctness). A
build-time metallib would need the offline `metal` compiler, which on
current Xcode is a separate download (absent on this machine) and not
guaranteed on CI images; runtime compilation needs only Metal.framework and
costs a fraction of a second once per process, negligible against any
training run. Linux builds compile neither and link stubs; `metal_mode()`
can never become true there.

**Numerics.** Safe math mode keeps IEEE semantics, and the precise
math-function setting makes an unqualified `tanh`, `exp`, `log` or `sqrt`
the `metal::precise` one: over every 97th float, within 2.3e-7 relative
of the CPU's libm. The GPU flushes subnormal inputs to zero in either
mode (`log` of a subnormal is -inf, `1/x` is inf), where the CPU does not;
no kernel feeds a subnormal to one of these on a path that matters (the
norms add eps first, the softmax sums are at least 1). Multiply-adds may still be fused into one rounding, by the
Metal compiler and by clang on the CPU (`-ffp-contract=on` is its default),
so elementwise results that are a product plus a sum can differ from the CPU
by an ulp of the product; disabling contraction in the kernels
(`#pragma METAL fp contract(off)`) was tried and only moved the mismatch to
the expressions clang fuses. Single-rounding elementwise results (add, mul,
scale, gathers, bias adds, masks) are bitwise equal to the CPU's.

**Determinism.** Every reduction runs in a fixed order independent of
scheduling: row reductions use a fixed-width threadgroup tree, column
reductions (bias, LayerNorm gamma and beta, positional embeddings) sum fixed
blocks of rows and then the block partials in order, the loss and the
gradient norm reduce through fixed-size chunks. No kernel uses float
atomics. The embedding backward is a segmented reduction: the CPU already
reads the token ids to validate them, so it also builds, by stable counting
sort, the list of positions per distinct token, and each GPU thread owns one
(token, column) and sums its positions in increasing order: the same order
as the CPU loop. The batched GEMM walks K in a fixed order; MPS is assumed
deterministic for a fixed shape, which the determinism test checks. Same
inputs twice give bitwise identical outputs.

**Dropout masks match the CPU bit for bit.** The CPU mask is a pure function
of (seed, stream, position): each 65,536-element block reseeds xorshift128+
through splitmix64, then steps it once per four 16-bit lanes. The generator
is linear over GF(2), so the GPU splits each block across 256 threads: a
thread starts from the block's seed state multiplied by a precomputed
128x128 bit matrix `T^(64k)` (256 matrices, 512 KB, built once on the CPU),
then steps 64 times. Stream indices are reserved through the same counter in
the same order as the CPU path. Identical masks make the CPU/Metal parity
test meaningful with dropout on.

**Fused loss.** `Variable::cross_entropy(targets)` is the trainer's loss.
In CPU mode it is exactly `log_softmax()->nll_loss(targets)`, the same two
nodes as before. In Metal mode it is one node that keeps the logits and a
per-row log-sum-exp and writes `(softmax - onehot) * g / n` directly into
the logits gradient, instead of materializing log-probabilities and a dense
one-hot gradient (2 x 131 MB per 70M micro-batch). `log_softmax` and
`nll_loss` still have GPU implementations for other callers.

## Autograd integration

Each op keeps one entry point and branches once on `metal_mode()`:

    if (metal_mode()) return metal_graph::gelu(shared_from_this());
    ...unchanged CPU code...

The Metal versions build the same graph nodes with the same children in the
same order, and their backward closures encode GPU work exactly as the CPU
closures run CPU work, so backward traversal, gradient accumulation order
and node retirement are shared. The Variable ops' versions live in
`metal_graph.cpp`; the fused module ops (attention, LayerNorm/RMSNorm, the
embeddings, the tied logits projection) keep theirs as file-local functions
beside their CPU code, a deviation from the first plan: each module's two
paths then read side by side, and the Metal path reuses the module's own
helpers (the RoPE table builder) instead of exporting them. The optimizer
branches per operation, and the trainer has a Metal step that reads the
losses only after AdamW is encoded. The CPU branches are untouched apart
from fences on Tensor accessors, `compute_grad_norm` hoisting `raw()`, and
the trainer, eval and bench calling `cross_entropy`, which is the same
`log_softmax()->nll_loss()` pair in CPU mode. The determinism probe hashes
(`gpt2 b5f8ffd5bbbd37d5`, `modern c27c183eba166512`) are unchanged.

One bug class is specific to this design: a backward closure outlives the
forward's stack frame, so any helper it copies must capture by value. A
by-reference capture in the attention backward read dead stack slots, and
the parity test caught it at the first updated step (loss off by 3e-2, and
different on every run).

## Correctness strategy and results

- `MetalStream`: GEMM chains encode with no wait and the first read waits
  once; CPU writes wait for queued GPU writers; tensors never handed to the
  GPU never wait; freed storage is not reused under a queued kernel; the
  batched strided GEMM and MPS against CPU BLAS at attention and odd shapes.
- `MetalKernelsVsCPU`: every kernel against the CPU implementation or its
  formula. Single-rounding outputs and all dropout masks match bit for bit;
  fused multiply-adds within an ulp of the product; transcendentals within
  1e-5 relative (observed worst 0.85 of that, in GELU's backward, where
  1 - tanh^2 cancels); reordered reductions within a bound scaled to their
  length (observed at most 0.09 of it). The activations are also run where
  they saturate: |x| from 6 to 60 in steps of 0.01 and magnitudes to 1e12,
  which the random inputs from [-6, 6] never reach.
- `MetalGradientChecking`: central differences through matmul, bias add,
  GELU, LayerNorm, SiLU gating, dropout and the fused loss on the GPU.
- `MetalTrainingParity`: GPT-2 and modern models (d64, 2 layers, vocab 97)
  trained 8 steps from one seed with dropout 0.1, accumulation 2 and clipping
  every step. Observed: every micro-batch loss within 2e-6 of the CPU's (a
  few ulps at 4.8), evaluation loss identical, gradient norms within 2e-7
  relative, mean parameter difference 1e-7. The bound is 2e-4 on losses,
  100x the observed drift and 100x below what a wrong kernel produced. The
  worst single parameter differs by up to 4e-4: weights with a zero or
  near-zero true gradient (the key biases, since softmax ignores a constant
  added to a row) receive rounding noise that Adam normalizes into full-size
  steps, so single weights are bounded by one lr, the mean by rounding. A
  second Metal run is bitwise identical, and each step waits once.
  `MetalTrainingParityThrottled` repeats it with 3-dispatch command buffers
  and a lead of 2.
- `MetalTrainerParity`: the Trainer itself in both modes; logged losses
  agree. `MetalInferenceParity`: KV-cached CPU decoding against the Metal
  forward on 3D and 2D input.
- Every Metal test exits 77 (skip) without a Metal device;
  `-DGRAD_METAL_BACKEND=OFF` builds the stubs on macOS to check that
  configuration, which is what Linux builds.
- Soak runs, `benchmarks/metal_soak.sh`: a preset trained from one seed on
  both devices for thousands of steps, reporting the first non-finite loss
  or gradient norm, gradient-norm spikes, step-time drift, memory footprint
  over the run, and the loss agreement at every step and evaluation. The
  tests above use tiny models for a few steps, so a failure that needs a
  trained model's value ranges passes all of them; the soak is where the one
  below was found. Results (2026-09-27, M3 Pro), all finite at every step:
  - `small` on Shakespeare (tokenizer v2), all 8,000 steps on both devices:
    final train loss 3.3455 on Metal and 3.3730 on the CPU, final val loss
    4.5613 and 4.5621. The per-step train loss difference grows as ordinary
    fp drift with dropout on, from a mean of 0.003 over the first 1,000
    steps to 0.017 over the last (largest single step 0.074); the 31 val
    losses differ by at most 0.028. No gradient-norm spikes; the largest
    norm is 4.59, at step 9 of warmup. Metal step time 52 ms from start to
    end, footprint flat at 1.3 GB.
  - `medium` (70M, GPT-2 block with GELU) on TinyStories, tokenizer v1,
    1,500 steps on Metal: train loss 3.2267 at step 1499, val loss 3.2023
    at step 1500, largest gradient norm 3.54 (step 2), no spikes, 1,120 ms
    per step over the last 500 (1,125 in the measurements below),
    footprint flat at 5.1 GB.
  - `modern` (69M, Llama block) on TinyStories, tokenizer v2, 1,000 steps
    on Metal: train loss 2.7257 at step 999, val loss 2.6764 at step 1000,
    largest gradient norm 3.50 (step 34), no spikes, 1,082 ms per step at
    the end, footprint flat at 4.7 GB.
  - `modern` on TinyStories, tokenizer v1, 600 steps on Metal against the
    CPU run that trained the released 69M checkpoint (same seed, same
    tokenizer): every step's loss within 1e-5, the metrics log's last
    digit (mean 1.3e-6), and the step-500 val loss identical, 4.25862.
  - Logged gradient norms differ between the devices by 2 to 8%, from
    step 0: the CPU trainer's logged norm (`compute_grad_norm`) adds 22M
    to 70M squares into one float, and small terms vanish against the
    running sum. On one 22M backward it reads 1.5256 against 1.5754 in
    double precision; the Metal reduction reads 1.5754278, and the CPU
    clip itself (per-tensor `sdot`) 1.5754268, so training is unaffected.
    Only CPU-mode logs are low, and they are left as they are here.
- `GRAD_METAL_CHECK=1` (`op_check.h`) localizes a non-finite value: every
  op that records a graph node waits for the GPU after its forward and its
  backward and scans what it wrote, the optimizer scans its state, and the
  first bad value throws after naming the step, the op and its index in the
  step, the tensor, the first bad index and whether the op's inputs were
  finite. `GRAD_METAL_CHECK_FROM=N` starts at step N so a run reaches an
  event at full speed; `GRAD_METAL_CHECK_DUMP=DIR` writes the op's tensors.
  Off, it costs a load and a not-taken branch per op. `OpCheck` tests it
  on both devices.

### Known failure found by soak and its fix

What: `grad train data/shakespeare.txt small --device metal` turned NaN at
step 2558, deterministically, where the CPU run stayed finite. With
`GRAD_METAL_CHECK` the first non-finite value was the last block's GELU
forward: input 10.4155 (finite), output NaN. Its only other input of 10 or
more, 10.0512, came out as 5.0256, half of what it should be. It was the
first time any GELU input had reached 10: the largest had grown from 8.0
at step 1274 to 9.98 at step 2535. Resumed from the step-2500 state (which
draws different batches, since a resume reseeds the loader by step), the
run failed the same way at step 2650, in the same FFN hidden unit (column
1800 of 2048), and bitwise identically on a second resume.

Why: GELU is `0.5x(1 + tanh(k(x + a x^3)))`, and at x = 10 the tanh
argument is 43.7. The kernels were compiled with `mathMode = Safe`, the
macOS 15 replacement for `fastMathEnabled = NO`, but that setting governs
only arithmetic; which namespace an unqualified `tanh` resolves to is a
second option, `mathFloatingPointFunctions`, whose default is
`metal::fast`. `fast::tanh` returns 0 for arguments in [43.67, 44.36)
and NaN from 44.36 to infinity (every float from 1 to 128 and a sample of
those above, on an M3 Pro), where `precise::tanh` and the CPU's `vvtanhf`
saturate to 1. So GELU returned x/2 for inputs in [10.0, 10.4), NaN above,
and its backward a derivative near 58 instead of 1. The NaN reached the
loss and, through AdamW, every weight. The first run also hit a GPU fault
(command buffers discarded as victims of GPU recovery) near step 5090; a
second run of the unfixed build stayed NaN but ran all 8,000 steps with a
flat footprint, so that fault did not come from the NaN deterministically.

Why the tests missed it: the kernel tests drew activation inputs from
[-6, 6], where the tanh argument stays below 13, and the parity tests
train tiny models for 8 steps, whose activations never grow that large.
Every check passed because every check stayed inside the range where the
fast and precise functions agree to 1e-5.

Fix: `mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise`,
which makes every unqualified transcendental in the kernels the precise
one, as this note always said they were (throughput unchanged within
noise: 14,143 against 14,110 tok/s at 22M, and the 70M step times above). `MetalKernelsVsCPU` now runs the activations through their
saturation range, and fails without the fix.

## What remains

- A per-kernel GPU time breakdown (Xcode Metal capture), to rank the
  optimization targets below by measured cost.
- Performance work the measurements are expected to motivate: a faster
  attention (the batched GEMM is a plain 32x32 simdgroup tile, and the full
  S x S scores are computed and masked rather than only the causal half, or
  a fused flash-style kernel), multi-tensor AdamW and gradient-norm launches
  (one dispatch per parameter today), and dropout masks stored as bytes.
- The fp16 operand option exists only for the CPU-mode offload.
- Generation stays on the CPU. It is latency-bound single-token work where
  the resident stream's advantage, batching many dispatches, does not apply.

## Measurements (2026-09-27, M3 Pro)

Measured after the 35-hour run finished, on an idle machine (each run gated
on a 1-minute load average below 2.5 and repeated if another heavy process
appeared), 5 trials. Raw JSON: `benchmarks/results/2026-09-27-m3pro-metal/`.
Each item states what was predicted before measuring, and what happened.

1. **22M, `grad bench --device metal`.** Predicted above PyTorch MPS, 13-17k
   tok/s. Measured **14,760 tok/s** (two rounds: 14,830 and 14,687), against
   6,380 for CPU mode and 9,464 for PyTorch MPS: 2.3x and 1.56x. Inside the
   predicted range.
2. **70M, `time_train_steps.sh --device metal`.** Predicted 5,000-7,000
   tok/s. Measured **7,282** (`medium`, 1,125 ms/step) and **7,509**
   (`modern`, 1,091 ms/step), against CPU 2,542 / 2,602 and PyTorch MPS
   6,794 / 6,008. Slightly above the predicted range. The prediction assumed
   parity with PyTorch on shared MPS GEMMs; the rest of the step is cheaper
   here than there.
3. **Syncs.** 0 per step inside a bench trial, as predicted. 935 dispatches
   in 29 command buffers per 22M step. (Syncs per `grad train` step were not
   counted separately; the trainer reads the losses and gradient norm once
   per step by design.)
4. **Sweeps.** Throughput stayed within 3% (14,519 to 14,934 tok/s) across
   `GRAD_METAL_COMMIT` 8-128 and `GRAD_METAL_MAX_IN_FLIGHT` 4-64: flat, as
   predicted if the GPU is the bottleneck. About 28 run-ahead waits per
   step confirm that the CPU encodes faster than the GPU executes. The
   defaults stay.
5. **Memory.** Predicted a plateau after the first step. Measured a flat
   5,069 MB `phys_footprint` over a 30-step 70M `medium` run, against the
   CPU mode's 1.3-3.1 GB sawtooth. It is a plateau, as predicted, but
   higher than "CPU peak plus deferred frees": the pool keeps every block
   size's high-water mark for the life of the process. Returning cached
   blocks between steps is an option if memory ever matters more than
   allocation speed.
6. **Parity at scale.** Not on the original list, and added because the
   speedup is large enough to need it: at 70M from the same seed, Metal and
   CPU losses agree to the metrics log's 6 significant digits for 16 steps
   of `medium` and 226 steps of `modern`.
7. **GPU time split.** Not yet taken (needs an Xcode Metal capture).
