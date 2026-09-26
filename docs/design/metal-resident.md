# Metal-resident execution

Status: implemented on `feature/metal-resident` (`--device metal`), tested
for correctness against the CPU, and not yet measured. CPU mode, the default,
is unchanged bit for bit.

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
compiled once per process with `newLibraryWithSource` in safe math mode. A
build-time metallib would need the offline `metal` compiler, which on
current Xcode is a separate download (absent on this machine) and not
guaranteed on CI images; runtime compilation needs only Metal.framework and
costs a fraction of a second once per process, negligible against any
training run. Linux builds compile neither and link stubs; `metal_mode()`
can never become true there.

**Numerics.** Safe math mode keeps IEEE semantics and precise
transcendentals. Multiply-adds may still be fused into one rounding, by the
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
  1e-5 relative (observed worst 0.3 of that); reordered reductions within a
  bound scaled to their length (observed at most 0.09 of it).
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

## What remains

- Measurement, below. Nothing about performance is known yet.
- Performance work the measurements are expected to motivate: a faster
  attention (the batched GEMM is a plain 32x32 simdgroup tile, and the full
  S x S scores are computed and masked rather than only the causal half, or
  a fused flash-style kernel), multi-tensor AdamW and gradient-norm launches
  (one dispatch per parameter today), and dropout masks stored as bytes.
- The fp16 operand option exists only for the CPU-mode offload.
- Generation stays on the CPU. It is latency-bound single-token work where
  the resident stream's advantage, batching many dispatches, does not apply.

## Measurements to take when the machine is idle

Nothing below has been run; the machine was busy with a 35-hour training
run while this was built. Protocol: idle machine, `nice` off, 5 trials.
BENCHMARKS.md carries the same list as a table.

1. `grad bench --device metal --steps 20 --trials 5 --json` vs
   `grad bench` (CPU) at 22M; compare with PyTorch MPS (9,464 tok/s).
   *Expected:* above PyTorch MPS. The 22M step is ~100 GFLOP of GEMM
   (~30 ms at ~3.5 TFLOPS) plus ~900 dispatches; if dispatch overhead is
   ~10-20 us each on the GPU the step lands at 45-60 ms, 13-17k tok/s.
2. `benchmarks/time_train_steps.sh` with `--device metal` for `medium` and
   `modern` at 70M; compare with CPU 2,542 / 2,602 and PyTorch MPS 6,794 /
   6,008 tok/s. *Expected:* 5,000-7,000 tok/s; the step is GEMM-bound
   (~850 GFLOP per micro-batch), and MPS GEMMs are the same kernels PyTorch
   uses.
3. Syncs per optimizer step from `metal::stream_stats()` (bench prints it).
   *Expected:* 0 within a bench trial and 1 per `grad train` step. More
   means a CPU read slipped into the step.
4. Command buffers, dispatches and run-ahead waits per step, and sweeps of
   `GRAD_METAL_COMMIT` (8, 16, 32, 64, 128) and `GRAD_METAL_MAX_IN_FLIGHT`
   (4, 16, 64). *Expected:* flat beyond 16 and 8 respectively if the GPU is
   the bottleneck; run-ahead waits every step mean the CPU encodes faster
   than the GPU executes, which is the goal.
5. Peak memory (`footprint`, not RSS: GPU-shared pages) at 70M vs CPU.
   *Expected:* CPU peak plus at most `GRAD_METAL_MAX_IN_FLIGHT` command
   buffers' worth of deferred frees; the allocator caches blocks by size,
   so a plateau after the first step, not growth.
6. GPU time split (Xcode Metal capture or `MTLCommandBuffer.GPUStartTime`):
   GEMM vs attention kernels vs elementwise. *Expected:* the batched
   attention GEMM and dropout generation are the first optimization targets.
