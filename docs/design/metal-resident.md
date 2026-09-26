# Metal-resident execution

Status: in progress on `feature/metal-resident`. CPU mode, the default, is
unchanged bit for bit.

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
to drain the stream; the stream counts these (`metal::stream_stats()`), and
the expected count is **one per optimizer step** (the trainer reads the
micro-batch losses and the gradient norm together after encoding AdamW) and
one per evaluation batch. Pointers returned by `raw()` stay valid for CPU
use only until the tensor is next handed to the GPU; the gradient-check
tests, which poke parameters through such pointers, do so between syncs.

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

**Errors.** A command buffer failure is recorded by its completion handler
and rethrown by the next fence or sync, on the thread that fences. Nothing
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
are embedded into the library as a string at configure time, and are
compiled once per process with `newLibraryWithSource` in safe math mode. A
build-time metallib would need the offline `metal` compiler, which on
current Xcode is a separate download (absent on this machine) and not
guaranteed on CI images; runtime compilation needs only Metal.framework and
costs a fraction of a second once per process, negligible against any
training run. Linux builds compile neither and link stubs; `metal_mode()`
can never become true there.

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

    if (metal_mode()) return metal_graph::gelu(self);   // encodes forward, closure encodes backward
    ...unchanged CPU code...

The Metal versions live in `metal_graph.cpp`, build the same graph nodes
with the same children in the same order, and their backward closures
encode GPU work exactly as the CPU closures run CPU work, so backward
traversal, gradient accumulation order and node retirement are shared. The
CPU branches are untouched; the only edits on the CPU side are fences on
Tensor accessors and hoisting `raw()` in `compute_grad_norm`. The
determinism probe hashes (`gpt2 b5f8ffd5bbbd37d5`, `modern
c27c183eba166512`) are the check that CPU training is bit-identical.

## Correctness strategy

- Per-kernel tests against the CPU implementation within fp32 tolerances
  (reductions reorder, so not bitwise; bounds scale with the reduction
  length, as in MetalMatmulVsCPU), and dropout masks checked for exact
  equality.
- Gradient checks through GPU ops (central differences on a small graph).
- GPU determinism: the same seeded training run twice, parameter hashes
  equal.
- End-to-end parity: tiny GPT-2 and modern models trained N steps from one
  seed on the CPU and on Metal, dropout on, losses within a tolerance
  justified by the observed per-step drift.
- Every Metal test exits 77 (skip) when no Metal device is present, so the
  Linux jobs pass.

## Measurements to take when the machine is idle

Nothing below has been run; the machine was busy with a 35-hour training
run while this was built. Protocol: idle machine, `nice` off, 5 trials.

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
   *Expected:* 1. More means a CPU read slipped into the step.
4. Command buffers and dispatches per step, and the `GRAD_METAL_COMMIT`
   sweep (8, 16, 32, 64, 128). *Expected:* flat beyond 16 if the GPU is
   the bottleneck.
5. Peak memory (`footprint`, not RSS: GPU-shared pages) at 70M vs CPU.
   *Expected:* CPU peak plus one command buffer's worth of deferred
   frees; the allocator caches blocks, so a plateau, not growth.
6. GPU time split (Xcode Metal capture or `MTLCommandBuffer.GPUStartTime`):
   GEMM vs attention kernels vs elementwise. *Expected:* the batched
   attention GEMM and dropout generation are the first optimization targets.
