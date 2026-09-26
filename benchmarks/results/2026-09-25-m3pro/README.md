# Head-to-head records, 2026-09-25 (Apple M3 Pro)

Raw JSON behind the "Head-to-head: PyTorch 2.14 on M3 Pro" section of
[BENCHMARKS.md](../../../BENCHMARKS.md).

- Machine: Apple M3 Pro (12-core CPU), macOS 26.5.1, fp32 throughout.
- grad.cpp: `e11ec3e`, AppleClang 21, Release (`-O3 -march=native`), Accelerate.
- PyTorch: 2.14.0 on Python 3.14.7, from `benchmarks/pytorch_baseline.py`.
- Every measurement waited for a 1-minute load average below 2.5 and was
  discarded and repeated if another heavy process ran during it (a first
  pass was contaminated by an unrelated test suite and is not included).

| file | what |
|---|---|
| `grad_small_{1,2}.json` | `grad bench --steps 20 --trials 5`, two rounds |
| `pt_small_{cpu,mps}_{1,2}.json` | `--preset small --steps 20 --warmup 3 --trials 5`, two rounds each |
| `pt_{medium,modern}_cpu.json` | `--steps 3 --warmup 1 --trials 5` |
| `pt_{medium,modern}_mps.json` | `--steps 5 --warmup 2 --trials 5` |

grad.cpp at 70M has no `bench` config; its numbers come from timing real
`grad train` steps with the method in
[`benchmarks/time_train_steps.sh`](../../time_train_steps.sh) (median of
steps 5 to ~35): `medium` 3,223 ms/step (2,542 tok/s, n=31), `modern`
3,148 ms/step (2,602 tok/s, n=32). The `medium` figure matches the 2,540
tok/s the 40,000-step TinyStories run sustained.
