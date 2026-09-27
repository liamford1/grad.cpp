# Metal-resident records, 2026-09-27 (Apple M3 Pro)

Raw JSON behind the "Metal-resident mode" section of
[BENCHMARKS.md](../../../BENCHMARKS.md). Machine: Apple M3 Pro (12-core CPU),
macOS 26.5.1, fp32. grad.cpp `069fcfb` (feature/metal-resident merged with
main), AppleClang 21, Release. Each run waited for a 1-minute load average
below 2.5 and would have been repeated if another heavy process appeared.

| file | what |
|---|---|
| `cpu_small_{1,2}.json` | `grad bench --device cpu --steps 20 --trials 5`, two rounds |
| `metal_small_{1,2}.json` | `grad bench --device metal --steps 20 --trials 5`, two rounds, including per-step stream counters |

The 70M numbers come from `benchmarks/time_train_steps.sh <grad>
data/tinystories.txt {medium,modern} 60 --device metal` (median of steps 5
to 60): `medium` 1,125 ms/step (7,282 tok/s, n=57), `modern` 1,091 ms/step
(7,509 tok/s, n=59). Memory at 70M: `footprint` sampled every 2s over 30
steps of `medium` read a constant 5,069 MB; CPU mode sampled every 0.5s read
1,288 to 3,052 MB.
