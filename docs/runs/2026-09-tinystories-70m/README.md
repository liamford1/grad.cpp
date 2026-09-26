# Run report: 70M GPT on TinyStories (September 2026)

The largest model trained with grad.cpp so far: a 69.8M-parameter GPT-2-style decoder trained from scratch for 40,000 optimizer steps (328M tokens) on one Apple M3 Pro (12-core CPU), with every forward and backward pass going through this repo's own autograd engine.

| | |
|---|---|
| Model | `medium` preset: d768 × 8 layers × 12 heads, FFN 3072 (GELU), learned positions, tied embeddings |
| Parameters | 69,778,944 |
| Tokenizer | BPE, 16,000 tokens, trained by `grad prepare` on the first 32MB of the corpus |
| Data | `TinyStories-train.txt` (1.92GB, 382M tokens), contiguous 95/5 train/val split |
| Batch | 8 sequences × 256 tokens × 4 accumulation steps = 8,192 tokens per optimizer step |
| Optimizer | AdamW (β 0.9/0.999, weight decay 0.1 on matrices), lr 3e-4, 1,000 warmup steps, cosine to 3e-5, clip 5.0, no dropout |
| Budget | 40,000 steps, 327.7M tokens (0.90 epochs) |
| Compute | 37.1 hours of training across four sessions, 2026-09-13 to 2026-09-21 |
| Result | **held-out loss 1.693, perplexity 5.44** (`grad eval`, 1M uniformly sampled validation tokens); train sample 1.668 |

![Training and validation loss](loss.svg)

The orange line is the trainer's in-loop validation on a fixed 65K-token slice, which turned out to be easier than the split as a whole (see [below](#the-in-loop-validation-number-was-optimistic)). The representative held-out loss is 1.693.

## Results

| checkpoint | split | tokens scored | loss | perplexity |
|---|---|---:|---:|---:|
| `tinystories_best.bin` (step 39,500) | validation | 1,024,000 | **1.693** | **5.44** |
| `tinystories_best.bin` (step 39,500) | train | 1,024,000 | 1.668 | 5.30 |
| `tinystories_final.bin` (step 40,000) | validation | 1,024,000 | 1.693 | 5.43 |
| `tinystories_final.bin` (step 40,000) | train | 1,024,000 | 1.668 | 5.30 |

Both splits are scored by `grad eval` (fixed seed, so both checkpoints see the same windows) on uniform random samples of non-overlapping 256-token windows. The validation loss sits 0.025 nats above the training loss, a small generalization gap. That is expected: the run covered 0.9 epochs, so it saw most training windows at most once and had little opportunity to memorize.

Training was clean. The loss curve has no spikes, and the gradient norm never reached the clip threshold (maximum 3.36 raw, median 0.68 over the last 1,000 steps).

### The in-loop validation number was optimistic

During the run, the trainer scored the first 32 batches (65,536 tokens) of the held-out split every 500 steps. Those numbers fell monotonically at all 79 evaluations and ended at **1.438 (perplexity 4.21)**. That slice scores 0.25 nats better than the split as a whole. The first few hundred stories after the split point are easier than average, and a fixed prefix cannot show it. The curve is internally consistent, since the slice was the same every time, so it is still a good record of how the run progressed:

| step | tokens seen | in-loop val loss (fixed 65K-token slice) |
|---:|---:|---:|
| 500 | 4.1M | 4.731 |
| 1,000 | 8.2M | 3.586 |
| 2,000 | 16.4M | 2.749 |
| 5,000 | 41.0M | 2.140 |
| 10,000 | 81.9M | 1.864 |
| 20,000 | 163.8M | 1.635 |
| 30,000 | 245.8M | 1.499 |
| 39,500 | 323.6M | 1.438 |

The gap was found while writing this report, by adding `grad eval` and comparing the two. The trainer now evaluates a fixed set of windows spread evenly across the whole split, so future runs report a representative figure in the loop.

### Comparability

The held-out split is the last 5% of `TinyStories-train.txt`, not TinyStories' official validation file. Per-token loss also depends on the tokenizer. These numbers are therefore not directly comparable with published TinyStories results.

## Samples

Sampled at temperature 0.8 from `tinystories_best.bin`, the first draw for each prompt, unedited. Output is shown up to the model's end-of-story marker or the 150-token limit (`…`).

> **Once upon a time, there was a little dragon who** wanted to meet someone else. He flew around and saw an old lady. She smiled at the dragon, and said, "Hello! My name is Frank." The big queen thought this sounded like fun, so she asked Frank if he would join her for some fun. Then, they became best friends. They went on adventures together, learning to be as friendly with one another. One day, the little dragon wanted to make a mess in the forest. So, he flew down and landed beside a big tree. He asked if anyone could help him clean up. The old lady was very kind, so she said yes. Frank took hold of all his toys, dried them up, wiped them off, and put away his toys. Then they both smiled at each other. And Frank said, "Thank you for your kindness!"

> **Tom and his sister Mia found a box in the garden.** They opened it to reveal an amazing surprise - it was a brand new puppy! He had soft fur and big, furry eyes. One day Sam said, "Let's take him for a walk." Mia laughed with delight, she couldn't believe it. However, as they were walking down the street, something surprising happened. Suddenly, dark clouds started to form in from the sky. Big drops of water began to pour down all over them. The puppy was now very wet. …

> **The old owl looked at the moon and said,** "Yes, I will help you. But you must promise to keep it quiet on the ground." The bird thanked him, and flew away with a smile on its face. From then on, every day the little bird would fly around looking for more things in her dream. She was always so happy when she could fly.

The grammar is fluent and the stories have a recognizable shape, with a setup, an event and a resolution. Coherence across more than a few sentences is where a model of this size and budget still fails: names drift (Tom becomes Sam), and characters appear without introduction (the old lady introduces herself as Frank, and "the big queen" arrives from nowhere).

## How the run went

| session | steps | training time | ended by |
|---|---|---:|---|
| 1 | 0 to 17,111 | 16.5 h | Ctrl-C (planned pause) |
| 2 | 17,112 to 33,960 | 15.2 h | Ctrl-C (planned pause) |
| 3 | 33,961 to 36,706 | 2.5 h | Ctrl-C (planned pause) |
| 4 | 36,707 to 39,999 | 2.9 h | completed |

`train_supervised.sh` ran every session: it resumes from `<prefix>_resume_{model,state}.bin` (weights, both Adam moments, schedule position and best val loss) and restarts the trainer after a crash with backoff. It never had to restart one. All three stops were deliberate Ctrl-C pauses to free the machine, and each one resumed with no change in the loss curve (the dashed lines on the chart). The resumed loader draws from a seed derived from the step position, so a resume does not replay batches the model has already seen.

**Throughput.** Median step time was 3.23s (10th to 90th percentile 3.20 to 3.44s), about **2,540 tokens/s** of training, fp32. The standard 6 × parameters × tokens estimate puts the run at 1.4 × 10¹⁷ FLOPs, about 1.0 TFLOP/s sustained over 37.1 hours. Only the 50-GFLOP logits matmul crosses the Metal threshold at this size. Everything else, including attention, ran on the CPU through Accelerate/AMX (see [BENCHMARKS.md](../../../BENCHMARKS.md) #7 and #11).

**Memory.** Resident memory at the end of each step had a median of 1.8GB and a maximum of 2.3GB for the whole run, with no upward drift. That is the trough between steps. The peak `phys_footprint` inside a step is about 4.1GB (BENCHMARKS.md #9); the metrics CSV does not record it.

## Limitations of this run

- **Undertrained by compute-optimal standards.** 328M tokens is 4.7 tokens per parameter against the Chinchilla rule of thumb of about 20. The validation loss was still falling when the schedule ended.
- **The tokenizer drops whitespace structure.** Pre-tokenization splits on whitespace, so newlines and runs of spaces never reach the model. TinyStories is mostly prose, so the effect here is small, but it is a real defect, and tokenizer v2 in the roadmap fixes it.
- **The validation split is the file's own tail**, not TinyStories' official validation file (see Comparability).
- **In-loop validation scored a fixed prefix of the split**, which read 0.25 nats optimistic (see Results). The trainer has since been fixed.

## Weights

`tinystories_best.bin` and its tokenizer are published as the [`tinystories-70m` release](https://github.com/liamford1/grad.cpp/releases/tag/tinystories-70m), with SHA-256 checksums and usage.

## Reproduce

```bash
curl -L -o data/tinystories.txt \
  "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
./build/grad prepare data/tinystories.txt 16000
./train_supervised.sh data/tinystories.txt medium      # resumable; re-run after any stop
./build/grad watch tinystories                          # live dashboard, second terminal
./build/grad eval tinystories_best.bin data/tinystories.txt 16000 256 500
python3 tools/plot_run.py tinystories_metrics.csv loss.svg
```

[`metrics.csv`](metrics.csv) is the run's complete per-step record: loss, learning rate, gradient norm, step time and memory for all 40,000 steps, plus every evaluation. The chart and every number on this page come from it, from the supervisor log, or from `grad eval`.
