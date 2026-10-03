#!/usr/bin/env bash
# Multi-seed architecture A/B: the GPT-2 block (medium) against the Llama
# block (modern), several seeds each, same corpus, tokenizer, budget and
# device, run back to back through train_supervised.sh so each run survives
# crashes and resumes. Safe to re-run: finished runs are skipped.
#
# Usage:  tools/ab_seeds.sh [steps] [seeds...]
#   e.g.  GRAD_DEVICE=metal nohup tools/ab_seeds.sh 10000 1 2 3 > ab_seeds.nohup.out 2>&1 &
#
# Defaults: 10000 steps, seeds 1 2 3, tokenizer v2. Six runs of 10k steps
# take about 18 hours on an M3 Pro GPU (1.09 s/step). Output prefixes:
# tinystories_v2_s<seed> (medium) and tinystories_v2_modern_s<seed> (modern).
set -uo pipefail

STEPS="${1:-10000}"; shift || true
[ $# -eq 0 ] && set -- 1 2 3
SEEDS=("$@")

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
export GRAD_TOKENIZER="${GRAD_TOKENIZER:-v2}" GRAD_STEPS="$STEPS"

for seed in "${SEEDS[@]}"; do
    for preset in medium modern; do
        suffix=""; [ "$preset" = modern ] && suffix="_modern"
        prefix="tinystories_v2${suffix}_s${seed}"
        if [ -f "${prefix}_final.bin" ]; then
            echo "[$(date -u +%FT%TZ)] skip $prefix (finished)"; continue
        fi
        echo "[$(date -u +%FT%TZ)] start $prefix"
        GRAD_SEED="$seed" ./train_supervised.sh data/tinystories.txt "$preset" || {
            echo "[$(date -u +%FT%TZ)] $prefix did not finish (rc=$?); stopping queue"; exit 1; }
    done
done
echo "[$(date -u +%FT%TZ)] all runs finished"
