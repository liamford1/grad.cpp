#!/bin/bash
# Times real training steps of a preset: starts `grad train`, waits until
# <steps> optimizer steps are in the metrics CSV, stops it with SIGINT (which
# saves resume state, so run it in a scratch directory), and reports the
# median step time over steps >= 5 (the first few include warmup effects).
# `grad bench` covers the 22M config; this covers the larger presets, whose
# step time is dominated by the same code path a full run uses.
#
# Usage: benchmarks/time_train_steps.sh <grad-binary> <corpus.txt> <preset> [steps] [train flags...]
#   e.g. (in a scratch dir with data/ linked)
#        benchmarks/time_train_steps.sh ~/grad.cpp/build/grad data/tinystories.txt medium 35
#        benchmarks/time_train_steps.sh ~/grad.cpp/build/grad data/tinystories.txt medium 35 --device metal
set -uo pipefail
BIN="${1:?usage: time_train_steps.sh <grad> <corpus> <preset> [steps]}"
CORPUS="${2:?corpus}"; PRESET="${3:?preset}"; STEPS="${4:-35}"

"$BIN" train "$CORPUS" "$PRESET" "${@:5}" > time_train_steps.log 2>&1 &
pid=$!
csv=""
while kill -0 "$pid" 2>/dev/null; do
    csv=$(ls -t ./*_metrics.csv 2>/dev/null | head -1)
    if [ -n "$csv" ] && [ "$(grep -c '^t,' "$csv")" -ge "$STEPS" ]; then break; fi
    sleep 5
done
kill -INT "$pid" 2>/dev/null; wait "$pid"
[ -n "$csv" ] || { echo "no metrics CSV written; see time_train_steps.log" >&2; exit 1; }

# t,<step>,<loss>,<lr>,<grad_norm>,<step_ms>,<mem_mb>,<wall_s>; m row field 3 = tokens/step
tokens=$(awk -F, '$1=="m"{print $3; exit}' "$csv")
awk -F, '$1=="t" && $2>=5 {print $6}' "$csv" | sort -n | awk -v tok="$tokens" \
    '{a[NR]=$1} END {m=a[int((NR+1)/2)]; printf "median: %d ms/step, %.0f tok/s (n=%d steps)\n", m, tok*1000/m, NR}'
