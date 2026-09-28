#!/bin/bash
# Soak test for Metal-resident mode (docs/design/metal-resident.md,
# Correctness): trains one preset from one seed on the CPU and on Metal for
# <steps> optimizer steps, then reports for each device the first
# non-finite loss or gradient norm, the largest gradient norms and the drift
# in step time, and across the two the loss agreement at every step and
# every evaluation. Each run's memory footprint (footprint(1)) is sampled
# while it trains. The unit and parity tests run tiny models for a
# few steps; failures that need a trained model's value ranges, such as an
# activation that only saturates after thousands of steps, show up here.
#
# Usage: benchmarks/metal_soak.sh <grad-binary> <corpus.txt> <preset> <steps> [train flags...]
#   e.g. benchmarks/metal_soak.sh build/grad data/shakespeare.txt small 8000
#        benchmarks/metal_soak.sh build/grad data/tinystories.txt medium 1500 --tokenizer v1
#
# Both runs use the default seed unless the flags pass --seed. Each device
# trains in $SOAK_DIR/<device> (default ./soak; replaced on every run),
# whose data/ links every file beside the corpus whose name starts with the
# corpus's (the corpus, its .bin token files, tokenizer files), so whatever
# a run creates lands there, never beside the corpus. A run is stopped with
# SIGINT, which saves a resume pair, once it has logged <steps> steps; with
# <steps> at the preset's length or more it runs to the end.
#
#   SOAK_DEVICES="metal cpu"  devices to run, one after the other so each
#                             run's step times are its own (default both)
#   SOAK_SAMPLE_S=60          seconds between footprint samples
#   GRAD_METAL_CHECK=1 and the other GRAD_* switches pass through.
#
# The exit status is 1 if any run produced a non-finite loss or gradient
# norm or failed, else 0; loss agreement is reported, not judged, since
# with dropout on the two devices drift apart by ordinary rounding.
set -uo pipefail
BIN="${1:?usage: metal_soak.sh <grad> <corpus> <preset> <steps> [train flags...]}"
CORPUS="${2:?corpus}"
PRESET="${3:?preset}"
STEPS="${4:?steps}"
shift 4
DIR="${SOAK_DIR:-soak}"
DEVICES="${SOAK_DEVICES:-metal cpu}"
SAMPLE_S="${SOAK_SAMPLE_S:-60}"

BIN=$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")
corpus_dir=$(cd "$(dirname "$CORPUS")" && pwd)
corpus_name=$(basename "$CORPUS")
[ -f "$corpus_dir/$corpus_name" ] || { echo "no corpus at $CORPUS" >&2; exit 2; }

rows() { awk -F, '$1=="t"{c++} END{print c+0}' "$1" 2>/dev/null || echo 0; }

run_device() {
    local device=$1 work="$DIR/$1"
    shift
    rm -rf "$work"
    mkdir -p "$work/data"
    for f in "$corpus_dir/$corpus_name"*; do ln -s "$f" "$work/data/"; done
    (
        cd "$work" || exit 1
        "$BIN" train "data/$corpus_name" "$PRESET" --device "$device" "$@" > train.log 2>&1 &
        pid=$!
        : > footprint.log
        last_sample=0
        while kill -0 "$pid" 2>/dev/null; do
            csv=$(find . -maxdepth 1 -name '*_metrics.csv' | head -1)
            if [ -n "$csv" ] && [ "$(rows "$csv")" -ge "$STEPS" ]; then
                kill -INT "$pid" 2>/dev/null
                break
            fi
            now=$(date +%s)
            if [ $((now - last_sample)) -ge "$SAMPLE_S" ]; then
                bytes=$(footprint -f bytes --noCategories "$pid" 2>/dev/null \
                        | sed -n 's/.*Footprint: \([0-9]*\) B.*/\1/p' | head -1)
                [ -n "$bytes" ] && echo "$now $([ -n "$csv" ] && rows "$csv" || echo 0) $bytes" >> footprint.log
                last_sample=$now
            fi
            sleep 1
        done
        wait "$pid"
        echo $? > exit_status
    )
}

for device in $DEVICES; do
    echo "soak: $PRESET on $device for $STEPS steps in $DIR/$device"
    run_device "$device" "$@"
done

python3 - "$DIR" "$STEPS" $DEVICES <<'EOF'
import glob, math, os, statistics, sys

soak_dir, steps, devices = sys.argv[1], int(sys.argv[2]), sys.argv[3:]

def load(device):
    csvs = glob.glob(os.path.join(soak_dir, device, "*_metrics.csv"))
    train, evals = {}, {}
    if csvs:
        for line in open(csvs[0]):
            f = line.strip().split(",")
            if f[0] == "t":
                train[int(f[1])] = (float(f[2]), float(f[4]), int(f[5]))
            elif f[0] == "e":
                evals[int(f[1])] = float(f[2])
    status_file = os.path.join(soak_dir, device, "exit_status")
    status = open(status_file).read().strip() if os.path.exists(status_file) else "?"
    return train, evals, status

failed = False
runs = {}
for device in devices:
    train, evals, status = load(device)
    runs[device] = (train, evals)
    print(f"\n== {device}: {len(train)} steps logged, exit status {status}")
    if not train:
        failed = True
        continue
    bad = [s for s, (loss, gn, _) in sorted(train.items())
           if not (math.isfinite(loss) and math.isfinite(gn))]
    if bad:
        failed = True
        print(f"  FIRST NON-FINITE at step {bad[0]} ({len(bad)} non-finite steps)")
    else:
        print("  loss and gradient norm finite at every step")
    # SIGINT (130) is how a short run ends; anything else is a failure.
    if status not in ("0", "130", "?") and len(train) < steps:
        failed = True
    # Polling overshoots <steps> by a step or two; report at the same step.
    last = steps - 1 if steps - 1 in train else max(train)
    print(f"  train loss {train[last][0]:.4f} at step {last}")
    if evals:
        e = max(evals)
        print(f"  last val loss {evals[e]:.4f} at step {e}")
    for line in open(os.path.join(soak_dir, device, "train.log"), errors="replace"):
        if line.startswith("Final val loss"):
            print("  " + line.strip())
    finite = [(gn, s) for s, (_, gn, _) in train.items() if math.isfinite(gn)]
    top = sorted(finite, reverse=True)[:3]
    print("  largest grad norms: " + ", ".join(f"{gn:.3f} (step {s})" for gn, s in top))
    # A spike: more than 3x the median of the previous 100 steps.
    order = sorted(train)
    spikes = []
    for i, s in enumerate(order[100:], start=100):
        window = [train[order[j]][1] for j in range(i - 100, i) if math.isfinite(train[order[j]][1])]
        if window and train[s][1] > 3 * statistics.median(window):
            spikes.append(s)
    print(f"  grad norm spikes (> 3x trailing median): {len(spikes)}"
          + (f", first at step {spikes[0]}" if spikes else ""))
    ms = [train[s][2] for s in order if s >= 5]
    if len(ms) >= 20:
        tenth = max(len(ms) // 10, 1)
        head, tail = statistics.median(ms[:tenth]), statistics.median(ms[-tenth:])
        print(f"  step time: median {head:.0f} ms over the first tenth, {tail:.0f} ms over the "
              f"last ({(tail / head - 1) * 100:+.1f}%)")
    fp = os.path.join(soak_dir, device, "footprint.log")
    if os.path.exists(fp):
        samples = [tuple(map(int, l.split())) for l in open(fp) if l.strip()]
        if samples:
            mb = [b / 2**20 for _, _, b in samples]
            print(f"  footprint: {len(samples)} samples, {mb[0]:.0f} MB at step {samples[0][1]}, "
                  f"max {max(mb):.0f} MB, {mb[-1]:.0f} MB at step {samples[-1][1]}")

if len(runs) == 2:
    (a, (ta, ea)), (b, (tb, eb)) = runs.items()
    common = sorted(set(ta) & set(tb))
    both = [s for s in common if math.isfinite(ta[s][0]) and math.isfinite(tb[s][0])]
    if both:
        diffs = [(abs(ta[s][0] - tb[s][0]), s) for s in both]
        worst = max(diffs)
        print(f"\n== {a} vs {b}: {len(both)} common finite steps")
        print(f"  train loss |diff|: max {worst[0]:.4g} (step {worst[1]}), "
              f"mean {statistics.mean(d for d, _ in diffs):.4g}")
        for lo in range(0, max(both) + 1, max(len(both) // 8, 1)):
            window = [d for d, s in diffs if lo <= s < lo + max(len(both) // 8, 1)]
            if window:
                print(f"    steps {lo}+: mean |diff| {statistics.mean(window):.4g}")
    ecommon = sorted(set(ea) & set(eb))
    if ecommon:
        ediffs = [(abs(ea[s] - eb[s]), s) for s in ecommon]
        worst = max(ediffs)
        print(f"  val loss |diff| over {len(ecommon)} evals: max {worst[0]:.4g} (step {worst[1]})")
        print("    " + ", ".join(f"{s}: {ea[s]:.4f}/{eb[s]:.4f}" for s in ecommon))

sys.exit(1 if failed else 0)
EOF
