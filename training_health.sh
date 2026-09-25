#!/usr/bin/env bash
# Health monitor for a grad.cpp training run.
#
# Emits one line per noteworthy training event. Stdout is the event stream:
# stay quiet while the run is healthy, speak up on anything actionable, and
# emit a heartbeat often enough that silence never has to be interpreted.
# Reads <prefix>_metrics.csv (written by the trainer) and, if present,
# <prefix>_supervisor.log (written by train_supervised.sh). Read-only.
#
# Usage:  ./training_health.sh <prefix> [poll_seconds] [heartbeat_every_n_polls]
#   e.g.  ./training_health.sh tinystories_modern
set -uo pipefail

PREFIX="${1:?usage: training_health.sh <prefix> [poll_s] [hb_polls]}"
POLL="${2:-120}"
HB_EVERY="${3:-15}"          # 15 * 120s = heartbeat every ~30 min

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV="$REPO/${PREFIX}_metrics.csv"
SUPLOG="$REPO/${PREFIX}_supervisor.log"

# Expected peak is ~5.4GB (BENCHMARKS #8). 9GB means something regressed;
# this machine has 18GB and OOM is how the first medium run died.
MEM_WARN_MB=9000
STALL_POLLS=5                # no new step across 5 polls (~10 min) = stalled
GRAD_WARN=50                 # clip is 5.0; raw norms this high are pathological

say() { echo "[$(date +%H:%M)] $*"; }

last_step=-1; stall=0; poll=0; warned_mem=0; warned_grad=0
base_ms=""; last_restarts=0; last_val=""

while true; do
    poll=$(( poll + 1 ))

    if [ ! -f "$CSV" ]; then
        [ $(( poll % HB_EVERY )) -eq 0 ] && say "waiting: no metrics CSV yet ($PREFIX)"
        sleep "$POLL"; continue
    fi

    # Last training row: t,step,loss,lr,grad_norm,step_ms,mem_mb,wall_s
    row=$(grep '^t,' "$CSV" | tail -1)
    if [ -z "$row" ]; then sleep "$POLL"; continue; fi
    IFS=, read -r _ step loss lr gnorm step_ms mem_mb wall <<<"$row"

    # Run length from the meta row: m,total_steps,tokens_per_step,params,desc.
    # It repeats on every resume; the last one is current.
    total=$(grep '^m,' "$CSV" | tail -1 | cut -d, -f2)

    # --- hard failure: non-finite loss. The run is dead, say so loudly. ---
    case "$loss" in
        *nan*|*NaN*|*inf*|*Inf*)
            say "RED FLAG: loss is $loss at step $step. Training has diverged; stop the run."
            sleep "$POLL"; continue ;;
    esac

    # --- stall detection: silence must not look like health ---
    if [ "$step" = "$last_step" ]; then
        stall=$(( stall + 1 ))
        if [ "$stall" -eq "$STALL_POLLS" ]; then
            if pgrep -f "grad train" >/dev/null 2>&1; then
                say "RED FLAG: no step progress in ~$(( STALL_POLLS * POLL / 60 ))m but process is alive (hung?). Step stuck at $step."
            else
                say "RED FLAG: no step progress and no 'grad train' process. Run died; supervisor should be resuming."
            fi
        fi
    else
        stall=0
    fi
    last_step="$step"

    # --- crashloop: supervisor restarting repeatedly ---
    if [ -f "$SUPLOG" ]; then
        # grep -c prints 0 itself on no match (and exits 1), so only an
        # unreadable log needs the default.
        restarts=$(grep -c "backing off then resuming" "$SUPLOG" 2>/dev/null)
        restarts=${restarts:-0}
        if [ "$restarts" -gt "$last_restarts" ]; then
            say "RED FLAG: supervisor restarted the run (total restarts: $restarts). Check ${PREFIX}_supervisor.log."
            last_restarts="$restarts"
        fi
    fi

    # --- memory: the historical failure mode on this machine ---
    if [ "${mem_mb%.*}" -gt "$MEM_WARN_MB" ] 2>/dev/null && [ "$warned_mem" -eq 0 ]; then
        say "RED FLAG: memory ${mem_mb}MB exceeds ${MEM_WARN_MB}MB (expected peak ~5400MB). OOM risk."
        warned_mem=1
    fi

    # --- gradient explosion (clip is 5.0; raw norm this high means trouble) ---
    if awk -v g="$gnorm" -v t="$GRAD_WARN" 'BEGIN{exit !(g+0 > t)}' && [ "$warned_grad" -eq 0 ]; then
        say "RED FLAG: grad_norm $gnorm at step $step (clip 5.0). LR may be too high."
        warned_grad=1
    fi

    # --- thermal / throughput degradation vs the first measured steps ---
    [ -z "$base_ms" ] && [ "$step" -gt 200 ] 2>/dev/null && base_ms="$step_ms"
    if [ -n "$base_ms" ] && awk -v c="$step_ms" -v b="$base_ms" 'BEGIN{exit !(b>0 && c > b*1.6)}'; then
        say "NOTE: step time ${step_ms}ms vs ~${base_ms}ms baseline (>60% slower). Likely thermal throttling."
        base_ms="$step_ms"   # re-baseline so this fires on further degradation, not every poll
    fi

    # --- validation: the signal that actually says whether it is learning ---
    vrow=$(grep '^e,' "$CSV" | tail -1)
    if [ -n "$vrow" ]; then
        IFS=, read -r _ vstep vloss _ <<<"$vrow"
        if [ "$vstep" != "${last_val%%:*}" ]; then
            prev="${last_val##*:}"
            if [ -n "$prev" ] && awk -v n="$vloss" -v p="$prev" 'BEGIN{exit !(n > p + 0.15)}'; then
                say "NOTE: val loss rose $prev -> $vloss at step $vstep. Watch for divergence."
            fi
            last_val="$vstep:$vloss"
        fi
    fi

    # --- heartbeat ---
    if [ $(( poll % HB_EVERY )) -eq 0 ]; then
        v="${last_val##*:}"; [ -z "$v" ] && v="n/a"
        if [ "${total:-0}" -gt 0 ] 2>/dev/null; then
            pct=$(awk -v s="$step" -v t="$total" 'BEGIN{printf "%.1f", (s+1)/t*100}')
            eta=$(awk -v s="$step" -v t="$total" -v ms="$step_ms" 'BEGIN{printf "%.1f", (t-s-1)*ms/1000/3600}')
            say "ok: step $step/$total (${pct}%) loss=$loss val=$v grad=$gnorm ${step_ms}ms ${mem_mb}MB eta=${eta}h"
        else
            say "ok: step $step (no meta row, total unknown) loss=$loss val=$v grad=$gnorm ${step_ms}ms ${mem_mb}MB"
        fi
    fi

    sleep "$POLL"
done
