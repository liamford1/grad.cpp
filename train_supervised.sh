#!/usr/bin/env bash
# Self-resuming supervisor for a grad.cpp training run.
#
# Survives the process dying, the box rebooting, a session reset, or an OOM
# kill. On every start it asks the checkpoint how far the run got and either
# resumes from there or starts fresh. Safe to run repeatedly; safe to put in
# @reboot cron. No babysitting, no orchestrator, no me.
#
# Usage:  ./train_supervised.sh <corpus.txt> <preset> [max_restarts]
#   e.g.  ./train_supervised.sh data/tinystories.txt medium
set -uo pipefail

CORPUS="${1:?usage: train_supervised.sh <corpus.txt> <preset> [max_restarts]}"
PRESET="${2:?usage: train_supervised.sh <corpus.txt> <preset> [max_restarts]}"
MAX_RESTARTS="${3:-500}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$REPO/build/grad"
BASE="$(basename "$CORPUS" .txt)"
# `modern` writes its checkpoints under a separate prefix so both lineages coexist.
PREFIX="$BASE"; [ "$PRESET" = modern ] && PREFIX="${BASE}_modern"
LOG="$REPO/${PREFIX}_supervisor.log"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

# Single instance. Two supervisors racing on one checkpoint pair corrupts it.
# mkdir is atomic on every POSIX filesystem, and unlike flock it exists on
# macOS as well as Linux. Stale lock from a hard kill is cleared by PID check.
LOCKDIR="$REPO/.${PREFIX}.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
    if [ -f "$LOCKDIR/pid" ] && kill -0 "$(cat "$LOCKDIR/pid")" 2>/dev/null; then
        log "another supervisor (pid $(cat "$LOCKDIR/pid")) is running; exiting"
        exit 0
    fi
    log "clearing stale lock from a previous kill"
    rm -rf "$LOCKDIR"; mkdir "$LOCKDIR" || { log "FATAL: cannot acquire lock"; exit 1; }
fi
echo $$ > "$LOCKDIR/pid"
trap 'rm -rf "$LOCKDIR"' EXIT

[ -x "$BIN" ] || { log "FATAL: $BIN missing. Build first."; exit 1; }
[ -f "$CORPUS" ] || { log "FATAL: corpus $CORPUS missing."; exit 1; }

log "supervising: preset=$PRESET prefix=$PREFIX corpus=$CORPUS"

for (( attempt = 1; attempt <= MAX_RESTARTS; attempt++ )); do
    # Resume only if the trainer actually left state behind; otherwise cold start.
    if [ -f "$REPO/${PREFIX}_resume_state.bin" ]; then
        INIT="resume"
    else
        INIT=""
    fi
    log "attempt $attempt/$MAX_RESTARTS (init='${INIT:-scratch}')"

    "$BIN" train "$CORPUS" "$PRESET" $INIT >>"$LOG" 2>&1
    rc=$?

    # The trainer exits 0 both on completion and on a SIGINT pause (it saves
    # resume state and returns 0 either way), so tell them apart by artifact:
    # only a finished run writes <prefix>_final.bin. Either way, do not restart.
    if [ $rc -eq 0 ]; then
        if [ -f "$REPO/${PREFIX}_final.bin" ]; then
            log "training finished: ${PREFIX}_final.bin  best: ${PREFIX}_best.bin"
        else
            log "paused by SIGINT; resume state saved. Re-run this script to continue."
        fi
        exit 0
    fi

    # 130 = SIGINT. A human pressed Ctrl-C; respect it rather than fighting them.
    if [ $rc -eq 130 ]; then
        log "interrupted by SIGINT; resume state saved. Not restarting."
        exit 130
    fi

    log "exited rc=$rc; backing off then resuming from last eval checkpoint"
    # Back off so a deterministic crash (bad corpus, no disk) doesn't spin hot.
    sleep $(( attempt < 10 ? attempt * 10 : 100 ))
done

log "hit max restarts ($MAX_RESTARTS) without finishing; stopping"
exit 1
