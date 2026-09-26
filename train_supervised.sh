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
#
# The corpus path is relative to the repo root, where the trainer runs and
# writes its checkpoints. GRAD_TOKENIZER=v1|v2 picks the tokenizer lineage;
# unset, it is the one that already has a run for this corpus and preset,
# else the one whose tokenizer file exists, else v2.
set -uo pipefail

CORPUS="${1:?usage: train_supervised.sh <corpus.txt> <preset> [max_restarts]}"
PRESET="${2:?usage: train_supervised.sh <corpus.txt> <preset> [max_restarts]}"
MAX_RESTARTS="${3:-500}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$REPO/build/grad"
cd "$REPO" || exit 1
# Checkpoint prefix, derived exactly as src/cli/train.cpp does: the corpus
# filename minus its last extension, then "_modern" for the modern-architecture
# presets and "_fast" for the fast ones, so lineages coexist on one corpus.
# v2 runs add "_v2" after the corpus name.
BASE="$(basename "$CORPUS")"; BASE="${BASE%.*}"
SUFFIX=""
case "$PRESET" in modern|fast-modern) SUFFIX="${SUFFIX}_modern" ;; esac
case "$PRESET" in fast*) SUFFIX="${SUFFIX}_fast" ;; esac

has_run() { [ -f "${BASE}$1${SUFFIX}_resume_state.bin" ] || [ -f "${BASE}$1${SUFFIX}_final.bin" ]; }
has_v1_tokenizer() {
    compgen -G "${CORPUS}.tokenizer_*.cache" >/dev/null \
        || { [ "$CORPUS" = data/shakespeare.txt ] && compgen -G "tokenizer_*.cache" >/dev/null; }
}
has_v2_tokenizer() { compgen -G "${CORPUS}.bytebpe_*.tok" >/dev/null; }

TOKENIZER="${GRAD_TOKENIZER:-}"
if [ -z "$TOKENIZER" ]; then
    if has_run "" && has_run "_v2"; then
        echo "FATAL: both a v1 and a v2 run exist for ${BASE}${SUFFIX}; set GRAD_TOKENIZER" >&2; exit 1
    elif has_run ""; then TOKENIZER=v1
    elif has_run "_v2"; then TOKENIZER=v2
    elif has_v1_tokenizer && has_v2_tokenizer; then
        echo "FATAL: v1 and v2 tokenizers both exist for $CORPUS; set GRAD_TOKENIZER" >&2; exit 1
    elif has_v1_tokenizer; then TOKENIZER=v1
    else TOKENIZER=v2
    fi
fi
case "$TOKENIZER" in
    v1) PREFIX="${BASE}${SUFFIX}" ;;
    v2) PREFIX="${BASE}_v2${SUFFIX}" ;;
    *) echo "FATAL: GRAD_TOKENIZER must be v1 or v2, got '$TOKENIZER'" >&2; exit 1 ;;
esac
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

log "supervising: preset=$PRESET tokenizer=$TOKENIZER prefix=$PREFIX corpus=$CORPUS"

for (( attempt = 1; attempt <= MAX_RESTARTS; attempt++ )); do
    # Resume only if the trainer actually left state behind; otherwise cold start.
    if [ -f "$REPO/${PREFIX}_resume_state.bin" ]; then
        INIT="resume"
    else
        INIT=""
    fi
    log "attempt $attempt/$MAX_RESTARTS (init='${INIT:-scratch}')"

    "$BIN" train "$CORPUS" "$PRESET" $INIT --tokenizer "$TOKENIZER" >>"$LOG" 2>&1
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

    # 2 = the command line was rejected (unknown preset, bad flag). Retrying
    # the same arguments cannot succeed, so stop instead of crashlooping.
    if [ $rc -eq 2 ]; then
        log "FATAL: grad rejected the command line (see above); not restarting"
        exit 2
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
