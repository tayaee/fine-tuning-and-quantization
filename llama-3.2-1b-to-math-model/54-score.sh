#!/usr/bin/env bash
# 54-score.sh [mini|full] (STRAT env, 기본 single) — Stage 7c wrapper. eval-<mode>-<strat>/*.jsonl 채점표 출력.
# (54-score.py 자체가 결과표를 stdout에 찍고 score.json 아티팩트를 남긴다.)
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
STRAT="${STRAT:-single}"
case "$STRAT" in single|ddp|fsdp) ;; *) echo "STRAT must be single|ddp|fsdp" >&2; exit 1;; esac
m1b_repro_echo
EVALDIR="$M1B_OUTPUTS/eval-$MODE-$STRAT"
SCORE="$EVALDIR/score.json"
m1b_require "$EVALDIR"
m1b_lock "$EVALDIR"
_inputs=()
for _f in "$EVALDIR"/*.jsonl; do [ -e "$_f" ] && _inputs+=("$_f"); done
if [ -f "$SCORE" ] && [ "${#_inputs[@]}" -gt 0 ] && m1b_fresh "$SCORE" "${_inputs[@]}"; then
  m1b_log "skip: fresh $SCORE (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; cat "$SCORE")
  exit 0
fi
m1b_log "evaldir=$EVALDIR"
uv run "$M1B_ROOT/54-score.py" --mode "$MODE" --strat "$STRAT"

echo ---- result ----
(set -x; ls -l "$M1B_OUTPUTS/eval-$MODE-$STRAT/")
