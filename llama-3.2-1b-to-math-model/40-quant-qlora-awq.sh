#!/usr/bin/env bash
# 40-quant-qlora-awq.sh [mini|full] [calib] (STRAT env, 기본 single) — Stage 5a wrapper. QLoRA-merged→AWQ 4bit.
# 입력: synthetic-qlora-<mode>-$STRAT-merged → 출력: ...-merged-awq
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
STRAT="${STRAT:-single}"
case "$STRAT" in single|ddp|fsdp) ;; *) echo "STRAT must be single|ddp|fsdp" >&2; exit 1;; esac
m1b_repro_echo
CALIB="${2:-$CALIB_N}"
: "${CALIB_FILE:=$M1B_DATASETS/gsm8k-calibration-256.jsonl}"
SRC="$M1B_MODELS/synthetic-qlora-$MODE-$STRAT-merged"
OUT="$M1B_MODELS/synthetic-qlora-$MODE-$STRAT-merged-awq"
m1b_lock "$OUT"
m1b_require "$SRC/config.json" "$CALIB_FILE"
if [ -f "$OUT/config.json" ] && m1b_fresh "$OUT/config.json" "$SRC/config.json" "$CALIB_FILE"; then
  m1b_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -l "$OUT")
  exit 0
fi
m1b_log "calib=$CALIB src=$CALIB_FILE out=$OUT"
(set -x; uv run "$M1B_ROOT/40-quant-qlora-awq.py" --mode "$MODE" --strat "$STRAT" --calib "$CALIB" --calib-file "$CALIB_FILE")

echo ---- result ----
(set -x; ls -l "$OUT")
