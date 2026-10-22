#!/usr/bin/env bash
# 12-prep-calibration.sh [smoke|full] [n] [seed] — 한국어 혼합 → raon-calibration-<n>.jsonl.
# smoke 기본 8 (10분 목표), full 기본 256.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-smoke}"
N="${2:-$CALIB_N}"
SEED="${3:-42}"
rq_repro_echo
OUT="$RQ_DATASETS/raon-calibration-$N.jsonl"
if rq_fresh "$OUT" "$RQ_ROOT/12-prep-calibration.py"; then
  rq_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -lh "$OUT")
  (set -x; wc -l "$OUT")
  exit 0
fi
export UV_NO_SYNC=1; unset VIRTUAL_ENV
(set -x; uv run "$RQ_ROOT/12-prep-calibration.py" --n "$N" --seed "$SEED")

echo ---- result ----
(set -x; ls -lh "$OUT")
(set -x; wc -l "$OUT")
