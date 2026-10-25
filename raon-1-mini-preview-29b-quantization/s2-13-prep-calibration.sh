#!/usr/bin/env bash
# s2-13-prep-calibration.sh [smoke|full] [n] [seed] — 한국어 혼합 → raon-calibration-<n>.jsonl.
# smoke 기본 8 (10분 목표), full 기본 256.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-smoke}"
N="${2:-$CALIB_N}"
SEED="${3:-42}"
rq_repro_echo
rq_banner "s2-13-prep-calibration" "Stage 2/9 - build the calibration set (Korean HF datasets -> datasets/raon-calibration-<N>.jsonl)"
OUT="$RQ_DATASETS/raon-calibration-$N.jsonl"
# 동시 build(8종 wrapper 병렬 실행)가 같은 calib 파일을 동시에 만들지 않게 한다.
# 먼저 들어온 쪽이 생성하고, 기다린 쪽은 freshness로 즉시 skip한다.
rq_lock "$RQ_DATASETS"
if rq_fresh "$OUT" "$RQ_ROOT/s2-13-prep-calibration.py"; then
  rq_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -lh "$OUT")
  (set -x; wc -l "$OUT")
  exit 0
fi
export UV_NO_SYNC=1; unset VIRTUAL_ENV
(set -x; uv run "$RQ_ROOT/s2-13-prep-calibration.py" --n "$N" --seed "$SEED")

echo ---- result ----
(set -x; ls -lh "$OUT")
(set -x; wc -l "$OUT")
