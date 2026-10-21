#!/usr/bin/env bash
# 12-prep-calibration.sh [n] [seed] — gsm8k-train → gsm8k-calibration-<n>.jsonl.
# AWQ 예제 정석(256) 기본. 40/50-awq가 이 파일을 캘리브레이션으로 읽는다.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${MODE:-mini}"
N="${1:-256}"
SEED="${2:-42}"
m1b_repro_echo
IN="$M1B_DATASETS/gsm8k-train.jsonl"
OUT="$M1B_DATASETS/gsm8k-calibration-$N.jsonl"
m1b_require "$IN"
if m1b_fresh "$OUT" "$IN"; then
  m1b_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -lh "$OUT")
  (set -x; wc -l "$OUT")
  exit 0
fi
export UV_NO_SYNC=1; unset VIRTUAL_ENV
uv run "$M1B_ROOT/12-prep-calibration.py" --n "$N" --seed "$SEED"

echo ---- result ----
(set -x; ls -lh "$OUT")
(set -x; wc -l "$OUT")
