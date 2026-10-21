#!/usr/bin/env bash
# 24-postprocess.sh [mini|full] — Stage 2c wrapper. 생성 CSV 후처리→SFT JSONL.
# 입력: generated-<mode>.csv → 출력: synthetic-<mode>.jsonl
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
m1b_repro_echo
IN="$M1B_DATASETS/generated-$MODE.csv"
OUT="$M1B_DATASETS/synthetic-$MODE.jsonl"
m1b_require "$IN"
if m1b_fresh "$OUT" "$IN"; then
  m1b_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -lh "$OUT")
  (set -x; wc -l "$OUT")
  exit 0
fi
m1b_log "in=$IN out=$OUT"
uv run "$M1B_ROOT/24-postprocess.py" --mode "$MODE"

echo ---- result ----
(set -x; ls -lh "$M1B_DATASETS/synthetic-$MODE.jsonl")
(set -x; wc -l "$M1B_DATASETS/synthetic-$MODE.jsonl")
