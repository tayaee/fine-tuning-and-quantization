#!/usr/bin/env bash
# 18-select-candidates.sh [mini|full] [seed] — Stage 2a wrapper. Alpaca→GSM8K 유사 후보 선별.
# 출력: alpaca-embeddings.parquet (캐시) + candidates-<mode>.parquet
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
SEED="${2:-1}"
m1b_repro_echo
IN="$M1B_DATASETS/gsm8k-train.jsonl"
OUT="$M1B_DATASETS/candidates-$MODE.parquet"
EMBCACHE="$M1B_DATASETS/alpaca-embeddings.parquet"
m1b_require "$IN"
if m1b_fresh "$OUT" "$IN"; then
  m1b_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -lh "$EMBCACHE")
  (set -x; ls -lh "$OUT")
  exit 0
fi
m1b_log "seed=$SEED"
(set -x; uv run "$M1B_ROOT/18-select-candidates.py" --mode "$MODE" --seed "$SEED")

echo ---- result ----
(set -x; ls -lh "$M1B_DATASETS/alpaca-embeddings.parquet") 
(set -x; ls -lh "$M1B_DATASETS/candidates-$MODE.parquet")
