#!/usr/bin/env bash
# s3-01-quant-awq-w3-transformers.sh [smoke|full] [calib] — 원본→AWQ-W3A16-g128 (품질 위험, smoke 먼저).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_banner "s3-01-quant-awq-w3-transformers" "Stage 3 - AWQ smoothing + GPTQ W3A16 quantization (models/raon-awq-w3-transformers-<mode>)"
CALIB="${2:-$CALIB_N}"
: "${CALIB_FILE:=$RQ_DATASETS/raon-calibration-$CALIB.jsonl}"
OUT="$RQ_MODELS/raon-awq-w3-transformers-$MODE"
rq_lock "$OUT"
: "${RQ_SRC:=$RQ_MODELS/raon-bf16}"
rq_require "$CALIB_FILE" "$RQ_SRC/config.json"
if [ -f "$OUT/config.json" ] \
  && ! grep -q '"format":[[:space:]]*"dense"' "$OUT/config.json" \
  && rq_fresh "$OUT/config.json" "$CALIB_FILE" "$RQ_SRC/config.json" "$RQ_ROOT/s3-01-quant-awq-w3-transformers.py"; then
  rq_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -l "$OUT")
  exit 0
fi
rq_log "target=awq-w3-transformers calib=$CALIB seq=$SEQ_LEN src=$CALIB_FILE out=$OUT"
(set -x; uv run "$RQ_ROOT/s3-01-quant-awq-w3-transformers.py" --mode "$MODE" --calib "$CALIB" --calib-file "$CALIB_FILE" --seq-len "$SEQ_LEN")

echo ---- result ----
(set -x; ls -l "$OUT")
