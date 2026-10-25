#!/usr/bin/env bash
# s5-28-quant-nvfp4-awq.sh [smoke|full] [calib] — 원본→NVFP4 (AWQ smoothing).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-smoke}"
rq_banner "s5-28-quant-nvfp4-awq" "Stage 5/9 - AWQ smoothing + NVFP4 quantization (models/raon-nvfp4-awq-<mode>)"
CALIB="${2:-$CALIB_N}"
: "${CALIB_FILE:=$RQ_DATASETS/raon-calibration-$CALIB.jsonl}"
OUT="$RQ_MODELS/raon-nvfp4-awq-$MODE"
rq_lock "$OUT"
: "${RQ_SRC:=$RQ_MODELS/raon-bf16}"
rq_require "$CALIB_FILE" "$RQ_SRC/config.json"
if [ -f "$OUT/config.json" ] \
  && ! grep -q '"format":[[:space:]]*"dense"' "$OUT/config.json" \
  && rq_fresh "$OUT/config.json" "$CALIB_FILE" "$RQ_SRC/config.json" "$RQ_ROOT/s5-28-quant-nvfp4-awq.py"; then
  rq_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -l "$OUT")
  exit 0
fi
rq_log "target=nvfp4-awq calib=$CALIB seq=$SEQ_LEN src=$CALIB_FILE out=$OUT"
(set -x; uv run "$RQ_ROOT/s5-28-quant-nvfp4-awq.py" --mode "$MODE" --calib "$CALIB" --calib-file "$CALIB_FILE" --seq-len "$SEQ_LEN")

echo ---- result ----
(set -x; ls -l "$OUT")
