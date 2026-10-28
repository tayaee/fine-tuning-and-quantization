#!/usr/bin/env bash
# s3-01-quant-nvfp4-awq-vllm.sh [smoke|full] [calib] — BF16→NVFP4 (AWQ Lite, ModelOpt).
# 산출물 표지가 modelopt_fp4라 vLLM 포크 nvfp4칸 적재 대상이다 (s3-01 transformers판과 구분).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_banner "s3-01-quant-nvfp4-awq-vllm" "Stage 3 - AWQ Lite + NVFP4 quantization, ModelOpt (models/raon-nvfp4-awq-vllm-<mode>)"
CALIB="${2:-$CALIB_N}"
: "${CALIB_FILE:=$RQ_DATASETS/raon-calibration-$CALIB.jsonl}"
OUT="$RQ_MODELS/raon-nvfp4-awq-vllm-$MODE"
rq_lock "$OUT"
: "${RQ_SRC:=$RQ_MODELS/raon-bf16}"
rq_require "$CALIB_FILE" "$RQ_SRC/config.json"
if [ -f "$OUT/config.json" ] \
  && grep -q '"quant_method":[[:space:]]*"modelopt_fp4"' "$OUT/config.json" \
  && rq_fresh "$OUT/config.json" "$CALIB_FILE" "$RQ_SRC/config.json" "$RQ_ROOT/s3-01-quant-nvfp4-awq-vllm.py"; then
  rq_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -l "$OUT")
  exit 0
fi
rq_log "target=nvfp4-awq-vllm calib=$CALIB seq=$SEQ_LEN src=$CALIB_FILE out=$OUT"
export UV_NO_SYNC=1; unset VIRTUAL_ENV
(set -x; uv run "$RQ_ROOT/s3-01-quant-nvfp4-awq-vllm.py" --mode "$MODE" --calib "$CALIB" --calib-file "$CALIB_FILE" --seq-len "$SEQ_LEN")

echo ---- result ----
(set -x; ls -l "$OUT")
