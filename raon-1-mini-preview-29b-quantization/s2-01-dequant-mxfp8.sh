#!/usr/bin/env bash
# s2-01-dequant-mxfp8.sh — MXFP8 베이스 → BF16 복원 (32/33/35/36의 선행 조건).
# 출력: models/raon-bf16 (mode 독립, 1회).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_repro_echo
rq_banner "s2-01-dequant-mxfp8" "Stage 2 - restore the BF16 base model from the MXFP8 checkpoint (models/raon-bf16)"
OUT="$RQ_MODELS/raon-bf16"
rq_lock "$OUT"
if [ -f "$OUT/config.json" ] && [ "${FORCE:-0}" != 1 ] \
  && ! grep -q '"quantization_config"' "$OUT/config.json" \
  && grep -q 'return nn.Linear(in_features, out_features, bias=False)' "$OUT/modeling_raon1_mini.py" 2>/dev/null \
  && grep -q 'get_calibrate_all_experts_flag' "$OUT/modeling_raon1_mini.py" 2>/dev/null \
  && [ "$OUT/config.json" -nt "$RQ_ROOT/s2-01-dequant-mxfp8.py" ]; then
  rq_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -l "$OUT" | head -12)
  exit 0
fi
export UV_NO_SYNC=1; unset VIRTUAL_ENV
(set -x; uv run "$RQ_ROOT/s2-01-dequant-mxfp8.py")

echo ---- result ----
(set -x; ls -l "$OUT" | head -12)
