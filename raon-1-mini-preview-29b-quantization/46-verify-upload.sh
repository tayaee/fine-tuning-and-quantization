#!/usr/bin/env bash
# 46-verify-upload.sh [smoke|full] [targets] — 업로드 전 모델·지표·README·vLLM 호환성 종합 감사.
# 43-prep-upload.sh 실행 후, 49-upload-hf.sh 실행 전에 모든 산출물의 결함 여부를 전수 점검한다.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-smoke}"
TARGETS="${2:-${TARGETS:-all}}"
: "${ALLOW_MISSING_METRICS:=0}"
: "${ALLOW_DISCARD:=0}"
: "${ALLOW_TINY:=0}"
: "${SKIP_BF16:=0}"
EXTRA_ARGS=()
[ "$ALLOW_MISSING_METRICS" = 1 ] && EXTRA_ARGS+=(--allow-missing-metrics)
[ "$ALLOW_DISCARD" = 1 ] && EXTRA_ARGS+=(--allow-discard)
[ "$ALLOW_TINY" = 1 ] && EXTRA_ARGS+=(--allow-tiny)
[ "$SKIP_BF16" = 1 ] && EXTRA_ARGS+=(--skip-bf16)
rq_repro_echo
rq_log "verify-upload mode=$MODE targets=$TARGETS"
(set -x; uv run "$RQ_ROOT/46-verify-upload.py" --mode "$MODE" --targets "$TARGETS" "${EXTRA_ARGS[@]}")

echo ---- result ----
(set -x; cat "$RQ_OUTPUTS/verify-upload-$MODE.json")
