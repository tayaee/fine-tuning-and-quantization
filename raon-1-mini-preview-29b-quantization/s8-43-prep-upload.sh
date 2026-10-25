#!/usr/bin/env bash
# s8-43-prep-upload.sh [smoke|full] [targets] — 업로드 전 모델 디렉터리 검증 + README.md 생성.
# s8-49-upload-hf.py(업로드 전담)가 요구하는 전제 조건을 여기서 만든다.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-smoke}"
TARGETS="${2:-${TARGETS:-all}}"
: "${ALLOW_MISSING_METRICS:=0}"
ALLOW_MISSING=""
[ "$ALLOW_MISSING_METRICS" = 1 ] && ALLOW_MISSING="--allow-missing-metrics"
rq_repro_echo
rq_banner "s8-43-prep-upload" "Stage 8/9 - verify the model directory and generate the HuggingFace model card (README.md)"
rq_lock "$RQ_OUTPUTS"
rq_log "prep-upload targets=$TARGETS allow_missing=$ALLOW_MISSING_METRICS"
(set -x; uv run "$RQ_ROOT/s8-43-prep-upload.py" --mode "$MODE" --targets "$TARGETS" $ALLOW_MISSING)

echo ---- result ----
(set -x; ls -l "$RQ_MODELS"/raon-*-"$MODE"/README.md)
(set -x; cat "$RQ_OUTPUTS/prep-upload-$MODE.json")