#!/usr/bin/env bash
# s8-49-upload-hf.sh [smoke|full] [targets] [extra args] — HF Hub 업로드 (tayaee/*).
# 전제: s8-43-prep-upload.sh가 README.md를 만들어 둔 상태여야 한다 (없으면 즉시 중단).
#   DRY_RUN=1 ./s8-49-upload-hf.sh full          # 네트워크 없이 준비 상태만 검증
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-smoke}"
DRY=""
if [ "${2:-}" = "--dry-run" ]; then
  TARGETS="${TARGETS:-all}"
  DRY="--dry-run"
else
  TARGETS="${2:-${TARGETS:-all}}"
  [ "${3:-}" = "--dry-run" ] && DRY="--dry-run"
fi
[ "${DRY_RUN:-0}" = 1 ] && DRY="--dry-run"
rq_repro_echo
rq_banner "s8-49-upload-hf" "Stage 8/9 - upload the prepared model folder to the HuggingFace Hub"
if [ "${SKIP_VERIFY:-0}" != 1 ]; then
  "$RQ_ROOT/s8-46-verify-upload.sh" "$MODE" "$TARGETS"
fi
(set -x; uv run "$RQ_ROOT/s8-49-upload-hf.py" --mode "$MODE" --targets "$TARGETS" $DRY)

echo ---- result ----
rq_log "uploaded targets=$TARGETS dry_run=${DRY:-0} (see URLs above)"