#!/usr/bin/env bash
# s6-00-clean-serving-workspace.sh — 고객 시나리오(transformers) 작업 디렉터리 정리.
# README Usage-transformers 예제가 쓰는 /tmp/raon-1-mini-preview-nvfp4-awq-transformers 를
# 있으면 삭제한다 (venv 포함). 없으면 아무 것도 하지 않고 정상 종료한다.
# 사용: ./s6-00-clean-serving-workspace.sh
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_banner "s6-00-clean-serving-workspace" "Stage 6 - clean the customer transformers workspace"
WORKSPACE="/tmp/raon-1-mini-preview-nvfp4-awq-transformers"
if [ -e "$WORKSPACE" ]; then
  (set -x; rm -rf -- "$WORKSPACE")
  rq_log "removed $WORKSPACE"
else
  rq_log "nothing to clean ($WORKSPACE 없음)"
fi
