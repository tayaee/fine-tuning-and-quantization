#!/usr/bin/env bash
# s6-00-clean-serving-workspace.sh — 고객 시나리오(transformers) 작업 디렉터리 정리.
# README Usage-transformers 예제가 쓰는 작업 디렉터리를 찾아 삭제한다 (venv 포함).
# 경로는 snippets/<target>/usage-transformers.sh 의 `mkdir -p <dir>` 에서 뽑는다 —
# 카드와 정리 스크립트가 갈라져 조용히 어긋나지 않게 (카드가 바꾸면 이 스크립트도 따라간다).
# 없으면 아무 것도 하지 않고 정상 종료한다.
# 사용: ./s6-00-clean-serving-workspace.sh
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_banner "s6-00-clean-serving-workspace" "Stage 6 - clean the customer transformers workspace"
# 폴백(스니펫이 없을 때): 카드에 박혀 있던 기본 경로.
DEFAULT_WORKSPACE="/tmp/raon-1-mini-preview-nvfp4-awq-transformers"
WORKSPACES=()
for _snip in "$RQ_ROOT"/snippets/*/usage-transformers.sh; do
  [ -f "$_snip" ] || continue
  _p="$(sed -nE 's|^[[:space:]]*mkdir[[:space:]]+-p[[:space:]]+"?([^"[:space:]]+)"?.*|\1|p' "$_snip" | head -n 1)"
  [ -n "$_p" ] || continue
  case " ${WORKSPACES[*]:-} " in *" $_p "*) continue ;; esac
  WORKSPACES+=("$_p")
done
[ "${#WORKSPACES[@]}" -gt 0 ] || WORKSPACES=("$DEFAULT_WORKSPACE")
for _ws in "${WORKSPACES[@]}"; do
  if [ -e "$_ws" ]; then
    (set -x; rm -rf -- "$_ws")
    rq_log "removed $_ws"
  else
    rq_log "nothing to clean ($_ws 없음)"
  fi
done
