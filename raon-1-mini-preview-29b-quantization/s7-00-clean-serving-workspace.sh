#!/usr/bin/env bash
# s7-00-clean-serving-workspace.sh — 고객 시나리오(vLLM) 작업 디렉터리 정리.
# README vLLM 예제(소스 빌드 + 서빙)가 쓰는 작업 디렉터리를 찾아 삭제한다 (빌드 venv 포함).
# 경로는 snippets/<target>/build-vllm-gb10.sh 의 `mkdir -p <dir>` / `VENV=<dir>` 에서 뽑는다 —
# 카드와 정리 스크립트가 갈라져 조용히 어긋나지 않게 (카드가 바뀌면 이 스크립트도 따라간다).
# 없으면 아무 것도 하지 않고 정상 종료한다.
# 사용: ./s7-00-clean-serving-workspace.sh
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_banner "s7-00-clean-serving-workspace" "Stage 7 - clean the customer vLLM workspace"
# 폴백(스니펫이 없을 때): 카드에 박혀 있던 기본 경로.
DEFAULT_WORKSPACE="/tmp/raon-1-mini-preview-nvfp4-awq-vllm"
WORKSPACES=()
for _snip in "$RQ_ROOT"/snippets/*/build-vllm-gb10.sh; do
  [ -f "$_snip" ] || continue
  _p="$(sed -nE 's|^[[:space:]]*mkdir[[:space:]]+-p[[:space:]]+"?([^"[:space:]]+)"?.*|\1|p' "$_snip" | head -n 1)"
  if [ -z "$_p" ]; then
    # VENV=/path/to/.venv → 작업 디렉터리는 그 상위
    _v="$(sed -n 's|^[[:space:]]*VENV=\([^[:space:]]*\).*|\1|p' "$_snip" | head -n 1)"
    [ -n "$_v" ] && _p="$(dirname "$_v")"
  fi
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
