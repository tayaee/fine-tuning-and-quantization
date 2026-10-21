#!/usr/bin/env bash
# 20-build-prompts.sh [mini|full] — Stage 2b wrapper. 후보→template 프롬프트 생성.
# 출력: prompts-<mode>.parquet. PUSH=1이면 Hub(tayaee/alpaca_syntheticdatagen_prompt-<mode>) 업로드.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
ARGS=()
[ "${PUSH:-0}" = 1 ] && ARGS+=(--push)
m1b_repro_echo
IN="$M1B_DATASETS/candidates-$MODE.parquet"
OUT="$M1B_DATASETS/prompts-$MODE.parquet"
m1b_require "$IN"
if [ "${PUSH:-0}" != 1 ] && m1b_fresh "$OUT" "$IN"; then
  m1b_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -lh "$OUT")
  exit 0
fi
m1b_log "push=${PUSH:-0}"
uv run "$M1B_ROOT/20-build-prompts.py" --mode "$MODE" "${ARGS[@]}"

echo ---- result ----
(set -x; ls -lh "$M1B_DATASETS/prompts-$MODE.parquet")
