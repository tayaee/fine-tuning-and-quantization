#!/usr/bin/env bash
# s0-07-picall-control.sh [smoke|full] — pi 툴콜 대조군 (Stage 0 절차 3).
# 기준 모델(minimax-m3)을 ~/.pi/agent/models.json에 등록하고 고정 질문으로
# pi의 도구 호출 왕복이 동작하는지 확인한다. s0-19(우리 모델) 판정의 대조군이다.
# MINIMAX_API_KEY가 없으면 SKIP(실패 아님).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
source "$RQ_ROOT/config/s0-pi.inc"
rq_repro_echo
rq_banner "s0-07-picall-control" "Stage 0 - pi tool-call control with the reference model (fixed question)"
if [ -z "${MINIMAX_API_KEY:-}" ]; then
  rq_log "SKIP: MINIMAX_API_KEY 없음 — 대조군 생략 (s0-19 판정 시 감안)"
  rq_stage_record 0 base pass "pi-control:skipped(no MINIMAX_API_KEY)"
  exit 0
fi
s0_pi_register "minimax-s0" "https://api.minimax.io/v1" "minimax-m3" "$MINIMAX_API_KEY"
PROBE="$(s0_pi_probe_dir)"
if s0_pi_ask "minimax-s0" "minimax-m3" "$PROBE"; then
  rq_stage_record 0 base pass "pi-control:ok"
else
  rq_stage_record 0 base fail "pi-control:fail"
  exit 1
fi
