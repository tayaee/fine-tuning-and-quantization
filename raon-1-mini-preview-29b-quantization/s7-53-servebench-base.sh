#!/usr/bin/env bash
# s7-53-servebench-base.sh [mode] — 베이스 모델 서빙벤치 (전용 포트).
# 기동 → 메모리(before_load) → 로딩 대기 → 메모리(after_warmup) →
# C=1,8,32 스위프 → 메모리(before_teardown) → 종료 → 메모리(after_teardown).
# 결과: checks.json servebench.targets.base (+ 메모리 4지점).
# 사용: ./s7-53-servebench-base.sh full
set -uo pipefail
cd "$(dirname "$0")"
MODE_ARG="${1:-${MODE:-smoke}}"
source "$(dirname "$0")/config/common.env" "$MODE_ARG"
rq_banner "s7-53-servebench-base" "Stage 7/9 - serve base model on dedicated port and sweep throughput"
PORT="${SERVEBENCH_BASE_PORT:-8101}"
BASE_URL="${BASE_URL:-http://localhost:${PORT}/v1}"
export PORT BASE_URL
source "$(dirname "$0")/servebench-common.inc"
LOAD_TIMEOUT="${LOAD_TIMEOUT:-1800}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-1800}"
SWEEP="${SERVEBENCH_SWEEP:-1,8,32}"
export SERVE_BENCH_N="${SERVE_BENCH_N:-32}" SERVE_BENCH_MAXTOK="${SERVE_BENCH_MAXTOK:-256}"
mkdir -p logs
trap 'kill_server; exit 130' INT TERM

if [ -n "$(pids_on_port)" ]; then
  echo "Port ${PORT}가 이미 사용 중($(pids_on_port)) — 기존 서버를 먼저 정리하라." >&2
  exit 1
fi
# base는 dequant 1회로 고정 → 서빙벤치도 전체 파이프라인에서 딱 1회.
# 이미 C 스위프 완측 기록이 있으면 skip (FORCE=1이면 재측정).
if [ "${FORCE:-0}" != 1 ]; then
  _have="$(MOD="$MODE" SWP="$SWEEP" python3 -c "
import json, os
try:
    sb = json.load(open(os.environ['RQ_SHARED'] + '/outputs/eval-' + os.environ['MOD'] + '/checks.json')).get('servebench', {})
    base = (sb.get('targets') or {}).get('base', {})
    need = [c.strip() for c in os.environ.get('SWP', '1,8,32').split(',')]
    ok = all((base.get(c) or {}).get('tok_per_s_p50') is not None for c in need) and bool((base.get('memory') or {}))
    print('yes' if ok else 'no')
except (OSError, json.JSONDecodeError):
    print('no')
")"
  if [ "$_have" = yes ]; then
    echo "[skip] base servebench 기록 이미 존재 (FORCE=1로 재측정)."
    exit 0
  fi
fi
LOGF="logs/53.servebench.${MODE}.base.log"
rm -f "$LOGF" "$LOGF.serve"
if serve_and_sweep "$MODE" "base" "none" "$LOGF"; then
  merge_memory "$MODE" "base" "${LOGF%.log}.mem.jsonl"
else
  echo "[FAIL] base servebench — 로그: $LOGF / $LOGF.serve"
  exit 1
fi
echo "########## 결과 ##########"
tail -5 "$LOGF"
cat "${LOGF%.log}.mem.jsonl"
