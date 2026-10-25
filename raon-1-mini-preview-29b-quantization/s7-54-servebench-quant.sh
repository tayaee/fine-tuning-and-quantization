#!/usr/bin/env bash
# s7-54-servebench-quant.sh [mode] [target] — 양자화 모델 서빙벤치 (전용 포트).
# base와 동일 절차: 기동 → 메모리 4지점 → C=1,8,32 스위프 → 종료.
# 결과는 checks.json servebench.targets.<target>에 누적된다
# (s7-52가 기존 기록을 보존하므로 base 결과가 지워지지 않는다).
# 사용: ./s7-54-servebench-quant.sh full nvfp4-awq
set -uo pipefail
cd "$(dirname "$0")"
MODE_ARG="${1:-${MODE:-smoke}}"
TARGET="${2:-${TARGETS:-nvfp4-awq}}"
source "$(dirname "$0")/config/common.env" "$MODE_ARG"
rq_banner "s7-54-servebench-quant" "Stage 7/9 - serve quantized model on dedicated port and sweep throughput"
case "$TARGET" in
  base) echo "base는 s7-53-servebench-base.sh를 쓰라." >&2; exit 1 ;;
  awq-w3|awq-w5|nvfp4-awq|nvfp4-gptq) ;;
  *) echo "target: awq-w3|awq-w5|nvfp4-awq|nvfp4-gptq" >&2; exit 1 ;;
esac
PORT="${SERVEBENCH_QUANT_PORT:-8102}"
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
LOGF="logs/54.servebench.${MODE}.${TARGET}.log"
rm -f "$LOGF" "$LOGF.serve"
if serve_and_sweep "$MODE" "$TARGET" "compressed-tensors" "$LOGF"; then
  merge_memory "$MODE" "$TARGET" "${LOGF%.log}.mem.jsonl"
else
  echo "[FAIL] $TARGET servebench — 로그: $LOGF / $LOGF.serve"
  exit 1
fi
echo "########## 결과 ##########"
tail -5 "$LOGF"
cat "${LOGF%.log}.mem.jsonl"
