#!/usr/bin/env bash
# s0-21-stop-server-with-vllm-fork.sh — s0-13 vLLM 서버 정리 단독 유틸리티 (Stage 0).
# pidfile(기본 logs/s0-serve-base.pid) 프로세스를 내리고 포트 해제를 확인한다.
# 정상 플로우에서는 s0-22-bench-latency-with-vllm-fork.sh가 정리한다.
# crashes/중단 후 잔존 서버 수동 정리에 쓴다. PORT/PIDF 환경변수로 오버라이드.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_repro_echo
rq_banner "s0-21-stop-server-with-vllm-fork" "Stage 0 - stop the vLLM base server left by s0-13"
PORT="${PORT:-8000}"
PIDF="${PIDF:-$RQ_ROOT/logs/s0-serve-base.pid}"
if [ -f "$PIDF" ]; then
  case "$(cat "$PIDF" 2>/dev/null)" in
    docker:*)
      _c="$(sed 's/^docker://' "$PIDF")"
      docker logs --tail 10 "$_c" 2>/dev/null || true
      docker rm -f "$_c" >/dev/null 2>&1 || true
      rq_log "stopped container $_c" ;;
    *)
      if kill -0 "$(cat "$PIDF")" 2>/dev/null; then
        kill "$(cat "$PIDF")" 2>/dev/null || true
        rq_log "stop signal sent pid=$(cat "$PIDF")"
      fi ;;
  esac
fi
rm -f "$PIDF"
for _i in $(seq 1 30); do
  ss -ltn 2>/dev/null | grep -q ":${PORT} " || { echo "Port ${PORT} is free."; exit 0; }
  sleep 1
done
echo "FAIL: port ${PORT} still occupied after stop" >&2
exit 1
