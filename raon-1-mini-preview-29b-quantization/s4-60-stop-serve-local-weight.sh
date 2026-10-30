#!/usr/bin/env bash
# s4-60-stop-serve-local-weight.sh — s4-40/45가 띄운 로컬 서빙 서버를 내린다 (in-process 측정 전 메모리 확보).
# pidfile 기준이라 s4-40/45가 띄운 프로세스만 죽인다. 서버가 없어도 정상 종료한다.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${MODE:-smoke}"
rq_banner "s4-60-stop-serve-local-weight" "Stage 4 - stop the local serving server"
stopped=0
for pf in "$RQ_ROOT/logs/s4-transformers.pid" "$RQ_ROOT/logs/s4-vllm-fork.pid"; do
  [ -f "$pf" ] || continue
  pid="$(cat "$pf" 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 12); do kill -0 "$pid" 2>/dev/null || break; sleep 5; done
    kill -9 "$pid" 2>/dev/null || true
    rq_log "stopped pid=$pid ($pf)"
    stopped=$((stopped + 1))
  fi
  rm -f "$pf"
done
rq_log "s4-60 OK stopped=$stopped"
