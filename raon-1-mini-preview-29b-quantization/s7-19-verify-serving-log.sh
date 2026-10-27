#!/usr/bin/env bash
# s7-19-verify-serving-log.sh [smoke|full] [log-dir] — s7-13/s7-16 로그 기반 CAT 감사.
# logs/s7-13-<slug>.log + logs/s7-16-<slug>.log를 읽어 서빙 chat·pi 통과 여부를 판정하고
# outputs/serve-<mode>.json에 기록한다. 로그가 없으면 SKIP(사유 기록). gate-serving의 근거.
# 사용: ./s7-19-verify-serving-log.sh full
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
LOGDIR="${2:-$RQ_ROOT/logs}"
rq_repro_echo
rq_banner "s7-19-verify-serving-log" "Stage 7 - verify the serving logs and write outputs/serve-<mode>.json"
(set -x; uv run "$RQ_ROOT/s7-19-verify-serving-log.py" --mode "$MODE" --log-dir "$LOGDIR")

echo ---- result ----
(set -x; cat "$RQ_OUTPUTS/serve-$MODE.json" 2>/dev/null | head -20)
