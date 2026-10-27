#!/usr/bin/env bash
# s8-76-verify-serving-log.sh [smoke|full] [log-dir] — s8-70 로그 기반 배포 후 검증.
# logs/67.<slug>.log를 읽어 서빙 정상 응답 여부를 판정하고 outputs/serve-<mode>.json에 기록한다.
# 로그가 없으면 SKIP(사유 기록). gate-serving의 근거.
# 사용: ./s8-76-verify-serving-log.sh full
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
LOGDIR="${2:-$RQ_ROOT/logs}"
rq_repro_echo
rq_banner "s8-76-verify-serving-log" "Stage 8/9 - verify the serving logs and write outputs/serve-<mode>.json"
(set -x; uv run "$RQ_ROOT/s8-76-verify-serving-log.py" --mode "$MODE" --log-dir "$LOGDIR")

echo ---- result ----
(set -x; cat "$RQ_OUTPUTS/serve-$MODE.json" 2>/dev/null | head -20)
