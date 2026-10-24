#!/usr/bin/env bash
# 07-acceptance.sh [smoke|full] — Stage 1(요구사항·수용 기준) 실행기.
# config/acceptance.toml을 검증해 outputs/acceptance-<mode>.json(계약 스냅샷)을 만든다.
# 이 스크립트는 Stage 1의 유일한 산출물 생성자이며, 이후 모든 게이트가 이 파일을 기준으로 삼는다.
# 사용: ./07-acceptance.sh smoke
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-smoke}"
rq_repro_echo

# 2노드 동시 실행 시 한쪽만 쓰고 나머지는 대기 후 스킵(계약은 동일하므로 결과도 동일).
rq_lock "$RQ_OUTPUTS/acceptance"

( set -x; uv run "$RQ_ROOT/07-acceptance.py" --mode "$MODE" )

echo ---- result ----
( set -x; ls -l "$RQ_OUTPUTS/acceptance-$MODE.json" )
