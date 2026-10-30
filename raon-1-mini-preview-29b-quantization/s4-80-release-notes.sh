#!/usr/bin/env bash
# s4-80-release-notes.sh [smoke|full] [--print] — SDLC 릴리스 노트 생성 (Stage 7 산출물).
# 단계 원장 + 수용 기준 + 실측 지표를 outputs/release-notes-<mode>.md로 굽는다.
# 사용: ./s4-80-release-notes.sh full
#       ./s4-80-release-notes.sh full --print   # 화면에도 출력
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
EXTRA=""
[ "${2:-}" = "--print" ] && EXTRA="--print"
rq_repro_echo
rq_banner "s4-80-release-notes" "Stage 4 - build the release notes from the stage ledger and the measured metrics"
(set -x; uv run "$RQ_ROOT/s4-80-release-notes.py" --mode "$MODE" $EXTRA)

echo ---- result ----
(set -x; ls -lh "$RQ_OUTPUTS/release-notes-$MODE.md")
