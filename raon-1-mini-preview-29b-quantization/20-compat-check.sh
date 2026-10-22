#!/usr/bin/env bash
# 20-compat-check.sh — 양자화 전제조건 점검. 32~36 실행 전 필수.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-smoke}"
rq_repro_echo
export UV_NO_SYNC=1; unset VIRTUAL_ENV
(set -x; uv run "$RQ_ROOT/20-compat-check.py")
