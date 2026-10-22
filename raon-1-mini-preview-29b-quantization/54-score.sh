#!/usr/bin/env bash
# 54-score.sh [smoke|full] — eval 스모크 집계.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-smoke}"
rq_repro_echo
(set -x; uv run "$RQ_ROOT/54-score.py" --mode "$MODE")
