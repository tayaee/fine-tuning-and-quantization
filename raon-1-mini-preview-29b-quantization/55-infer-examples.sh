#!/usr/bin/env bash
# 55-infer-examples.sh [n] — 서빙 중 서버에 스모크 추론 (터미널2).
set -euo pipefail
source "$(dirname "$0")/config/common.env" smoke
N="${1:-3}"
rq_repro_echo
(set -x; uv run "$RQ_ROOT/55-infer-examples.py" --n "$N")
