#!/usr/bin/env bash
# s9-16-inference-test.sh [n] — 서빙 중 서버에 스모크 추론 (터미널2).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${MODE:-smoke}"
N="${1:-3}"
rq_repro_echo
rq_banner "s9-16-inference-test" "Stage 9 - run inference examples against a running server"
(set -x; uv run "$RQ_ROOT/s9-16-inference-test.py" --n "$N")
