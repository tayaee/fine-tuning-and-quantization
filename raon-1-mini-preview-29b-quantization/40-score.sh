#!/usr/bin/env bash
# 40-score.sh [smoke|full] — eval 스모크 집계.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-smoke}"
rq_repro_echo
# 37-eval.sh와 같은 lock — 동시 score가 eval jsonl을 쓰는 중을 읽는 것을 막는다
# (40-score는 lock 없이 jsonl을 line-by-line 파싱해 부분줄에서 깨질 수 있다).
rq_lock "$RQ_OUTPUTS/eval-$MODE"
(set -x; uv run "$RQ_ROOT/40-score.py" --mode "$MODE")
