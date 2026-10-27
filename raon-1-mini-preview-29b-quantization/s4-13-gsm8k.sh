#!/usr/bin/env bash
# s4-13-gsm8k.sh [smoke|full] — eval 스모크 집계.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_repro_echo
rq_banner "s4-13-gsm8k" "Stage 4 - aggregate eval outputs into outputs/eval-<mode>/score.json"
# s4-07-eval-base.sh/s4-10-eval-quantized.sh와 같은 lock — 동시 score가 eval jsonl을 쓰는 중을 읽는 것을 막는다
# (s4-13-gsm8k는 lock 없이 jsonl을 line-by-line 파싱해 부분줄에서 깨질 수 있다).
rq_lock "$RQ_OUTPUTS/eval-$MODE"
(set -x; uv run "$RQ_ROOT/s4-13-gsm8k.py" --mode "$MODE")
