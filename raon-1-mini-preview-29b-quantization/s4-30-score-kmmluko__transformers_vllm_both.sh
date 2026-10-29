#!/usr/bin/env bash
# s4-30-score-kmmluko__transformers_vllm_both.sh [smoke|full] — 표준 한국어 KMMLU 채점 + 수용 기준 대조 (Tier 2).
# s4-10-answer-benchmarks-base__transformers-only.sh/s4-15-answer-benchmarks-quantized__transformers-only.sh가 만든 응답(eval-<mode>/<target>.jsonl) 중 bench=kmmluko 행을 채점하고,
# base 대비 하락 허용폭(acceptance.toml의 kmmluko_acc_drop_max)과 대조한다.
# 결과: outputs/eval-<mode>/checks.json 의 "kmmluko" 섹션.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_repro_echo
rq_banner "s4-30-score-kmmluko__transformers_vllm_both" "Stage 4 - score the standard Korean KMMLU check and judge it against the acceptance contract"

( set -x; uv run "$RQ_ROOT/s4-30-score-kmmluko__transformers_vllm_both.py" --mode "$MODE" )

echo ---- result ----
( set -x; cat "$RQ_OUTPUTS/eval-$MODE/checks.json" )
