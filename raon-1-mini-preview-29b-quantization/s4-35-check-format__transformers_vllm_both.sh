#!/usr/bin/env bash
# s4-35-check-format__transformers_vllm_both.sh [smoke|full] [--prepare] — 지시/형식 준수 스위트 (Tier 2).
#   --prepare : 고정 케이스 프롬프트 세트(prompts-format.jsonl)만 만든다 (eval 단계 앞)
#   (기본)    : s4-10-answer-benchmarks-base__transformers-only.sh/s4-15-answer-benchmarks-quantized__transformers-only.sh가 생성한 응답을 검사해 pass_rate를 산출한다
# 결과: outputs/eval-<mode>/checks.json 의 "format" 섹션.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
shift || true   # 첫 인자(mode)를 소비한다 — 남은 인자가 --prepare인지 본다
rq_repro_echo
rq_banner "s4-35-check-format__transformers_vllm_both" "Stage 4 - instruction/format compliance suite (language, JSON, markers, length, bullets)"

EXTRA=""
[ "${1:-}" = "--prepare" ] && EXTRA="--prepare"
( set -x; uv run "$RQ_ROOT/s4-35-check-format__transformers_vllm_both.py" --mode "$MODE" $EXTRA )

echo ---- result ----
if [ -n "$EXTRA" ]; then
  # prepare 모드에서는 아직 checks.json이 없다(채점은 하지 않는다) — 실패로 보지 않는다.
  ( set -x; wc -l "$RQ_OUTPUTS/eval-$MODE/prompts-format.jsonl" )
else
  ( set -x; cat "$RQ_OUTPUTS/eval-$MODE/checks.json" 2>/dev/null | head -40 ) || true
fi
