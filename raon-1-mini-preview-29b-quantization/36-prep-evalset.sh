#!/usr/bin/env bash
# 36-prep-evalset.sh [smoke|full] — 표준 GSM8K 평가 세트 준비 (Stage 7 입력).
# openai/gsm8k(main/test) 표준 테스트셋에서 결정론적 샘플을 뽑아
# outputs/eval-<mode>/prompts-gsm8k.jsonl + evalset-<mode>.json(매니페스트)를 만든다.
# 사용: ./36-prep-evalset.sh full        # GSM8K_N 환경변수로 표본 크기 지정(0=전체 1319)
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-smoke}"
rq_repro_echo

N="${GSM8K_N:-0}"
rq_lock "$RQ_OUTPUTS/eval-$MODE"

( set -x; uv run "$RQ_ROOT/36-prep-evalset.py" --mode "$MODE" --n "$N" )

echo ---- result ----
( set -x; wc -l "$RQ_OUTPUTS/eval-$MODE/prompts-gsm8k.jsonl"; cat "$RQ_OUTPUTS/eval-$MODE/evalset-$MODE.json" )
