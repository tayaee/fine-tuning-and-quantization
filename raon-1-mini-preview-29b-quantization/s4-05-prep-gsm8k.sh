#!/usr/bin/env bash
# s4-05-prep-gsm8k.sh [smoke|full] — 표준 GSM8K 평가 세트 준비 (Stage 7 입력).
# openai/gsm8k(main/test) 표준 테스트셋에서 결정론적 샘플을 뽑아
# outputs/eval-<mode>/prompts-gsm8k.jsonl + evalset-<mode>.json(매니페스트)를 만든다.
# 사용: ./s4-05-prep-gsm8k.sh full        # GSM8K_N 환경변수로 표본 크기 지정(0=전체 1319)
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_repro_echo
rq_banner "s4-05-prep-gsm8k" "Stage 4 - build the standard GSM8K evaluation set (openai/gsm8k test split, deterministic sample)"

N="${GSM8K_N:-0}"
rq_lock "$RQ_OUTPUTS/eval-$MODE"

( set -x; uv run "$RQ_ROOT/s4-05-prep-gsm8k.py" --mode "$MODE" --n "$N" )

echo ---- result ----
( set -x; wc -l "$RQ_OUTPUTS/eval-$MODE/prompts-gsm8k.jsonl"; cat "$RQ_OUTPUTS/eval-$MODE/evalset-$MODE.json" )
