#!/usr/bin/env bash
# s4-16-prep-kmmluko.sh [smoke|full] — 표준 한국어 KMMLU 평가 세트 준비 (Tier 2: 한국어 커버리지).
# HAERAE-HUB/KMMLU(표준 한국어 4지선다 MCQ)에서 과목별 표본을 결정론적으로 뽑아
# outputs/eval-<mode>/prompts-kmmluko.jsonl 을 만든다. 생성은 s4-07-answer-benchmarks-base.sh/s4-10-answer-benchmarks-quantized.sh, 채점은 s4-19-score-kmmluko.sh.
# 환경변수: KMMLU_N(표본 수, 기본 40) · KMMLU_SUBJECTS(과목 목록) · KMMLU_SEED
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_repro_echo
rq_banner "s4-16-prep-kmmluko" "Stage 4 - build the standard Korean KMMLU evaluation set (HAERAE-HUB/KMMLU, deterministic sample)"

N="${KMMLU_N:-40}"
SUBJECTS="${KMMLU_SUBJECTS:-Biology,Chemistry,Computer-Science,Economics,Korean-History,Law,Math,Management}"
rq_lock "$RQ_OUTPUTS/eval-$MODE"

( set -x; uv run "$RQ_ROOT/s4-16-prep-kmmluko.py" --mode "$MODE" --n "$N" --subjects "$SUBJECTS" )

echo ---- result ----
( set -x; wc -l "$RQ_OUTPUTS/eval-$MODE/prompts-kmmluko.jsonl" )
