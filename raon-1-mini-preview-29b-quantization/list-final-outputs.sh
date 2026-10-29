#!/usr/bin/env bash
# list-final-outputs.sh [README.md] — 최종 산출물 10개 모델 디렉토리 목록.
#   인자 없음 : 디렉토리 안의 파일 나열 (존재하는 것만)
#   인자 있음 : 해당 파일만 나열 (예: README.md, 존재하는 것만)
# 10개 헤딩은 항상 전체 경로로 표시한다. `ls`를 쓰지 않는다.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
RQ_SHARED="${RQ_SHARED:-/rosenas/data/AIML/$(basename "$ROOT")-shared}"
RQ_MODELS="$RQ_SHARED/models"

FILTER="${1:-}"
i=0
for t in awq-w3-transformers awq-w5-transformers nvfp4-awq-vllm nvfp4-awq-transformers nvfp4-gptq-transformers ; do
  for mode in smoke full; do
    i=$((i + 1))
    d="$RQ_MODELS/raon-$t-$mode"
    echo "### [$i/10] $d"
    if [ -z "$FILTER" ]; then
      ls -l $d 2>/dev/null || true
    else
      ls -l $d/$FILTER 2>/dev/null || true
    fi
    echo ""
  done
done
