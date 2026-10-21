#!/usr/bin/env bash
# 62-infer-examples.sh [n|model] [n] — Stage 8c wrapper. 서빙 중 모델에 추론 예제 요청.
# $1이 숫자면 개수(서빙 중 첫 모델 자동 사용), 모델명이면 해당 모델 사용.
# 인자 없이 실행하면 복사-붙여넣기용 단축 명령을 보여준다.
# 예: ./62-infer-examples.sh                              # 단축 명령만 출력
#     ./62-infer-examples.sh 3                             # 권장: $BASE_URL/models 첫 모델 자동 사용
#     ./62-infer-examples.sh tayaee/Llama-3.2-1B-math-fft-mini 3
#     ./62-infer-examples.sh tayaee/Llama-3.2-1B-math-fft 3  # full은 -mini 없음
#     DRY_RUN=1 ./62-infer-examples.sh 1                   # 프롬프트만 출력
#     BASE_URL=http://localhost:8000/v1 ./62-infer-examples.sh 3
# (추론 텍스트 자체가 결과이므로 별도 ls 없음.)
set -euo pipefail
if [ $# -eq 0 ]; then
  echo "copy & paste (terminal2, 60 기동 후):"
  echo "./62-infer-examples.sh 3"
  echo "DRY_RUN=1 ./62-infer-examples.sh 1  # 프롬프트만"
  exit 0
fi
source "$(dirname "$0")/config/common.env" "${1:+mini}"
if [[ "${1}" =~ ^[0-9]+$ ]]; then
  MODEL="-"; N="${1}"
else
  MODEL="${1:?usage: $0 [n|model] [n]}"; N="${2:-3}"
fi
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
ARGS=()
[ "${DRY_RUN:-0}" = 1 ] && ARGS+=(--dry-run)
m1b_repro_echo
m1b_log "model=$MODEL base_url=$BASE_URL n=$N"
echo + uv run "$M1B_ROOT/62-infer-examples.py" --model "$MODEL" --base-url "$BASE_URL" --n "$N" "${ARGS[@]}"
uv run "$M1B_ROOT/62-infer-examples.py" --model "$MODEL" --base-url "$BASE_URL" --n "$N" "${ARGS[@]}"
