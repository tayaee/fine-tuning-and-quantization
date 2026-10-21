#!/usr/bin/env bash
# 61-serve-gguf.sh [mini|full] [target] (STRAT env, 기본 single) — Stage 8b-gguf. llama-server OpenAI-호환 GGUF 서빙.
# vLLM은 GGUF 미지원(실험적 OOT 플러그인 필요, 52-eval-gguf.py도 llama.cpp 평가) → GGUF 정석은 llama.cpp llama-server.
# target: fft-gguf|qlora-gguf (기본 fft-gguf; gguf=fft-gguf 약칭). QTYPE env (기본 q8_0).
# 인자 없이 실행하면 복사-붙여넣기용 실행 명령 4줄을 보여준다.
# 예: ./61-serve-gguf.sh                        # 실행 명령 목록만 출력 (서빙 안 함)
#     ./61-serve-gguf.sh mini fft-gguf
#     QTYPE=q4_k_m ./61-serve-gguf.sh mini fft-gguf  # GGUF 변종 선택 (기본 q8_0)
#     ./61-serve-gguf.sh full qlora-gguf
#     # 터미널2: ./62-infer-examples.sh 3 (60과 동일, /v1/models+/v1/completions 호환)
set -euo pipefail
if [ $# -eq 0 ]; then
  source "$(dirname "$0")/config/common.env" mini
  STRAT="${STRAT:-single}"
  echo "[STRAT=$STRAT] copy & paste (terminal1, GGUF llama-server):"
  for spec in "full fft-gguf" "full qlora-gguf" "mini fft-gguf" "mini qlora-gguf"; do
    read -r m t <<< "$spec"
    case "$t" in
      fft-gguf)   SUF="synthetic-fft-$m-$STRAT-gguf/model-${QTYPE:-q8_0}.gguf" ;;
      qlora-gguf) SUF="synthetic-qlora-$m-$STRAT-merged-gguf/model-${QTYPE:-q8_0}.gguf" ;;
    esac
    printf '%-42s # %s\n' "./61-serve-gguf.sh $m $t" "$M1B_MODELS/$SUF"
  done
  exit 0
fi
source "$(dirname "$0")/config/common.env" "${1:-mini}"
STRAT="${STRAT:-single}"
case "$STRAT" in single|ddp|fsdp) ;; *) echo "STRAT must be single|ddp|fsdp" >&2; exit 1;; esac
TARGET="${2:-fft-gguf}"
case "$TARGET" in
  gguf|fft-gguf) TARGET="fft-gguf" ;;
  qlora-gguf) ;;
  *) echo "target: fft-gguf|qlora-gguf (gguf=fft-gguf)" >&2; exit 1 ;;
esac
QTYPE="$(printf '%s' "${QTYPE:-q8_0}" | tr '[:upper:]' '[:lower:]')"
case "$TARGET" in
  fft-gguf)   SUF="synthetic-fft-$MODE-$STRAT-gguf/model-${QTYPE}.gguf" ;;
  qlora-gguf) SUF="synthetic-qlora-$MODE-$STRAT-merged-gguf/model-${QTYPE}.gguf" ;;
esac
FILE="$M1B_MODELS/$SUF"
[ -f "$FILE" ] || { echo "missing GGUF file: $FILE (36/44 먼저 실행, QTYPE=$QTYPE 확인)" >&2; exit 1; }

# HF repo명 규칙은 58-upload-hf.py repo_name()과 동일 (single 동결, full 무표기) — 60/64와 동일.
MODEL_SLUG="${BASE_MODEL##*/}"
REPO="tayaee/${MODEL_SLUG}-math-${TARGET}"
[ "$STRAT" != single ] && REPO="$REPO-$STRAT"
[ "$MODE" != full ] && REPO="$REPO-$MODE"

LLAMACPP="${LLAMACPP:-$HOME/git/llama.cpp}"
[ -x "$LLAMACPP/build/bin/llama-server" ] || { echo "llama-server 없음: $LLAMACPP/build/bin/llama-server (llama.cpp에서 cmake --build로 빌드)" >&2; exit 1; }
PORT="${PORT:-8000}"
m1b_repro_echo
m1b_log "serving gguf file=$FILE alias=$REPO port=$PORT ctx=${LLAMACPP_CTX:-2048} ngl=${NGL:-999}"
set -x
exec "$LLAMACPP/build/bin/llama-server" -m "$FILE" \
  -a "$REPO" \
  --host 0.0.0.0 --port "$PORT" \
  -c "${LLAMACPP_CTX:-2048}" \
  --n-gpu-layers "${NGL:-999}"
