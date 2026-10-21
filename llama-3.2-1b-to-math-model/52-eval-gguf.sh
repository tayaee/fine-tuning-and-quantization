#!/usr/bin/env bash
# 52-eval-gguf.sh [mini|full] [target...] (STRAT env, 기본 single) — Stage 7b-gguf wrapper. llama-completion greedy GSM8K 추론.
# target 기본 all = GGUF 10종 (fft/qlora × {q8_0,q6_k,q5_k_m,q4_k_m,q3_k_m}).
# 전제: llama-completion 빌드 ($LLAMACPP/build/bin/llama-completion). 없으면 전 타깃 SKIP (rc=0).
# (llama-cli는 신버전(b9544+)에서 대화형으로 진입하므로 사용 금지.)
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
STRAT="${STRAT:-single}"
case "$STRAT" in single|ddp|fsdp) ;; *) echo "STRAT must be single|ddp|fsdp" >&2; exit 1;; esac
TARGETS="${2:-all}"
[ "$TARGETS" = all ] && TARGETS="fft-gguf-q8_0 fft-gguf-q6_k fft-gguf-q5_k_m fft-gguf-q4_k_m fft-gguf-q3_k_m qlora-gguf-q8_0 qlora-gguf-q6_k qlora-gguf-q5_k_m qlora-gguf-q4_k_m qlora-gguf-q3_k_m"
N="$EVAL_N"  # mini 10 / full 0=전체 (common.env)
LLAMACPP="${LLAMACPP:-$HOME/git/llama.cpp}"
m1b_repro_echo
if [ ! -x "$LLAMACPP/build/bin/llama-completion" ]; then
  m1b_log "WARN: no llama-completion, GGUF eval SKIP (build llama.cpp llama-completion target)"
  (set -x; ls -l "$M1B_OUTPUTS/eval-$MODE-$STRAT/" 2>/dev/null || true)
  exit 0
fi
TEST="$M1B_DATASETS/gsm8k-test.jsonl"
m1b_require "$TEST"
# <target> → GGUF 파일 (52-eval-gguf.py TARGETS 규칙 미러)
gguf_file() {
  case "$1" in
    fft-gguf-*) echo "$M1B_MODELS/synthetic-fft-$MODE-$STRAT-gguf/model-${1#fft-gguf-}.gguf" ;;
    qlora-gguf-*) echo "$M1B_MODELS/synthetic-qlora-$MODE-$STRAT-merged-gguf/model-${1#qlora-gguf-}.gguf" ;;
  esac
}
m1b_log "targets=$TARGETS n=$N"
m1b_lock "$M1B_OUTPUTS/eval-$MODE-$STRAT"
for t in $TARGETS; do
  out="$M1B_OUTPUTS/eval-$MODE-$STRAT/$t.jsonl"
  g="$(gguf_file "$t")"
  if [ -n "$g" ] && [ -f "$g" ] && [ -f "$out" ] && m1b_fresh "$out" "$g" "$TEST"; then
    m1b_log "skip: fresh $out (FORCE=1 to rebuild)"; continue
  fi
  (set -x; uv run "$M1B_ROOT/52-eval-gguf.py" --mode "$MODE" --strat "$STRAT" --target "$t" --n "$N")
done

echo ---- result ----
(set -x; ls -l "$M1B_OUTPUTS/eval-$MODE-$STRAT/")
(set -x; wc -l "$M1B_OUTPUTS/eval-$MODE-$STRAT/"*.jsonl)
