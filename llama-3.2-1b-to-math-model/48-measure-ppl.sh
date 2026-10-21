#!/usr/bin/env bash
# 48-measure-ppl.sh [mini|full] [targets] (STRAT env, 기본 single) — Stage 6 wrapper. PPL 측정 + 판정.
# targets: all|base,fft,qlora,fft-gptq,... (콤마 구분, 기본 all).
# 기준: 부모 FP 대비 Δ<0.3 Accept / 0.3~1.0 Conditional(GSM8K 2차) / ≥1.0 Discard.
# GGUF 10종은 llama-perplexity로 측정 (미빌드 시 해당 타깃만 SKIP).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
STRAT="${STRAT:-single}"
case "$STRAT" in single|ddp|fsdp) ;; *) echo "STRAT must be single|ddp|fsdp" >&2; exit 1;; esac
N="${PPL_N:-32}"
TARGETS="${2:-all}"
m1b_repro_echo
SUMMARY="$M1B_OUTPUTS/ppl-$MODE-$STRAT/summary.json"
m1b_lock "${SUMMARY%/*}"
if [ "${FORCE:-0}" != 1 ] && [ -f "$SUMMARY" ]; then
  # 이 strat 산출물 중 SUMMARY보다新しい 것이 하나도 없으면 스킵.
  # $M1B_MODELS 전체를 보면 타 strat 빌드 때마다 전부 재측정된다 (실측:
  # ddp 빌드 후 single 60이 재실행됨). *-MODE-STRAT* 한정이 정확하다.
  if [ -z "$(find "$M1B_MODELS" -maxdepth 1 -name "*-$MODE-$STRAT*" -newer "$SUMMARY" 2>/dev/null | head -1)" ]; then
    m1b_log "skip: fresh $SUMMARY (FORCE=1 to rebuild)"
    echo ---- result ----
    (set -x; ls -l "$M1B_OUTPUTS/ppl-$MODE-$STRAT/")
    (set -x; cat "$SUMMARY")
    exit 0
  fi
fi
m1b_log "ppl_n=$N targets=$TARGETS"
(set -x; uv run "$M1B_ROOT/48-measure-ppl.py" --mode "$MODE" --strat "$STRAT" --n "$N" --targets "$TARGETS")

echo ---- result ----
(set -x; ls -l "$M1B_OUTPUTS/ppl-$MODE-$STRAT/")
(set -x; cat "$M1B_OUTPUTS/ppl-$MODE-$STRAT/summary.json")
