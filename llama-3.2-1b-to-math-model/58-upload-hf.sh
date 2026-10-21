#!/usr/bin/env bash
# 58-upload-hf.sh [mini|full] [targets] (STRAT env, 기본 single) — Stage 7a. HF Hub 업로드 (tayaee/*).
# targets: all|fft|qlora|fft-gptq|...|qlora-gguf (콤마 구분 가능, 기본 all).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
STRAT="${STRAT:-single}"
case "$STRAT" in single|ddp|fsdp) ;; *) echo "STRAT must be single|ddp|fsdp" >&2; exit 1;; esac
m1b_repro_echo
(set -x; uv run "$M1B_ROOT/58-upload-hf.py" --mode "$MODE" --strat "$STRAT" --targets "${2:-all}")

echo ---- result ----
m1b_log "uploaded targets=${2:-all} (see URLs above)"
