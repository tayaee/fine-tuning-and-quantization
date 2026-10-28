#!/usr/bin/env bash
# run-s1-to-s4-full-nvfp4-awq-transformers.sh — full × nvfp4-awq-transformers 단일 타깃 원샷(빌드→PPL→eval).
# 범위: Stage s1~s4 (setup → acceptance → calib → compat → dequant → quant → ppl → eval)
# 타깃: nvfp4-awq-transformers — AWQ smoothing + NVFP4 (W4A4) (s3-07-quant-nvfp4-awq-transformers.sh)
# 단일 (mode × target) 빌드 진입점:
#   dequant(BF16 복원) → quant(nvfp4-awq-transformers) → PPL(base 대비 Δ) → eval → score
# 사용: ./run-s1-to-s4-full-nvfp4-awq-transformers.sh
#   STEPS=quant,ppl ./run-s1-to-s4-full-nvfp4-awq-transformers.sh          # 일부 스텝만
#   FORCE=1 ./run-s1-to-s4-full-nvfp4-awq-transformers.sh                   # freshness 무시(강제 재계산)
#   SKIP_SETUP=1 ./run-s1-to-s4-full-nvfp4-awq-transformers.sh             # s1-01-setup 생략
#   QUANT_CPU_ONLY=1 ./run-s1-to-s4-full-nvfp4-awq-transformers.sh          # 양자화만 GPU 숨김(PPL/eval은 GPU)
#   STEPS=prep,verify ./run-s1-to-s4-full-nvfp4-awq-transformers.sh        # 모델카드(s4-31) + 업로드 전 감사(s4-34)까지
# 게이트: gate-quant(패킹 계약·global_scale) → gate-ppl(유한 PPL) → gate-eval(응답 생성)
# 상세: build-ppl-eval.inc / PROCEDURE.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
export BPE_MODE=full BPE_TARGET=nvfp4-awq-transformers
source "$ROOT/config/common.env" "$BPE_MODE"
source "$ROOT/build-ppl-eval.inc"
rq_banner "run-s1-to-s4-full-nvfp4-awq-transformers" "s1-s4 pipeline for a single target (setup -> acceptance -> calib -> compat -> dequant -> quant -> ppl -> eval)"
rq_bpe_main
