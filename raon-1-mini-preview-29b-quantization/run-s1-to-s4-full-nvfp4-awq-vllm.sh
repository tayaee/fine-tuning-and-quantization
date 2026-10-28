#!/usr/bin/env bash
# run-s1-to-s4-full-nvfp4-awq-vllm.sh — full × nvfp4-awq-vllm 단일 타깃 원샷(빌드→PPL→eval).
# 범위: Stage s1~s4 (setup → acceptance → calib → compat → dequant → quant → ppl → eval)
# 타깃: nvfp4-awq-vllm — AWQ Lite + NVFP4, ModelOpt (s3-06, modelopt_fp4 표지 → vLLM 포크 적재 대상)
# 단일 (mode × target) 빌드 진입점:
#   dequant(BF16 복원) → quant(nvfp4-awq-vllm) → PPL(base 대비 Δ) → eval → score
# 사용: ./run-s1-to-s4-full-nvfp4-awq-vllm.sh
#   STEPS=quant,ppl ./run-s1-to-s4-full-nvfp4-awq-vllm.sh    # 일부 스텝만
#   FORCE=1 ./run-s1-to-s4-full-nvfp4-awq-vllm.sh             # freshness 무시(강제 재계산)
#   SKIP_SETUP=1 ./run-s1-to-s4-full-nvfp4-awq-vllm.sh       # s1-01-setup 생략
#   QUANT_CPU_ONLY=1 ./run-s1-to-s4-full-nvfp4-awq-vllm.sh    # 양자화만 GPU 숨김(PPL/eval은 GPU)
#   STEPS=prep,verify ./run-s1-to-s4-full-nvfp4-awq-vllm.sh  # 모델카드(s4-31) + 업로드 전 감사(s4-34)까지
# 게이트: gate-quant(표지·expert 계약) → gate-ppl(유한 PPL) → gate-eval(응답 생성)
# 상세: build-ppl-eval.inc / PROCEDURE.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
export BPE_MODE=full BPE_TARGET=nvfp4-awq-vllm
source "$ROOT/config/common.env" "$BPE_MODE"
source "$ROOT/build-ppl-eval.inc"
rq_banner "run-s1-to-s4-full-nvfp4-awq-vllm" "s1-s4 pipeline for a single target (setup -> acceptance -> calib -> compat -> dequant -> quant -> ppl -> eval)"
rq_bpe_main
