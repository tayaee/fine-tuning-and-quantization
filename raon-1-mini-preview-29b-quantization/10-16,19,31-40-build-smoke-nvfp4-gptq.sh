#!/usr/bin/env bash
# 10-16,19,31-40-build-smoke-nvfp4-gptq.sh — smoke × nvfp4-gptq 단일 타깃 원샷(빌드→PPL→eval).
# 스텝 번호: 10-setup · 13-prep-calibration · 16-compat-check · 19-dequant-mxfp8 · 31-quant-nvfp4-gptq · 34-measure-ppl · 37-eval · 40-score
# 타깃: nvfp4-gptq — GPTQModifier NVFP4 (W4A4) (31-quant-nvfp4-gptq.sh)
# run-smoke.sh의 TARGETS/STEPS 조합을 한 타깃으로 좁힌 버전:
#   dequant(BF16 복원) → quant(nvfp4-gptq) → PPL(base 대비 Δ) → eval → score
# 사용: ./10-16,19,31-40-build-smoke-nvfp4-gptq.sh
#   STEPS=quant,ppl ./10-16,19,31-40-build-smoke-nvfp4-gptq.sh          # 일부 스텝만
#   FORCE=1 ./10-16,19,31-40-build-smoke-nvfp4-gptq.sh                   # freshness 무시(강제 재계산)
#   SKIP_SETUP=1 ./10-16,19,31-40-build-smoke-nvfp4-gptq.sh             # 10-setup 생략
#   QUANT_CPU_ONLY=1 ./10-16,19,31-40-build-smoke-nvfp4-gptq.sh          # 양자화만 GPU 숨김(PPL/eval은 GPU)
#   STEPS=prep,verify ./10-16,19,31-40-build-smoke-nvfp4-gptq.sh        # 모델카드(43) + 업로드 전 감사(46)까지
# 게이트: gate-quant(패킹 계약·global_scale) → gate-ppl(유한 PPL) → gate-eval(응답 생성)
# 상세: build-ppl-eval.inc / PROCEDURE.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
export BPE_MODE=smoke BPE_TARGET=nvfp4-gptq
source "$ROOT/config/common.env" "$BPE_MODE"
source "$ROOT/build-ppl-eval.inc"
rq_banner "build-smoke-nvfp4-gptq" "9-stage pipeline for a single target (setup -> acceptance -> calib -> compat -> dequant -> quant -> ppl -> eval)"
rq_bpe_main
