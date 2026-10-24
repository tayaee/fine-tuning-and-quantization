#!/usr/bin/env bash
# 10-16,19,28-40-build-smoke-nvfp4-awq.sh — smoke × nvfp4-awq 단일 타깃 원샷(빌드→PPL→eval).
# 스텝 번호: 10-setup · 13-prep-calibration · 16-compat-check · 19-dequant-mxfp8 · 28-quant-nvfp4-awq · 34-measure-ppl · 37-eval · 40-score
# 타깃: nvfp4-awq — AWQ smoothing + NVFP4 (W4A4) (28-quant-nvfp4-awq.sh)
# run-smoke.sh의 TARGETS/STEPS 조합을 한 타깃으로 좁힌 버전:
#   dequant(BF16 복원) → quant(nvfp4-awq) → PPL(base 대비 Δ) → eval → score
# 사용: ./10-16,19,28-40-build-smoke-nvfp4-awq.sh
#   STEPS=quant,ppl ./10-16,19,28-40-build-smoke-nvfp4-awq.sh          # 일부 스텝만
#   FORCE=1 ./10-16,19,28-40-build-smoke-nvfp4-awq.sh                   # freshness 무시(강제 재계산)
#   SKIP_SETUP=1 ./10-16,19,28-40-build-smoke-nvfp4-awq.sh             # 10-setup 생략
#   QUANT_CPU_ONLY=1 ./10-16,19,28-40-build-smoke-nvfp4-awq.sh          # 양자화만 GPU 숨김(PPL/eval은 GPU)
#   STEPS=prep,verify ./10-16,19,28-40-build-smoke-nvfp4-awq.sh        # 모델카드(43) + 업로드 전 감사(46)까지
# 게이트: gate-quant(패킹 계약·global_scale) → gate-ppl(유한 PPL) → gate-eval(응답 생성)
# 상세: build-ppl-eval.inc / PROCEDURE.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
export BPE_MODE=smoke BPE_TARGET=nvfp4-awq
source "$ROOT/config/common.env" "$BPE_MODE"
source "$ROOT/build-ppl-eval.inc"
rq_bpe_main
