#!/usr/bin/env bash
# run-s0-to-s7-smoke-awq-w5.sh — smoke × awq-w5 단일 타깃 원샷(빌드→PPL→eval).
# 범위: Stage 0~7 (setup → acceptance → calib → compat → dequant → quant → ppl → eval)
# 타깃: awq-w5 — AWQ smoothing + GPTQ W5 (W5A16-g128) (s5-25-quant-awq-w5.sh)
# run-smoke.sh의 TARGETS/STEPS 조합을 한 타깃(Stage 0~7)으로 좁힌 버전:
#   dequant(BF16 복원) → quant(awq-w5) → PPL(base 대비 Δ) → eval → score
# 사용: ./run-s0-to-s7-smoke-awq-w5.sh
#   STEPS=quant,ppl ./run-s0-to-s7-smoke-awq-w5.sh          # 일부 스텝만
#   FORCE=1 ./run-s0-to-s7-smoke-awq-w5.sh                   # freshness 무시(강제 재계산)
#   SKIP_SETUP=1 ./run-s0-to-s7-smoke-awq-w5.sh             # s0-10-setup 생략
#   QUANT_CPU_ONLY=1 ./run-s0-to-s7-smoke-awq-w5.sh          # 양자화만 GPU 숨김(PPL/eval은 GPU)
#   STEPS=prep,verify ./run-s0-to-s7-smoke-awq-w5.sh        # 모델카드(43) + 업로드 전 감사(46)까지
# 게이트: gate-quant(패킹 계약·global_scale) → gate-ppl(유한 PPL) → gate-eval(응답 생성)
# 상세: build-ppl-eval.inc / PROCEDURE.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
export BPE_MODE=smoke BPE_TARGET=awq-w5
source "$ROOT/config/common.env" "$BPE_MODE"
source "$ROOT/build-ppl-eval.inc"
rq_banner "run-s0-to-s7-smoke-awq-w5" "Stage 0-7 pipeline for a single target (setup -> acceptance -> calib -> compat -> dequant -> quant -> ppl -> eval)"
rq_bpe_main
