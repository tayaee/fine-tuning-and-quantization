#!/usr/bin/env bash
# run-s1-to-s4-smoke-nvfp4-awq-vllm.sh — smoke × nvfp4-awq-vllm 단일 타깃 원샷(빌드→PPL→eval).
# AWQ Lite + NVFP4 (ModelOpt). STEPS 필터 없이 전 구간 실행한다 (부분 실행은 각 스크립트 직접 호출).
#   FORCE=1 ./run-s1-to-s4-smoke-nvfp4-awq-vllm.sh             # freshness 무시(강제 재계산)
#   QUANT_CPU_ONLY=1 ./run-s1-to-s4-smoke-nvfp4-awq-vllm.sh    # 양자화만 GPU 숨김
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
MODE="smoke"; TARGET="nvfp4-awq-vllm"
QSCRIPT="s3-01-quant-nvfp4-awq-vllm.sh"; QFORMAT="mopt-nvfp4"
source "$ROOT/config/common.env" "$MODE"
# vLLM 전용 포맷: transformers 지표(PPL/eval) 불가 → 카드·감사 지표 면제, 품질 게이트는 s7 서빙
export ALLOW_MISSING_METRICS=1
rq_banner "run-s1-to-s4-smoke-nvfp4-awq-vllm" "s1-s4 pipeline for a single target (setup -> acceptance -> calib -> compat -> dequant -> quant -> ppl -> eval)"
rq_repro_echo

./s1-01-setup-environment.sh "$MODE"
./s1-04-setup-acceptance-criteria.sh "$MODE"
rq_stage_record 1 acceptance pass "$RQ_OUTPUTS/acceptance-$MODE.json"
rq_gate_acceptance "$MODE"
./s1-07-prep-calibration.sh "$MODE"
./s1-10-preflight-check.sh "$MODE"
./s2-01-dequant-mxfp8-into-bf16.sh "$MODE"
if [ "${QUANT_CPU_ONLY:-0}" = 1 ]; then QC="env CUDA_VISIBLE_DEVICES="; else QC="env"; fi
$QC ./$QSCRIPT "$MODE"
rq_gate_quant "$RQ_MODELS/raon-$TARGET-$MODE" "$QFORMAT"
rq_stage_target 3 quant pass "$TARGET" "$RQ_MODELS/raon-$TARGET-$MODE"
./s4-01-measure-ppl.sh "$MODE" "base,$TARGET"
rq_gate_ppl "$RQ_OUTPUTS/ppl-$MODE/$TARGET.json" "$TARGET"
rq_stage_record 4 measure pass "$RQ_OUTPUTS/ppl-$MODE/summary.json"
./s4-04-prep-gsm8k.sh "$MODE"
./s4-16-prep-kmmluko.sh "$MODE"
./s4-22-check-format.sh "$MODE" --prepare
./s4-07-answer-benchmarks-base.sh "$MODE"
./s4-10-answer-benchmarks-quantized.sh "$MODE" "$TARGET"
./s4-13-aggregate-gsm8k.sh "$MODE"
./s4-19-score-kmmluko.sh "$MODE"
./s4-22-check-format.sh "$MODE"
rq_gate_eval "$RQ_OUTPUTS/eval-$MODE/$TARGET.jsonl" "$TARGET" "$MODE"
rq_gate_kmmluko "$MODE" "$TARGET"
rq_gate_format "$MODE" "$TARGET"
rq_stage_record 4 eval pass "$RQ_OUTPUTS/eval-$MODE/score.json"
./s4-31-update-readme.sh "$MODE" "$TARGET"
rq_stage_record 4 eval pass "$RQ_OUTPUTS/prep-upload-$MODE.json"
./s4-34-verify-upload.sh "$MODE" "$TARGET"
rq_gate_release "$MODE"
echo "run-s1-to-s4-smoke-nvfp4-awq-vllm done"
