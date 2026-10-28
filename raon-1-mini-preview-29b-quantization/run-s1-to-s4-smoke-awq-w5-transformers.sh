#!/usr/bin/env bash
# run-s1-to-s4-smoke-awq-w5-transformers.sh — smoke × awq-w5-transformers 단일 타깃 원샷(빌드→PPL→eval).
# AWQ W5. STEPS 필터 없이 전 구간 실행한다 (부분 실행은 각 스크립트 직접 호출).
#   FORCE=1 ./run-s1-to-s4-smoke-awq-w5-transformers.sh             # freshness 무시(강제 재계산)
#   QUANT_CPU_ONLY=1 ./run-s1-to-s4-smoke-awq-w5-transformers.sh    # 양자화만 GPU 숨김
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
MODE="smoke"; TARGET="awq-w5-transformers"
QSCRIPT="s3-01-quant-awq-w5-transformers.sh"; QFORMAT="pack-quantized"
source "$ROOT/config/common.env" "$MODE"
rq_banner "run-s1-to-s4-smoke-awq-w5-transformers" "s1-s4 pipeline for a single target (setup -> acceptance -> calib -> compat -> dequant -> quant -> ppl -> eval)"
rq_repro_echo

./s1-01-setup.sh "$MODE"
./s1-04-acceptance.sh "$MODE"
rq_stage_record 1 acceptance pass "$RQ_OUTPUTS/acceptance-$MODE.json"
rq_gate_acceptance "$MODE"
./s1-07-prep-calibration.sh "$MODE"
./s1-10-compat-check.sh "$MODE"
./s2-01-dequant-mxfp8.sh "$MODE"
if [ "${QUANT_CPU_ONLY:-0}" = 1 ]; then QC="env CUDA_VISIBLE_DEVICES="; else QC="env"; fi
$QC ./$QSCRIPT "$MODE"
rq_gate_quant "$RQ_MODELS/raon-$TARGET-$MODE" "$QFORMAT"
rq_stage_target 3 quant pass "$TARGET" "$RQ_MODELS/raon-$TARGET-$MODE"
./s4-01-measure-ppl.sh "$MODE" "base,$TARGET"
rq_gate_ppl "$RQ_OUTPUTS/ppl-$MODE/$TARGET.json" "$TARGET"
rq_stage_record 4 measure pass "$RQ_OUTPUTS/ppl-$MODE/summary.json"
./s4-04-prep-gsm8k.sh "$MODE"
./s4-16-prep-kmmluko.sh "$MODE"
./s4-22-format.sh "$MODE" --prepare
./s4-07-eval-base.sh "$MODE"
./s4-10-eval-quantized.sh "$MODE" "$TARGET"
./s4-13-gsm8k.sh "$MODE"
./s4-19-kmmluko.sh "$MODE"
./s4-22-format.sh "$MODE"
rq_gate_eval "$RQ_OUTPUTS/eval-$MODE/$TARGET.jsonl" "$TARGET" "$MODE"
rq_gate_kmmluko "$MODE" "$TARGET"
rq_gate_format "$MODE" "$TARGET"
rq_stage_record 4 eval pass "$RQ_OUTPUTS/eval-$MODE/score.json"
./s4-31-prep-upload.sh "$MODE" "$TARGET"
rq_stage_record 4 eval pass "$RQ_OUTPUTS/prep-upload-$MODE.json"
./s4-34-verify-upload.sh "$MODE" "$TARGET"
rq_gate_release "$MODE"
echo "run-s1-to-s4-smoke-awq-w5-transformers done"
