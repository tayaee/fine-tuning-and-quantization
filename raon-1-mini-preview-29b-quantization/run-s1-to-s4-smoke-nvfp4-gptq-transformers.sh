#!/usr/bin/env bash
# run-s1-to-s4-smoke-nvfp4-gptq-transformers.sh — smoke × nvfp4-gptq-transformers 단일 타깃 원샷(빌드→PPL→eval).
# GPTQ NVFP4. STEPS 필터 없이 전 구간 실행한다 (부분 실행은 각 스크립트 직접 호출).
#   FORCE=1 ./run-s1-to-s4-smoke-nvfp4-gptq-transformers.sh             # freshness 무시(강제 재계산)
#   QUANT_CPU_ONLY=1 ./run-s1-to-s4-smoke-nvfp4-gptq-transformers.sh    # 양자화만 GPU 숨김
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
MODE="smoke"; TARGET="nvfp4-gptq-transformers"
QSCRIPT="s3-01-quant-nvfp4-gptq-transformers.sh"; QFORMAT="nvfp4-pack-quantized"
source "$ROOT/config/common.env" "$MODE"
rq_banner "run-s1-to-s4-smoke-nvfp4-gptq-transformers" "s1-s4 pipeline for a single target (setup -> acceptance -> calib -> compat -> dequant -> quant -> ppl -> eval)"
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
./s4-00-measure-ppl__transformers-only.sh "$MODE" "base,$TARGET"
rq_gate_ppl "$RQ_OUTPUTS/ppl-$MODE/$TARGET.json" "$TARGET"
rq_stage_record 4 measure pass "$RQ_OUTPUTS/ppl-$MODE/summary.json"
./s4-05-prep-gsm8k.sh "$MODE"
./s4-25-prep-kmmluko.sh "$MODE"
./s4-35-check-format__transformers_vllm_both.sh "$MODE" --prepare
./s4-10-answer-benchmarks-base__transformers-only.sh "$MODE"
./s4-15-answer-benchmarks-quantized__transformers-only.sh "$MODE" "$TARGET"
./s4-20-aggregate-gsm8k__transformers_vllm_both.sh "$MODE"
./s4-30-score-kmmluko__transformers_vllm_both.sh "$MODE"
./s4-35-check-format__transformers_vllm_both.sh "$MODE"
rq_gate_eval "$RQ_OUTPUTS/eval-$MODE/$TARGET.jsonl" "$TARGET" "$MODE"
rq_gate_kmmluko "$MODE" "$TARGET"
rq_gate_format "$MODE" "$TARGET"
rq_stage_record 4 eval pass "$RQ_OUTPUTS/eval-$MODE/score.json"
# 서빙 벤치용 로컬 서버 (in-process 측정 이후 — 모델 이중 적재 방지). 기동 실패해도 계속 (s4-50이 skip 기록).
./s4-40-serve-local-weight-with-transformers.sh "$MODE" "$TARGET" || rq_log "WARN: local server start failed — s4-50 will skip"
SERVE_BENCH_TARGET="$TARGET" ./s4-50-servebench__vllm-only.sh "$MODE"
./s4-55-stop-serve-local-weight.sh
./s4-65-update-readme.sh "$MODE" "$TARGET"
rq_stage_record 4 eval pass "$RQ_OUTPUTS/prep-upload-$MODE.json"
./s4-70-verify-upload.sh "$MODE" "$TARGET"
rq_gate_release "$MODE"
echo "run-s1-to-s4-smoke-nvfp4-gptq-transformers done"
