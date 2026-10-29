#!/usr/bin/env bash
# run-s4-metrics-via-serving.sh [smoke|full] [target] — 서빙 스택으로 품질 지표를 재고 카드를 갱신한다.
#
# transformers 로 적재 불가한 타깃(nvfp4-awq-vllm = modelopt_fp4)의 GSM8K/KMMLU/형식 지표를
# 실제 서빙 스택(vLLM 포크)에서 측정하고, 그 값으로 모델 카드를 다시 만든다.
# (base 비교값은 이미 eval-<mode>/score.json·checks.json 에 있다 — transformers in-process 측정.)
#
# 순서: s4-40(서빙) → s4-45(서버로 벤치 응답) → s4-50(서빙 벤치) → s4-55(정리)
#       → s4-20/30/35(집계) → 게이트 → s4-65(카드) → s4-70(감사) → (s5-01 로 재게시)
#
# 전제: vLLM GB10 포크가 빌드돼 있어야 한다.
#   ./s7-00-clean-serving-workspace.sh <mode>
#   ./s7-02-verify-readme-with-nvfp4-awq-vllm-and-vllm-fork.sh (카드 블록이 Step1 빌드까지 포함) <mode>
# (없으면 s4-40 이 KRAFTON 사전빌드 포크로 폴백하는데, GB10(sm_121) 에서는 엔진 init 에서 죽는다.)
#
# 재측정: s4-40 은 servebench 기록이 있으면 서버 기동을 건너뛴다 → FORCE=1 로 강제한다.
#
# 사용:
#   ./run-s4-metrics-via-serving.sh full nvfp4-awq-vllm
#   ./s5-01-upload-to-hf.sh full nvfp4-awq-vllm    # 갱신된 카드로 재게시
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
MODE="${1:-${MODE:-full}}"
TARGET="${2:-nvfp4-awq-vllm}"
case "$MODE" in smoke|full) ;; *) echo "usage: $0 [smoke|full] [target]" >&2; exit 2 ;; esac
source "$ROOT/config/common.env" "$MODE"
# PPL 은 transformers 로 불가 → 카드에서 PPL 행 자체가 빠진다 (카드 생성기가 포맷별로 처리).
export ALLOW_MISSING_METRICS=1
export EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-gsm8k,kmmluko,format}"
rq_repro_echo
rq_banner "run-s4-metrics-via-serving" "measure quality metrics on the serving stack and refresh the model card ($TARGET, $MODE)"

# 1) 서빙 — GB10 소스 빌드 포크 우선 (s4-40 이 venv 존재 여부로 자동 선택)
./s4-40-serve-local-weight-with-vllm-fork.sh "$MODE" "$TARGET"
# 2) 서버로 벤치 응답 생성 (s4-07/s4-10 과 동일 스키마 → 집계 단계가 그대로 동작)
./s4-45-answer-via-server.sh "$MODE" "$TARGET"
# 3) 서빙 처리량 (카드의 성능 행)
SERVE_BENCH_TARGET="$TARGET" ./s4-50-servebench__vllm-only.sh "$MODE"
# 4) 서버 정리 (GPU 메모리 반납)
./s4-55-stop-serve-local-weight.sh
# 5) 집계 — base 비교값과 나란히 놓인다
./s4-20-aggregate-gsm8k__transformers_vllm_both.sh "$MODE"
./s4-30-score-kmmluko__transformers_vllm_both.sh "$MODE"
./s4-35-check-format__transformers_vllm_both.sh "$MODE"
# 6) 계약 대조 (허용폭 초과면 여기서 막힌다)
rq_gate_eval "$RQ_OUTPUTS/eval-$MODE/$TARGET.jsonl" "$TARGET" "$MODE"
rq_gate_kmmluko "$MODE" "$TARGET"
rq_gate_format "$MODE" "$TARGET"
rq_stage_record 4 eval pass "$RQ_OUTPUTS/eval-$MODE/score.json"
# 7) 카드 재생성 + 감사
./s4-65-update-readme.sh "$MODE" "$TARGET"
rq_stage_record 4 eval pass "$RQ_OUTPUTS/prep-upload-$MODE.json"
./s4-70-verify-upload.sh "$MODE" "$TARGET"
rq_gate_release "$MODE"
echo "run-s4-metrics-via-serving done — 다음: ./s5-01-upload-to-hf.sh $MODE $TARGET"
