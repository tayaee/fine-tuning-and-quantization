#!/usr/bin/env bash
# s4-70-update-readme.sh [smoke|full] [targets] — 업로드 전 모델 디렉터리 검증 + README.md 생성.
# targets 필수 (미지정 시 사용법 출력 후 종료, TARGETS=all로 전체).
# s5-00-upload-to-hf.py(업로드 전담)가 요구하는 전제 조건을 여기서 만든다.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
TARGETS="$(rq_require_targets "./s4-70-update-readme.sh $MODE" "${2:-}" awq-w3-transformers awq-w5-transformers nvfp4-awq-vllm nvfp4-awq-transformers nvfp4-gptq-transformers)" || exit 2
: "${ALLOW_MISSING_METRICS:=0}"
ALLOW_MISSING=""
[ "$ALLOW_MISSING_METRICS" = 1 ] && ALLOW_MISSING="--allow-missing-metrics"
rq_repro_echo
rq_banner "s4-70-update-readme" "Stage 4 - verify the model directory and generate the HuggingFace model card (README.md)"
rq_lock "$RQ_OUTPUTS"
rq_log "prep-upload targets=$TARGETS allow_missing=$ALLOW_MISSING_METRICS"
PREP_LOG="$(mktemp)"
(set -x; uv run "$RQ_ROOT/s4-70-update-readme.py" --mode "$MODE" --targets "$TARGETS" $ALLOW_MISSING 2>&1 | tee "$PREP_LOG") && RC=0 || RC=$?
if [ "$RC" != 0 ]; then
  echo "==================== PREP FAILED (rc=$RC) ===================="
  echo "모델 카드 미생성 — 아래 FAIL 해소 후 재실행 (vLLM 전용 타깃은 ALLOW_MISSING_METRICS=1)"
  grep -E "^FAIL " "$PREP_LOG" | head -n 5
  echo "==============================================================="
  rm -f "$PREP_LOG"
  exit "$RC"
fi
rm -f "$PREP_LOG"

echo ---- result ----
# 요청한 타깃의 카드만 보여준다 (모드 전체 glob 금지 — 건드리지 않은 타깃이 섞임)
_targets_disp="$TARGETS"
if [ "$_targets_disp" = all ]; then
  _targets_disp="awq-w3-transformers awq-w5-transformers nvfp4-awq-vllm nvfp4-awq-transformers nvfp4-gptq-transformers"
fi
for _t in $(echo "$_targets_disp" | tr ',' ' '); do
  (set -x; ls -l "$RQ_MODELS/raon-$_t-$MODE/README.md")
done
(set -x; ls -l "$RQ_MODELS/README-$MODE.md")
(set -x; cat "$RQ_OUTPUTS/prep-upload-$MODE.json")