#!/bin/bash
# s9-19-vllm-serve-all-models.sh [mode] [n] — 전 타깃 순회 서빙+추론 (단일 터미널).
# 대상은 TARGETS 환경변수로 필수 지정 (미지정 시 사용법 출력 후 종료, TARGETS=all로 전체).
# 모델마다: serve 기동 → 로딩 대기 → s9-16-inference-test → logs/16.<slug>.log → kill.
# 29B 로드 감안: LOAD_TIMEOUT 기본 1200.
# 사용: TARGETS=all ./s9-19-vllm-serve-all-models.sh smoke 3
#       TARGETS=nvfp4-awq ./s9-19-vllm-serve-all-models.sh full 3
set -uo pipefail
cd "$(dirname "$0")"
MODE_ARG="${1:-${MODE:-smoke}}"
N="${2:-3}"
source "$(dirname "$0")/config/common.env" "$MODE_ARG"
RQ_WANT="$(rq_require_targets "./s9-19-vllm-serve-all-models.sh $MODE 3" "" base awq-w3 awq-w5 nvfp4-awq nvfp4-gptq)" || exit 2
rq_banner "s9-19-vllm-serve-all-models" "Stage 9 - serve every target in sequence and smoke-test inference"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://localhost:${PORT}/v1}"
export PORT BASE_URL
# TARGETS 환경변수(필수)를 순회 목록으로 편다. all이면 5종 전부.
TARGETS=()
for _t in $(echo "$RQ_WANT" | tr ',' ' '); do
  if [ "$_t" = all ]; then TARGETS=(base awq-w3 awq-w5 nvfp4-awq nvfp4-gptq); break; fi
  case " base awq-w3 awq-w5 nvfp4-awq nvfp4-gptq " in
    *" $_t "*) TARGETS+=("$_t") ;;
    *) echo "Unknown TARGETS: $_t (allowed: base awq-w3 awq-w5 nvfp4-awq nvfp4-gptq | all)" >&2; exit 2 ;;
  esac
done
LOAD_TIMEOUT="${LOAD_TIMEOUT:-1200}"
INFER_TIMEOUT="${INFER_TIMEOUT:-900}"
mkdir -p logs

slugify() { sed -e 's;/;__;g' | tr 'A-Z' 'a-z'; }
expected_slug() { local s="tayaee/${BASE_MODEL##*/}-$2"; [ "$1" = smoke ] && s="$s-smoke"; printf '%s' "$s" | slugify; }
pids_on_port() {
  ss -ltnp 2>/dev/null | grep ":${PORT} " | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u
}
CHILD=""
kill_server() {
  if [ -n "${CHILD:-}" ] && kill -0 "$CHILD" 2>/dev/null; then kill "$CHILD" 2>/dev/null; fi
  local i pids
  for i in $(seq 1 15); do
    pids="$(pids_on_port)"
    [ -z "$pids" ] && break
    for p in $pids; do kill "$p" 2>/dev/null; done
    sleep 1
  done
  pids="$(pids_on_port)"
  if [ -n "$pids" ]; then
    echo "[warn] port ${PORT} 점유 잔존 → KILL: $pids"
    for p in $pids; do kill -9 "$p" 2>/dev/null; done
    sleep 2
  fi
  if [ -n "${CHILD:-}" ]; then wait "$CHILD" 2>/dev/null; fi
  CHILD=""
}
wait_for_port_free() {
  local i=0
  while [ "$i" -lt 60 ]; do
    ss -ltn 2>/dev/null | grep -q ":${PORT} " || { echo "Port ${PORT} is free."; return 0; }
    sleep 1; i=$((i + 1))
  done
  return 1
}
trap 'kill_server; exit 130' INT TERM

run_one() {
  local mode="$1" target="$2"
  local expected=""; expected="$(expected_slug "$mode" "$target")"
  local logf="logs/16.${expected}.log"
  local servelog="logs/13.${expected}.serve.log"
  if [ -f "$logf" ]; then echo "[skip] $mode $target — $logf 이미 있음"; return 0; fi
  rm -f "$logf.tmp"
  echo "=========================================="
  echo "Model: mode=$mode target=$target (expect $expected, n=$N)"
  echo "=========================================="
  # Preflight (2026-10-26 진단): krafton-ai/vllm@raon-1-mini-preview 포크의
  # Raon1MiniForCausalLMConfig 후크는 ModelOpt NVFP4가 아닌 체크포인트의
  # quantization_config를 삭제한다 → llmcompressor 산출물(compressed-tensors)은
  # get_quant_config에서 TypeError로 사망한다. 미리 알리고 서빙 로그를 남긴다.
  case "$target" in
    awq-w3|awq-w5|nvfp4-awq|nvfp4-gptq)
      _suf="raon-$target-$mode"
      if [ -f "$RQ_MODELS/$_suf/config.json" ] \
        && grep -q '"quant_method": "compressed-tensors"' "$RQ_MODELS/$_suf/config.json" 2>/dev/null; then
        echo "[warn] $mode $target — compressed-tensors PTQ 체크포인트는 핀된 포크 서빙과"
        echo "       비호환 가능 (MXFP8-release/ModelOpt 프로파일만 지원, 2026-10-26 확인)."
        echo "       서빙 시도 후 실패 원인은 $servelog 에 보존된다."
      fi
      unset _suf
      ;;
  esac
  rm -f "$servelog"
  eval "./s9-13-vllm-serve-one.sh $mode $target" >"$servelog" 2>&1 &
  CHILD=$!
  local elapsed=0 loaded=0 slug=""
  while [ "$elapsed" -lt "$LOAD_TIMEOUT" ]; do
    if ! kill -0 "$CHILD" 2>/dev/null; then
      echo "[SERVE-FAIL] $mode $target — 서버 프로세스가 로딩 중 사망 (상세: $servelog)"
      tail -n 15 "$servelog" 2>/dev/null | grep -E "Error|error" | head -n 5 || true
      CHILD=""; kill_server; wait_for_port_free || true
      { echo "[SERVE-FAIL] $(date --iso-8601=s) mode=$mode target=$target serve_log=$servelog"; } > "$logf.tmp" && mv "$logf.tmp" "$logf"
      return 1
    fi
    slug="$(./model-slug.sh 2>/dev/null | tr -d '[:space:]')"
    if [ -n "$slug" ] && [ "$slug" != "null" ]; then loaded=1; echo "Model loaded: $slug (${elapsed}s)."; break; fi
    sleep 5; elapsed=$((elapsed + 5))
  done
  if [ "$loaded" -eq 0 ]; then
    echo "[SERVE-FAIL] $mode $target — 로딩 타임아웃 (${LOAD_TIMEOUT}s, 상세: $servelog)"
    kill_server; wait_for_port_free || true
    { echo "[SERVE-FAIL] $(date --iso-8601=s) mode=$mode target=$target serve_log=$servelog"; } > "$logf.tmp" && mv "$logf.tmp" "$logf"
    return 1
  fi
  echo "[infer] $slug (n=$N, timeout ${INFER_TIMEOUT}s) → $logf"
  {
    echo "[model] $slug $(date --iso-8601=s) mode=$mode target=$target"
    if command -v timeout >/dev/null 2>&1; then timeout "$INFER_TIMEOUT" ./s9-16-inference-test.sh "$N"
    else ./s9-16-inference-test.sh "$N"; fi
  } > "$logf.tmp" 2>&1 && rc=0 || rc=$?
  if [ "$rc" -eq 0 ]; then echo "[OK] $slug" >> "$logf.tmp"
  elif [ "$rc" -eq 124 ]; then echo "[INFER-FAIL] timeout ${INFER_TIMEOUT}s $(date --iso-8601=s)" >> "$logf.tmp"
  else echo "[INFER-FAIL] rc=$rc $(date --iso-8601=s)" >> "$logf.tmp"; fi
  mv "$logf.tmp" "$logf"
  tail -n 3 "$logf"
  kill_server; wait_for_port_free || true
  echo "[done] $mode $target"
  return 0
}

if [ -n "$(pids_on_port)" ]; then
  echo "Port ${PORT}가 이미 사용 중($(pids_on_port)) — 기존 서버를 먼저 정리하라." >&2
  exit 1
fi
for t in "${TARGETS[@]}"; do run_one "$MODE" "$t" || true; done

echo "########## 결과 ##########"
ok=0; fail=0
for t in "${TARGETS[@]}"; do
  s="$(expected_slug "$MODE" "$t")"; f="logs/16.${s}.log"
  if [ ! -f "$f" ]; then echo "[MISSING] $MODE $t"; fail=$((fail + 1)); continue; fi
  if grep -q "SERVE-FAIL\|INFER-FAIL" "$f"; then echo "[FAIL] $MODE $t ($f)"; fail=$((fail + 1)); else echo "[OK] $MODE $t"; ok=$((ok + 1)); fi
done
echo "OK=$ok FAIL/MISSING=$fail"
[ "$fail" -eq 0 ]
