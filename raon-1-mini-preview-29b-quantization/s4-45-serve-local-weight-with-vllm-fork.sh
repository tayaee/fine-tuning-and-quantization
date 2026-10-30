#!/usr/bin/env bash
# s4-45-serve-local-weight-with-vllm-fork.sh [mode] [target] — 로컬 가중치를 KRAFTON vLLM 포크로 서빙.
# s4-55-servebench가 붙는 endpoint를 연다. in-process 측정(s4-00/10/15/20) 이후에 실행할 것 (모델 이중 적재 방지).
# target: base|nvfp4-awq-vllm (기본 nvfp4-awq-vllm). compressed-tensors 타깃은 포크 로드 미검증이라 거부한다.
# PORT 기본 8000 (s4-55 기본값과 동일). 첫 기동은 커널 컴파일로 ~10분. 서버는 상주, s4-60이 내린다.
# 이미 잰 타깃×동시성이면 기동 생략 (FORCE=1이면 강제).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
TARGET="${2:-nvfp4-awq-vllm}"
case "$TARGET" in
  base) SUF="raon-bf16" ;;
  nvfp4-awq-vllm) SUF="raon-$TARGET-$MODE" ;;
  *) echo "target: base|nvfp4-awq-vllm (compressed-tensors 타깃은 s4-40-serve-local-weight-with-transformers.sh)" >&2; exit 1 ;;
esac
MODEL="$RQ_MODELS/$SUF"
[ -f "$MODEL/config.json" ] || { echo "missing model: $MODEL (quant/dequant를 먼저 실행하라)" >&2; exit 1; }
PORT="${PORT:-8000}"
VLLM_REF="${VLLM_REF:-raon-1-mini-preview}"
SERVLOG="$RQ_ROOT/logs/s4-45-serve-vllm-fork.log"
PIDF="$RQ_ROOT/logs/s4-vllm-fork.pid"
TIMEOUT="${S4_LOAD_TIMEOUT:-1800}"
# freshness: 이미 잰 타깃×동시성이면 서버 기동 생략.
if [ "${FORCE:-0}" != 1 ] && "$RQ_PY" - "$MODE" "$TARGET" "${SERVE_BENCH_C:-1}" <<'PYEOF'; then
import json, os, sys
mode, target, c = sys.argv[1], sys.argv[2], str(sys.argv[3])
try:
    d = json.load(open(os.path.join(os.environ["RQ_OUTPUTS"], f"eval-{mode}", "checks.json")))
except (OSError, ValueError):
    sys.exit(1)
run = ((d.get("servebench") or {}).get("targets") or {}).get(target) or {}
sys.exit(0 if run.get(c, {}).get("tok_per_s_p50") is not None else 1)
PYEOF
  rq_log "skip: servebench already recorded for $TARGET c=${SERVE_BENCH_C:-1} (FORCE=1 to re-serve)"
  exit 0
fi
rq_repro_echo
rq_banner "s4-45-serve-local-weight-with-vllm-fork" "Stage 4 - serve local weights with the KRAFTON vLLM fork (endpoint for servebench)"
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "REFUSE: port $PORT busy — 다른 서버가 떠 있음 (끄거나 PORT를 바꿔라)" >&2; exit 1
fi
rm -f "$SERVLOG"
# ---- 서버 실행 바이너리 선택 --------------------------------------------------
# GB10(sm_121)에서는 KRAFTON 사전빌드 포크가 엔진 init 에서 죽는다 (2026-10-29 실측:
# logs/s4-45-serve-vllm-fork.log "Engine core initialization failed"). 그래서
# 소스 빌드 포크(tayaee/raon-vllm-spark@gb10-support)를 쓴다 — s7-20-verify-readme-with-nvfp4-awq-vllm-and-vllm-fork.sh (카드 블록이 Step1 빌드까지 포함)
# 가 만든 venv. 없으면 사전빌드 포크로 폴백 (B300/RTX 5090 계열에서만 동작).
#
# 플래그는 카드(snippets/nvfp4-awq-vllm/serve-vllm.sh)의 서빙 설정과 맞춘다 —
# servebench 수치가 '게시된 카드 그대로'의 조건이어야 한다.
GB10_VENV="${GB10_VENV:-/tmp/raon-1-mini-preview-nvfp4-awq-vllm/.venv}"
SERVE_ARGS=(
  serve "$MODEL"
  --host 127.0.0.1 --port "$PORT"
  --served-model-name "$TARGET"
  --tensor-parallel-size "${TP:-1}"
  --max-model-len "${VLLM_MAX_LEN:-262144}"
  --max-num-batched-tokens 16384
  --max-num-seqs 128
  --gpu-memory-utilization "${GPU_UTIL:-0.85}"
  --kv-cache-dtype fp8_e4m3
  --enable-prefix-caching
  --reasoning-parser raon_1
  --tool-call-parser raon_1
  --enable-auto-tool-choice
  --trust-remote-code
  --default-chat-template-kwargs '{"enable_thinking": true, "truncate_history_thinking": false}'
)
if [ -x "$GB10_VENV/bin/vllm" ]; then
  rq_log "serve binary: $GB10_VENV/bin/vllm (GB10 source build)"
  nohup "$GB10_VENV/bin/vllm" "${SERVE_ARGS[@]}" >"$SERVLOG" 2>&1 &
else
  rq_log "serve binary: precompiled KRAFTON fork (GB10 에서는 엔진 init 실패 — ./s7-20-verify-readme-with-nvfp4-awq-vllm-and-vllm-fork.sh (카드 블록이 Step1 빌드까지 포함) 를 먼저)"
  VLLM_USE_PRECOMPILED=1 nohup uvx --from "vllm @ git+https://github.com/krafton-ai/vllm.git@${VLLM_REF}" \
    vllm "${SERVE_ARGS[@]}" >"$SERVLOG" 2>&1 &
fi
echo $! >"$PIDF"
rq_log "serve start target=$TARGET port=$PORT pid=$(cat "$PIDF") log=$SERVLOG fork=krafton-ai/vllm@$VLLM_REF"
elapsed=0
while [ "$elapsed" -lt "$TIMEOUT" ]; do
  if curl -s --max-time 3 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q '"id"'; then
    echo "Endpoint ready: http://localhost:${PORT}/v1 (target=$TARGET, ${elapsed}s)."
    break
  fi
  if grep -qE "Traceback \(most recent|Error" "$SERVLOG" 2>/dev/null; then
    echo "serve died early (see $SERVLOG)"
    tail -n 5 "$SERVLOG"
    exit 1
  fi
  sleep 5; elapsed=$((elapsed + 5))
done
curl -s --max-time 3 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q '"id"' \
  || { echo "FAIL: endpoint 미기동 (${TIMEOUT}s, 상세: $SERVLOG)" >&2; exit 1; }
echo ---- result ----
rq_log "s4-45 OK endpoint=http://localhost:${PORT}/v1 (server kept for s4-50, pidfile=$PIDF)"
