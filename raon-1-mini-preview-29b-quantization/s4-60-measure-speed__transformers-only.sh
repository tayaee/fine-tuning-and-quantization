#!/usr/bin/env bash
# s4-60-measure-speed__transformers-only.sh [mode] [targets] — transformers 인프로세스 처리량 측정 (Tier 3).
# stock vLLM은 Raon 커스텀 아키텍처를 서빙하지 못하므로, HF 카드의 정석 스택
# (transformers + trust_remote_code)으로 같은 워크(s4-50 고정 8프롬프트)를 순차 측정한다.
# vLLM 서버 불필요. 결과: outputs/eval-<mode>/bench-transformers-<mode>.json
# + checks.json servebench.targets (backend=transformers 표기, 카드 렌더용).
# 사용: ./s4-60-measure-speed__transformers-only.sh full base,nvfp4-awq-transformers  (targets 필수)
# 양자화 타깃만 지정해도 base 기록이 없으면 base를 자동 선행 측정한다 (카드 2줄 보장).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
TARGETS="$(rq_require_targets "./s4-60-measure-speed__transformers-only.sh $MODE" "${2:-}" base awq-w3-transformers awq-w5-transformers nvfp4-awq-vllm nvfp4-awq-transformers nvfp4-gptq-transformers)" || exit 2
rq_repro_echo
rq_banner "s4-60-measure-speed__transformers-only" "Stage 4 - measure in-process throughput with transformers (no server)"
N="${TRANSFORMERS_BENCH_N:-${SERVE_BENCH_N:-8}}"
MAXTOK="${TRANSFORMERS_BENCH_MAXTOK:-${SERVE_BENCH_MAXTOK:-256}}"
# s4-24/s4-26 서버가 떠 있으면 모델 이중 적재(OOM 위험) — 먼저 내리고 실행할 것.
for _pf in "$RQ_ROOT/logs/s4-transformers.pid" "$RQ_ROOT/logs/s4-vllm-fork.pid"; do
  _pid="$(cat "$_pf" 2>/dev/null || true)"
  if [ -n "$_pid" ] && kill -0 "$_pid" 2>/dev/null; then
    rq_log "WARN: local server alive (pid=$_pid, $_pf) — in-process 측정과 모델 이중 적재됨. s4-55-stop-serve-local-weight.sh로 먼저 내릴 것"
  fi
done
unset _pf _pid
# s4-20-aggregate-gsm8k와 같은 lock — checks.json을 쓰는 동안 eval 집계와 겹치지 않게.
rq_lock "$RQ_OUTPUTS/eval-$MODE"
( set -x; uv run "$RQ_ROOT/s4-60-measure-speed__transformers-only.py" --mode "$MODE" --targets "$TARGETS" --n "$N" --max-tokens "$MAXTOK" )
