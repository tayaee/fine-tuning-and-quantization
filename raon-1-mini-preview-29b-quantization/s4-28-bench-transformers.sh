#!/usr/bin/env bash
# s4-28-bench-transformers.sh [mode] [targets] — transformers 인프로세스 처리량 측정 (Tier 3).
# stock vLLM은 Raon 커스텀 아키텍처를 서빙하지 못하므로, HF 카드의 정석 스택
# (transformers + trust_remote_code)으로 같은 워크(s4-25 고정 8프롬프트)를 순차 측정한다.
# vLLM 서버 불필요. 결과: outputs/eval-<mode>/bench-transformers-<mode>.json
# + checks.json servebench.targets (backend=transformers 표기, 카드 렌더용).
# 사용: ./s4-28-bench-transformers.sh full base,nvfp4-awq  (targets 필수)
# 양자화 타깃만 지정해도 base 기록이 없으면 base를 자동 선행 측정한다 (카드 2줄 보장).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
TARGETS="$(rq_require_targets "./s4-28-bench-transformers.sh $MODE" "${2:-}" base awq-w3 awq-w5 nvfp4-awq nvfp4-gptq)" || exit 2
rq_repro_echo
rq_banner "s4-28-bench-transformers" "Stage 4 - measure in-process throughput with transformers (no server)"
N="${TRANSFORMERS_BENCH_N:-${SERVE_BENCH_N:-8}}"
MAXTOK="${TRANSFORMERS_BENCH_MAXTOK:-${SERVE_BENCH_MAXTOK:-256}}"
# s4-13-gsm8k와 같은 lock — checks.json을 쓰는 동안 eval 집계와 겹치지 않게.
rq_lock "$RQ_OUTPUTS/eval-$MODE"
( set -x; uv run "$RQ_ROOT/s4-28-bench-transformers.py" --mode "$MODE" --targets "$TARGETS" --n "$N" --max-tokens "$MAXTOK" )
