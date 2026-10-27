#!/usr/bin/env bash
# s7-52-servebench.sh [smoke|full] — 서빙 처리량 측정 (Tier 3: 배포 성능 증거).
# 이미 떠 있는 OpenAI-호환 서버(s8-64-vllm-serve-one.sh)에 요청을 보내 TTFT와 출력 tok/s를 잰다.
# 서버가 없으면 skip(사유 기록). 결과: outputs/eval-<mode>/servebench-<mode>.json
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_repro_echo
rq_banner "s7-52-servebench" "Stage 7/9 - measure serving throughput (TTFT, output tokens/s) against a running server"

N="${SERVE_BENCH_N:-8}"
C="${SERVE_BENCH_C:-1}"
( set -x; uv run "$RQ_ROOT/s7-52-servebench.py" --mode "$MODE" --n "$N" --concurrency "$C" )

echo ---- result ----
( set -x; cat "$RQ_OUTPUTS/eval-$MODE/servebench-$MODE.json" 2>/dev/null | head -30 )
