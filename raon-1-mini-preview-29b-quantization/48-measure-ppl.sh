#!/usr/bin/env bash
# 48-measure-ppl.sh [smoke|full] [targets] — PPL 측정 + 판정.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-smoke}"
N="${PPL_N:-8}"
TARGETS="${2:-all}"
rq_repro_echo
SUMMARY="$RQ_OUTPUTS/ppl-$MODE/summary.json"
rq_lock "${SUMMARY%/*}"
if [ "${FORCE:-0}" != 1 ] && [ -f "$SUMMARY" ]; then
  if [ -z "$(find "$RQ_MODELS" -maxdepth 1 -name "raon-*-$MODE" -newer "$SUMMARY" 2>/dev/null | head -1)" ]; then
    rq_log "skip: fresh $SUMMARY (FORCE=1 to rebuild)"
    echo ---- result ----
    (set -x; cat "$SUMMARY")
    exit 0
  fi
fi
rq_log "ppl_n=$N targets=$TARGETS"
(set -x; uv run "$RQ_ROOT/48-measure-ppl.py" --mode "$MODE" --n "$N" --targets "$TARGETS")

echo ---- result ----
(set -x; cat "$SUMMARY")
