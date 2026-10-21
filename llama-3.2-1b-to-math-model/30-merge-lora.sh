#!/usr/bin/env bash
# 30-merge-lora.sh [mini|full] [base-model] (STRAT env, 기본 single) — Stage 3c wrapper. QLoRA 어댑터 병합.
# 입력: synthetic-qlora-<mode>-$STRAT → 출력: synthetic-qlora-<mode>-$STRAT-merged
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
STRAT="${STRAT:-single}"
case "$STRAT" in single|ddp|fsdp) ;; *) echo "STRAT must be single|ddp|fsdp" >&2; exit 1;; esac
m1b_repro_echo
BASE="${2:-$BASE_MODEL}"
SRC="$M1B_MODELS/synthetic-qlora-$MODE-$STRAT"
OUT="$M1B_MODELS/synthetic-qlora-$MODE-$STRAT-merged"
m1b_lock "$OUT"
m1b_require "$SRC/adapter_config.json"
if [ -f "$OUT/config.json" ] && m1b_fresh "$OUT/config.json" "$SRC/adapter_config.json"; then
  m1b_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -l "$OUT")
  exit 0
fi
mkdir -p "$OUT"
m1b_log "base=$BASE rev=$BASE_REVISION out=$OUT"
(set -x; uv run "$M1B_ROOT/30-merge-lora.py" --mode "$MODE" --strat "$STRAT" --base "$BASE" --revision "$BASE_REVISION")

echo ---- result ----
(set -x; ls -l "$OUT")
