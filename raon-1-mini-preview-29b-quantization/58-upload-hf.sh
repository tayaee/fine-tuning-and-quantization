#!/usr/bin/env bash
# 58-upload-hf.sh [smoke|full] [targets] — HF Hub 업로드 (tayaee/*).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-smoke}"
rq_repro_echo
(set -x; uv run "$RQ_ROOT/58-upload-hf.py" --mode "$MODE" --targets "${2:-all}")

echo ---- result ----
rq_log "uploaded targets=${2:-all} (see URLs above)"
