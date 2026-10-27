#!/usr/bin/env bash
# s0-01-load-transformers.sh [smoke|full] — 베이스모델 접근+메타데이터 확인 (Stage 0 절차 1).
# BASE_MODEL(HF gated, trust_remote_code 필수)의 config·tokenizer가 풀리는지 확인한다.
# 가중치 변경 없음. 전체 가중치 로드는 s0-04-chat-transformers에서 검증한다.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_repro_echo
rq_banner "s0-01-load-transformers" "Stage 0 - check base model access (config/tokenizer via transformers)"
export UV_NO_SYNC=1; unset VIRTUAL_ENV
RQ_BASE="${BASE_MODEL:?BASE_MODEL 미설정}"
(set -x; uv run python - "$RQ_BASE" <<'PY'
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = sys.argv[1]
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print("tokenizer OK", type(tok).__name__, "vocab=", tok.vocab_size)
import transformers
print("transformers", transformers.__version__)
PY
)
rq_stage_record 0 base pass "base-transformers-load:$RQ_BASE"
echo ---- result ----
rq_log "s0-01 OK base=$RQ_BASE (tokenizer/config load; full weights in s0-04)"
