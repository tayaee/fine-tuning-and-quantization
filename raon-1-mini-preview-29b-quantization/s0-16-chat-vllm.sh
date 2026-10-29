#!/usr/bin/env bash
# s0-16-chat-vllm.sh [smoke|full] — vLLM 서빙 베이스로 chat 기능 검증 (Stage 0 절차 6).
# 전제: s0-13-serve-base-from-hf-with-vllm-fork.sh가 서버를 띄운 상태. PORT(기본 8000).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_repro_echo
rq_banner "s0-16-chat-vllm" "Stage 0 - verify chat on the vLLM-served base model"
export UV_NO_SYNC=1; unset VIRTUAL_ENV
PORT="${PORT:-8000}"
rq_require_serving "$PORT" "$RQ_ROOT/logs/s0-13-serve-base.log" "s0-13-serve-base-from-hf-with-vllm-fork.sh" || {
  rc=$?
  if [ "$rc" = 2 ]; then rq_stage_record 0 base skipped "chat-vllm:no-server"; exit 0; fi
  rq_stage_record 0 base fail "chat-vllm:server-failed"; exit 1
}
OUT="$RQ_OUTPUTS/s0-$MODE.json"
(set -x; uv run python - "$PORT" "$OUT" <<'PY'
import json, sys
from openai import OpenAI
port, out_path = sys.argv[1], sys.argv[2]
client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
model_id = client.models.list().data[0].id
r = client.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": "사과 3개가 있고 배는 사과보다 2개 많습니다. 배는 몇 개입니까?"}],
    max_tokens=256, temperature=0.6, top_p=0.95)
text = (r.choices[0].message.content or "")
ok = bool(text.strip())
print("CHAT", "OK" if ok else "EMPTY", ":", text[:200])
try:
    prev = json.load(open(out_path))
except (OSError, ValueError):
    prev = {"step": "s0-chat-vllm"}
prev.update({"chat_vllm": {"model": model_id, "ok": ok, "response_head": text[:500]}})
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(prev, f, ensure_ascii=False, indent=2)
if not ok:
    raise SystemExit("FAIL: 빈 응답")
PY
)
rq_stage_record 0 base pass "$OUT"
echo ---- result ----
rq_log "s0-16 OK (see $OUT)"
