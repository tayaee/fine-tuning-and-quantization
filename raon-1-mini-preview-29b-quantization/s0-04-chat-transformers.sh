#!/usr/bin/env bash
# s0-04-chat-transformers.sh [smoke|full] — transformers 서빙 endpoint로 chat 검증 (Stage 0 절차 2).
# 전제: s0-01-load-transformers.sh가 서버를 띄운 상태. PORT(기본 8002).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_repro_echo
rq_banner "s0-04-chat-transformers" "Stage 0 - verify chat on the transformers-served base model"
export UV_NO_SYNC=1; unset VIRTUAL_ENV
PORT="${PORT:-8002}"
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
    max_tokens=256, temperature=0)
text = (r.choices[0].message.content or "")
ok = bool(text.strip())
print("CHAT", "OK" if ok else "EMPTY", ":", text[:200])
try:
    prev = json.load(open(out_path))
except (OSError, ValueError):
    prev = {"step": "s0-chat-transformers"}
prev.update({"chat_transformers": {"model": model_id, "ok": ok,
                                   "response_head": text[:500]}})
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(prev, f, ensure_ascii=False, indent=2)
if not ok:
    raise SystemExit("FAIL: 빈 응답")
PY
)
rq_stage_record 0 base pass "$OUT"
echo ---- result ----
rq_log "s0-04 OK (see $OUT)"
