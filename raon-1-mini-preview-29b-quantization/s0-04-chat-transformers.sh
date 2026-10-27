#!/usr/bin/env bash
# s0-04-chat-transformers.sh [smoke|full] — transformers 로딩본으로 chat 기능 검증 (Stage 0 절차 2).
# chat template + greedy 단문 생성으로 한국어 응답이 나오는지 확인한다 (GPU).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_repro_echo
rq_banner "s0-04-chat-transformers" "Stage 0 - verify chat generation on the transformers-loaded base model"
export UV_NO_SYNC=1; unset VIRTUAL_ENV
RQ_BASE="${BASE_MODEL:?BASE_MODEL 미설정}"
OUT="$RQ_OUTPUTS/s0-$MODE.json"
(set -x; uv run python - "$RQ_BASE" "$OUT" <<'PY'
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id, out_path = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
messages = [{"role": "user", "content": "사과 3개가 있고 배는 사과보다 2개 많습니다. 배는 몇 개입니까?"}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                 return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[1]
gen = model.generate(**inputs, max_new_tokens=128, do_sample=False)
text = tok.decode(gen[0][input_len:], skip_special_tokens=True)
ok = bool(text.strip())
print("CHAT", "OK" if ok else "EMPTY", ":", text[:200])
payload = {"step": "s0-04-chat-transformers", "model": model_id,
           "ok": ok, "response_head": text[:500]}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
if not ok:
    raise SystemExit("FAIL: 빈 응답 — chat template/디코더 확인 필요")
PY
)
rq_stage_record 0 base pass "$OUT"
echo ---- result ----
rq_log "s0-04 OK (see $OUT)"
