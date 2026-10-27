#!/usr/bin/env bash
# s6-13-chat.sh [mode] [model] — 고객 transformers 로딩본으로 chat 검증 (Stage 6).
# model: HF repo id ("tayaee/...-nvfp4-awq") | short target | all (기본 nvfp4-awq).
# 각 모델을 Hub→다운로드→transformers 로드→chat template greedy 생성→응답 존재 판정.
# logs/s6-13-<tag>.log + outputs/s6-<mode>/chat-<tag>.json. 산출물 변경 없음.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
source "$RQ_ROOT/config/s6-customer.inc"
rq_repro_echo
rq_banner "s6-13-chat" "Stage 6 - verify chat on customer-loaded models (transformers)"
export UV_NO_SYNC=1; unset VIRTUAL_ENV
rq_hf_parse "${2:-nvfp4-awq}" "$MODE"
FAIL=0
for _e in $RQ_HF_TARGETS_RUN; do
  case "$_e" in
    */*) REPO="$_e"; TAG="$(printf '%s' "$_e" | tr '/:' '__')" ;;
    *) REPO="$(s6_repo "$_e" "$MODE")"; TAG="$_e" ;;
  esac
  t="$TAG"
  LOGF="$RQ_ROOT/logs/s6-13-$t.log"
  OUT="$RQ_OUTPUTS/s6-$MODE/chat-$t.json"
  mkdir -p "$RQ_OUTPUTS/s6-$MODE"
  echo "=========================================="
  echo "Model: mode=$MODE target=$t (repo $REPO)"
  echo "=========================================="
  if ! LOCAL="$(s6_download "$REPO")"; then
    { echo "[SERVE-FAIL] $(date --iso-8601=s) mode=$MODE target=$t repo=$REPO (download 실패)"; } > "$LOGF"
    rq_stage_record 6 cat-tf fail "customer-chat:$t"
    FAIL=1; continue
  fi
  {
    echo "[model] $REPO $(date --iso-8601=s) mode=$MODE target=$t"
    uv run python - "$LOCAL" "$REPO" "$OUT" <<'PY'
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
local_dir, repo, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    local_dir, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
messages = [{"role": "user", "content": "사과 3개가 있고 배는 사과보다 2개 많습니다. 배는 몇 개입니까?"}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                 return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[1]
gen = model.generate(**inputs, max_new_tokens=128, do_sample=False)
text = tok.decode(gen[0][input_len:], skip_special_tokens=True)
ok = bool(text.strip())
print("CHAT", "OK" if ok else "EMPTY", ":", text[:200])
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"target": repo, "ok": ok, "response_head": text[:500]},
              f, ensure_ascii=False, indent=2)
if not ok:
    raise SystemExit("FAIL: 빈 응답")
PY
  } > "$LOGF" 2>&1 && rc=0 || rc=$?
  if [ "$rc" -eq 0 ]; then echo "[OK] $t" >> "$LOGF"; tail -n 2 "$LOGF"
  else echo "[CHAT-FAIL] rc=$rc $(date --iso-8601=s)" >> "$LOGF"; FAIL=1; fi
  if [ "$rc" -eq 0 ]; then rq_stage_record 6 cat-tf pass "$OUT"
  else rq_stage_record 6 cat-tf fail "customer-chat:$t"; fi
done
[ "$FAIL" -eq 0 ]
