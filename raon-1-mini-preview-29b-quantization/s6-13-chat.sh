#!/usr/bin/env bash
# s6-13-chat.sh [mode] [model] — 고객 transformers 로딩본으로 chat 검증 (Stage 6).
# model: HF repo id ("tayaee/...-nvfp4-awq-transformers") | short target | all (기본 nvfp4-awq-transformers).
# 각 모델을 Hub→다운로드→transformers 로드→chat template greedy 생성→응답 존재 판정.
# logs/s6-13-<tag>.log + outputs/s6-<mode>/chat-<tag>.json. 산출물 변경 없음.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
source "$RQ_ROOT/config/s6-customer.inc"
rq_repro_echo
rq_banner "s6-13-chat" "Stage 6 - verify chat on customer-loaded models (transformers)"
export UV_NO_SYNC=1; unset VIRTUAL_ENV
rq_hf_parse "${2:-nvfp4-awq-transformers}" "$MODE"
FAIL=0
for _e in $RQ_HF_TARGETS_RUN; do
  case "$_e" in
    */*) REPO="$_e"; TAG="$(printf '%s' "$_e" | tr '/:' '__')" ;;
    *) REPO="$(s6_repo "$_e" "$MODE")"; TAG="$_e" ;;
  esac
  t="$TAG"
  LOGF="$RQ_ROOT/logs/s6-13-$t.log"
  _tgt="$(rq_target_of "$t")"
  if [ -n "$_tgt" ] && rq_skip_transformers "$_tgt"; then
    echo "[SKIP] mode=$MODE target=$t — transformers 적재 불가(vLLM 전용)" | tee "$LOGF"
    rq_stage_record 6 cat-tf pass "customer-skip:$t"
    continue
  fi
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
    # 카드의 app-transformers.py 스니펫과 동일 파일 실행 (s6 검증 = README instruction 검증).
    # 스니펫이 없는 타깃(직접 repo 지정 등)은 기존 인라인 검사로 폴백한다.
    _snip_tgt="$_tgt"
    if [ -z "$_snip_tgt" ]; then
      for _c in awq-w3-transformers awq-w5-transformers nvfp4-awq-vllm nvfp4-awq-transformers nvfp4-gptq-transformers; do
        case "$t" in *"$_c"*) _snip_tgt="$_c" ;; esac
      done
    fi
    _snip="$RQ_ROOT/snippets/$_snip_tgt/app-transformers.py"
    if [ -n "$_snip_tgt" ] && [ -f "$_snip" ]; then
      echo "[snippet] $_snip (repo=$REPO)"
      sed "s|__REPO_ID__|$REPO|g" "$_snip" > "$OUTDIR/app-$t.py"
      uv run python "$OUTDIR/app-$t.py" > "$OUTDIR/app-$t.out" 2>&1; rc=$?
      cat "$OUTDIR/app-$t.out"
      if [ "$rc" -eq 0 ] && awk '/^Answer:$/{f=1;next} f && NF{found=1} END{exit !found}' "$OUTDIR/app-$t.out"; then
        _ans="$(awk '/^Answer:$/{f=1;next} f{print}' "$OUTDIR/app-$t.out" | head -c 500)"
        "$RQ_PY" - "$REPO" "$OUT" "$_ans" <<'PY'
import json, sys
repo, out_path, ans = sys.argv[1], sys.argv[2], sys.argv[3]
ok = bool(ans.strip())
print("CHAT", "OK" if ok else "EMPTY", ":", ans[:200])
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"target": repo, "ok": ok, "response_head": ans[:500]},
              f, ensure_ascii=False, indent=2)
if not ok:
    raise SystemExit("FAIL: 빈 응답")
PY
        rc=$?
      else
        echo "FAIL: 스니펫 실행 실패 또는 빈 Answer (rc=$rc)"
        rc=1
      fi
    else
      echo "[fallback] 인라인 chat 검사 (스니펫 없음: $_snip)"
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
      rc=$?
    fi
  } > "$LOGF" 2>&1 && rc=0 || rc=$?
  if [ "$rc" -eq 0 ]; then echo "[OK] $t" >> "$LOGF"; tail -n 2 "$LOGF"
  else echo "[CHAT-FAIL] rc=$rc $(date --iso-8601=s)" >> "$LOGF"; FAIL=1; fi
  if [ "$rc" -eq 0 ]; then rq_stage_record 6 cat-tf pass "$OUT"
  else rq_stage_record 6 cat-tf fail "customer-chat:$t"; fi
done
[ "$FAIL" -eq 0 ]
