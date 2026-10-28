#!/usr/bin/env bash
# s6-07-test-nvfp4-awq-transformers-with-transformers.sh [smoke|full] — 고객 시나리오: Hub에서 내려받아 transformers 로딩 (nvfp4-awq-transformers).
# README Usage-transformers와 동일한 절차 (읽기 전용, 산출물 변경 없음).
# smoke는 -smoke repo, full만 정식 repo를 본다. repo 미발행이면 FAIL(고객이 받을 수 없음).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
source "$RQ_ROOT/config/s6-customer.inc"
rq_repro_echo
rq_banner "s6-07-test-nvfp4-awq-with-transformers" "Stage 6 - customer scenario: download from Hub and load with transformers (nvfp4-awq-transformers)"
export UV_NO_SYNC=1; unset VIRTUAL_ENV
TARGET="nvfp4-awq-transformers"
REPO="$(s6_repo "$TARGET" "$MODE")"
OUTDIR="$RQ_OUTPUTS/s6-$MODE"
mkdir -p "$OUTDIR"
rq_log "repo=$REPO"
if ! LOCAL="$(s6_download "$REPO")"; then
  echo "FAIL: Hub repo 없음/다운로드 실패: $REPO (미발행 타깃)" >&2
  rq_stage_record 6 cat-tf fail "customer-download:$TARGET"
  exit 1
fi
(set -x; uv run python - "$LOCAL" "$REPO" "$OUTDIR/$TARGET.json" <<'PY'
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
local_dir, repo, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    local_dir, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
prompt = "What is 2+2? 한국어로 답하세요."
inputs = tok(prompt, return_tensors="pt").to(model.device)
gen = model.generate(**inputs, max_new_tokens=32, do_sample=False)
text = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
ok = True  # 로딩 자체가 검증 (생성 텍스트 품질은 s6-13에서 판정)
print("LOAD OK", type(model).__name__, ":", text[:120])
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"target": repo, "local_dir": local_dir, "ok": ok,
                "response_head": text[:500]}, f, ensure_ascii=False, indent=2)
PY
)
rq_stage_record 6 cat-tf pass "$OUTDIR/$TARGET.json"
echo ---- result ----
rq_log "s6-07 OK target=$TARGET (see $OUTDIR/$TARGET.json)"
