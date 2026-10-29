#!/usr/bin/env bash
# s6-02-verify-awq-w5-transformers-with-transformers.sh [smoke|full] — README Usage-transformers의 model.generate 예제를 그대로 실행 (awq-w5-transformers).
# 실행 내용은 snippets/awq-w5-transformers/app-transformers.py 와 동일하다 (카드에 박힌 예제 그 자체, repo id만 치환).
# 사용: ./s6-02-verify-awq-w5-transformers-with-transformers.sh smoke
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
source "$RQ_ROOT/config/s6-customer.inc"
rq_repro_echo
rq_banner "s6-02-verify-awq-w5-transformers-with-transformers" "Stage 6 - run the README transformers generate example as-is (awq-w5-transformers)"
export UV_NO_SYNC=1; unset VIRTUAL_ENV
TARGET="awq-w5-transformers"
REPO="$(s6_repo "$TARGET" "$MODE")"
OUTDIR="$RQ_OUTPUTS/s6-$MODE"
mkdir -p "$OUTDIR"
rq_log "repo=$REPO"
if ! LOCAL="$(s6_download "$REPO")"; then
  echo "[SERVE-FAIL] $(date --iso-8601=s) mode=$MODE target=$TARGET repo=$REPO (다운로드 실패)" >&2
  rq_stage_record 6 cat-tf fail "customer-download:$TARGET"
  exit 1
fi
APP="$OUTDIR/app-$TARGET.py"
sed "s|__REPO_ID__|$REPO|g" "$RQ_ROOT/snippets/$TARGET/app-transformers.py" > "$APP"
rq_log "snippet $APP (repo=$REPO)"
( set -x; uv run python "$APP" ) > "$OUTDIR/app-$TARGET.out" 2>&1; rc=$?
cat "$OUTDIR/app-$TARGET.out"
if [ "$rc" -eq 0 ] && awk '/^Answer:{f=1;next} f && NF{found=1} END{exit !found}' "$OUTDIR/app-$TARGET.out"; then
  _ans="$(awk '/^Answer:{f=1;next} f{print}' "$OUTDIR/app-$TARGET.out" | head -c 500)"
  "$RQ_PY" - "$REPO" "$OUTDIR/$TARGET.json" "$_ans" <<'PY'
import json, sys
repo, out_path, ans = sys.argv[1], sys.argv[2], sys.argv[3]
ok = bool(ans.strip())
print("VERIFY", "OK" if ok else "EMPTY", ":", ans[:200])
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"target": repo, "ok": ok, "response_head": ans[:500]},
              f, ensure_ascii=False, indent=2)
if not ok:
    raise SystemExit("FAIL: 빈 Answer")
PY
  rc=$?
else
  echo "FAIL: 스니펫 실행 실패 또는 빈 Answer (rc=$rc)" >&2
  rc=1
fi
if [ "$rc" -eq 0 ]; then rq_stage_record 6 cat-tf pass "$OUTDIR/$TARGET.json"
else rq_stage_record 6 cat-tf fail "customer-verify:$TARGET"; fi
echo ---- result ----
rq_log "s6-02 DONE target=$TARGET (see $OUTDIR/$TARGET.json)"
[ "$rc" -eq 0 ]
