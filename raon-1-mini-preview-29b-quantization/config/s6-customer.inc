# config/s6-customer.inc — s6 고객 시나리오 공용 헬퍼 (Hub 다운로드 경로 규칙).
# 고객이 하는 그대로: Hub repo → 로컬 내려받기 → transformers 로딩.
# 다운로드는 $RQ_SHARED/hf-customer/ 아래에 repo slug별로 1회 (snapshot 재사용).
s6_repo() {
  local target="$1" mode="${2:-$MODE}"
  local repo
  repo="tayaee/${BASE_MODEL##*/}-$target"
  [ "$mode" = smoke ] && repo="$repo-smoke"
  printf '%s' "$repo"
}

s6_customer_dir() {
  printf '%s' "$RQ_SHARED/hf-customer/$(printf '%s' "$1" | tr '/:' '__')"
}

# s6_download <repo> — Hub 스냅샷을 받아 로컬 디렉터리 경로를 출력한다.
# repo가 없으면 내려받기 실패 → 호출자가 FAIL 처리한다.
s6_download() {
  local repo="$1" dest
  dest="$(s6_customer_dir "$repo")"
  if [ -f "$dest/config.json" ]; then
    printf '%s' "$dest"
    return 0
  fi
  RQ_PY="${RQ_PY:-python3}"
  "$RQ_PY" - "$repo" "$dest" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, local_dir=dest)
print(dest, file=sys.stderr)
PY
  [ -f "$dest/config.json" ] || return 1
  printf '%s' "$dest"
}

# s6_readme_verify <target> — Hub README "## Usage - transformers" 를 바이트 그대로 실행한다.
# 고객 절차 그대로: Hub의 README.md를 받아 블록 0(셋업 bash)을 실행하고, 블록 1(python)을
# 작업 디렉터리에 app.py로 저장한 뒤 블록 2(`python app.py`)를 같은 셸 세션에서 실행한다.
# - 블록 0/1은 snippets/<target>/ 와 바이트 비교 (다르면 카드 드리프트 → FAIL, 실행 안 함).
# - 작업 디렉터리(블록 0의 mkdir 경로)가 이미 있으면 FAIL → ./s6-00-clean-serving-workspace.sh 먼저.
#   (README는 새 디렉터리 전제: 기존 .venv가 있으면 `uv venv`가 대화형 확인에서 멈춘다.)
# - 실행 환경은 env -i 로 비운 고객 셸 (HOME/USER/LANG/TERM/PATH(uv, nvcc), HF_TOKEN만 전달).
#   이 repo의 HF_HOME·PYTORCH_CUDA_ALLOC_CONF·VIRTUAL_ENV 등은 새지 않는다 → HF 캐시는 ~/.cache/huggingface.
# - HF_TOKEN은 필수: README 블록 0의 `hf auth login`은 토큰이 이미 있으면 no-op으로 끝나지만,
#   토큰이 없으면 OAuth 디바이스 플로우(브라우저 승인 대기)로 들어가 무한 대기한다 → 사전 차단.
# - 세션 전체에 timeout (S6_README_TIMEOUT, 기본 7200s). 걸려도 파이프라인이 영영 멈추지 않게.
# s6 의 정의: **Hub 카드의 '## Usage - transformers' 코드 예제를 검증한다.**
#   (transformers 로 적재 불가한 타깃(modelopt_fp4)은 카드가 transformers 예제를 갖지
#    않으므로 s6 검증 대상이 아니다 — 그런 타깃은 s7(vLLM)에서 검증한다.)
# - 판정: 세션 rc=0 이고 "Answer:" 뒤에 비어있지 않은 텍스트. 산출물 outputs/s6-<mode>/<target>.json.
s6_readme_verify() {
  local target="$1" repo outdir rdir readme ws runner out rc=0 t0 _uv tmo
  repo="$(s6_repo "$target" "$MODE")"
  outdir="$RQ_OUTPUTS/s6-$MODE"
  rdir="$outdir/readme-$target"
  readme="$rdir/README.md"
  runner="$rdir/customer-session.sh"
  out="$outdir/app-$target.out"
  rm -rf -- "$rdir"; mkdir -p "$rdir"
  rq_log "repo=$repo readme=https://huggingface.co/$repo/raw/main/README.md"
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "FAIL: HF_TOKEN 미설정 — README 블록 0의 'hf auth login'은 토큰이 없으면 인터랙티브 OAuth로 무한 대기한다" >&2
    rq_stage_record 6 cat-tf fail "customer-readme-no-token:$target"
    return 1
  fi
  # gated repo 는 토큰 없으면 README조차 401 이다 → 토큰을 실어 받는다.
  # (repo 가 미발행이면 401/404 로 막힌다 — 그건 "미발행 repo면 FAIL" 계약이다.)
  if ! curl -fsSL -H "Authorization: Bearer $HF_TOKEN" \
      "https://huggingface.co/$repo/raw/main/README.md" -o "$readme"; then
    echo "[SERVE-FAIL] $(date --iso-8601=s) mode=$MODE target=$target repo=$repo" >&2
    echo "           (Hub README 다운로드 실패 — repo 미발행이거나 gated 접근 권한이 없다)" >&2
    rq_stage_record 6 cat-tf fail "customer-readme:$target"
    return 1
  fi
  if ! "$RQ_PY" "$RQ_ROOT/config/readme-blocks.py" "$readme" "Usage - transformers" "$rdir" "$repo" \
      "$RQ_ROOT/snippets/$target/usage-transformers.sh" \
      "$RQ_ROOT/snippets/$target/app-transformers.py" -; then
    rq_stage_record 6 cat-tf fail "customer-readme-drift:$target"
    return 1
  fi
  [ -f "$rdir/block-0.sh" ] && [ -f "$rdir/block-1.py" ] && [ -f "$rdir/block-2.sh" ] || {
    echo "FAIL: README transformers 블록 구성이 (bash, python, bash) 가 아님" >&2
    rq_stage_record 6 cat-tf fail "customer-readme-shape:$target"
    return 1
  }
  ws="$(sed -nE 's|^[[:space:]]*mkdir[[:space:]]+-p[[:space:]]+"?([^"[:space:]]+)"?.*|\1|p' "$rdir/block-0.sh" | head -n 1)"
  if [ -z "$ws" ]; then
    echo "FAIL: 블록 0에서 작업 디렉터리(mkdir -p ...)를 찾지 못함" >&2
    rq_stage_record 6 cat-tf fail "customer-readme-shape:$target"
    return 1
  fi
  if [ -e "$ws" ]; then
    echo "FAIL: 작업 디렉터리 $ws 가 이미 있음 — ./s6-00-clean-serving-workspace.sh 를 먼저 실행" >&2
    return 1
  fi
  # 고객 세션 = 블록 0 원문 + (app.py 저장) + 블록 2 원문. 블록 원문은 cat으로 이어붙여 바이트 보존.
  {
    cat "$rdir/block-0.sh"
    printf 'cp -- %q app.py  # [harness] README python 블록을 app.py로 저장 (고객의 복사-붙여넣기)\n' "$rdir/block-1.py"
    cat "$rdir/block-2.sh"
  } > "$runner"
  echo "---- customer session ($runner) ----"
  cat "$runner"
  echo "------------------------------------"
  _uv="$(dirname "$(command -v uv)")"
  tmo="${S6_README_TIMEOUT:-7200}"
  t0=$(date +%s)
  timeout --foreground "$tmo" \
    env -i HOME="$HOME" USER="${USER:-$(id -un)}" LOGNAME="${LOGNAME:-$(id -un)}" \
      LANG="${LANG:-C.UTF-8}" TERM="${TERM:-dumb}" \
      PATH="$_uv:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
      HF_TOKEN="$HF_TOKEN" \
      bash -e -o pipefail "$runner" < /dev/null 2>&1 | tee "$out" || rc=$?
  local elapsed=$(( $(date +%s) - t0 ))
  if [ "$rc" -eq 0 ] && ! cmp -s "$ws/app.py" "$rdir/block-1.py"; then
    echo "FAIL: $ws/app.py 가 README python 블록과 바이트 불일치" >&2; rc=1
  fi
  if [ "$rc" -eq 0 ] && awk '/^Answer:$/{f=1;next} f && NF{found=1} END{exit !found}' "$out"; then
    "$RQ_PY" - "$repo" "$outdir/$target.json" "$rdir/blocks.json" "$elapsed" "$out" <<'PY' || rc=$?
import json, sys
repo, out_path, blocks, elapsed, log = sys.argv[1:6]
text = open(log, encoding="utf-8", errors="replace").read()
ans = text.split("\nAnswer:\n", 1)[-1].strip()
ok = bool(ans)
print("VERIFY", "OK" if ok else "EMPTY", ":", ans[:200])
meta = json.load(open(blocks, encoding="utf-8"))
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"target": repo, "ok": ok, "as_is": True, "elapsed_s": int(elapsed),
               "readme_sha256": meta["readme_sha256"], "blocks": meta["blocks"],
               "response_head": ans[:500]}, f, ensure_ascii=False, indent=2)
raise SystemExit(0 if ok else 1)
PY
  else
    if [ "$rc" -eq 124 ]; then
      echo "FAIL: README 세션 타임아웃 (${tmo}s 초과) — log $out" >&2
    else
      echo "FAIL: README 세션 실패 또는 빈 Answer (rc=$rc, ${elapsed}s, log $out)" >&2
    fi
    [ "$rc" -eq 0 ] && rc=1
  fi
  if [ "$rc" -eq 0 ]; then rq_stage_record 6 cat-tf pass "$outdir/$target.json"
  else rq_stage_record 6 cat-tf fail "customer-verify:$target"; fi
  return "$rc"
}
