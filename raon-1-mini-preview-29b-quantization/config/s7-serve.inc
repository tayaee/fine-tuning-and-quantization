# config/s7-serve.inc — s7-02 (README as-is 검증) 공용 로직.
#
# s7 의 정의: **Hub 카드의 '## Usage - vLLM' 코드 예제를 as-is 그대로 실행한다.**
#   1) 카드 검증: Hub README 의 '## Usage - vLLM' 블록을 snippets/ 와 **바이트 비교**
#      (drift 이면 FAIL — 카드가 실제 명령과 달라졌으면 고객이 복사해 실행할 것이
#       검증된 것과 다르다).
#   2) 실행: **Hub 카드에서 그대로 뽑은 블록**을 README 에 적힌 순서대로 실행한다
#      (nvfp4-awq-vllm 은 Step 1 소스 빌드 → Step 2 서빙, 2블록). 빌드 스크립트를
#      별도 스텝으로 떼어내지 않는다 — 카드가 두 단계를 붙여 놓았으므로 그대로 돌린다.
#   3) transformers 적재 불가 타깃(modelopt_fp4)은 카드의 '## Usage - transformers' 에
#      실행 예제가 없어야 한다 (있으면 FAIL — 그 예제는 반드시 실패한다).
#
# 원칙: **카드의 명령은 손대지 않는다.** 여기서 하는 일은 검증·추출·실행뿐이고,
# (사전 점검으로 막지 못하는 것은 vLLM 엔진 기동 실패이고, 그것은 s7-13이 판정한다.)

# s7_verify_card <target> <repo> — Hub 카드 ↔ snippets/ 바이트 대조 + 블록 추출.
#   추출된 블록: <outdir>/block-<i>.sh  (Hub 카드 원본 바이트)
s7_verify_card() {
  local target="$1" repo="$2" rdir
  local -a snippets=()
  rdir="$RQ_OUTPUTS/s7-$MODE/readme-$target"
  rm -rf -- "$rdir"; mkdir -p "$rdir"
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "FAIL: HF_TOKEN 미설정 — gated repo 의 카드를 읽을 수 없다" >&2
    return 1
  fi
  if ! curl -fsSL -H "Authorization: Bearer $HF_TOKEN" \
      "https://huggingface.co/$repo/raw/main/README.md" -o "$rdir/README.md"; then
    echo "FAIL: Hub README 다운로드 실패 — repo 미발행이거나 gated 접근 권한이 없다 ($repo)" >&2
    return 1
  fi
  # '## Usage - vLLM' 의 블록 순서 = [소스 빌드 스니펫(있으면), 서빙 스니펫]
  [ -f "$RQ_ROOT/snippets/$target/build-vllm-gb10.sh" ] \
    && snippets+=("$RQ_ROOT/snippets/$target/build-vllm-gb10.sh")
  snippets+=("$RQ_ROOT/snippets/$target/serve-vllm.sh")
  if ! "$RQ_PY" "$RQ_ROOT/config/readme-blocks.py" "$rdir/README.md" "Usage - vLLM" "$rdir" "$repo" \
      "${snippets[@]}"; then
    echo "FAIL: Hub 카드의 '## Usage - vLLM' 이 snippets/ 와 바이트 불일치 — 카드 재업로드 필요" >&2
    return 1
  fi
  # transformers 적재 불가 타깃의 카드에도 transformers 실행 예제가 있을 수 있다
  # (modelopt_fp4 는 예제 안에서 BF16 로 변환한다) → 예제 유무만 확인한다.
  if rq_skip_transformers "$target"; then
    "$RQ_PY" "$RQ_ROOT/config/readme-blocks.py" "$rdir/README.md" "Usage - transformers" "$rdir" "$repo" \
        --expect-blocks >/dev/null 2>&1 \
      || { echo "FAIL: 카드의 '## Usage - transformers' 에 실행 예제가 없다" >&2; return 1; }
    rq_log "OK: 카드에 transformers 실행 예제 있음 (BF16 변환 경로)"
  fi
  return 0
}

# s7_run_card_blocks <rdir> — 카드의 vLLM 블록을 README 순서대로 실행.
#   마지막 블록(서빙)은 상주 프로세스라 여기서 끝나지 않는다 — 그 판정은 s7-13/s7-16 이 한다.
s7_run_card_blocks() {
  local rdir="$1" f n=0
  shopt -s nullglob
  for f in "$rdir"/block-*.sh; do
    n=$((n + 1))
    echo "---- README 블록 $n: $f ----"
    cat "$f"
    echo "---- 실행 ----"
    bash "$f"
  done
  shopt -u nullglob
  [ "$n" -gt 0 ] || { echo "FAIL: 실행할 카드 블록이 없다 ($rdir)" >&2; return 1; }
  return 0
}

# s7_serve_frozen <target> [mode]
#   mode: smoke|full (기본 full). repo id 는 mode 에 따라 조립한다.
#     - 이전에는 full 로 고정돼 있어, s7-13 이 smoke 에서 이 래퍼를 부르면
#       full 모델을 서빙했다 (조용히 잘못된 모델 검증).
s7_serve_frozen() {
  local target="$1" mode="${2:-full}" repo rdir
  case "$mode" in smoke|full) ;; *) echo "usage: s7_serve_frozen <target> [smoke|full]" >&2; return 2 ;; esac
  repo="tayaee/${BASE_MODEL##*/}-$target"
  [ "$mode" = smoke ] && repo="$repo-smoke"
  rdir="$RQ_OUTPUTS/s7-$MODE/readme-$target"
  # 카드 검증 (바이트 as-is) — s7 의 본래 목적.
  s7_verify_card "$target" "$repo" || return 1
  if [ "${DRY_RUN:-0}" = 1 ]; then
    # 실행 없이 렌더링된 명령만 본다 (repo id 치환 검증용).
    sed "s|__REPO_ID__|$repo|g" "$RQ_ROOT/snippets/$target/serve-vllm.sh"
    return 0
  fi
  rq_log "repo=$repo (mode=$mode) — 카드의 '## Usage - vLLM' 블록을 순서대로 실행"
  s7_run_card_blocks "$rdir"
}
