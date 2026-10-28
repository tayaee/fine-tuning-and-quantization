# config/s6-customer.inc — s6 고객 시나리오 공용 헬퍼 (Hub 다운로드 경로 규칙).
# 고객이 하는 그대로: Hub repo → 로컬 내려받기 → transformers 로딩.
# 다운로드는 $RQ_SHARED/hf-customer/ 아래에 repo slug별로 1회 (snapshot 재사용).
s6_repo() {
  local target="$1" mode="${2:-$MODE}"
  local repo
  if [ "$target" = nvfp4-awq-modelopt ]; then
    repo="tayaee/Raon-1-mini-preview-NVFP4-AWQ"  # 시장 관례 대문자 (s3-06 ModelOpt판)
  else
    repo="tayaee/${BASE_MODEL##*/}-$target"
  fi
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
