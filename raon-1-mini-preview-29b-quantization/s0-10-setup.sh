#!/usr/bin/env bash
# s0-10-setup.sh — 양 노드에서 1회씩 실행. 학습 없음.
# 재현성: uv.lock + .python-version 기준.
set -euo pipefail
cd "$(dirname "$0")"
source "$(dirname "$0")/config/common.env"
rq_repro_echo
rq_banner "s0-10-setup" "Stage 0/9 - prepare the environment (uv sync, llmcompressor/compressed-tensors pins, HF login, GPU/NVFP4 check)"

rq_log "repo=$REPO_ROOT rq_root=$RQ_ROOT infra=$INFRA python=$(python3 --version 2>&1) uv=$(uv --version 2>&1)"
rq_log "pins: BASE_MODEL=$BASE_MODEL BASE_REVISION=${BASE_REVISION:-(floating, 첫 다운로드 후 고정)}"

# 동시 build(8종 wrapper 병렬 실행)가 venv를 동시에 변조하지 않게 한다.
# lock은 .venv를 포함한 저장소 단위로 잡는다(먼저 들어온 쪽이 setup, 나머지는 대기→skip).
rq_lock "$RQ_ROOT"

# 이미 준비된 환경이면 건너뛴다 (build 8종이 매번 uv sync/pip install을 반복하지 않도록).
# 판정 기준: uv.lock 해시 + 설치된 핀 버전(llmcompressor/compressed-tensors).
STAMP="$RQ_ROOT/.venv/.rq-setup-stamp"
_llm_ver="$(uv pip show llmcompressor 2>/dev/null | awk '/^Version:/{print $2}')"
_ct_ver="$(uv pip show compressed-tensors 2>/dev/null | awk '/^Version:/{print $2}')"
_uvlock_sha="$(sha256sum "$RQ_ROOT/uv.lock" 2>/dev/null | awk '{print $1}')"
if [ "${FORCE:-0}" != 1 ] \
  && rq_stamp_ok "$STAMP" uvlock "$_uvlock_sha" \
  && rq_stamp_ok "$STAMP" llmcompressor "$_llm_ver" \
  && rq_stamp_ok "$STAMP" compressed_tensors "$_ct_ver"; then
  rq_log "skip: env already ready (llmcompressor=$_llm_ver compressed-tensors=$_ct_ver uv.lock=${_uvlock_sha:0:12})"
  echo ---- result ----
  (set -x; ls -lh "$RQ_DATASETS/")
  exit 0
fi

if [ -n "${UV_MIN:-}" ]; then
  _uvv="$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)"
  if [ -n "$_uvv" ] && [ "$(printf '%s\n%s\n' "$UV_MIN" "$_uvv" | sort -V | head -n 1)" != "$UV_MIN" ]; then
    echo "REFUSE: uv $_uvv < min $UV_MIN — uv를 업그레이드하라" >&2; exit 1
  fi
  unset _uvv
fi

echo + uv sync --frozen
uv sync --frozen

# write 도구로 생성된 .sh는 실행 비트가 없을 수 있어 여기서 보장 (126 방지).
chmod +x "$RQ_ROOT"/*.sh

# llmcompressor는 resolver 충돌 회피를 위해 --no-deps로 별도 설치.
echo + uv pip install --no-deps "llmcompressor==0.14.0"
uv pip install --no-deps "llmcompressor==0.14.0"

# compressed-tensors는 lock(vllm==0.31.0 → ==0.17.0)과 llmcompressor(==0.19.0 요구)가 충돌한다.
# 파이프라인은 llmcompressor import가 필수라 0.19.0으로 명시 상향 (vLLM 0.31 서빙·로딩 동작 확인済み).
# common.env가 UV_NO_SYNC=1이라 파이프 중 되돌려지지 않는다. bare `uv sync`는 0.17.0으로
# 되돌리므로 환경 재구축은 이 스크립트 경로로만 수행한다.
echo + uv pip install "compressed-tensors==0.19.0"
uv pip install "compressed-tensors==0.19.0"

rq_log "compressed-tensors=$(uv pip show compressed-tensors 2>/dev/null | awk '/^Version:/{print $2}')"
rq_log "vllm=$(uv pip show vllm 2>/dev/null | awk '/^Version:/{print $2}')"

mkdir -p "$RQ_SHARED"/{datasets,models,outputs,hf-cache,vllm-cache}

rq_log "HF login check (Raon gated — 자동 승인 후 login 필요)"
(set -x; "$HF_BIN" auth whoami 2>/dev/null) || echo "-> 'uv run hf auth login' required"

rq_log "GPU check (NVFP4는 Blackwell SM>=10 필요)"
nvidia-smi -L 2>/dev/null || echo "-> no nvidia-smi, check on DGX OS"
(set -x; "$VENV_BIN/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())" 2>/dev/null || true)

rq_log "OK. shared=$RQ_SHARED (compare ls on spark1/spark2)"

# 다음 build가 setup을 건너뛸 수 있도록 준비 상태를 기록한다.
_llm_ver="$(uv pip show llmcompressor 2>/dev/null | awk '/^Version:/{print $2}')"
_ct_ver="$(uv pip show compressed-tensors 2>/dev/null | awk '/^Version:/{print $2}')"
rq_stamp_write "$STAMP" uvlock "$_uvlock_sha" llmcompressor "$_llm_ver" compressed_tensors "$_ct_ver"
rq_log "setup stamp -> $STAMP"

echo ---- result ----
(set -x; ls -lh "$RQ_DATASETS/")
