#!/usr/bin/env bash
# 10-setup.sh — 양 노드에서 1회씩 실행. 학습 없음.
# 재현성: uv.lock + .python-version 기준.
set -euo pipefail
cd "$(dirname "$0")"
source "$(dirname "$0")/config/common.env"
rq_repro_echo

rq_log "repo=$REPO_ROOT rq_root=$RQ_ROOT infra=$INFRA python=$(python3 --version 2>&1) uv=$(uv --version 2>&1)"
rq_log "pins: BASE_MODEL=$BASE_MODEL BASE_REVISION=${BASE_REVISION:-(floating, 첫 다운로드 후 고정)}"
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

echo ---- result ----
(set -x; ls -lh "$RQ_DATASETS/")
