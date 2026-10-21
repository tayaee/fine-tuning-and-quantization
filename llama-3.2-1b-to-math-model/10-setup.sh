#!/usr/bin/env bash
# 10-setup.sh — 양 노드에서 1회씩 실행. 학습 없음.
# 재현성: uv.lock + .python-version + requirements-frozen.txt 기준.
# 상세 핀 목록은 REPRODUCIBILITY.md.
set -euo pipefail
cd "$(dirname "$0")"  # uv가 math-1b-sft-and-quantization/pyproject.toml을 프로젝트로 인식하도록 고정
source "$(dirname "$0")/config/common.env"
m1b_repro_echo

m1b_log "repo=$REPO_ROOT m1b_root=$M1B_ROOT infra=$INFRA python=$(python3 --version 2>&1) uv=$(uv --version 2>&1)"
m1b_log "pins: BASE_REVISION=$BASE_REVISION TEACHER_REVISION=$TEACHER_REVISION MASKED_REV=$MASKED_REV LLAMACPP_REV=$LLAMACPP_REV"
if [ -n "${UV_MIN:-}" ]; then
  _uvv="$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)"
  if [ -n "$_uvv" ] && [ "$(printf '%s\n%s\n' "$UV_MIN" "$_uvv" | sort -V | head -n 1)" != "$UV_MIN" ]; then
    echo "REFUSE: uv $_uvv < min $UV_MIN — uv를 업그레이드하라" >&2; exit 1
  fi
  unset _uvv
fi

# 락 기준 설치 ( --frozen: lock과 pyproject 불일치 시 실패 = 드리프트 즉시 감지 ).
echo + uv sync --frozen
uv sync --frozen

# llmcompressor는 resolver 충돌 회피를 위해 --no-deps로 별도 설치 (상세: pyproject.toml NOTE).
# lock 밖이므로 버전을 명시 고정. 2026-10-19 실측: 0.14.0.
echo + uv pip install --no-deps "llmcompressor==0.14.0"
uv pip install --no-deps "llmcompressor==0.14.0"

# compressed-tensors는 lock(vllm==0.31.0 → ==0.17.0)을 기준으로 사용한다.
# 과거 0.19.0 별도 설치는 uv sync가 되돌려 실측 0.17.0으로 동작 확인됨 — 오버라이드 삭제.
# (--no-deps llmcompressor가 의존성을 안 맞추므로 vLLM AWQ/FP8 로딩은 50-eval에서 확인.)
m1b_log "compressed-tensors=$(uv pip show compressed-tensors 2>/dev/null | awk '/^Version:/{print $2}')"

# torchaudio는 lock의 transitive로 남아도 제거하지 않는다 (제거해도 다음 sync에서
# 복원되어 드리프트만 유발. vLLM warmup에 불필요해도 존재 자체는 무해).

mkdir -p "$M1B_SHARED"/{datasets,models,outputs,hf-cache,vllm-cache}

m1b_log "upstream check: $MASKED (pinned $MASKED_REV)"
if [ ! -d "$MASKED/.git" ]; then
  git clone https://github.com/changyuchen347/maskedthought "$MASKED" \
    || m1b_log "WARN: clone failed, clone manually if offline"
fi
if [ -d "$MASKED/.git" ]; then
  _mr="$(git -C "$MASKED" rev-parse HEAD 2>/dev/null || true)"
  if [ "$_mr" != "$MASKED_REV" ]; then
    m1b_log "WARN: maskedthought HEAD=$_mr != pinned $MASKED_REV (참조용이므로 경고만)"
  fi
  unset _mr
fi

m1b_log "copy vendored data to shared datasets (if missing, via .tmp+rename)"
for _bn in gsm8k-train.jsonl gsm8k-test.jsonl; do
  if [ ! -f "$M1B_DATASETS/$_bn" ]; then
    cp "$M1B_DATA/$_bn" "$M1B_DATASETS/$_bn.tmp" && mv "$M1B_DATASETS/$_bn.tmp" "$M1B_DATASETS/$_bn"
  fi
done
unset _bn

m1b_log "HF pinned revisions (cache pre-check, download는 각 stage가 revision으로 수행)"
for _r in "unsloth/Llama-3.2-1B:$BASE_REVISION" "meta-llama/Llama-3.1-8B-Instruct:$TEACHER_REVISION"; do
  m1b_log "want ${_r}"
done
unset _r

m1b_log "HF login check (Llama gated)"

(set -x; "$HF_BIN" auth whoami 2>/dev/null) || echo "-> 'uv run hf auth login' required"

m1b_log "GPU check"
nvidia-smi -L 2>/dev/null || echo "-> no nvidia-smi, check on DGX OS"
(set -x; "$VENV_BIN/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())" 2>/dev/null || true)

m1b_log "llamacpp check (pinned $LLAMACPP_REV)"
if [ -d "${LLAMACPP:-$HOME/git/llama.cpp}/.git" ]; then
  _lr="$(git -C "${LLAMACPP:-$HOME/git/llama.cpp}" rev-parse HEAD 2>/dev/null || true)"
  if [ "$_lr" != "$LLAMACPP_REV" ]; then
    m1b_log "WARN: llama.cpp HEAD=$_lr != pinned $LLAMACPP_REV (GGUF 바이트 동일성 보장 안 됨)"
  fi
  unset _lr
else
  m1b_log "WARN: no llama.cpp checkout (GGUF 단계에서 필요)"
fi

m1b_log "OK. shared=$M1B_SHARED (compare ls on spark1/spark2)"

echo ---- result ----
(set -x; ls -lh "$M1B_DATASETS/")
