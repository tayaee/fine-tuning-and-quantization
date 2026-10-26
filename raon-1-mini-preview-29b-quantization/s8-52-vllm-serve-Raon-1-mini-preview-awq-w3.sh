#!/usr/bin/env bash
# s8-52-vllm-serve-Raon-1-mini-preview-awq-w3.sh — README 모델카드(full)의 vLLM 명령을 그대로 실행한다.
# 사용: ./s8-52-vllm-serve-Raon-1-mini-preview-awq-w3.sh  (Hub repo에서 내려받아 서빙, 첫 기동은 커널 컴파일로 ~10분)
set -euo pipefail
VLLM_REF="raon-1-mini-preview"
VLLM_USE_PRECOMPILED=1 uvx --from "vllm @ git+https://github.com/krafton-ai/vllm.git@${VLLM_REF}" \
  vllm serve tayaee/Raon-1-mini-preview-awq-w3 \
  --host 127.0.0.1 --port 8000 \
  --served-model-name tayaee/Raon-1-mini-preview-awq-w3 \
  --tensor-parallel-size 1 \
  --quantization compressed-tensors \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code
