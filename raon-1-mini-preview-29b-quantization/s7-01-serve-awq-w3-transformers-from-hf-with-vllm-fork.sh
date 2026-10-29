#!/usr/bin/env bash
# s7-01-serve-awq-w3-with-vllm-fork-from-hf-with-vllm-fork.sh — README 모델카드(full)의 vLLM 명령을 그대로 실행한다 (production, awq-w3-transformers).
# 사용: ./s7-01-serve-awq-w3-with-vllm-fork-from-hf-with-vllm-fork.sh  (Hub repo에서 내려받아 서빙, 첫 기동은 커널 컴파일로 ~10분)
VLLM_REF="raon-1-mini-preview"
VLLM_USE_PRECOMPILED=1 uvx --from "vllm @ git+https://github.com/krafton-ai/vllm.git@${VLLM_REF}" \
  vllm serve tayaee/Raon-1-mini-preview-awq-w3-transformers \
  --host 127.0.0.1 --port 8000 \
  --served-model-name tayaee/Raon-1-mini-preview-awq-w3-transformers \
  --tensor-parallel-size 1 \
  --quantization compressed-tensors \
  --max-model-len 262144 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --reasoning-parser raon_1 \
  --tool-call-parser raon_1 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking": true, "truncate_history_thinking": false}'
