#!/usr/bin/env bash
# s7-01-serve-nvfp4-awq-vllm-with-vllm-fork-from-hf-with-vllm-fork.sh — README 모델카드(full)의 vLLM 명령을 그대로 실행한다 (production, nvfp4-awq-vllm).
# ModelOpt NVFP4는 --quantization 없이 자동 감지된다 (compressed-tensors 지정 금지).
# 사용: ./s7-01-serve-nvfp4-awq-vllm-with-vllm-fork-from-hf-with-vllm-fork.sh  (Hub repo에서 내려받아 서빙, 첫 기동은 커널 컴파일로 ~10분)
# GB10: VLLM_REF에 gb10-support 패치가 있어야 한다. KRAFTON 원본에는 sm_121이 없어서
# vsplit에서 죽는다. 당장은 로컬 빌드(~/git/raon-vllm-spark-gb10-support/scripts/gb10/serve-tayaee-nvfp4.sh)로
# 검증하고, 포크에 머지된 뒤 이 스크립트의 VLLM_REF를 올린다.
VLLM_REF="raon-1-mini-preview"
VLLM_USE_PRECOMPILED=1 uvx --from "vllm @ git+https://github.com/krafton-ai/vllm.git@${VLLM_REF}" \
  vllm serve tayaee/Raon-1-mini-preview-nvfp4-awq-vllm \
  --host 127.0.0.1 --port 8000 \
  --served-model-name tayaee/Raon-1-mini-preview-nvfp4-awq-vllm \
  --tensor-parallel-size 1 \
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
