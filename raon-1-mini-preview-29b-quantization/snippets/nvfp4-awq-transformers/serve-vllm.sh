VLLM_REF="raon-1-mini-preview"
VLLM_USE_PRECOMPILED=1 uvx --from "vllm @ git+https://github.com/krafton-ai/vllm.git@${VLLM_REF}" \
  vllm serve __REPO_ID__ \
  --host 127.0.0.1 --port 8000 \
  --served-model-name __REPO_ID__ \
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
