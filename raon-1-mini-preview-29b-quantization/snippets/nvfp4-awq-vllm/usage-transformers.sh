mkdir -p /tmp/raon-1-mini-preview-nvfp4-awq-vllm && cd /tmp/raon-1-mini-preview-nvfp4-awq-vllm
uv venv --python 3.12 .venv && source .venv/bin/activate  # ignore any active venv
uv pip install "torch" --index-url https://download.pytorch.org/whl/cu130  # GB10/Blackwell; CPU-only hosts can skip the index-url
uv pip install "transformers==5.14.1" accelerate huggingface_hub safetensors
uvx --from huggingface_hub hf auth login  # the base repo is gated (approval is automatic)
uvx --from huggingface_hub hf download __REPO_ID__ nvfp4_to_bf16.py --local-dir .
