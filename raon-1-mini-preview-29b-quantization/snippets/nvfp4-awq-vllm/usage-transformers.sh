mkdir -p /tmp/raon-1-mini-preview-nvfp4-awq-transformers && cd /tmp/raon-1-mini-preview-nvfp4-awq-transformers
uv venv --python 3.12 .venv && source .venv/bin/activate  # ignore any active venv
uv pip install "torch" --index-url https://download.pytorch.org/whl/cu130  # GB10/Blackwell; CPU-only hosts can skip the index-url
uv pip install "transformers==5.14.1" "nvidia-modelopt" accelerate huggingface_hub
uvx --from huggingface_hub hf auth login  # the base repo is gated (approval is automatic)
