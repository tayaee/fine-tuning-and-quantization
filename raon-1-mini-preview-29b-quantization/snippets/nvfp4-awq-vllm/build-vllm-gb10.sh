mkdir -p /tmp/raon-1-mini-preview-nvfp4-awq-vllm && cd /tmp/raon-1-mini-preview-nvfp4-awq-vllm
VENV=/tmp/raon-1-mini-preview-nvfp4-awq-vllm/.venv
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$VENV/bin/python" setuptools wheel packaging ninja cmake setuptools_rust setuptools_scm
MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="12.1" uv pip install --python "$VENV/bin/python" --no-build-isolation "vllm @ git+https://github.com/tayaee/raon-vllm-spark.git@gb10-support"
"$VENV/bin/python" -c "import vllm; print('vllm', vllm.__version__)"
