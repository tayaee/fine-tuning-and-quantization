#!/usr/bin/env bash
# run-s5-to-s7-full-nvfp4-gptq-transformers.sh — s5~s7 단일 타깃 원샷(업로드→고객 인수).
# 전제: run-s1-to-s4-full-nvfp4-gptq-transformers.sh 완료 (prep/verify 포함).
#   ./s5-00-upload-to-hf.sh full nvfp4-gptq-transformers
#   ./s6-00-clean-serving-workspace.sh full
#   ./s6-20-verify-readme-with-nvfp4-gptq-transformers-and-transformers.sh full
#   ./s7-30-chat.sh full nvfp4-gptq-transformers
#   ./s7-35-pi-tool-call-test.sh full nvfp4-gptq-transformers
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
: "${ALLOW_MISSING_METRICS:=0}"
./s5-00-upload-to-hf.sh full nvfp4-gptq-transformers
./s6-00-clean-serving-workspace.sh full
./s6-20-verify-readme-with-nvfp4-gptq-transformers-and-transformers.sh full
./s7-30-chat.sh full nvfp4-gptq-transformers
./s7-35-pi-tool-call-test.sh full nvfp4-gptq-transformers
echo "run-s5-to-s7-full-nvfp4-gptq-transformers done"
