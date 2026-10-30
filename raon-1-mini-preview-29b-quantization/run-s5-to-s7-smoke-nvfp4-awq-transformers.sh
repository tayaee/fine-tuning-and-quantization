#!/usr/bin/env bash
# run-s5-to-s7-smoke-nvfp4-awq-transformers.sh — s5~s7 단일 타깃 원샷(업로드→고객 인수).
# 전제: run-s1-to-s4-smoke-nvfp4-awq-transformers.sh 완료 (prep/verify 포함).
#   ./s5-00-upload-to-hf.sh smoke nvfp4-awq-transformers
#   ./s6-00-clean-serving-workspace.sh smoke
#   ./s6-15-verify-readme-with-nvfp4-awq-transformers-and-transformers.sh smoke
#   ./s7-30-chat.sh smoke nvfp4-awq-transformers
#   ./s7-35-pi-tool-call-test.sh smoke nvfp4-awq-transformers
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
ALLOW_MISSING_METRICS=1  # smoke는 지표 없이 배관만 검증
./s5-00-upload-to-hf.sh smoke nvfp4-awq-transformers
./s6-00-clean-serving-workspace.sh smoke
./s6-15-verify-readme-with-nvfp4-awq-transformers-and-transformers.sh smoke
./s7-30-chat.sh smoke nvfp4-awq-transformers
./s7-35-pi-tool-call-test.sh smoke nvfp4-awq-transformers
echo "run-s5-to-s7-smoke-nvfp4-awq-transformers done"
