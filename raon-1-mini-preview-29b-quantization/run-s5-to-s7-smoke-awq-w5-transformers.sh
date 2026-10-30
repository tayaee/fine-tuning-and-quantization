#!/usr/bin/env bash
# run-s5-to-s7-smoke-awq-w5-transformers.sh — s5~s7 단일 타깃 원샷(업로드→고객 인수).
# 전제: run-s1-to-s4-smoke-awq-w5-transformers.sh 완료 (prep/verify 포함).
#   ./s5-00-upload-to-hf.sh smoke awq-w5-transformers
#   ./s6-00-clean-serving-workspace.sh smoke
#   ./s6-10-verify-readme-with-awq-w5-transformers-and-transformers.sh smoke
#   ./s7-30-chat.sh smoke awq-w5-transformers
#   ./s7-35-pi-tool-call-test.sh smoke awq-w5-transformers
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
ALLOW_MISSING_METRICS=1  # smoke는 지표 없이 배관만 검증
./s5-00-upload-to-hf.sh smoke awq-w5-transformers
./s6-00-clean-serving-workspace.sh smoke
./s6-10-verify-readme-with-awq-w5-transformers-and-transformers.sh smoke
./s7-30-chat.sh smoke awq-w5-transformers
./s7-35-pi-tool-call-test.sh smoke awq-w5-transformers
echo "run-s5-to-s7-smoke-awq-w5-transformers done"
