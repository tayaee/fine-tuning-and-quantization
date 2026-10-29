#!/usr/bin/env bash
# run-s5-to-s7-full-awq-w5-transformers.sh — s5~s7 단일 타깃 원샷(업로드→고객 인수).
# 전제: run-s1-to-s4-full-awq-w5-transformers.sh 완료 (prep/verify 포함).
#   ./s5-01-upload-to-hf.sh full awq-w5-transformers
#   ./s6-00-clean-serving-workspace.sh full
#   ./s6-02-verify-readme-with-awq-w5-transformers-and-transformers.sh full
#   ./s7-13-chat.sh full awq-w5-transformers
#   ./s7-16-pi-tool-call-test.sh full awq-w5-transformers
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
: "${ALLOW_MISSING_METRICS:=0}"
./s5-01-upload-to-hf.sh full awq-w5-transformers
./s6-00-clean-serving-workspace.sh full
./s6-02-verify-readme-with-awq-w5-transformers-and-transformers.sh full
./s7-13-chat.sh full awq-w5-transformers
./s7-16-pi-tool-call-test.sh full awq-w5-transformers
echo "run-s5-to-s7-full-awq-w5-transformers done"
