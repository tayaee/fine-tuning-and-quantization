#!/usr/bin/env bash
# run-s5-to-s7-full-nvfp4-awq-transformers.sh — s5~s7 단일 타깃 원샷(업로드→고객 인수).
# 전제: run-s1-to-s4-full-nvfp4-awq-transformers.sh 완료 (prep/verify 포함).
#   ./s5-01-upload-hf.sh full nvfp4-awq-transformers
#   ./s6-01-test-nvfp4-awq-transformers.sh full
#   ./s6-13-chat.sh full nvfp4-awq-transformers
#   ./s6-16-pi-tool-call-test.sh full nvfp4-awq-transformers
#   ./s7-13-chat.sh full nvfp4-awq-transformers
#   ./s7-16-pi-tool-call-test.sh full nvfp4-awq-transformers
#   ./s7-19-verify-serving-log.sh full
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
: "${ALLOW_MISSING_METRICS:=0}"
./s5-01-upload-hf.sh full nvfp4-awq-transformers
./s6-01-test-nvfp4-awq-transformers.sh full
./s6-13-chat.sh full nvfp4-awq-transformers
./s6-16-pi-tool-call-test.sh full nvfp4-awq-transformers
./s7-13-chat.sh full nvfp4-awq-transformers
./s7-16-pi-tool-call-test.sh full nvfp4-awq-transformers
./s7-19-verify-serving-log.sh full
echo "run-s5-to-s7-full-nvfp4-awq-transformers done"
