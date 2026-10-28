#!/usr/bin/env bash
# run-s5-to-s7-smoke-nvfp4-gptq-transformers.sh — s5~s7 단일 타깃 원샷(업로드→고객 인수).
# 전제: run-s1-to-s4-smoke-nvfp4-gptq-transformers.sh 완료 (prep/verify 포함).
#   ./s5-01-upload-hf.sh smoke nvfp4-gptq-transformers
#   ./s6-01-test-nvfp4-gptq-transformers.sh smoke
#   ./s6-13-chat.sh smoke nvfp4-gptq-transformers
#   ./s6-16-pi-tool-call-test.sh smoke nvfp4-gptq-transformers
#   ./s7-13-chat.sh smoke nvfp4-gptq-transformers
#   ./s7-16-pi-tool-call-test.sh smoke nvfp4-gptq-transformers
#   ./s7-19-verify-serving-log.sh smoke
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
ALLOW_MISSING_METRICS=1  # smoke는 지표 없이 배관만 검증
./s5-01-upload-hf.sh smoke nvfp4-gptq-transformers
./s6-01-test-nvfp4-gptq-transformers.sh smoke
./s6-13-chat.sh smoke nvfp4-gptq-transformers
./s6-16-pi-tool-call-test.sh smoke nvfp4-gptq-transformers
./s7-13-chat.sh smoke nvfp4-gptq-transformers
./s7-16-pi-tool-call-test.sh smoke nvfp4-gptq-transformers
./s7-19-verify-serving-log.sh smoke
echo "run-s5-to-s7-smoke-nvfp4-gptq-transformers done"
