#!/usr/bin/env bash
# run-s5-to-s7-smoke-nvfp4-awq-vllm.sh — s5~s7 단일 타깃 원샷(업로드→고객 인수).
# 전제: run-s1-to-s4-smoke-nvfp4-awq-vllm.sh 완료 (prep/verify 포함).
#   ./s5-01-upload-to-hf.sh smoke nvfp4-awq-vllm
#   ./s6-02-verify-nvfp4-awq-vllm-with-transformers.sh smoke
#   ./s6-13-chat.sh smoke nvfp4-awq-vllm
#   ./s6-16-pi-tool-call-test.sh smoke nvfp4-awq-vllm
#   ./s7-13-chat.sh smoke nvfp4-awq-vllm
#   ./s7-16-pi-tool-call-test.sh smoke nvfp4-awq-vllm
#   set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
export ALLOW_MISSING_METRICS=1  # smoke는 지표 없이 배관만 검증
# GB10 로컬 포크 빌드가 있으면 s7-13이 krafton 대신 로컬로 서빙
[ -x "$HOME/git/raon-vllm-spark-gb10-support/.venv/bin/vllm" ] && export FORK_ROOT="$HOME/git/raon-vllm-spark-gb10-support"
./s5-01-upload-to-hf.sh smoke nvfp4-awq-vllm
./s6-02-verify-nvfp4-awq-vllm-with-transformers.sh smoke
./s6-13-chat.sh smoke nvfp4-awq-vllm
./s6-16-pi-tool-call-test.sh smoke nvfp4-awq-vllm
./s7-13-chat.sh smoke nvfp4-awq-vllm
./s7-16-pi-tool-call-test.sh smoke nvfp4-awq-vllm
echo "run-s5-to-s7-smoke-nvfp4-awq-vllm done"
