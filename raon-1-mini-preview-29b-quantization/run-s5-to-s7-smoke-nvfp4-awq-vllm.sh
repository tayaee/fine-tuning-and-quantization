#!/usr/bin/env bash
# run-s5-to-s7-smoke-nvfp4-awq-vllm.sh — s5~s7 단일 타깃 원샷(업로드→고객 인수).
# 전제: run-s1-to-s4-smoke-nvfp4-awq-vllm.sh 완료 (prep/verify 포함).
#   ./s5-01-upload-to-hf.sh smoke nvfp4-awq-vllm
#   (카드 Step 1 포크 빌드는 s7-02 안에서 카드의 블록 순서대로 실행된다)
#   SOURCE=hf ./s7-13-chat.sh smoke nvfp4-awq-vllm
#   ./s7-16-pi-tool-call-test.sh smoke nvfp4-awq-vllm
#
# 이 타깃은 transformers 로 적재 불가(modelopt_fp4) → s6(transformers) 대상이 아니다.
# 고객 인수 검증은 전부 s7(Hub 카드의 '## Usage - vLLM' 예제 → 서빙)에서 한다.
#   - 카드 Step 1(GB10 vLLM 포크 소스 빌드)은 s7-02 안에서 카드의 블록 순서대로 실행된다.
#   - SOURCE=hf: **Hub 에 게시된 repo** 로 서빙한다 (고객이 받는 경로).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
export ALLOW_MISSING_METRICS=1  # smoke는 지표 없이 배관만 검증
./s5-01-upload-to-hf.sh smoke nvfp4-awq-vllm

SOURCE=hf ./s7-13-chat.sh smoke nvfp4-awq-vllm
./s7-16-pi-tool-call-test.sh smoke nvfp4-awq-vllm
echo "run-s5-to-s7-smoke-nvfp4-awq-vllm done"
