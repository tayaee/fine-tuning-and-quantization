#!/usr/bin/env bash
# s7-02-verify-readme-with-awq-w3-transformers-and-vllm-fork.sh [smoke|full] — Hub 카드의 '## Usage - vLLM' 코드 예제를 as-is 그대로 실행한다 (awq-w3-transformers).
#
# 검증 순서:
#   1) Hub README 의 '## Usage - vLLM' 블록을 snippets/ 와 바이트 비교 (drift 이면 FAIL).
#   2) Hub 카드에서 뽑은 블록을 README 순서대로 실행 (nvfp4-awq-vllm 은 Step1 빌드 → Step2 서빙).
#      마지막 블록(서빙)은 상주 프로세스라 여기서 끝나지 않는다 — 실제 응답 검증은
#      s7-13-chat.sh(채팅) / s7-16-pi-tool-call-test.sh(툴콜)가 맡는다.
#
# 사용: ./s7-02-verify-readme-with-awq-w3-transformers-and-vllm-fork.sh full
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-full}}"
source "$RQ_ROOT/config/s7-serve.inc"
s7_serve_frozen "awq-w3-transformers" "$MODE"
