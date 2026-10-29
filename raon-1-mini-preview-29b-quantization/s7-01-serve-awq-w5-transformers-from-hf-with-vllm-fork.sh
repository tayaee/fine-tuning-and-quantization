#!/usr/bin/env bash
# s7-01-serve-awq-w5-transformers-from-hf-with-vllm-fork.sh — README 모델카드(full)의 vLLM 명령을 그대로 실행한다 (production, awq-w5-transformers).
# 실행 내용은 snippets/awq-w5-transformers/serve-vllm.sh 와 동일하다 (카드에 박힌 명령 그 자체,
# repo id만 full로 고정). 카드 수정 없이 명령을 바꾸려면 스니펫을 고쳐라.
# 사용: ./s7-01-serve-awq-w5-transformers-from-hf-with-vllm-fork.sh  (Hub repo에서 내려받아 서빙, 첫 기동은 커널 컴파일로 ~10분)
exec bash <(sed "s|__REPO_ID__|tayaee/Raon-1-mini-preview-awq-w5-transformers|g" "$(dirname "$0")/snippets/awq-w5-transformers/serve-vllm.sh")
