#!/usr/bin/env bash
# s6-02-verify-readme-with-awq-w3-transformers-and-transformers.sh [smoke|full] — Hub README "Usage - transformers"를 바이트 그대로 실행 (awq-w3-transformers).
# Hub에서 README.md를 받아 셋업 bash 블록 → app.py 저장 → `python app.py` 를 env -i 고객 셸에서
# 원문 그대로 실행하고, 비어있지 않은 Answer로 판정한다 (config/s6-customer.inc: s6_readme_verify).
# 블록은 snippets/awq-w3-transformers/ 와 바이트 비교한다 (다르면 카드 드리프트 → 실행 없이 FAIL).
# 사용: ./s6-00-clean-serving-workspace.sh && ./s6-02-verify-readme-with-awq-w3-transformers-and-transformers.sh full
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
source "$RQ_ROOT/config/s6-customer.inc"
rq_repro_echo
rq_banner "s6-02-verify-readme-with-awq-w3-transformers-and-transformers" "Stage 6 - run the Hub README transformers usage verbatim (awq-w3-transformers)"
TARGET="awq-w3-transformers"
rc=0; s6_readme_verify "$TARGET" || rc=$?
echo ---- result ----
rq_log "s6-02 $([ "$rc" -eq 0 ] && echo OK || echo FAIL) target=$TARGET (see $RQ_OUTPUTS/s6-$MODE/$TARGET.json)"
exit "$rc"
