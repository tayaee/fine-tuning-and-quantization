#!/usr/bin/env bash
# verify-issue-10.sh — issue-10(문서 정렬) 승인 기준 기계 검증
#   1. docs/SDLC.md 존재 + 8개 그룹/게이트/멱등성 표
#   2. PROCEDURE.md / README.md / cheatsheet.md 에 새 스크립트 반영
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0

[ -f "$ROOT/docs/SDLC.md" ] || { echo "FAIL: docs/SDLC.md 없음"; fail=1; }
for tok in '8개 그룹' 'Stage 1' 'rq_gate_quant' 'acceptance.toml' '멱등' 'stage-ledger' \
           'release-notes' 'issue'; do
  grep -qF -- "$tok" "$ROOT/docs/SDLC.md" || { echo "FAIL: docs/SDLC.md에 '$tok' 없음"; fail=1; }
done
echo "OK: docs/SDLC.md"

for tok in 's1-04-acceptance' '8개 그룹' 'gate-quant'; do
  grep -qF -- "$tok" "$ROOT/PROCEDURE.md" || { echo "FAIL: PROCEDURE.md에 '$tok' 없음"; fail=1; }
done
for tok in 's1-04-acceptance' 'docs/SDLC.md'; do
  grep -qF -- "$tok" "$ROOT/README.md" || { echo "FAIL: README.md에 '$tok' 없음"; fail=1; }
done
for tok in 's1-04-acceptance' 's4-37-release-notes'; do
  grep -qF -- "$tok" "$ROOT/cheatsheet.md" || { echo "FAIL: cheatsheet.md에 '$tok' 없음"; fail=1; }
done
echo "OK: PROCEDURE/README/cheatsheet 정렬"

exit $fail
