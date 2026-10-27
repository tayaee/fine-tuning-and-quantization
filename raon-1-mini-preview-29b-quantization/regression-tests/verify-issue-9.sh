#!/usr/bin/env bash
# verify-issue-9.sh — issue-9(배포 후 검증 게이트) 승인 기준 기계 검증
#   1. s7-19-verify-serving-log.py 존재 + 로그 판정 + 리포트 생성
#   2. gates.inc 의 rq_gate_serving(로그 미수행은 skip)
#   3. 단위 테스트 (tests/test_sdlc.py)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

[ -f "$ROOT/s7-19-verify-serving-log.py" ] || { echo "FAIL: s7-19-verify-serving-log.py 없음"; fail=1; }
for tok in 'SERVE-FAIL' 'INFER-FAIL' 'serve-' 'def inspect_log(' 'def expected_slug('; do
  grep -qF -- "$tok" "$ROOT/s7-19-verify-serving-log.py" || { echo "FAIL: s7-19-verify-serving-log.py에 '$tok' 없음"; fail=1; }
done
grep -qF 'rq_gate_serving() {' "$ROOT/gates.inc" || { echo "FAIL: gates.inc에 rq_gate_serving 없음"; fail=1; }
grep -q 'SKIP' "$ROOT/gates.inc" || { echo "FAIL: gate-serving이 skip 처리를 하지 않음"; fail=1; }
python3 -c "import ast;ast.parse(open('$ROOT/s7-19-verify-serving-log.py').read())" || { echo "FAIL: s7-19-verify-serving-log.py 문법 오류"; fail=1; }
echo "OK: s7-19-verify-serving-log.py + rq_gate_serving"

if "$PY" -c "import pytest" >/dev/null 2>&1; then
  if "$PY" -m pytest "$ROOT/tests/test_sdlc.py" -q >/tmp/verify-issue-9.log 2>&1; then
    echo "OK: tests/test_sdlc.py ($(grep -oE '[0-9]+ passed' /tmp/verify-issue-9.log | head -1))"
  else
    echo "FAIL: 단위 테스트 실패"; tail -20 /tmp/verify-issue-9.log; fail=1
  fi
else
  if "$PY" "$ROOT/tests/test_sdlc.py" >/tmp/verify-issue-9.log 2>&1; then
    echo "OK: tests/test_sdlc.py ($(grep -c '^PASS' /tmp/verify-issue-9.log) passed)"
  else
    echo "FAIL: 단위 테스트 실패"; grep -E '^FAIL' /tmp/verify-issue-9.log || true; fail=1
  fi
fi

exit $fail
