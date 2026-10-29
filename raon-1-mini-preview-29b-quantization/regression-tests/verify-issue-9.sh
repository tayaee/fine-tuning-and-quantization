#!/usr/bin/env bash
# verify-issue-9.sh — issue-9(배포 후 검증 게이트) 승인 기준 기계 검증
#   (s7-19 감사는 삭제됨 — 기동 검사는 s7-13이, 판정은 s7-13/16 로그가 직접 맡는다)
#   1. gates.inc 의 rq_gate_serving 존재 (휴면)
#   2. 단위 테스트 (tests/test_sdlc.py)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

grep -qF 'rq_gate_serving() {' "$ROOT/gates.inc" || { echo "FAIL: gates.inc에 rq_gate_serving 없음"; fail=1; }
grep -q 'SKIP' "$ROOT/gates.inc" || { echo "FAIL: gate-serving이 skip 처리를 하지 않음"; fail=1; }
[ -f "$ROOT/s7-19-verify-serving-log.py" ] && { echo "FAIL: 삭제된 s7-19-verify-serving-log.py 잔존"; fail=1; }
echo "OK: rq_gate_serving (s7-19 삭제됨)"

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
