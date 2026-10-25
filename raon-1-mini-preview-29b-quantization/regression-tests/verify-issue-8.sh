#!/usr/bin/env bash
# verify-issue-8.sh — issue-8(단계 원장 + 릴리스 노트) 승인 기준 기계 검증
#   1. sdlc.py 원장 API(락·타깃 집계) + CLI
#   2. common.env 의 rq_stage_record/rq_stage_target/rq_stage_of
#   3. run-*.sh / build-ppl-eval.inc 가 단계마다 원장 기록
#   4. s8-61-release-notes.py 존재 + 릴리스 노트 생성
#   5. 단위 테스트 (tests/test_sdlc.py)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

for tok in 'def update_ledger(' 'def read_ledger(' 'fcntl.LOCK_EX' 'stage-ledger-' \
           'def _cmd_ledger(' 'def _cmd_show(' 'targets'; do
  grep -qF -- "$tok" "$ROOT/sdlc.py" || { echo "FAIL: sdlc.py에 '$tok' 없음"; fail=1; }
done
for tok in 'rq_stage_record()' 'rq_stage_target()' 'rq_stage_of()'; do
  grep -qF -- "$tok" "$ROOT/config/common.env" || { echo "FAIL: common.env에 '$tok' 없음"; fail=1; }
done
for f in run-smoke.sh run-full.sh build-ppl-eval.inc; do
  grep -q 'rq_stage_record' "$ROOT/$f" || { echo "FAIL: $f 가 원장을 기록하지 않음"; fail=1; }
done
[ -f "$ROOT/s8-61-release-notes.py" ] || { echo "FAIL: s8-61-release-notes.py 없음"; fail=1; }
for tok in 'release-notes-' '수용 기준' 'def render(' 'def _stage_rows(' 'def _metric_rows('; do
  grep -qF -- "$tok" "$ROOT/s8-61-release-notes.py" || { echo "FAIL: s8-61-release-notes.py에 '$tok' 없음"; fail=1; }
done
echo "OK: 원장 API + 파이프라인 기록 + s8-61-release-notes.py"

if "$PY" -c "import pytest" >/dev/null 2>&1; then
  if "$PY" -m pytest "$ROOT/tests/test_sdlc.py" -q >/tmp/verify-issue-8.log 2>&1; then
    echo "OK: tests/test_sdlc.py ($(grep -oE '[0-9]+ passed' /tmp/verify-issue-8.log | head -1))"
  else
    echo "FAIL: 단위 테스트 실패"; tail -20 /tmp/verify-issue-8.log; fail=1
  fi
else
  if "$PY" "$ROOT/tests/test_sdlc.py" >/tmp/verify-issue-8.log 2>&1; then
    echo "OK: tests/test_sdlc.py ($(grep -c '^PASS' /tmp/verify-issue-8.log) passed)"
  else
    echo "FAIL: 단위 테스트 실패"; grep -E '^FAIL' /tmp/verify-issue-8.log || true; fail=1
  fi
fi

exit $fail
