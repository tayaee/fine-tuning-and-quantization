#!/usr/bin/env bash
# verify-issue-19.sh — issue-19(모델 카드 = base+대상 1개 / 전체 버전 비교는 요약 리포트) 승인 기준 기계 검증
#   1. 모델 카드 렌더링이 base + 대상 1개만 표에 실린다
#   2. 요약 리포트 models/README-<mode>.md 가 생성되고 전체 버전을 담는다
#   3. 업로드 전 감사(46-verify)가 "다른 버전 행 금지"를 검사한다
#   4. 단위 테스트
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

# 1. 카드의 표 행은 base + 대상만
grep -q 'def render_readme(target: str' "$ROOT/s4-65-update-readme.py" \
  || { echo "FAIL: 모델 카드 렌더가 대상 1개가 아님"; fail=1; }
grep -q 'def render_summary' "$ROOT/s4-65-update-readme.py" \
  || { echo "FAIL: 전체 버전 비교 리포트(render_summary) 없음"; fail=1; }
grep -q 'def summary_path' "$ROOT/s4-65-update-readme.py" \
  || { echo "FAIL: 요약 리포트 경로 헬퍼(summary_path) 없음"; fail=1; }
echo "OK: 카드(대상 1개) + 요약 리포트 분리"

# 2. 요약 경로가 models/README-<mode>.md
grep -q 'README-{mode}.md' "$ROOT/s4-65-update-readme.py" \
  || { echo "FAIL: 요약 리포트가 models/README-<mode>.md 가 아님"; fail=1; }
grep -q 'summary_path(mode)' "$ROOT/s4-65-update-readme.py" \
  || { echo "FAIL: prepare_target가 요약 리포트를 쓰지 않음"; fail=1; }
echo "OK: 요약 경로 models/README-<mode>.md"

# 3. 업로드 전 감사 가드
grep -q '다른 양자화 버전 행' "$ROOT/s4-70-verify-upload.py" \
  || { echo "FAIL: 46-verify에 '다른 버전 행 금지' 검사 없음"; fail=1; }
grep -q 'README-{mode}.md' "$ROOT/s4-70-verify-upload.py" \
  || { echo "FAIL: 46-verify가 요약 리포트 존재를 확인하지 않음"; fail=1; }
echo "OK: 업로드 전 감사 가드"

# 4. 단위 테스트
if "$PY" -c "import pytest" >/dev/null 2>&1; then
  "$PY" -m pytest "$ROOT/tests/test_prep_and_upload.py" -q >/tmp/verify-issue-19.log 2>&1 \
    || { echo "FAIL: tests/test_prep_and_upload.py 실패"; tail -20 /tmp/verify-issue-19.log; fail=1; }
else
  "$PY" "$ROOT/tests/test_prep_and_upload.py" >/tmp/verify-issue-19.log 2>&1 \
    || { echo "FAIL: tests/test_prep_and_upload.py 실패"; grep -E '^FAIL' /tmp/verify-issue-19.log || true; fail=1; }
fi
echo "OK: 단위 테스트 (test_prep_and_upload)"

exit $fail
