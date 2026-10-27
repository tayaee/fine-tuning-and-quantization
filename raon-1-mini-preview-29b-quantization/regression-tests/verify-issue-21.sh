#!/usr/bin/env bash
# verify-issue-21.sh — issue-21(일부 벤치 재생성 시 GSM8K 응답 소실 + PPL delta 3자리) 승인 기준 기계 검증
#   1. s4-07-eval.py가 이번에 재생성하지 않는 벤치의 기존 응답을 <target>.jsonl에 보존한다
#   2. s4-07-eval-base/s4-10-eval-quantized의 freshness가 "요청 벤치 응답 존재"까지 본다
#   3. s4-31-prep-upload.py가 GSM8K 정확도 없는 카드를 거부하고, PPL delta를 3자리로 쓴다
#   4. 단위 테스트
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

# 1. 응답 보존
grep -q 'def preserved_rows(main_path: str, regenerated: set\[str\]) -> list\[dict\]:' "$ROOT/s4-07-eval.py" \
  || { echo "FAIL: s4-07-eval.py에 preserved_rows() 없음"; fail=1; }
grep -q 'kept = preserved_rows(main_path' "$ROOT/s4-07-eval.py" \
  || { echo "FAIL: run_target이 기존 벤치 응답을 보존하지 않음"; fail=1; }
echo "OK: 일부 벤치 재생성 시 다른 벤치 응답 보존"

# 2. freshness = 요청 벤치 응답 존재 (base/quantized 양쪽)
for _sh in s4-07-eval-base.sh s4-10-eval-quantized.sh; do
  grep -q '_missing_bench' "$ROOT/$_sh" \
    || { echo "FAIL: $_sh freshness가 벤치 존재를 검사하지 않음"; fail=1; }
  grep -q 'has no rows for bench(es)' "$ROOT/$_sh" \
    || { echo "FAIL: $_sh 벤치 누락 시 stale 로그 없음"; fail=1; }
  bash -n "$ROOT/$_sh" || { echo "FAIL: $_sh 문법 오류"; fail=1; }
done
echo "OK: freshness가 요청 벤치 응답 존재까지 확인"

# 3. 카드 가드 + PPL delta 3자리
grep -q 'no GSM8K accuracy for' "$ROOT/s4-31-prep-upload.py" \
  || { echo "FAIL: prep-upload가 GSM8K 정확도 누락을 거부하지 않음"; fail=1; }
grep -q "_fmt(delta, 3)" "$ROOT/s4-31-prep-upload.py" \
  || { echo "FAIL: 수용 기준 표의 PPL delta가 3자리가 아님"; fail=1; }
grep -q "_fmt(pr.get('delta'), 3)" "$ROOT/s4-31-prep-upload.py" \
  || { echo "FAIL: Check 1 표의 PPL delta가 3자리가 아님"; fail=1; }
echo "OK: GSM8K 누락 거부 · PPL delta 3자리"

# 4. 단위 테스트
if "$PY" -c "import pytest" >/dev/null 2>&1; then
  "$PY" -m pytest "$ROOT/tests/test_ppl_and_eval.py" "$ROOT/tests/test_prep_and_upload.py" \
    "$ROOT/tests/test_verify_upload.py" -q >/tmp/verify-issue-21.log 2>&1 \
    || { echo "FAIL: eval/카드/감사 단위 테스트 실패"; tail -20 /tmp/verify-issue-21.log; fail=1; }
  echo "OK: 단위 테스트 (eval + card + audit)"
else
  echo "SKIP: pytest 없음"
fi

exit $fail
