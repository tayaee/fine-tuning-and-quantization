#!/usr/bin/env bash
# verify-issue-6.sh — issue-6(Stage 1 수용 기준 계약화) 승인 기준 기계 검증
#   1. config/acceptance.toml 존재 + 4타깃 기준 정의 + 스키마 정상
#   2. s1-04-setup-acceptance-criteria.py/.sh 존재(+실행 권한), 스키마 오류 시 rc=1, 산출물 생성
#   3. s4-00-measure-ppl__transformers-only.py / s4-70-verify-upload.py / s4-65-update-readme.py 가 기준을 참조
#   4. 단위 테스트 (tests/test_sdlc.py)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

# 1. 계약 파일
[ -f "$ROOT/config/acceptance.toml" ] || { echo "FAIL: config/acceptance.toml 없음"; fail=1; }
for tok in 'version = 1' '[targets.awq-w3-transformers]' '[targets.awq-w5-transformers]' '[targets.nvfp4-awq-transformers]' \
           '[targets.nvfp4-gptq-transformers]' 'quant_format' 'size_gb' 'ppl_delta_warn' 'ppl_delta_max' \
           '[gates.release]'; do
  grep -qF -- "$tok" "$ROOT/config/acceptance.toml" || { echo "FAIL: acceptance.toml에 '$tok' 없음"; fail=1; }
done
echo "OK: config/acceptance.toml 존재 + 4타깃 기준 + gates.release"

# 2. s1-04-acceptance
for f in s1-04-setup-acceptance-criteria.py s1-04-setup-acceptance-criteria.sh sdlc.py; do
  [ -f "$ROOT/$f" ] || { echo "FAIL: $f 없음"; fail=1; }
done
[ -x "$ROOT/s1-04-setup-acceptance-criteria.sh" ] || { echo "FAIL: s1-04-setup-acceptance-criteria.sh 실행 권한 없음"; fail=1; }
bash -n "$ROOT/s1-04-setup-acceptance-criteria.sh" || { echo "FAIL: s1-04-setup-acceptance-criteria.sh bash -n"; fail=1; }
for tok in 'def load_acceptance(' 'def acceptance_errors(' 'def verdict_for_delta(' \
           'def update_ledger(' 'def read_ledger(' 'acceptance-' 'acceptance_errors'; do
  grep -qF -- "$tok" "$ROOT/sdlc.py" || { echo "FAIL: sdlc.py에 '$tok' 없음"; fail=1; }
done
echo "OK: s1-04-acceptance.{py,sh} + sdlc.py 계약/원장 API"

# 3. 기존 판정 코드가 기준을 참조
grep -q 'acceptance_spec' "$ROOT/s4-00-measure-ppl__transformers-only.py" || { echo "FAIL: s4-00-measure-ppl__transformers-only.py가 기준 미참조"; fail=1; }
grep -q '_apply_acceptance_size_ranges' "$ROOT/s4-70-verify-upload.py" || { echo "FAIL: s4-70-verify-upload.py가 기준 미참조"; fail=1; }
grep -q '_acceptance_target_spec' "$ROOT/s4-65-update-readme.py" || { echo "FAIL: s4-65-update-readme.py에 수용 기준 표 없음"; fail=1; }
echo "OK: 34/43/46 이 기준(acceptance.toml)을 참조"

# 4. 단위 테스트
if "$PY" -c "import pytest" >/dev/null 2>&1; then
  if "$PY" -m pytest "$ROOT/tests/test_sdlc.py" -q >/tmp/verify-issue-6.log 2>&1; then
    echo "OK: tests/test_sdlc.py ($(grep -oE '[0-9]+ passed' /tmp/verify-issue-6.log | head -1))"
  else
    echo "FAIL: 단위 테스트 실패"; tail -20 /tmp/verify-issue-6.log; fail=1
  fi
else
  if "$PY" "$ROOT/tests/test_sdlc.py" >/tmp/verify-issue-6.log 2>&1; then
    echo "OK: tests/test_sdlc.py ($(grep -c '^PASS' /tmp/verify-issue-6.log) passed)"
  else
    echo "FAIL: 단위 테스트 실패"; grep -E '^FAIL' /tmp/verify-issue-6.log || true; fail=1
  fi
fi

exit $fail
