#!/usr/bin/env bash
# verify-issue-20.sh — issue-20(카드 표의 Model 컬럼 = 정식 HF 모델명) 승인 기준 기계 검증
#   1. repo id 규칙이 업로드 스크립트(s5-01)와 동일하다
#   2. 카드의 Model 컬럼·제목·사용 예제가 repo id를 쓴다
#   3. 업로드 전 감사가 repo id 기준으로 검사한다
#   4. 단위 테스트
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

# 1. repo id 규칙 일치 (양쪽 다 "<HF_ID>/<BASE_SLUG>-<target>" + smoke 접미사)
for f in s4-65-update-readme.py s4-70-verify-upload.py; do
  grep -q 'def hf_repo_id(target: str, mode: str) -> str:' "$ROOT/$f" \
    || { echo "FAIL: $f 에 hf_repo_id() 없음"; fail=1; }
  grep -q 'BASE_SLUG = os.environ.get("BASE_MODEL"' "$ROOT/$f" \
    || { echo "FAIL: $f 의 BASE_SLUG 규칙이 업로드 스크립트와 다름"; fail=1; }
done
grep -q 'name += "-smoke"' "$ROOT/s4-65-update-readme.py" \
  || { echo "FAIL: smoke 접미사 규칙 없음"; fail=1; }
echo "OK: repo id 규칙(43/46/49 동일)"

# 2. 카드가 repo id를 쓴다
grep -q 'def render_readme(target: str' "$ROOT/s4-65-update-readme.py" \
  || { echo "FAIL: 카드 렌더가 대상 1개가 아님"; fail=1; }
grep -q 'def model_label(target: str, mode: str) -> str:' "$ROOT/s4-65-update-readme.py" \
  || { echo "FAIL: model_label() 없음"; fail=1; }
grep -q '# {repo_id}' "$ROOT/s4-65-update-readme.py" \
  || { echo "FAIL: 카드 제목이 repo id가 아님"; fail=1; }
grep -q 'model_id = "{repo_id}"' "$ROOT/s4-65-update-readme.py" \
  || { echo "FAIL: transformers 예제가 repo id를 안 씀"; fail=1; }
grep -q '{vllm_serve_launcher} {repo_id}' "$ROOT/s4-65-update-readme.py" \
  || { echo "FAIL: vLLM 예제가 repo id를 안 씀"; fail=1; }
echo "OK: 카드 표·제목·사용 예제 = repo id"

# 3. 업로드 전 감사가 repo id 기준
grep -q 'repo_id = hf_repo_id(target, mode)' "$ROOT/s4-70-verify-upload.py" \
  || { echo "FAIL: 46-verify가 repo id 기준으로 검사하지 않음"; fail=1; }
grep -q 'other_id = hf_repo_id(other, mode)' "$ROOT/s4-70-verify-upload.py" \
  || { echo "FAIL: 46-verify의 타깃 혼입 검사가 repo id 기준이 아님"; fail=1; }
echo "OK: 업로드 전 감사 = repo id 기준"

# 4. 단위 테스트
if "$PY" -c "import pytest" >/dev/null 2>&1; then
  "$PY" -m pytest "$ROOT/tests/test_prep_and_upload.py" "$ROOT/tests/test_verify_upload.py" -q >/tmp/verify-issue-20.log 2>&1 \
    || { echo "FAIL: 카드/감사 단위 테스트 실패"; tail -20 /tmp/verify-issue-20.log; fail=1; }
else
  "$PY" "$ROOT/tests/test_prep_and_upload.py" >/tmp/verify-issue-20.log 2>&1 \
    || { echo "FAIL: tests/test_prep_and_upload.py 실패"; grep -E '^FAIL' /tmp/verify-issue-20.log || true; fail=1; }
fi
echo "OK: 단위 테스트 (card + audit)"

exit $fail
