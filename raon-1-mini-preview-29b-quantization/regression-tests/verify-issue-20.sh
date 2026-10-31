#!/usr/bin/env bash
# verify-issue-20.sh — issue-20(카드 표의 Model 컬럼 = 정식 HF 모델명) 승인 기준 기계 검증
#   1. repo id 규칙이 업로드 스크립트(s5-00)와 동일하다
#   2. 카드의 Model 컬럼·제목·사용 예제가 repo id를 쓴다
#   3. 업로드 전 감사가 repo id 기준으로 검사한다
#   4. 단위 테스트
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

# 1. repo id 규칙 일치 (단일 소스 sdlc.hf_repo_id — 카드·감사·업로드 3곳이 공유)
#    이전에는 각 스크립트가 자기 사본을 들고 있었고, vLLM 전용 타깃의 slug 가 어긋났다.
grep -q 'def hf_repo_id(target: str, mode: str = "full") -> str:' "$ROOT/sdlc.py" \
  || { echo "FAIL: sdlc.py 에 단일 소스 hf_repo_id() 없음"; fail=1; }
grep -q 'name += "-smoke"' "$ROOT/sdlc.py" \
  || { echo "FAIL: smoke 접미사 규칙 없음"; fail=1; }
grep -q 'Raon-1-mini-preview-NVFP4-AWQ-GB10-vLLM' "$ROOT/sdlc.py" \
  || { echo "FAIL: vLLM 전용 타깃의 정식 slug 가 sdlc.py 에 없음"; fail=1; }
for f in s4-70-update-readme.py s4-75-verify-upload.py; do
  grep -q 'from sdlc import hf_repo_id as _hf_repo_id' "$ROOT/$f" \
    || { echo "FAIL: $f 가 sdlc.hf_repo_id 를 쓰지 않는다 (규칙 분산)"; fail=1; }
done
grep -q 'from sdlc import hf_repo_id' "$ROOT/s5-00-upload-to-hf.py" \
  || { echo "FAIL: s5-00-upload-to-hf.py 가 sdlc.hf_repo_id 를 쓰지 않는다"; fail=1; }
echo "OK: repo id 규칙(단일 소스 · 카드/감사/업로드 동일)"

# 2. 카드가 repo id를 쓴다 (실행 예제는 snippets/ 단일 소스, __REPO_ID__ 자리)
grep -q 'def render_readme(target: str' "$ROOT/s4-70-update-readme.py" \
  || { echo "FAIL: 카드 렌더가 대상 1개가 아님"; fail=1; }
grep -q 'def model_label(target: str, mode: str) -> str:' "$ROOT/s4-70-update-readme.py" \
  || { echo "FAIL: model_label() 없음"; fail=1; }
grep -q '# {repo_id}' "$ROOT/s4-70-update-readme.py" \
  || { echo "FAIL: 카드 제목이 repo id가 아님"; fail=1; }
grep -q 'def load_snippet(' "$ROOT/s4-70-update-readme.py" \
  || { echo "FAIL: 스니펫 로더 없음"; fail=1; }
grep -q '__REPO_ID__", repo_id)' "$ROOT/s4-70-update-readme.py" \
  || { echo "FAIL: 스니펫 repo id 치환 없음"; fail=1; }
for _t in awq-w3-transformers awq-w5-transformers nvfp4-awq-transformers nvfp4-gptq-transformers; do
  grep -q 'model_id = "__REPO_ID__"' "$ROOT/snippets/$_t/app-transformers.py" \
    || { echo "FAIL: snippets/$_t/app-transformers.py에 repo id 자리 없음"; fail=1; }
  grep -q '__REPO_ID__' "$ROOT/snippets/$_t/serve-vllm.sh" \
    || { echo "FAIL: snippets/$_t/serve-vllm.sh에 repo id 자리 없음"; fail=1; }
done
# nvfp4-awq-vllm은 transformers 적재 불가(vLLM 전용)이므로 serve-vllm.sh만 검사
grep -q '__REPO_ID__' "$ROOT/snippets/nvfp4-awq-vllm/serve-vllm.sh" \
  || { echo "FAIL: snippets/nvfp4-awq-vllm/serve-vllm.sh에 repo id 자리 없음"; fail=1; }
echo "OK: 카드 표·제목·사용 예제 = repo id"

# 3. 업로드 전 감사가 repo id 기준
grep -q 'repo_id = hf_repo_id(target, mode)' "$ROOT/s4-75-verify-upload.py" \
  || { echo "FAIL: 46-verify가 repo id 기준으로 검사하지 않음"; fail=1; }
grep -q 'other_id = hf_repo_id(other, mode)' "$ROOT/s4-75-verify-upload.py" \
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
