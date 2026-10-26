#!/usr/bin/env bash
# verify-issue-12.sh — issue-12(TARGETS 필수화) 승인 기준 기계 검증
#   1. run-{smoke,full}.sh: TARGETS 미지정 → "TARGETS is not specified" 출력 + exit 2 (스텝 실행 0건)
#   2. TARGETS=all / 부분집합 / 알 수 없는 값 동작
#   3. 문서에 TARGETS=all 사용법 반영
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0

for f in run-smoke.sh run-full.sh; do
  bash -n "$ROOT/$f" || { echo "FAIL: $f bash -n"; fail=1; }
  grep -q 'RQ_TARGETS_ALL="awq-w3 awq-w5 nvfp4-awq nvfp4-gptq"' "$ROOT/$f" \
    || { echo "FAIL: $f 에 타깃 목록 정의 없음"; fail=1; }
  grep -q 'TARGETS is not specified' "$ROOT/$f" || { echo "FAIL: $f 에 TARGETS is not specified 안내 없음"; fail=1; }
  grep -q 'Unknown TARGETS' "$ROOT/$f" || { echo "FAIL: $f 에 알 수 없는 타깃 거부 없음"; fail=1; }
  # 가드가 setup 스텝보다 앞에 있어야 한다(스텝 0건 종료)
  _guard=$(grep -n 'TARGETS is not specified' "$ROOT/$f" | head -1 | cut -d: -f1)
  _setup=$(grep -n 'rq_step "10-setup"' "$ROOT/$f" | head -1 | cut -d: -f1)
  if [ -n "$_guard" ] && [ -n "$_setup" ] && [ "$_guard" -gt "$_setup" ]; then
    echo "FAIL: $f 의 TARGETS 가드가 s0-10-setup 뒤에 있다 (스텝 실행 0건이어야 함)"; fail=1
  fi
done
echo "OK: TARGETS 가드 위치·내용"

# 실제 동작: 미지정 → exit 2 + 안내 + 스텝 0건
out="$(cd "$ROOT" && ./run-smoke.sh 2>&1 || true)"
rc=0
( cd "$ROOT" && ./run-smoke.sh >/dev/null 2>&1 ) || rc=$?
[ "$rc" = 2 ] || { echo "FAIL: TARGETS is not specified 시 exit code=$rc (expected 2)"; fail=1; }
echo "$out" | grep -q "Available values" || { echo "FAIL: 가능 값 안내 없음"; fail=1; }
for t in awq-w3 awq-w5 nvfp4-awq nvfp4-gptq all; do
  echo "$out" | grep -q -- "- $t" || { echo "FAIL: 가능 값 안내에 '$t' 없음"; fail=1; }
done
echo "$out" | grep -q "step start:" && { echo "FAIL: TARGETS is not specified인데 스텝이 실행됨"; fail=1; }
echo "OK: TARGETS is not specified → exit 2, 스텝 0건"

# TARGETS=all 은 정상 진행(validate 스텝만, 무거운 스텝은 제외)
( cd "$ROOT" && TARGETS=all STEPS=validate ./run-smoke.sh >/tmp/verify-issue-12.log 2>&1 ) \
  || { echo "FAIL: TARGETS=all 실행 실패"; tail -10 /tmp/verify-issue-12.log; fail=1; }
echo "OK: TARGETS=all 실행"

# 알 수 없는 타깃 거부
( cd "$ROOT" && TARGETS=bogus ./run-smoke.sh >/dev/null 2>&1 ) && { echo "FAIL: Unknown TARGETS가 통과함"; fail=1; }
echo "OK: Unknown TARGETS 거부"

# 문서
for doc in README.md PROCEDURE.md cheatsheet.md; do
  grep -q "TARGETS=all" "$ROOT/$doc" || { echo "FAIL: $doc 에 TARGETS=all 예시 없음"; fail=1; }
done
echo "OK: 문서 반영"

exit $fail
