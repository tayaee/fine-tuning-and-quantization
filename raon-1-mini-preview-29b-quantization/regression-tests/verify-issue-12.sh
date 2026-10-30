#!/usr/bin/env bash
# verify-issue-12.sh — issue-12(TARGETS 필수화) 승인 기준 기계 검증
#   1. 구 모놀리식 러너(run-{smoke,full}.sh + .inc)는 archive/로 은퇴 (루트에 없음)
#   2. 개별 스크립트의 TARGETS 가드 spot check (미지정 → exit 2, 전수 검사는 verify-targets-guard.sh)
#   3. 문서에 TARGETS 사용법 반영 (개별 스크립트 + s5-00 기준)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0

# 1. 은퇴 러너는 archive/에만, 루트에는 없음
for f in run-smoke.sh run-full.sh run-smoke.inc run-full.inc; do
  [ -f "$ROOT/archive/$f" ] || { echo "FAIL: archive/$f 없음"; fail=1; }
  [ ! -e "$ROOT/$f" ] || { echo "FAIL: 은퇴 러너 $f 가 루트에 남아 있음"; fail=1; }
done
echo "OK: 모놀리식 러너 archive/ 은퇴"

# 2. per-script TARGETS 가드 spot check (s5-00-upload-to-hf.sh)
out="$(cd "$ROOT" && env -u TARGETS ./s5-00-upload-to-hf.sh smoke 2>&1 || true)"
echo "$out" | grep -q "TARGETS is not specified" || { echo "FAIL: s5-00 미지정 안내 없음"; fail=1; }
( cd "$ROOT" && ./s5-00-upload-to-hf.sh smoke >/dev/null 2>&1 ) && rc=0 || rc=$?
[ "$rc" = 2 ] || { echo "FAIL: s5-00 미지정 시 exit=$rc (expected 2)"; fail=1; }
echo "OK: per-script TARGETS 가드 spot check"

# 3. 문서
for doc in README.md PROCEDURE.md cheatsheet.md; do
  grep -q "TARGETS=all" "$ROOT/$doc" || { echo "FAIL: $doc 에 TARGETS=all 예시 없음"; fail=1; }
done
grep -q "archive/" "$ROOT/PROCEDURE.md" || { echo "FAIL: PROCEDURE.md에 러너 은퇴(archive/) 언급 없음"; fail=1; }
echo "OK: 문서 반영"

exit $fail
