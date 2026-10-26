#!/usr/bin/env bash
# verify-targets-guard.sh — 타깃 명시 강제(rq_require_targets) 기계 검증
#   1. 타깃을 받는 전 스크립트: $2·TARGETS 미지정 → "TARGETS is not specified" + exit 2 (작업 0건)
#   2. rq_require_targets 단위 동작 (명시값 통과 / 미지정 2 / TARGETS=all 통과)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0

GUARDED="s6-34-measure-ppl.sh s7-37-eval.sh
  s7-55-bench-transformers.sh s7-56-prep-upload.sh s7-57-verify-upload.sh
  s8-49-upload-hf.sh s8-58-serve-infer-all.sh"

for f in config/common.env run-smoke.sh run-full.sh $GUARDED; do
  bash -n "$ROOT/$f" || { echo "FAIL: $f bash -n"; fail=1; }
done
echo "OK: bash -n"

# 1. 미지정 → exit 2 + 사용법
for f in $GUARDED; do
  out="$(cd "$ROOT" && env -u TARGETS ./$f smoke 2>&1)" && rc=0 || rc=$?
  [ "$rc" = 2 ] || { echo "FAIL: $f 미지정 시 exit=$rc (expected 2)"; fail=1; continue; }
  echo "$out" | grep -q "TARGETS is not specified" || { echo "FAIL: $f 사용법 안내 없음"; fail=1; continue; }
  echo "$out" | grep -q "Available values" || { echo "FAIL: $f 가능 값 안내 없음"; fail=1; continue; }
  echo "$out" | grep -q "step start:" && { echo "FAIL: $f 미지정인데 스텝 실행됨"; fail=1; continue; }
  echo "OK: $f 미지정 → exit 2"
done

# 2. helper 단위 동작 (가벼움 — common.env만 source)
rc=0
( source "$ROOT/config/common.env" smoke >/dev/null 2>&1
  rq_require_targets "./x smoke" "" awq-w3 nvfp4-awq >/dev/null 2>&1 ) || rc=$?
[ "$rc" = 2 ] || { echo "FAIL: helper 미지정 status=$rc (expected 2)"; fail=1; }
echo "OK: helper 미지정 → 2"

v="$(source "$ROOT/config/common.env" smoke >/dev/null 2>&1; rq_require_targets "./x smoke" "nvfp4-awq" awq-w3 nvfp4-awq)"
[ "$v" = "nvfp4-awq" ] || { echo "FAIL: helper 명시값 반환='$v'"; fail=1; }
v="$(cd "$ROOT" && TARGETS=all bash -c 'source config/common.env smoke >/dev/null 2>&1; rq_require_targets "./x smoke" "" awq-w3 nvfp4-awq')"
[ "$v" = "all" ] || { echo "FAIL: helper TARGETS=all 반환='$v'"; fail=1; }
echo "OK: helper 명시값/all 통과"

exit $fail
