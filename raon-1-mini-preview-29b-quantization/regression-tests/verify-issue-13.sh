#!/usr/bin/env bash
# verify-issue-13.sh — issue-13(스크립트 시작 배너) 승인 기준 기계 검증
#   1. 모든 번호 스크립트가 시작 시 1줄 영어 설명을 출력한다
#   2. 배너 형식: rq_banner "<id>" "<영어 설명>"
#   3. 설명에 한글이 섞이지 않는다
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0

# 번호가 붙은 실행 스크립트만 대상(공용 모듈 .inc/.py 제외)
mapfile -t SCRIPTS < <(cd "$ROOT" && ls -1 s?-*.sh s0-s7-build-*.sh | sort)
[ "${#SCRIPTS[@]}" -ge 25 ] || { echo "FAIL: 대상 스크립트 수=${#SCRIPTS[@]} (25개 이상 기대)"; fail=1; }

for f in "${SCRIPTS[@]}"; do
  if ! grep -q 'rq_banner "' "$ROOT/$f"; then
    echo "FAIL: $f 에 시작 배너(rq_banner) 없음"; fail=1; continue
  fi
  desc=$(grep -m1 'rq_banner "' "$ROOT/$f" | sed -E 's/.*rq_banner "[^"]*" "(.*)"/\1/')
  [ -n "$desc" ] || { echo "FAIL: $f 배너 설명 파싱 실패"; fail=1; continue; }
  if printf '%s' "$desc" | grep -qP '[\x{AC00}-\x{D7A3}\x{1100}-\x{11FF}\x{3130}-\x{318F}]'; then
    echo "FAIL: $f 배너 설명에 한글 포함: $desc"; fail=1
  fi
  printf '%s' "$desc" | grep -qE '[A-Za-z]{4,}' || { echo "FAIL: $f 배너 설명이 영어 문장 아님: $desc"; fail=1; }
done
echo "OK: ${#SCRIPTS[@]}개 스크립트 시작 배너(영어 1줄)"

# 공용 헬퍼 존재 + run/build 진입점 배너
grep -q 'rq_banner() {' "$ROOT/config/common.env" || { echo "FAIL: common.env에 rq_banner 헬퍼 없음"; fail=1; }
for f in run-smoke.sh run-full.sh; do
  grep -q 'rq_banner "' "$ROOT/$f" || { echo "FAIL: $f 에 배너 없음"; fail=1; }
done
# 빌드 wrapper 8종은 각자 배너를 출력한다(공용 엔진 .inc는 중복 방지를 위해 배너 없음)
n_build=0
for f in "$ROOT"/s0-s7-build-*.sh; do
  grep -q 'rq_banner "' "$f" || { echo "FAIL: $(basename "$f") 에 배너 없음"; fail=1; }
  n_build=$((n_build + 1))
done
[ "$n_build" = 8 ] || { echo "FAIL: 빌드 wrapper 수=$n_build (8 기대)"; fail=1; }
for f in s8-61-release-notes.py s8-64-verify-serving.py; do
  grep -q 'print("\[s8-' "$ROOT/$f" || { echo "FAIL: $f 에 배너 출력 없음"; fail=1; }
done
echo "OK: 헬퍼 + 진입점 배너"

# 실제 출력 확인(07)
out="$(cd "$ROOT" && ./s1-07-acceptance.sh smoke 2>&1 || true)"
echo "$out" | grep -qE '^\[s1-07-acceptance\] Stage 1/9 - .*[A-Za-z]' \
  || { echo "FAIL: s1-07-acceptance.sh 시작 배너 출력 형식 불일치"; fail=1; }
echo "OK: s1-07-acceptance.sh 실제 출력"

exit $fail
