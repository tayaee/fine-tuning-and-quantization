#!/usr/bin/env bash
# verify-issue-13.sh — issue-13(스크립트 시작 배너) 승인 기준 기계 검증
#   1. 모든 번호 스크립트가 시작 시 1줄 영어 설명을 출력한다
#   2. 배너 형식: rq_banner "<id>" "<영어 설명>"
#   3. 설명에 한글이 섞이지 않는다
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0

# 번호가 붙은 실행 스크립트만 대상(공용 모듈 .inc/.py 제외).
# frozen 카드 래퍼(s7-02-verify-readme-with-<target>-and-vllm-fork.sh)는 README 명령
# as-is 실행이 설계라 배너 면제 (배너는 s7-13-chat이 찍는다).
# (구 명칭 s7-<num>-test-<target>.sh → 2026-10-28 개명. 패턴도 같이 갱신했다.)
mapfile -t SCRIPTS < <(cd "$ROOT" && ls -1 s?-*.sh run-s1-to-s4-*.sh | grep -vE 's7-02-verify-readme-with-.+-and-vllm-fork\.sh' | sort)
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

# 공용 헬퍼 존재 + build 진입점 배너 (구 run-{smoke,full}.sh는 archive/로 은퇴)
grep -q 'rq_banner() {' "$ROOT/config/common.env" || { echo "FAIL: common.env에 rq_banner 헬퍼 없음"; fail=1; }
for f in archive/run-smoke.sh archive/run-full.sh; do
  [ -f "$ROOT/$f" ] || { echo "FAIL: 은퇴 러너 $f 가 archive/에 없음"; fail=1; }
done
# 빌드 wrapper 8종은 각자 배너를 출력한다(공용 엔진 .inc는 중복 방지를 위해 배너 없음)
n_build=0
for f in "$ROOT"/run-s1-to-s4-*.sh; do
  grep -q 'rq_banner "' "$f" || { echo "FAIL: $(basename "$f") 에 배너 없음"; fail=1; }
  n_build=$((n_build + 1))
done
[ "$n_build" = 10 ] || { echo "FAIL: 빌드 wrapper 수=$n_build (10 기대)"; fail=1; }
for f in s4-75-release-notes.py; do
  grep -qE 'print\("\[s[0-9]-' "$ROOT/$f" || { echo "FAIL: $f 에 배너 출력 없음"; fail=1; }
done
echo "OK: 헬퍼 + 진입점 배너"

# 실제 출력 확인(07)
out="$(cd "$ROOT" && ./s1-04-setup-acceptance-criteria.sh smoke 2>&1 || true)"
echo "$out" | grep -qE '^\[s1-04-setup-acceptance-criteria\] Stage 1 - .*[A-Za-z]' \
  || { echo "FAIL: s1-04-setup-acceptance-criteria.sh 시작 배너 출력 형식 불일치"; fail=1; }
echo "OK: s1-04-setup-acceptance-criteria.sh 실제 출력"

exit $fail
