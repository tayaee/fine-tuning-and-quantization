#!/usr/bin/env bash
# verify-issue-18.sh — issue-18(모드 환경변수 MODE 무시 결함) 승인 기준 기계 검증
#   1. stage wrapper가 common.env에 "${1:-smoke}"로 하드코딩하지 않는다(환경변수 MODE를 죽이는 원인)
#   2. 모든 wrapper가 "${1:-${MODE:-smoke}}" 형태를 쓴다
#   3. 런타임: MODE=full이 반영되고, 명시적 인자가 여전히 우선한다
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0

# 1. 버그 패턴 잔존 0건 — common.env에 모드를 넘길 때 환경변수를 죽이면 안 된다
#    ("${1:-smoke}" / 리터럴 "smoke" 모두 금지)
for f in "$ROOT"/*.sh; do
  [ -f "$f" ] || continue
  line="$(grep -m1 'source .*common\.env' "$f" || true)"
  [ -n "$line" ] || continue
  case "$line" in
    *'${1:-smoke}'*|*'common.env" smoke'*|*'common.env smoke'*)
      echo "FAIL: $(basename "$f") 가 MODE를 하드코딩: $line"; fail=1 ;;
  esac
done
echo "OK: MODE 하드코딩 wrapper 0건"

# 2. stage wrapper(모드를 인자로 받는 20종)는 정식 형태를 쓴다
n=$(grep -l 'common\.env" "\${1:-\${MODE:-smoke}}"' "$ROOT"/s[0-9]-*.sh 2>/dev/null | wc -l)
[ "$n" -ge 20 ] || { echo "FAIL: \${1:-\${MODE:-smoke}} 형태 wrapper가 $n개뿐(20개 미만)"; fail=1; }
echo "OK: wrapper $n개가 \${1:-\${MODE:-smoke}} 형태"

# 3. 런타임 — 환경변수 반영 / 인자 우선 / 기본값
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
_mode() { RQ_SHARED="$tmp" MODE="$1" bash -c "source '$ROOT/config/common.env' ${2:+\"$2\"}; echo \"\$MODE\""; }
got="$(_mode full "")"
[ "$got" = "full" ] || { echo "FAIL: MODE=full 환경변수가 반영되지 않음( got=$got)"; fail=1; }
got="$(_mode full smoke)"
[ "$got" = "smoke" ] || { echo "FAIL: 명시적 인자가 환경변수를 덮어쓰지 않음( got=$got)"; fail=1; }
got="$(RQ_SHARED="$tmp" bash -c "source '$ROOT/config/common.env'; echo \"\$MODE\"")"
[ "$got" = "smoke" ] || { echo "FAIL: MODE 미지정 시 기본값이 smoke가 아님( got=$got)"; fail=1; }
echo "OK: 모드 해석(환경변수 반영·인자 우선·기본값)"

exit $fail
