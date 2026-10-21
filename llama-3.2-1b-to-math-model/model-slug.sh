#!/bin/sh
# 서버의 /v1/models 첫 id를 로그용 slug로 변환 (소문자, / → __).
# BASE_URL 기본 http://localhost:${PORT:-8000}/v1 — 60/61/62와 동일 서버를 본다.
# 구방식 호환: localhost 실패 시 spark1:${PORT} 로 fallback.
BASE_URL="${BASE_URL:-http://localhost:${PORT:-8000}/v1}"
ID="$(curl -s --max-time 3 "$BASE_URL/models" 2>/dev/null | jq -r '.data[0].id // empty' 2>/dev/null)"
if [ -z "$ID" ]; then
  ID="$(curl -s --max-time 3 "http://spark1:${PORT:-8000}/v1/models" 2>/dev/null | jq -r '.data[0].id // empty' 2>/dev/null)"
fi
[ -z "$ID" ] && exit 0
printf '%s' "$ID" | sed -e 's;/;__;g' | tr A-Z a-z
