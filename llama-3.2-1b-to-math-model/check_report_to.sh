#!/usr/bin/env bash
# check_report_to.sh — 2종 리포팅 연동(trackio, wandb) INSTALL + CONFIG 상태 green/red 표시.
#   ./check_report_to.sh                  # 1회 확인 (기본 2종)
#   ./check_report_to.sh --watch [초]      # 모두 green 될 때까지 반복 (기본 30초)
# 종료 코드: 0=2종 INSTALL·CONFIG 모두 READY, 1=미준비 있음.
set -uo pipefail
cd "$(dirname "$0")"

TWO="trackio,wandb"
WATCH=0; INTERVAL=30; REQ="$TWO"; OUT_ARG=(); OUT_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --watch) WATCH=1; [ "${2:-}" != "" ] && [[ "$2" =~ ^[0-9]+$ ]] && { INTERVAL="$2"; shift; }; shift ;;
    --report-to) REQ="$2"; shift 2 ;;
    --out) OUT_ARG=(--out "$2"); OUT_DIR="$2"; shift 2 ;;
    *) echo "usage: $0 [--watch [secs]] [--report-to csv] [--out dir]" >&2; exit 2 ;;
  esac
done
IFS=',' read -ra NAMES <<< "$REQ"

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then GREEN=$'\e[32m'; RED=$'\e[31m'; RST=$'\e[0m'; else GREEN=''; RED=''; RST=''; fi
ok() { printf '%b● READY%b' "$GREEN" "$RST"; }
bad() { printf '%b● NOT READY%b' "$RED" "$RST"; }

# config 점검: 전역 $raw (체커 출력) 사용. echo "이유"; return 0=READY.
config_check() {
  case "$1" in
    trackio) echo "추가 설정 불필요"; return 0 ;;  # 추가 설정 불필요
    wandb)
      if [ -n "${WANDB_API_KEY:-}" ]; then echo "API키: env 설정됨"; return 0; fi
      if [ -f ~/.netrc ] && grep -q "machine api.wandb.ai" ~/.netrc; then echo "API키: ~/.netrc 로그인됨"; return 0; fi
      echo "'wandb login' 필요 (WANDB_API_KEY도 미설정)"; return 1 ;;
  esac
  return 0
}

# 보고서 저장 위치 (정보 표시용 — 체커는 아무 파일도 쓰지 않음).
output_path() {
  case "$1" in
    trackio) echo "${HF_HOME:-~/.cache/huggingface}/trackio/trackio-math-1b-sft-and-quantization.db" ;;
    wandb) echo "cloud wandb.ai/${WANDB_PROJECT:-wandb-math-1b-sft-and-quantization} + ./wandb/" ;;
    *) echo "(벤더 문서 참조)" ;;
  esac
}

one_pass() {
  local raw rc n inst conf reason overall=0
  raw=$(uv run --no-sync check_report_to.py --report_to "$REQ" "${OUT_ARG[@]}" 2>&1)
  rc=$?
  echo "$raw" | grep -v "pytree"
  echo "---- install ----"
  declare -A INST_OK=() INST_WHY=()
  for n in "${NAMES[@]}"; do
    if echo "$raw" | grep -q "^\s*\[OK\] $n:"; then inst="$(ok)"; INST_OK[$n]=1;
    else
      inst="$(bad)"; INST_OK[$n]=0; overall=1
      if echo "$raw" | grep -q "^\s*\[BROKEN\] $n:"; then
        INST_WHY[$n]=$(echo "$raw" | grep "^\s*\[BROKEN\] $n:" | head -1 | sed 's/^[^]]*] [^:]*: //')
      elif echo "$raw" | grep -q "^\s*\[UNKNOWN\] $n:"; then INST_WHY[$n]="이름 오류(오타?)";
      else INST_WHY[$n]="패키지 미설치"; fi
    fi
    printf '%-12s install : %b\n' "$n" "$inst"
  done
  echo "---- config ----"
  for n in "${NAMES[@]}"; do
    if [ "${INST_OK[$n]}" = "1" ] && reason=$(config_check "$n"); then conf="$(ok)";
    else conf="$(bad)"; overall=1; [ "${INST_OK[$n]}" = "1" ] || reason="${INST_WHY[$n]}"; fi
    printf '%-12s config  : %b  (%s)\n' "$n" "$conf" "$reason"
  done
  echo "REPORT_TO=${REPORT_TO:-(미설정, 기본 trackio)}"
  echo "---- output ----"
  for n in "${NAMES[@]}"; do
    printf '%-12s output  : %s\n' "$n" "$(output_path "$n")"
  done
  [ "$overall" = "0" ] && echo "ALL GREEN."
  return $overall
}

if [ "$WATCH" = "1" ]; then
  while true; do
    echo "== $(date -u +%FT%TZ) =="
    if one_pass; then exit 0; fi
    echo "retry in ${INTERVAL}s ... (Ctrl-C to stop)"
    sleep "$INTERVAL"
  done
else
  one_pass
fi
