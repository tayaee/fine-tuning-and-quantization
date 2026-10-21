#!/usr/bin/env bash
# 56-report.sh [mini|full] [--format table|json] (STRAT env, 기본 single) — Stage 7d wrapper.
# PPL + score 종합 리포트.
# 입력: $M1B_OUTPUTS/ppl-$MODE-$STRAT/summary.json (48-measure-ppl.sh 산출),
#   $M1B_OUTPUTS/eval-$MODE-$STRAT/score.json (54-score.sh 산출).
# 출력: $M1B_OUTPUTS/report-$MODE-$STRAT/report.json + stdout 4컬럼 표
#   (모델 / ppl+개선여부 / score+랭킹 / HF 업로드 추천 Yes-No).
# FORMAT=json이면 stdout에 JSON만 (파이프용, 잡음 없음).
# (56-report.py 자체가 결과표를 stdout에 찍고 report.json 아티팩트를 남긴다.)
set -euo pipefail
MODE="mini"
FORMAT="${FORMAT:-table}"
while [ $# -gt 0 ]; do
  case "$1" in
    mini|full) MODE="$1"; shift ;;
    --format) FORMAT="${2:?--format needs table|json}"; shift 2 ;;
    -h|--help) echo "usage: $0 [mini|full] [--format table|json]"; exit 0 ;;
    *) echo "usage: $0 [mini|full] [--format table|json]" >&2; exit 1 ;;
  esac
done
source "$(dirname "$0")/config/common.env" "$MODE"
STRAT="${STRAT:-single}"
case "$STRAT" in single|ddp|fsdp) ;; *) echo "STRAT must be single|ddp|fsdp" >&2; exit 1;; esac
case "$FORMAT" in table|json) ;; *) echo "FORMAT must be table|json" >&2; exit 1;; esac
[ "$FORMAT" = json ] || m1b_repro_echo
PPLSUM="$M1B_OUTPUTS/ppl-$MODE-$STRAT/summary.json"
SCORE="$M1B_OUTPUTS/eval-$MODE-$STRAT/score.json"
REPORT="$M1B_OUTPUTS/report-$MODE-$STRAT/report.json"
m1b_require "$PPLSUM" "$SCORE"
m1b_lock "${REPORT%/*}"
if [ -f "$REPORT" ] && m1b_fresh "$REPORT" "$PPLSUM" "$SCORE"; then
  if [ "$FORMAT" = json ]; then
    cat "$REPORT"
  else
    m1b_log "skip: fresh $REPORT (FORCE=1 to rebuild)"
    uv run "$M1B_ROOT/56-report.py" --from-report "$REPORT" --format table
  fi
  exit 0
fi
uv run "$M1B_ROOT/56-report.py" --mode "$MODE" --strat "$STRAT" --format "$FORMAT"

if [ "$FORMAT" != json ]; then
  echo ---- result ----
  (set -x; ls -l "$M1B_OUTPUTS/report-$MODE-$STRAT/")
fi
