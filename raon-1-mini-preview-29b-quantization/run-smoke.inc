# run-smoke.inc — smoke 프리셋 (run-smoke.sh와 공유).
# 수동 사용: source ./run-smoke.inc → 이후 단계 스크립트가 인자 없이도 MODE를 따라간다.
export MODE=smoke
: "${INFRA:=dgx-spark-1x}"
case "$INFRA" in *2x|2x) : "${TP:=2}" ;; *) : "${TP:=1}" ;; esac
export INFRA TP
echo "[preset] MODE=$MODE INFRA=$INFRA TP=$TP"
