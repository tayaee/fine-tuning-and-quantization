# run-full.inc — full 프리셋 (run-full.sh와 공유).
export MODE=full
: "${INFRA:=dgx-spark-1x}"
case "$INFRA" in *2x|2x) : "${TP:=2}" ;; *) : "${TP:=1}" ;; esac
export INFRA TP
echo "[preset] MODE=$MODE INFRA=$INFRA TP=$TP"
