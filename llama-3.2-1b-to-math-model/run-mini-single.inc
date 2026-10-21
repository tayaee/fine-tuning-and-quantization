# run-mini-single.inc — mini single 프리셋 (run-mini-single.sh와 공유).
# 수동 사용: source ./run-mini-single.inc  → 이후 단계 스크립트(./14-baseline-train.sh 등)가
# 인자 없이도 MODE/STRAT/INFRA를 따라간다. run 스크립트도 이 파일을 source 한다.
# single은 항상 1x (외부 INFRA export가 있어도 무시).
export MODE=mini STRAT=single
export INFRA=dgx-spark-1x
export TP=1  # single은 노드 단독 1GPU (1노드=1GPU 하드웨어)
echo "[preset] MODE=$MODE STRAT=$STRAT INFRA=$INFRA TP=$TP"
