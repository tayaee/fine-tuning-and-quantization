# run-full-fsdp.inc — full fsdp 프리셋 (run-full-fsdp.sh와 공유).
# 수동 사용: 각 노드에서 source ./run-full-fsdp.inc  → 이후 단계 스크립트가
# 인자 없이도 MODE/STRAT/INFRA/MASTER_ADDR/NODE_RANK를 따라간다.
# run 스크립트도 이 파일을 source 한다.
# fsdp는 항상 2노드 (외부 INFRA export가 있어도 무시, ddp와 동일).
# 1노드=1GPU라 단일노드 FSDP는 무의미하다.
export MODE=full STRAT=fsdp
export INFRA=dgx-spark-2x
DIST=1  # 항상 분산 (run 스크립트 하단 분기에서 사용)
if [ "$DIST" = "1" ]; then
  # MASTER_ADDR 자동 검출 (미지정 시): CX7 fabric 우선, /etc/hosts·getent 기준, ping 확인
  if [ -z "${MASTER_ADDR:-}" ]; then
    for _h in spark1-p1-r0 spark1-p1-r1 spark1.local; do
      _ip="$(getent hosts "$_h" 2>/dev/null | awk '{print $1; exit}')"
      [ -z "$_ip" ] && continue
      if ! command -v ping >/dev/null 2>&1 || ping -c1 -W1 "$_ip" >/dev/null 2>&1; then
        MASTER_ADDR="$_h"
        echo "[$MODE][$STRAT] MASTER_ADDR auto-detected: $_h ($_ip)"
        break
      fi
    done
    unset _h _ip
  fi
  : "${MASTER_ADDR:?MASTER_ADDR 검출 실패 — 수동 지정 필요 (예: MASTER_ADDR=spark1-p1-r0)}"
  export MASTER_ADDR
  # NODE_RANK 자동 검출 (미지정 시): 호스트명 기준
  if [ -z "${NODE_RANK:-}" ]; then
    case "$(hostname)" in
      spark1*) NODE_RANK=0 ;;
      spark2*) NODE_RANK=1 ;;
      spark3*) NODE_RANK=2 ;;
      spark4*) NODE_RANK=3 ;;
      *) : "${NODE_RANK:?NODE_RANK 검출 실패 — 수동 지정 필요 (예: NODE_RANK=0|1)}" ;;
    esac
    export NODE_RANK
    echo "[$MODE][$STRAT] NODE_RANK auto-detected: $NODE_RANK ($(hostname))"
  fi
  export NODE_RANK
fi
: "${NODE_RANK:=0}"
export NODE_RANK
echo "[preset] MODE=$MODE STRAT=$STRAT INFRA=$INFRA DIST=$DIST NODE_RANK=$NODE_RANK"
