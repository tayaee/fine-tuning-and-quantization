#!/usr/bin/env bash
# run-all-all.sh — 6종 파이프라인 고정 순서로 순차 실행.
# 순서: run-mini-single → run-mini-ddp → run-full-single → run-full-ddp
#       → run-mini-fsdp → run-full-fsdp
# 실행: 양 노드에서 같은 명령 (spark1=rank0/master, spark2=rank1/worker).
#   ./run-all-all.sh
# single 2종은 rank0(spark1)에서만 실행한다 (rank1은 SKIP — 동일 OUT 중복 기록 방지).
# ddp/fsdp 4종은 양 노드에서 실행한다 (하위 스크립트 내부 rank 게이팅).
# 이 파일에서 수행하는 초기화:
#   - BASE_MODEL 기본값 unsloth/Llama-3.2-1B (spark1 common.env의 gated
#     meta-llama 기본값 회피, 양 노드 통일)
#   - NCCL CX7 fabric 고정 (NCCL_SOCKET_IFNAME/NCCL_IB_HCA — 2x 분산이 관리 LAN으로
#     새는 것 방지. 1x 실행에는 무해)
#   - NODE_RANK 자동 검출 (호스트명 기준, 미확인 호스트는 0)
#   - MASTER_ADDR는 하위 스크립트가 자동 검출 (/etc/hosts+ping)
# 환경변수: SKIP_SETUP=1 (10-setup 생략), STEPS=... (하위 스크립트에 그대로 전달),
#   RUNS=... (이 6개 중 일부만 실행, 콤마 구분. 예: RUNS=mini-ddp,mini-fsdp),
#   BASE_MODEL / NCCL_SOCKET_IFNAME / NCCL_IB_HCA (기본값 재정의 가능)
# NOTE: full 포함 전체는 수십 시간. STEPS/RUNS로 부분 실행 가능.
# 스텝별 소요시간: stdout + logs/timing-all-all.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"

# ---- 초기화 ----
export BASE_MODEL="${BASE_MODEL:-unsloth/Llama-3.2-1B}"  # 비gated 미러 (양 노드 통일)
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"  # CX7 p1-r0
export NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f1}"  # enp1s0f1np1 대응 HCA
export INFRA=dgx-spark-2x  # run-all-all은 2노드 오케스트레이터로 고정
export TP="${TP:-1}"  # 1노드=1GPU 하드웨어라 TP=2는 불가. vLLM 생성·평가는 항상 단독 실행
# (single 스크립트는 내부에서 1x 강제, ddp/fsdp는 2x 고정)
# MASTER_ADDR 자동 검출 (미지정 시): CX7 fabric 우선, /etc/hosts·getent 기준, ping 확인
if [ -z "${MASTER_ADDR:-}" ]; then
  for _h in spark1-p1-r0 spark1-p1-r1 spark1.local; do
    _ip="$(getent hosts "$_h" 2>/dev/null | awk '{print $1; exit}')"
    [ -z "$_ip" ] && continue
    if ! command -v ping >/dev/null 2>&1 || ping -c1 -W1 "$_ip" >/dev/null 2>&1; then
      MASTER_ADDR="$_h"
      echo "[all] MASTER_ADDR auto-detected: $_h ($_ip)"
      break
    fi
  done
  unset _h _ip
fi
: "${MASTER_ADDR:?MASTER_ADDR 검출 실패 — 수동 지정 필요 (예: MASTER_ADDR=spark1-p1-r0)}"
if [ -z "${NODE_RANK:-}" ]; then
  case "$(hostname)" in
    spark1*) NODE_RANK=0 ;;
    spark2*) NODE_RANK=1 ;;
    spark3*) NODE_RANK=2 ;;
    spark4*) NODE_RANK=3 ;;
    *) NODE_RANK=0 ;;
  esac
  export NODE_RANK
  echo "[all] NODE_RANK auto-detected: $NODE_RANK ($(hostname))"
fi
source "$ROOT/config/common.env" mini  # m1b_log + 공용 env (UV_NO_SYNC 등)
m1b_repro_echo

nvidia-smi -L >/dev/null 2>&1 || { m1b_log "ERROR: no GPU (nvidia-smi failed)"; exit 1; }

# ---- 스텝 타이밍 (stdout + TIMING_LOG) ----
TIMING_LOG="$ROOT/logs/timing-all-all.log"
mkdir -p "$ROOT/logs"
RUN_T0=$(date +%s)
m1b_step() { # <이름> <명령...> — 시작/종료/소요초를 stdout과 TIMING_LOG에 기록
  local _name="$1"; shift
  local _t0 _dt _rc=0
  _t0=$(date +%s)
  m1b_log "step start: $_name"
  "$@" || _rc=$?
  _dt=$(($(date +%s) - _t0))
  m1b_log "step done: $_name elapsed=${_dt}s rc=$_rc"
  # 0s 성공(스킵)은 미기록 — 로그가 스킵으로 뒤덮여 실측을 찾기 힘듦. 실패는 항상 기록.
  if [ "$_rc" -ne 0 ] || [ "$_dt" -gt 0 ]; then
    printf '%s\t%s\t%s\t%ss\trc=%d\n' "$(date -u +%FT%TZ)" "all-all" "$_name" "$_dt" "$_rc" >>"$TIMING_LOG"
  fi
  return $_rc
}
want_run() { # <런 이름> — RUNS=all(기본)이면 전부, 아니면 지정된 것만 실행
  case ",${RUNS:-all}," in *,all,*|*,"$1",*) return 0 ;; *) return 1 ;; esac
}
run_done() { # EXIT trap — 전체 소요시간 기록 (성공/실패 무관)
  m1b_log "run-all-all end rc=$? total=$(( $(date +%s) - RUN_T0 ))s (timing: $TIMING_LOG)"
}
trap 'M1B_RC=$?; m1b_repro; run_done' EXIT

is_rank0() { [ "${NODE_RANK:-0}" = "0" ]; }

m1b_log "run-all-all start (rank=$NODE_RANK runs=${RUNS:-all} steps=${STEPS:-all})"

if is_rank0; then
  want_run mini-single && m1b_step "run-mini-single" ./run-mini-single.sh
elif want_run mini-single; then
  m1b_log "SKIP run-mini-single (rank0 only)"
fi

want_run mini-ddp && m1b_step "run-mini-ddp" ./run-mini-ddp.sh

if is_rank0; then
  want_run full-single && m1b_step "run-full-single" ./run-full-single.sh
elif want_run full-single; then
  m1b_log "SKIP run-full-single (rank0 only)"
fi

want_run full-ddp && m1b_step "run-full-ddp" ./run-full-ddp.sh
want_run mini-fsdp && m1b_step "run-mini-fsdp" ./run-mini-fsdp.sh
want_run full-fsdp && m1b_step "run-full-fsdp" ./run-full-fsdp.sh

m1b_log "run-all-all done"
