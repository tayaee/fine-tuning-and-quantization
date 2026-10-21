#!/usr/bin/env bash
# run-full-ddp.sh — full 2노드 DDP 원샷 (spark1+spark2에서 같은 명령 실행).
# 순서: spark1(rank0)을 먼저 기동한 뒤 spark2(rank1)를 실행한다 (torchrun 랑데부).
#   ./run-full-ddp.sh   # MASTER_ADDR·NODE_RANK 자동 검출 (아래 규칙)
# 분산 대상: 학습만 (10-baseline, 30-fft, 31-qlora).
# 합성데이터·전략비교·merge·PTQ·PPL·평가는 rank0만 수행 (TP=1, cheatsheet 관례).
# rank1은 rank0의 synthetic-full.jsonl 생성을 기다렸다가 학습에 합류한다
# (22-teacher 10k 생성에 약 5~6시간, 대기 상한 9시간).
# 하류(merge→평가)는 같은 전략 산출물이 있어야 진행한다:
#   없으면 DDP 학습 + 12-compare까지만 수행하고 종료 (학습 스텝 먼저 실행).
# 사용: SKIP_SETUP=1 ./run-full-ddp.sh                 # 10-setup 생략
#   MASTER_ADDR=spark1-p1-r0 NODE_RANK=1 ./run-full-ddp.sh  # 수동 지정도 가능
#   STEPS=setup,baseline ./run-full-ddp.sh             # 최소 파이프라인 (분산 동작 확인용)
# 자동 검출: MASTER_ADDR는 CX7(spark1-p1-r0→p1-r1→LAN 순, /etc/hosts+ping 확인),
#   NODE_RANK는 호스트명(spark1→0, spark2→1, …) 기준.
# 스텝명: setup baseline syndata fft qlora compare merge quant ppl eval (기본 STEPS=all)
# 스텝별 소요시간: stdout + logs/timing-full-ddp.log
set -euo pipefail
# 프리셋은 run-full-ddp.inc에 (수동 실행: 각 노드에서 source ./run-full-ddp.inc 후 단계 스크립트 인자 없이 실행)
source "$(dirname "$0")/run-full-ddp.inc"
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
source "$ROOT/config/common.env" "$MODE"
m1b_repro_echo

# ---- 스텝 타이밍 (stdout + TIMING_LOG) ----
TIMING_LOG="$ROOT/logs/timing-$MODE-$STRAT.log"
mkdir -p "$ROOT/logs"
RUN_T0=$(date +%s)
m1b_step() { # <스텝명> <명령...> — 시작/종료/소요초를 stdout과 TIMING_LOG에 기록
  local _name="$1"; shift
  local _t0 _dt _rc=0
  _t0=$(date +%s)
  m1b_log "step start: $_name"
  "$@" || _rc=$?
  _dt=$(($(date +%s) - _t0))
  m1b_log "step done: $_name elapsed=${_dt}s rc=$_rc"
  # 0s 성공(스킵)은 미기록 — 로그가 스킵으로 뒤덮여 실측을 찾기 힘듦. 실패는 항상 기록.
  if [ "$_rc" -ne 0 ] || [ "$_dt" -gt 0 ]; then
    printf '%s\t%s\t%s\t%ss\trc=%d\n' "$(date -u +%FT%TZ)" "$MODE-$STRAT" "$_name" "$_dt" "$_rc" >>"$TIMING_LOG"
  fi
  return $_rc
}
want_step() { # <스텝명> — STEPS=all(기본)이면 전부, 아니면 지정된 것만 실행
  case ",${STEPS:-all}," in *,all,*|*,"$1",*) return 0 ;; *) return 1 ;; esac
}
run_done() { # EXIT trap — 전체 소요시간 기록 (성공/실패 무관)
  m1b_log "run-$MODE-$STRAT end rc=$? total=$(( $(date +%s) - RUN_T0 ))s (timing: $TIMING_LOG)"
}
trap 'M1B_RC=$?; m1b_repro; run_done' EXIT

is_rank0() { [ "$NODE_RANK" = "0" ]; }
wait_file() { # <path> <timeout_s> — 공유디스크에 파일이 생길 때까지 대기
  local p="$1" t="${2:-32400}" waited=0
  while [ ! -s "$p" ]; do
    sleep 30; waited=$((waited + 30))
    if [ "$waited" -ge "$t" ]; then m1b_log "ERROR: wait timeout (${t}s): $p"; exit 1; fi
  done
}

m1b_log "run-$MODE-$STRAT start (rank=$NODE_RANK master=$MASTER_ADDR steps=${STEPS:-all})"

if [ "${SKIP_SETUP:-0}" != 1 ] && want_step setup; then
  m1b_step "10-setup" ./10-setup.sh
fi

want_step baseline && m1b_step "10-baseline-$STRAT" ./14-baseline-train.sh "$MODE" "$STRAT"

if is_rank0; then
  export TP=1  # 추론계(vLLM 생성·평가)는 노드 단독 1GPU로 실행
  if want_step syndata; then
    m1b_step "18-select-candidates" ./18-select-candidates.sh "$MODE"
    m1b_step "20-build-prompts" ./20-build-prompts.sh "$MODE"
    m1b_step "22-teacher-generate" ./22-teacher-to-generate-syn-data.sh "$MODE"
    m1b_step "24-postprocess" ./24-postprocess.sh "$MODE"
  fi
else
  wait_file "$M1B_DATASETS/synthetic-$MODE.jsonl" 32400
fi

want_step fft && m1b_step "30-fft-$STRAT" ./26-fft-train.sh "$MODE" "$STRAT"
want_step qlora && m1b_step "31-qlora-$STRAT" ./28-qlora-train.sh "$MODE" "$STRAT"

is_rank0 || { m1b_log "run-$MODE-$STRAT done (rank=$NODE_RANK, train-only)"; exit 0; }
export TP=1

want_step compare && m1b_step "12-compare" ./16-compare-strategies.sh "$MODE"

if [ ! -d "$M1B_MODELS/synthetic-fft-$MODE-$STRAT" ] || [ ! -d "$M1B_MODELS/synthetic-qlora-$MODE-$STRAT" ]; then
  m1b_log "downstream SKIP: no -$STRAT output (run train steps first)"
  exit 0
fi

want_step merge && m1b_step "30-merge-lora" ./30-merge-lora.sh "$MODE"

if want_step quant; then
  m1b_step "32-quant-fft-awq" ./32-quant-fft-awq.sh "$MODE"
  m1b_step "34-quant-fft-fp8" ./34-quant-fft-fp8.sh "$MODE"
  m1b_step "38-quant-fft-gptq" ./38-quant-fft-gptq.sh "$MODE"
  m1b_step "40-quant-qlora-awq" ./40-quant-qlora-awq.sh "$MODE"
  m1b_step "42-quant-qlora-fp8" ./42-quant-qlora-fp8.sh "$MODE"
  m1b_step "46-quant-qlora-gptq" ./46-quant-qlora-gptq.sh "$MODE"
  if [ -x "${LLAMACPP:-$HOME/git/llama.cpp}/build/bin/llama-quantize" ]; then
    m1b_step "36-quant-fft-gguf" ./36-quant-fft-gguf.sh "$MODE"
    m1b_step "44-quant-qlora-gguf" ./44-quant-qlora-gguf.sh "$MODE"
  else
    m1b_log "WARN: no llama-quantize, GGUF SKIP (run 42/52 after build)"
  fi
fi

want_step ppl && m1b_step "48-measure-ppl" ./48-measure-ppl.sh "$MODE"
if want_step eval; then
  m1b_step "50-eval" ./50-eval.sh "$MODE"
  m1b_step "52-eval-gguf" ./52-eval-gguf.sh "$MODE"
  m1b_step "54-score" ./54-score.sh "$MODE"
fi

m1b_log "run-$MODE-$STRAT done"
# --- 선택 (수동, rank0) ---
# ./58-upload-hf.sh full
# ./60-serve-vllm.sh full fft            # 터미널1 (상주)
# ./62-infer-examples.sh tayaee/Llama-3.2-1B-math-fft-full  # 터미널2
