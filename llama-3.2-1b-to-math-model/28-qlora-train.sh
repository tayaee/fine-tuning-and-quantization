#!/usr/bin/env bash
# 28-qlora-train.sh [mini|full] [single|ddp|fsdp] — Stage 3b. 합성데이터 QLoRA.
# 전략 선택: $2 > $STRAT(환경) > single.
# 원본: train_QLoRA.sh. 포인트: 8bit 블록 비활성 + 4bit nf4 double_quant,
# target=q/k/v/o/gate/down/up(+embed,lm_head), 산출물은 어댑터만.
# NOTE: 원본 강의는 FSDP+QLoRA 불가 → DDP로 동작. fsdp 전략은 학습용으로만 수행.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
m1b_resolve_strat "${2:-${STRAT:-single}}"
# 분산 전략은 2x 필수 — 1x에서 돌리면 solo 성공을 분산 성공으로 착각한다 (실측).
# ddp는 resolve가 이미 거부, fsdp 1x 기능검증도 이 환경(1노드=1GPU)에선 무의미.
if [ "$STRAT" != single ] && [ "$PROFILE" != "2x" ]; then
  echo "REFUSE: strat=$STRAT needs INFRA=dgx-spark-2x (현재 PROFILE=$PROFILE)" >&2
  exit 1
fi
m1b_repro_echo
# 리포팅 구분명 기본값 (미설정 시만 적용 — 직접 export한 값 우선).
: "${WANDB_PROJECT:=wandb-math-1b-sft-and-quantization}"
: "${MLFLOW_EXPERIMENT_NAME:=mlflow-math-1b-sft-and-quantization}"
export WANDB_PROJECT MLFLOW_EXPERIMENT_NAME

OUT="$M1B_MODELS/synthetic-qlora-$MODE-$STRAT"
IN="$M1B_DATASETS/synthetic-$MODE.jsonl"
m1b_require "$IN"
if [ -f "$OUT/adapter_config.json" ] && m1b_fresh "$OUT/adapter_config.json" "$IN"; then
  m1b_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -l "$OUT")
  exit 0
fi
# 출력 디렉토리는 rank0만 생성 (양쪽 동시 mkdir -p가 공유FS에서 EEXIST로 죽는
# 레이스 실측). rank1은 OUT에 쓰지 않는다 (중간 ckpt·최종 저장 모두 rank0-only).
if [ "${NODE_RANK:-0}" = "0" ]; then mkdir -p "$OUT"; fi
m1b_log "train=$M1B_DATASETS/synthetic-$MODE.jsonl out=$OUT (adapter only) strat=$STRAT world=$WORLD accum=$ACCUM"
# rank별 학습 로그 (tmux 유실 대비 — 매 줄 [rankN] 접두, grep rank1 logs/* 로 확인).
TRAIN_LOG="$M1B_ROOT/logs/train-$MODE-$STRAT-rank${NODE_RANK:-0}.log"
echo "== $(date -u +%FT%TZ) $(hostname) rank=${NODE_RANK:-0} world=$WORLD strat=$STRAT ==" > "$TRAIN_LOG"
m1b_log "log: $TRAIN_LOG"

# WORLD==1 solo 학습은 중복 실행 금지 (같은 OUT 저장 레이스 — 실측).
# 분산(WORLD=2)은 양쪽이 다 돌아야 해서 락 없음.
[ "$WORLD" = "1" ] && m1b_lock "$OUT" nowait
(set -x; "${LAUNCH[@]}" "$M1B_ROOT/train-entry.py" --peft \
  --model "$BASE_MODEL" --revision "$BASE_REVISION" --train "$M1B_DATASETS/synthetic-$MODE.jsonl" \
  --out "$OUT" --epochs "$EPOCHS" --mode "$MODE" \
  --strategy "$STRAT" --accum "$ACCUM" --expect-world "$WORLD" \
  --max_rows "$GSM_ROWS" --report_to "${REPORT_TO:-trackio,wandb}") 2>&1 | sed -u "s/^/[rank${NODE_RANK:-0}]: /" | tee -a "$TRAIN_LOG"

echo ---- result ----
(set -x; ls -l "$OUT")
