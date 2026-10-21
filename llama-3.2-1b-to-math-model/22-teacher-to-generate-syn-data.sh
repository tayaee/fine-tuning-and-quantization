#!/usr/bin/env bash
# 22-generate.sh [mini|full] — Stage 2b. vLLM 합성 생성.
# 원본: data_gen_vllm.py (tp=4, max 8192, temp 0.5/top_p 0.8/top_k 5/rep 1.05).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"

MAXLEN=8192
# VLLM_GPU_MEM_UTIL: teacher 생성은 배치 작업이라 서빙급 예약 불필요.
# 8B 가중치 16GB + KV 8seq×8k ≈ 9GB + enforce_eager(그래프 없음) → 0.5(≈60GB)면
# 2배 여유. 0.8은 EngineCore 기동 회피용 과예약이었음 (DGX Spark 통합 128GB에서
# 0.9가 터져 0.8로 내렸던 것의 후속). export로 오버라이드 가능.
# 실행 중 잡에는 영향 없음 (vLLM 기동 시점에 고정).
: "${VLLM_GPU_MEM_UTIL:=0.5}"
# VLLM_MAX_NUM_SEQS=8: mini/full 통일. 8k 컨텍스트 시퀀스당 KV ~1GB,
# 8개 동시 = 8GB 수준이라 서빙급 선점(80GB)은 불필요.
: "${VLLM_MAX_NUM_SEQS:=8}"
m1b_repro_echo
IN="$M1B_DATASETS/prompts-$MODE.parquet"
OUT="$M1B_DATASETS/generated-$MODE.csv"
m1b_require "$IN"
if m1b_fresh "$OUT" "$IN"; then
  m1b_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -lh "$OUT")
  (set -x; wc -l "$OUT")
  exit 0
fi
m1b_log "teacher=$TEACHER_MODEL rev=$TEACHER_REVISION tp=$TP maxlen=$MAXLEN n=$SYNTH_N gpu_mem_util=$VLLM_GPU_MEM_UTIL max_num_seqs=$VLLM_MAX_NUM_SEQS"
(set -x; uv run "$M1B_ROOT/22-teacher-to-generate-syn-data.py" --mode "$MODE" --tp "$TP" --maxlen "$MAXLEN" \
  --teacher "$TEACHER_MODEL" --revision "$TEACHER_REVISION" --n "$SYNTH_N" --gpu-mem-util "$VLLM_GPU_MEM_UTIL" \
  --max-num-seqs "$VLLM_MAX_NUM_SEQS")

echo ---- result ----
(set -x; ls -lh "$M1B_DATASETS/generated-$MODE.csv")
(set -x; wc -l "$M1B_DATASETS/generated-$MODE.csv")
