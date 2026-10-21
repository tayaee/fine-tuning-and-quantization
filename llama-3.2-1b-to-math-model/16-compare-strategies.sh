#!/usr/bin/env bash
# 16-compare-strategies.sh [mini|full] [single|ddp|fsdp|all] — Stage 1 산출물 3전략 비교.
# single/ddp/fsdp 체크포인트의 safetensors 해시를 대조해
# "결과가 똑같은가?"를 직접 확인한다. config.json은 동일해야 정상,
# 가중치 해시는 effective batch를 맞춰도 bit-identical이 아닐 수 있다
# (allreduce vs reduce-scatter 합산 순서, dataloader 샤딩 차이).
# 전략 선택: $2 > $STRAT(환경) > all. single은 기준(reference)이라 항상 비교 대상에 둔다.
# NOTE: m1b_resolve_strat를 쓰지 않는다 — 읽기 전용 비교라 WORLD/LAUNCH가 필요 없고,
# ddp여도 1x 셸에서 동작해야 하기 때문 (resolve는 ddp에 PROFILE=2x를 요구).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"

WANT="${2:-${STRAT:-all}}"
case "$WANT" in single|ddp|fsdp|all) ;; *) echo "strategy must be single|ddp|fsdp|all" >&2; exit 1;; esac
if [ "$WANT" = "all" ]; then LIST="single ddp fsdp"; else LIST="$WANT"; fi
m1b_repro_echo

for s in $LIST; do
  d="$M1B_MODELS/base-gsm8k-$MODE-$s"
  echo "== $s ($d)"
  [ -d "$d" ] || { echo "  (missing, train that strategy first)"; continue; }
  cmp -s "$d/config.json" "$M1B_MODELS/base-gsm8k-$MODE-single/config.json" 2>/dev/null \
    && echo "  config: same as single" || echo "  config: differ/uncomparable"
  (cd "$d" && sha256sum ./*.safetensors 2>/dev/null | head -5) || echo "  (no safetensors)"
done
