#!/usr/bin/env bash
# run-full-all.sh — full 3전략 순차 실행 (single → ddp → fsdp).
# 양 노드에서 같은 명령. single은 spark1만, ddp/fsdp는 양 노드.
# STEPS/RUNS/SKIP_SETUP은 env로 하위 전달. NCCL fabric 고정은 run-all-all.sh 경로에서만.
# NOTE: full 전체는 수십 시간. STEPS/RUNS로 부분 실행 가능.
set -euo pipefail
cd "$(dirname "$0")"
case "$(hostname)" in spark1*) ./run-full-single.sh ;; esac
./run-full-ddp.sh
./run-full-fsdp.sh
