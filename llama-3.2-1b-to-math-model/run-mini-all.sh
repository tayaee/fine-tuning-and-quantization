#!/usr/bin/env bash
# run-mini-all.sh — mini 3전략 순차 실행 (single → ddp → fsdp).
# 양 노드에서 같은 명령. single은 spark1만, ddp/fsdp는 양 노드.
# STEPS/RUNS/SKIP_SETUP은 env로 하위 전달. NCCL fabric 고정은 run-all-all.sh 경로에서만.
set -euo pipefail
cd "$(dirname "$0")"
case "$(hostname)" in spark1*) ./run-mini-single.sh ;; esac
./run-mini-ddp.sh
./run-mini-fsdp.sh
