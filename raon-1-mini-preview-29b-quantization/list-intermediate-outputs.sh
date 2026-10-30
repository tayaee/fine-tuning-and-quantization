#!/usr/bin/env bash
# list-intermediate-outputs.sh — 중간 산출물 목록 (스텝별, 전체 경로 표시).
# 최종 산출물(10개 모델 디렉토리)은 list-final-outputs.sh 참조.
# 각 헤딩은 항상 표시하고, 존재하는 파일만 나열한다. `ls`를 쓰지 않는다.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
RQ_ROOT="$ROOT"
RQ_SHARED="${RQ_SHARED:-/rosenas/data/AIML/$(basename "$ROOT")-shared}"
RQ_MODELS="$RQ_SHARED/models"
RQ_OUTPUTS="$RQ_SHARED/outputs"
RQ_DATASETS="$RQ_SHARED/datasets"

show() { # 존재하는 경로만 출력
  for p in "$@"; do
    # shellcheck disable=SC2317
    [ -e "$p" ] && echo "$p" || true
  done
}

echo "### [s0] base-model checks — $RQ_OUTPUTS/s0-{smoke,full}.json + bench/pi probe"
for m in smoke full; do show "$RQ_OUTPUTS/s0-$m.json"; done
for f in "$RQ_OUTPUTS"/s0-bench-* "$RQ_OUTPUTS"/s0-pi-probe "$RQ_ROOT"/logs/s0-*.log; do
  # shellcheck disable=SC2317
  [ -e "$f" ] && echo "$f" || true
done

echo "### [s1-00] setup — $RQ_ROOT/.venv/.rq-setup-stamp"
show "$RQ_ROOT/.venv/.rq-setup-stamp"

echo "### [s1-05] acceptance contract — $RQ_OUTPUTS/acceptance-{smoke,full}.json"
for m in smoke full; do show "$RQ_OUTPUTS/acceptance-$m.json"; done

echo "### [s1-15] calibration set — $RQ_DATASETS/raon-calibration-*.jsonl"
for f in "$RQ_DATASETS"/raon-calibration-*.jsonl "$RQ_DATASETS"/raon-calibration-*.meta; do
  # shellcheck disable=SC2317
  [ -e "$f" ] && echo "$f" || true
done

echo "### [s1-15] preflight — $RQ_OUTPUTS/compat.{stamp,log}"
show "$RQ_OUTPUTS/compat.stamp" "$RQ_OUTPUTS/compat.log"

echo "### [s2-00] dequant (base bf16) — $RQ_MODELS/raon-bf16"
show "$RQ_MODELS/raon-bf16"

echo "### [s4-00] PPL — $RQ_OUTPUTS/ppl-{smoke,full}/*.json"
for m in smoke full; do
  for f in "$RQ_OUTPUTS/ppl-$m"/*.json; do
    # shellcheck disable=SC2317
    [ -e "$f" ] && echo "$f" || true
  done
done

echo "### [s4-05] gsm8k prep — $RQ_OUTPUTS/eval-{smoke,full}/{prompts-gsm8k.jsonl,evalset-<mode>.json,prompts-humaneval.jsonl,prompts-kmmlu_redux.jsonl}"
for m in smoke full; do
  show "$RQ_OUTPUTS/eval-$m/prompts-gsm8k.jsonl" "$RQ_OUTPUTS/eval-$m/evalset-$m.json" \
    "$RQ_OUTPUTS/eval-$m/prompts-humaneval.jsonl" "$RQ_OUTPUTS/eval-$m/prompts-kmmlu_redux.jsonl"
done

echo "### [s4-10] answers (base, transformers) — eval-<mode>/{base.jsonl,bench/*/base.jsonl}"
for m in smoke full; do
  show "$RQ_OUTPUTS/eval-$m/base.jsonl"
  for f in "$RQ_OUTPUTS/eval-$m"/bench/*/base.jsonl; do
    # shellcheck disable=SC2317
    [ -e "$f" ] && echo "$f" || true
  done
done

echo "### [s4-15] answers (quant, transformers) — eval-<mode>/<target>.jsonl + bench details"
for m in smoke full; do
  for t in awq-w3-transformers awq-w5-transformers nvfp4-awq-transformers nvfp4-gptq-transformers; do
    show "$RQ_OUTPUTS/eval-$m/$t.jsonl"
    for f in "$RQ_OUTPUTS/eval-$m"/bench/*/"$t.jsonl"; do
      # shellcheck disable=SC2317
      [ -e "$f" ] && echo "$f" || true
    done
  done
done

echo "### [s4-50] answers (vLLM-served) — eval-<mode>/nvfp4-awq-vllm.jsonl + bench details"
for m in smoke full; do
  show "$RQ_OUTPUTS/eval-$m/nvfp4-awq-vllm.jsonl"
  for f in "$RQ_OUTPUTS/eval-$m"/bench/*/nvfp4-awq-vllm.jsonl; do
    # shellcheck disable=SC2317
    [ -e "$f" ] && echo "$f" || true
  done
done

echo "### [s4-20] aggregate — $RQ_OUTPUTS/eval-{smoke,full}/score.json"
for m in smoke full; do show "$RQ_OUTPUTS/eval-$m/score.json"; done

echo "### [s4-25] kmmluko prep — $RQ_OUTPUTS/eval-{smoke,full}/{prompts-kmmluko.jsonl,evalset-kmmluko.json}"
for m in smoke full; do show "$RQ_OUTPUTS/eval-$m/prompts-kmmluko.jsonl" "$RQ_OUTPUTS/eval-$m/evalset-kmmluko.json"; done

echo "### [s4-30/35/55] checks — $RQ_OUTPUTS/eval-{smoke,full}/checks.json (+ servebench/bench-transformers)"
for m in smoke full; do
  show "$RQ_OUTPUTS/eval-$m/checks.json" "$RQ_OUTPUTS/eval-$m/prompts-format.jsonl" \
    "$RQ_OUTPUTS/eval-$m/servebench-$m.json" "$RQ_OUTPUTS/eval-$m/bench-transformers-$m.json"
done

echo "### [s4-40/45/60] local serve — transient, no output files (servers only)"
true

echo "### [s4-65] (see s4-55 bench-transformers — same file)"

echo "### [s4-70] prep-upload manifest — $RQ_OUTPUTS/prep-upload-{smoke,full}.json + $RQ_MODELS/README-{smoke,full}.md"
for m in smoke full; do show "$RQ_OUTPUTS/prep-upload-$m.json" "$RQ_MODELS/README-$m.md"; done

echo "### [s4-75] verify-upload — $RQ_OUTPUTS/verify-upload-{smoke,full}.json"
for m in smoke full; do show "$RQ_OUTPUTS/verify-upload-$m.json"; done

echo "### [s4-80] release notes — $RQ_OUTPUTS/release-notes-{smoke,full}.md"
for m in smoke full; do show "$RQ_OUTPUTS/release-notes-$m.md"; done

echo "### [s5-00] HF upload (remote, no local files)"
for t in awq-w3-transformers awq-w5-transformers nvfp4-awq-transformers nvfp4-gptq-transformers nvfp4-awq-vllm; do
  echo "https://huggingface.co/tayaee/Raon-1-mini-preview-$t"
done

echo "### [s6] customer transformers — $RQ_OUTPUTS/s6-{smoke,full}/*"
for f in "$RQ_OUTPUTS"/s6-*/*; do
  # shellcheck disable=SC2317
  [ -e "$f" ] && echo "$f" || true
done

echo "### [s7] customer vLLM — $RQ_ROOT/logs/s7-*.log + $RQ_OUTPUTS/serve-{smoke,full}.json"
for f in "$RQ_ROOT"/logs/s7-*.log "$RQ_OUTPUTS"/serve-smoke.json "$RQ_OUTPUTS"/serve-full.json; do
  # shellcheck disable=SC2317
  [ -e "$f" ] && echo "$f" || true
done

echo "### [ledger] stage ledger (all steps) — $RQ_OUTPUTS/stage-ledger-{smoke,full}.json"
for m in smoke full; do show "$RQ_OUTPUTS/stage-ledger-$m.json"; done
