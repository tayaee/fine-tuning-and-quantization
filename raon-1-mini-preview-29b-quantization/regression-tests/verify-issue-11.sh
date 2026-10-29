#!/usr/bin/env bash
# verify-issue-11.sh — issue-11(표준 GSM8K 평가 + 카드 2종 검사 + 배치 생성) 승인 기준 기계 검증
#   1. s4-04-prep-gsm8k.{py,sh} 존재(+실행 권한) + 표준 GSM8K 매니페스트 계약
#   2. s4-07-eval.py: BENCHMARKS=gsm8k 단일, 정식 프롬프트 필수, 배치 생성 지원
#   3. s4-31-prep-upload.py: 카드에 검사1(PPL)/검사2(GSM8K) 두 표 + GSM8K 판정
#   4. acceptance.toml: gsm8k_acc_drop_max
#   5. gates.inc: rq_gate_eval의 GSM8K 정확도 대조
#   6. 파이프라인 배선(36 단계) + 단위 테스트
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

# 1. s4-04-prep-gsm8k
for f in s4-04-prep-gsm8k.py s4-04-prep-gsm8k.sh; do
  [ -f "$ROOT/$f" ] || { echo "FAIL: $f 없음"; fail=1; }
done
[ -x "$ROOT/s4-04-prep-gsm8k.sh" ] || { echo "FAIL: s4-04-prep-gsm8k.sh 실행 권한 없음"; fail=1; }
bash -n "$ROOT/s4-04-prep-gsm8k.sh" || { echo "FAIL: s4-04-prep-gsm8k.sh bash -n"; fail=1; }
for tok in 'openai/gsm8k' 'GSM8K_REVISION' 'GSM8K_SEED' 'def _gold_of(' 'def sample_rows(' \
           'def build(' 'def is_fresh(' 'prompts-gsm8k.jsonl' 'evalset-' 'sha256' 'dataset_idx'; do
  grep -qF -- "$tok" "$ROOT/s4-04-prep-gsm8k.py" || { echo "FAIL: s4-04-prep-gsm8k.py에 '$tok' 없음"; fail=1; }
done
echo "OK: s4-04-prep-gsm8k.{py,sh} 존재 + 표준 GSM8K 계약"

# 2. s4-07-eval.py
grep -q 'BENCHMARKS = \["gsm8k"\]' "$ROOT/s4-07-eval.py" || { echo "FAIL: s4-07-eval.py BENCHMARKS가 gsm8k 단일이 아님"; fail=1; }
grep -q 's4-04-prep-gsm8k.sh' "$ROOT/s4-07-eval.py" || { echo "FAIL: s4-07-eval.py가 표준 프롬프트 미참조(실패 안내) 없음"; fail=1; }
for tok in 'batch_size' 'padding=bs > 1' 'padding_side = "left"' 'attention_mask=mask'; do
  grep -qF -- "$tok" "$ROOT/s4-07-eval.py" || { echo "FAIL: s4-07-eval.py에 '$tok' 없음"; fail=1; }
done
for _sh in s4-07-answer-benchmarks-base.sh s4-10-answer-benchmarks-quantized.sh; do
  grep -q 'EVAL_BENCHMARKS' "$ROOT/$_sh" || { echo "FAIL: $_sh에 EVAL_BENCHMARKS 없음"; fail=1; }
  grep -q '_missing_bench' "$ROOT/$_sh" || { echo "FAIL: $_sh freshness에 벤치 존재 검사 없음"; fail=1; }
  grep -q 'has no rows for bench' "$ROOT/$_sh" || { echo "FAIL: $_sh 누락 벤치 보고 없음"; fail=1; }
done
echo "OK: s4-07-answer-benchmarks-base/s4-10-answer-benchmarks-quantized 표준 GSM8K + 배치 생성"

# 3. s4-31-prep-upload.py 카드 2종 검사
for tok in '## Check 1 - PPL' '## Check 2 - GSM8K' '## Quantization spec' 'def _gsm8k_verdict(' 'def _gsm8k_acc_n(' 'def _evalset_meta('; do
  grep -qF -- "$tok" "$ROOT/s4-31-prep-upload.py" || { echo "FAIL: s4-31-prep-upload.py에 '$tok' 없음"; fail=1; }
done
echo "OK: 모델 카드 검사 2종(PPL·GSM8K)"

# 4. acceptance
grep -q 'gsm8k_acc_drop_max' "$ROOT/config/acceptance.toml" || { echo "FAIL: acceptance.toml에 gsm8k_acc_drop_max 없음"; fail=1; }
n_drop=$(grep -c 'gsm8k_acc_drop_max' "$ROOT/config/acceptance.toml")
[ "$n_drop" -ge 4 ] || { echo "FAIL: acceptance.toml의 gsm8k_acc_drop_max가 4타깃 미만($n_drop)"; fail=1; }
echo "OK: acceptance.toml GSM8K 기준"

# 5. gate
grep -q 'gsm8k_acc_drop_max' "$ROOT/gates.inc" || { echo "FAIL: gates.inc의 gate-eval이 GSM8K 기준 미사용"; fail=1; }
echo "OK: gate-eval GSM8K 대조"

# 6. 파이프라인 배선
for f in build-ppl-eval.inc; do
  grep -q 's4-04-prep-gsm8k.sh' "$ROOT/$f" || { echo "FAIL: $f 에 36 단계 없음"; fail=1; }
  bash -n "$ROOT/$f" || { echo "FAIL: $f bash -n"; fail=1; }
done
echo "OK: 36 단계 배선"

# 7. 단위 테스트
for t in tests/test_evalset.py tests/test_sdlc.py; do
  if "$PY" -c "import pytest" >/dev/null 2>&1; then
    "$PY" -m pytest "$ROOT/$t" -q >/tmp/verify-issue-11.log 2>&1 || { echo "FAIL: $t 실패"; tail -20 /tmp/verify-issue-11.log; fail=1; }
  else
    "$PY" "$ROOT/$t" >/tmp/verify-issue-11.log 2>&1 || { echo "FAIL: $t 실패"; grep -E '^FAIL' /tmp/verify-issue-11.log || true; fail=1; }
  fi
done
echo "OK: 단위 테스트 (test_evalset, test_sdlc)"

exit $fail
