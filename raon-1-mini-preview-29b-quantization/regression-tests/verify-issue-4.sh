#!/usr/bin/env bash
# verify-issue-4.sh — issue-4 승인 기준 기계 검증
#   1. s4-01-measure-ppl.py: --targets(콤마/공백/env)·--summarize-only·타겟별 json·summary 집계
#   2. s4-01-measure-ppl.sh: 타겟별 서브프로세스 + 마지막 집계
#   3. s4-07-eval.py: --backend {auto,vllm,transformers} + run_transformers + 폴백 판정
#   4. build-ppl-eval.inc: 타깃 전파 + QUANT_CPU_ONLY (구 run-{smoke,full}.sh는 archive/로 은퇴)
#   5. 단위 테스트 (tests/test_ppl_and_eval.py)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

# 1. s4-01-measure-ppl.py 계약
for tok in 'def parse_targets(' 'TARGET_ALIASES = {"bf16": "base"' \
           'def summarize(' 'def measure_targets(' 'def release_memory(' \
           'os.environ.get("TARGETS")' 'TARGETS = ["base", "awq-w5", "awq-w3", "nvfp4-awq", "nvfp4-gptq"]'; do
  grep -qF "$tok" "$ROOT/s4-01-measure-ppl.py" || { echo "FAIL: s4-01-measure-ppl.py에 '$tok' 없음"; fail=1; }
done
grep -q '"--summarize-only"' "$ROOT/s4-01-measure-ppl.py" || { echo "FAIL: --summarize-only 없음"; fail=1; }
grep -q 'f"{t}.json"' "$ROOT/s4-01-measure-ppl.py" || { echo "FAIL: 타겟별 <target>.json 저장 없음"; fail=1; }
grep -q 'summary.json' "$ROOT/s4-01-measure-ppl.py" || { echo "FAIL: summary.json 집계 없음"; fail=1; }
echo "OK: s4-01-measure-ppl.py 타겟 분리/집계 계약"

# 2. s4-01-measure-ppl.sh: 타겟마다 1 프로세스 + summarize-only
n_sub=$(grep -c 's4-01-measure-ppl.py" --mode "\$MODE" --n "\$N" --targets "\$t"' "$ROOT/s4-01-measure-ppl.sh" || true)
[ "$n_sub" = 1 ] || { echo "FAIL: s4-01-measure-ppl.sh 타겟별 서브프로세스 호출 수=$n_sub (expected 1, 루프 안)"; fail=1; }
grep -q -- '--summarize-only' "$ROOT/s4-01-measure-ppl.sh" || { echo "FAIL: s4-01-measure-ppl.sh에 --summarize-only 없음"; fail=1; }
# `all`을 명시 목록으로 펴야 5개 모델이 한 프로세스에 몰리지 않는다 (OOM 회피의 핵심)
grep -q 'PPL_ALL_TARGETS="base awq-w3 awq-w5 nvfp4-awq-modelopt nvfp4-awq nvfp4-gptq"' "$ROOT/s4-01-measure-ppl.sh" \
  || { echo "FAIL: s4-01-measure-ppl.sh가 'all'을 명시 목록으로 펼치지 않음 (한 프로세스로 5개 로드 → OOM)"; fail=1; }
if grep -qE '_t=all(\s|$)' "$ROOT/s4-01-measure-ppl.sh"; then
  echo "FAIL: s4-01-measure-ppl.sh가 'all'을 단일 타겟으로 통과시킴 (프로세스 격리 무효화)"; fail=1
fi
grep -q 'EVAL_ALL_TARGETS="awq-w3 awq-w5 nvfp4-awq-modelopt nvfp4-awq nvfp4-gptq"' "$ROOT/s4-10-eval-quantized.sh" \
  || { echo "FAIL: s4-10-eval-quantized.sh가 'all'을 명시 목록으로 펼치지 않음 (한 프로세스로 4개 로드)"; fail=1; }
if grep -v '^\s*#' "$ROOT/s4-10-eval-quantized.sh" | grep -q -- '--target all'; then
  echo "FAIL: s4-10-eval-quantized.sh에 --target all 단일 프로세스 경로가 남아 있음"; fail=1
fi
# 실제 루프가 몇 개 프로세스를 띄우는지 검증 (모델 로드 없이)
for _case in all "" nvfp4-awq nvfp4-gptq,nvfp4-awq; do
  _n=$(TARGETS="$_case" PPL_ALL_TARGETS="base awq-w3 awq-w5 nvfp4-awq-modelopt nvfp4-awq nvfp4-gptq" bash -c '
    _t=""
    for t in $(echo "$TARGETS" | tr "," " "); do
      if [ "$t" = all ]; then _t="$PPL_ALL_TARGETS"; break; fi
      _t="${_t:+$_t }$t"
    done
    [ -n "${_t:-}" ] || _t="$PPL_ALL_TARGETS"
    echo "$_t"')
  case "$_case" in
    all|"")           _want=5 ;;
    nvfp4-awq)         _want=1 ;;
    nvfp4-gptq,nvfp4-awq) _want=2 ;;
  esac
  _got=$(echo "$_n" | wc -w)
  [ "$_got" = "$_want" ] || { echo "FAIL: s4-01-measure-ppl.sh targets='$_case' → $_got procs (expected $_want)"; fail=1; }
done
echo "OK: s4-01-measure-ppl.sh/s4-10-eval-quantized.sh 프로세스 격리 + 'all' 펼침 + 캐시 집계"
bash -n "$ROOT/s4-01-measure-ppl.sh" || { echo "FAIL: s4-01-measure-ppl.sh bash -n"; fail=1; }
bash -n "$ROOT/s4-07-eval-base.sh" || { echo "FAIL: s4-07-eval-base.sh bash -n"; fail=1; }
bash -n "$ROOT/s4-10-eval-quantized.sh" || { echo "FAIL: s4-10-eval-quantized.sh bash -n"; fail=1; }

# 3. s4-07-eval.py 백엔드
for tok in 'def run_transformers(' 'def run_vllm(' 'def generate(' \
           'def is_unsupported_vllm_error(' 'def unsupported_vllm_error(' \
           'trust_remote_code=True' 'do_sample=False' 'max_new_tokens=max_tokens' 'use_cache=True'; do
  grep -qF "$tok" "$ROOT/s4-07-eval.py" || { echo "FAIL: s4-07-eval.py에 '$tok' 없음"; fail=1; }
done
grep -q '"--backend"' "$ROOT/s4-07-eval.py" || { echo "FAIL: --backend 인자 없음"; fail=1; }
grep -q 'RQ_EVAL_BACKEND' "$ROOT/s4-07-eval.py" || { echo "FAIL: RQ_EVAL_BACKEND env 폴백 없음"; fail=1; }
grep -q '"transformers"' "$ROOT/s4-07-eval.py" || { echo "FAIL: transformers 백엔드 선택지 없음"; fail=1; }
# 신규 토큰만 디코드해야 한다(단건: out[0][n_in:] / 배치: out[:, ids.size(1):])
grep -q 'decode(out\[0\]\[n_in:\])\|gen = out\[:, ids.size(1) :\]' "$ROOT/s4-07-eval.py" \
  || { echo "FAIL: 신규 토큰만 디코드하지 않음"; fail=1; }
echo "OK: s4-07-eval.py transformers 백엔드 + auto 폴백"

# 3b. s4-13-gsm8k.py 벤치마크 정확도 (issue-5가 score.json을 읽으므로 선행 조건)
for tok in 'def score_answer(' '"benchmarks"' 'score.json'; do
  grep -qF "$tok" "$ROOT/s4-13-gsm8k.py" || { echo "FAIL: s4-13-gsm8k.py에 '$tok' 없음"; fail=1; }
done
echo "OK: s4-13-gsm8k.py 벤치마크 정확도 산출"

# 4. build-ppl-eval.inc 파이프라인 연동 (단일 타깃 빌드 엔진)
f="$ROOT/build-ppl-eval.inc"
grep -q 's4-01-measure-ppl.sh "\$mode" "\$eval_targets"' "$f" \
  || { echo "FAIL: build 엔진이 TARGETS(=base 포함)를 s4-01에 전달하지 않음"; fail=1; }
grep -q 's4-07-eval-base.sh "\$mode"' "$f" \
  || { echo "FAIL: build 엔진이 base eval을 호출하지 않음"; fail=1; }
grep -q 's4-10-eval-quantized.sh "\$mode" "\$target"' "$f" \
  || { echo "FAIL: build 엔진이 target을 s4-10-eval-quantized에 전달하지 않음"; fail=1; }
grep -q 'QUANT_CPU_ONLY' "$f" || { echo "FAIL: build 엔진에 QUANT_CPU_ONLY 옵션 없음"; fail=1; }
grep -q 'quant_cmd=(env CUDA_VISIBLE_DEVICES=)' "$f" \
  || { echo "FAIL: build 엔진이 GPU 가시성을 자식 프로세스에만 제한하지 않음"; fail=1; }
grep -q '"s4-13-gsm8k" ./s4-13-gsm8k.sh "\$mode"' "$f" \
  || { echo "FAIL: build 엔진에서 s4-13-gsm8k 호출 체인 끊김"; fail=1; }
bash -n "$f" || { echo "FAIL: build-ppl-eval.inc bash -n"; fail=1; }
for w in "$ROOT"/run-s1-to-s4-*.sh; do
  grep -q 'source "\$ROOT/build-ppl-eval.inc"' "$w" \
    || { echo "FAIL: $(basename "$w")가 build 엔진을 source하지 않음"; fail=1; }
done
echo "OK: build-ppl-eval.inc 타깃 전파 + QUANT_CPU_ONLY + eval 체인"

# 5. 단위 테스트 (pytest 또는 단독 실행)
if "$PY" -c "import pytest" >/dev/null 2>&1; then
  if "$PY" -m pytest "$ROOT/tests/test_ppl_and_eval.py" -q >/tmp/verify-issue-4.log 2>&1; then
    echo "OK: tests/test_ppl_and_eval.py ($(grep -oE '[0-9]+ passed' /tmp/verify-issue-4.log | head -1))"
  else
    echo "FAIL: 단위 테스트 실패"; tail -20 /tmp/verify-issue-4.log; fail=1
  fi
else
  echo "WARN: pytest 없음 — 단독 러너로 대체"
  if "$PY" "$ROOT/tests/test_ppl_and_eval.py" >/tmp/verify-issue-4.log 2>&1; then
    echo "OK: tests/test_ppl_and_eval.py ($(grep -c '^PASS' /tmp/verify-issue-4.log) passed)"
  else
    echo "FAIL: 단위 테스트 실패"; grep -E '^(FAIL|SKIP)' /tmp/verify-issue-4.log || true; fail=1
  fi
fi

exit $fail