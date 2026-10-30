#!/usr/bin/env bash
# verify-issue-7.sh — issue-7(게이트 공통화) 승인 기준 기계 검증
#   1. gates.inc 존재 + 6종 게이트 정의
#   2. build-ppl-eval.inc에 gate-quant/gate-ppl/gate-eval 연결 + acceptance 단계
#      (구 run-{smoke,full}.sh는 archive/로 은퇴)
#   3. build-ppl-eval.inc 가 인라인 게이트를 버리고 rq_gate_* 사용
#   4. 단위 테스트 (tests/test_sdlc.py)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

# 1. gates.inc
[ -f "$ROOT/gates.inc" ] || { echo "FAIL: gates.inc 없음"; fail=1; }
for fn in rq_gate_acceptance rq_gate_quant rq_gate_ppl rq_gate_eval rq_gate_release rq_gate_serving; do
  grep -qF -- "$fn() {" "$ROOT/gates.inc" || { echo "FAIL: gates.inc에 '$fn' 없음"; fail=1; }
done
grep -q 'verify_compressed_output' "$ROOT/gates.inc" || { echo "FAIL: gate-quant가 verify_compressed_output 미사용"; fail=1; }
grep -q 'verdict_for_delta' "$ROOT/gates.inc" || { echo "FAIL: gate-ppl이 수용 기준 미대조"; fail=1; }
echo "OK: gates.inc 6종 게이트"

# 2. build 경로 연결
f="$ROOT/build-ppl-eval.inc"
grep -q '"gate-quant" rq_gate_quant' "$f" || { echo "FAIL: build 엔진에 gate-quant 없음"; fail=1; }
grep -q '"gate-ppl" rq_gate_ppl' "$f" || { echo "FAIL: build 엔진에 gate-ppl 없음"; fail=1; }
grep -q '"gate-eval" rq_gate_eval' "$f" || { echo "FAIL: build 엔진에 gate-eval 없음"; fail=1; }
grep -q '"s1-05-acceptance" ./s1-05-setup-acceptance-criteria.sh' "$f" || { echo "FAIL: build 엔진에 acceptance 단계 없음"; fail=1; }
bash -n "$f" || { echo "FAIL: build-ppl-eval.inc bash -n"; fail=1; }
echo "OK: build-ppl-eval.inc 게이트 + acceptance 단계"

# 3. build 경로
grep -q 'rq_gate_quant' "$ROOT/build-ppl-eval.inc" || { echo "FAIL: build-ppl-eval.inc에 rq_gate_quant 없음"; fail=1; }
grep -q 'rq_gate_ppl' "$ROOT/build-ppl-eval.inc" || { echo "FAIL: build-ppl-eval.inc에 rq_gate_ppl 없음"; fail=1; }
grep -q 'rq_gate_eval' "$ROOT/build-ppl-eval.inc" || { echo "FAIL: build-ppl-eval.inc에 rq_gate_eval 없음"; fail=1; }
if grep -q 'rq_bpe_gate_' "$ROOT/build-ppl-eval.inc"; then
  echo "FAIL: build-ppl-eval.inc에 인라인 게이트(rq_bpe_gate_*) 잔존"; fail=1
fi
grep -q 'gates.inc' "$ROOT/config/common.env" || { echo "FAIL: common.env가 gates.inc를 source하지 않음"; fail=1; }
bash -n "$ROOT/build-ppl-eval.inc" || { echo "FAIL: build-ppl-eval.inc bash -n"; fail=1; }
echo "OK: build-ppl-eval.inc 공용 게이트 사용"

# 4. 단위 테스트
if "$PY" -c "import pytest" >/dev/null 2>&1; then
  if "$PY" -m pytest "$ROOT/tests/test_sdlc.py" -q >/tmp/verify-issue-7.log 2>&1; then
    echo "OK: tests/test_sdlc.py ($(grep -oE '[0-9]+ passed' /tmp/verify-issue-7.log | head -1))"
  else
    echo "FAIL: 단위 테스트 실패"; tail -20 /tmp/verify-issue-7.log; fail=1
  fi
else
  if "$PY" "$ROOT/tests/test_sdlc.py" >/tmp/verify-issue-7.log 2>&1; then
    echo "OK: tests/test_sdlc.py ($(grep -c '^PASS' /tmp/verify-issue-7.log) passed)"
  else
    echo "FAIL: 단위 테스트 실패"; grep -E '^FAIL' /tmp/verify-issue-7.log || true; fail=1
  fi
fi

exit $fail
