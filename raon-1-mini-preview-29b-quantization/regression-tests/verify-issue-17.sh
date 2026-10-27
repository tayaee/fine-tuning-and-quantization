#!/usr/bin/env bash
# verify-issue-17.sh — issue-17(s7 평가 확장: 한국어 KMMLU·형식지시 준수·서빙 벤치 + 명시적 개명) 승인 기준 기계 검증
#   1. s4 평가 체계 존재 + 구 번호(s5/s6/s7/s8 접두) 0건
#   2. 신규 s7 스크립트 존재(43-prep-kmmluko, 46-kmmluko, 49-format, 52-servebench)
#   3. acceptance.toml: kmmluko_acc_drop_max / format_pass_rate_min
#   4. gates.inc: rq_gate_kmmluko / rq_gate_format
#   5. 카드(43-prep): Check 3(KMMLU) / Check 4(형식) / 서빙 성능 절
#   6. 파이프라인 배선 + 단위 테스트
#   7. 형식 검사 판정이 계약 하한(format_pass_rate_min) 기준 + prepare 모드 종료코드 0
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

# 1. s4 평가 체계 + 구 번호 0건
for f in s4-04-prep-gsm8k.sh s4-04-prep-gsm8k.py s4-13-gsm8k.sh s4-13-gsm8k.py \
         s4-16-prep-kmmluko.sh s4-16-prep-kmmluko.py s4-19-kmmluko.sh s4-19-kmmluko.py \
         s4-22-format.sh s4-22-format.py s4-25-servebench.sh s4-25-servebench.py; do
  [ -f "$ROOT/$f" ] || { echo "FAIL: $f 없음"; fail=1; }
done
if ls "$ROOT"/s8-*.sh "$ROOT"/s9-*.sh >/dev/null 2>&1; then
  echo "FAIL: 은퇴 번호 스크립트(s8/s9-*)가 루트에 남아 있음"; fail=1
fi
for f in s0-10-setup.sh s1-07-acceptance.sh s2-13-prep-calibration.sh s3-16-compat-check.sh \
         s4-19-dequant-mxfp8.sh s7-36-prep-evalset.sh s7-40-score.sh \
         s8-01-upload-hf.sh s9-01-vllm-serve-Raon-1-mini-preview-awq-w3.sh \
         s9-13-vllm-serve-one.sh s9-16-inference-test.sh s9-19-vllm-serve-all-models.sh \
         s9-25-verify-serving-log.sh; do
  [ ! -e "$ROOT/$f" ] || { echo "FAIL: 옛 이름 $f 가 남아 있음"; fail=1; }
done
echo "OK: s4 파일 체계 + 구 번호 0건"

# 2. 계약 기준
for tok in 'kmmluko_acc_drop_max' 'format_pass_rate_min'; do
  grep -q "$tok" "$ROOT/config/acceptance.toml" || { echo "FAIL: acceptance.toml에 '$tok' 없음"; fail=1; }
done
n=$(grep -c 'kmmluko_acc_drop_max' "$ROOT/config/acceptance.toml")
[ "$n" -ge 4 ] || { echo "FAIL: kmmluko_acc_drop_max가 4타깃 미만($n)"; fail=1; }
echo "OK: acceptance 계약(한국어·형식 기준)"

# 3. 게이트
for fn in rq_gate_kmmluko rq_gate_format; do
  grep -qF "$fn() {" "$ROOT/gates.inc" || { echo "FAIL: gates.inc에 '$fn' 없음"; fail=1; }
done
echo "OK: 게이트(한국어·형식)"

# 4. 카드 절
for tok in '## Check 3 - KMMLU' '## Check 4 - Instruction and format compliance' '## Serving performance'; do
  grep -qF "$tok" "$ROOT/s4-31-prep-upload.py" || { echo "FAIL: 카드에 '$tok' 없음"; fail=1; }
done
grep -qF '내용이 너무 짧음' "$ROOT/s4-34-verify-upload.py" || { echo "FAIL: 46-verify에 카드 실질 검사 없음"; fail=1; }
echo "OK: 카드 Check 3/4 + 서빙 성능"

# 5. 파이프라인 배선
for f in run-smoke.sh run-full.sh build-ppl-eval.inc; do
  for tok in 's4-16-prep-kmmluko.sh' 's4-19-kmmluko.sh' 's4-22-format.sh' 's4-25-servebench.sh'; do
    grep -qF "$tok" "$ROOT/$f" || { echo "FAIL: $f 에 '$tok' 배선 없음"; fail=1; }
  done
  bash -n "$ROOT/$f" || { echo "FAIL: $f bash -n"; fail=1; }
done
echo "OK: 파이프라인 배선"

# 6. 단위 테스트
if "$PY" -c "import pytest" >/dev/null 2>&1; then
  "$PY" -m pytest "$ROOT/tests/test_s7_checks.py" -q >/tmp/verify-issue-17.log 2>&1 \
    || { echo "FAIL: tests/test_s7_checks.py 실패"; tail -20 /tmp/verify-issue-17.log; fail=1; }
else
  "$PY" "$ROOT/tests/test_s7_checks.py" >/tmp/verify-issue-17.log 2>&1 \
    || { echo "FAIL: tests/test_s7_checks.py 실패"; grep -E '^FAIL' /tmp/verify-issue-17.log || true; fail=1; }
fi
echo "OK: 단위 테스트 (test_s7_checks)"

# 7. 계약 대조 판정(가짜 통과 방지) + prepare 종료코드
grep -qF 'pass_rate_min' "$ROOT/s4-22-format.py" \
  || { echo "FAIL: s4-22-format.py가 계약 하한(pass_rate_min)을 쓰지 않음"; fail=1; }
grep -qF '"verdict"' "$ROOT/s4-22-format.py" \
  || { echo "FAIL: s4-22-format.py가 verdict를 산출하지 않음"; fail=1; }
# 판정은 통과율이 아니라 계약 하한으로 — 모델 카드가 verdict를 그대로 쓴다
grep -qF 'checks.get("format") or {}).get("verdict")' "$ROOT/s4-31-prep-upload.py" \
  || { echo "FAIL: s4-31-prep-upload.py가 형식 verdict를 계약 기준으로 대조하지 않음"; fail=1; }
_tmprq="$(mktemp -d)"
if RQ_SHARED="$_tmprq" "$ROOT/s4-22-format.sh" smoke --prepare >/dev/null 2>&1; then
  :
else
  echo "FAIL: s4-22-format.sh --prepare 가 0이 아닌 상태로 종료(파이프라인 중단)"; fail=1
fi
rm -rf "$_tmprq"
echo "OK: 형식 판정 = 계약 대조 + prepare 종료코드"

exit $fail
