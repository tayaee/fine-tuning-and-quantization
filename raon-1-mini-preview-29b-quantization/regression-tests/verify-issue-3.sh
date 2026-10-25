#!/usr/bin/env bash
# verify-issue-3.sh — issue-3 승인 기준 기계 검증
#   1. quant_common.verify_compressed_output 존재/계약 정의
#   2. 단위 테스트(정상 nvfp4/pack 픽스처 통과, dense 폴백 등 거부)
#   3. 4개 PTQ 스크립트가 commit_dir 전에 verify_compressed_output 호출
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

# 1. verify_compressed_output 정의 + 포맷별 계약
grep -q '^def verify_compressed_output(' "$ROOT/quant_common.py" \
  || { echo "FAIL: quant_common.verify_compressed_output 정의 없음"; fail=1; }
for tok in 'PACK_QUANTIZED = "pack-quantized"' \
           'NVFP4_PACK_QUANTIZED = "nvfp4-pack-quantized"' \
           '_PackedContract("I32", None, False, False)' \
           '_PackedContract("U8", "F8_E4M3", True, True)' \
           'configuration_raon1_mini.py' \
           'modeling_raon1_mini.py'; do
  grep -qF "$tok" "$ROOT/quant_common.py" \
    || { echo "FAIL: quant_common.py에 '$tok' 없음"; fail=1; }
done
grep -q 'weight_packed' "$ROOT/quant_common.py" || { echo "FAIL: weight_packed 검증 없음"; fail=1; }
grep -q 'weight_global_scale' "$ROOT/quant_common.py" || { echo "FAIL: weight_global_scale 검증 없음"; fail=1; }
grep -q 'weight_zero_point' "$ROOT/quant_common.py" || { echo "FAIL: weight_zero_point 검증 없음"; fail=1; }
echo "OK: quant_common.verify_compressed_output + 포맷 계약 정의"

# 2. 단위 테스트 (dense 폴백 거부 / 픽스처 통과 / custom code 누락 거부)
if "$PY" "$ROOT/tests/test_verify_compressed_output.py" >/tmp/verify-issue-3.log 2>&1; then
  echo "OK: tests/test_verify_compressed_output.py ($(grep -c '^PASS' /tmp/verify-issue-3.log) passed)"
else
  echo "FAIL: 단위 테스트 실패"; grep -E '^(FAIL|Traceback)' /tmp/verify-issue-3.log || true; fail=1
fi

# 3. PTQ 스크립트 4종: commit_dir 이전에 verify_compressed_output 호출
check_call() {  # <script> <expected constant>
  local f="$1" const="$2"
  local vline cline
  vline=$(grep -n "verify_compressed_output(tmp, $const)" "$ROOT/$f" || true)
  cline=$(grep -n "commit_dir(tmp, out)" "$ROOT/$f" || true)
  if [ -z "$vline" ]; then
    echo "FAIL: $f 에서 verify_compressed_output(tmp, $const) 호출 없음"; return 1
  fi
  if [ -z "$cline" ]; then
    echo "FAIL: $f 에 commit_dir 호출 없음"; return 1
  fi
  if [ "${vline%%:*}" -ge "${cline%%:*}" ]; then
    echo "FAIL: $f 에서 verify(line ${vline%%:*}) 가 commit_dir(line ${cline%%:*}) 이후"; return 1
  fi
  echo "OK: $f verify(line ${vline%%:*}) < commit_dir(line ${cline%%:*}) [$const]"
}
check_call s5-25-quant-awq-w5.py PACK_QUANTIZED || fail=1
check_call s5-22-quant-awq-w3.py PACK_QUANTIZED || fail=1
check_call s5-28-quant-nvfp4-awq.py NVFP4_PACK_QUANTIZED || fail=1
check_call s5-31-quant-nvfp4-gptq.py NVFP4_PACK_QUANTIZED || fail=1

# 4. PTQ 쉘 래퍼 4종: dense 폴백 디렉터리나 재생성된 raon-bf16을 fresh로 건너뛰지 않음
for sh in s5-25-quant-awq-w5.sh s5-22-quant-awq-w3.sh s5-28-quant-nvfp4-awq.sh s5-31-quant-nvfp4-gptq.sh; do
  grep -q '"dense"' "$ROOT/$sh" || { echo "FAIL: $sh fresh-skip이 format:dense를 거부하지 않음"; fail=1; }
  grep -q 'rq_fresh "\$OUT/config.json" "\$CALIB_FILE" "\$RQ_SRC/config.json"' "$ROOT/$sh" \
    || { echo "FAIL: $sh rq_fresh가 \$RQ_SRC/config.json(raon-bf16)을 확인하지 않음"; fail=1; }
  bash -n "$ROOT/$sh" || { echo "FAIL: $sh bash -n"; fail=1; }
done
echo "OK: PTQ .sh 4종 fresh-skip 가드 (dense 거부 + raon-bf16 타임스탬프 연동)"

exit $fail