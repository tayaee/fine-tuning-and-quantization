#!/usr/bin/env bash
# verify-issue-2.sh — issue-2 승인 기준 기계 검증
#   1. s2-01-dequant-mxfp8.py에 MXFP8BlockDequantizer(E8M0 2**(byte-127)) 정의
#   2. _apply_tracing_shim에 _routed_projection → nn.Linear 치환이 있고,
#      commit_dir 전에 verify_dequantized_bf16이 호출된다
#   3. 단위 테스트 (E8M0 수치 일치 / nn.Linear 치환 / 폭주·잔존 scale_inv 거부)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

# 1. E8M0 디퀀타이저 정의
for tok in 'class MXFP8BlockDequantizer(FP8BlockDequantizer)' \
           'def _create_dequantized_weight(self, weight, scale_inv, dtype=None)' \
           'torch.exp2(_e8m0_scale_bytes(scale_inv).to(torch.float32) - 127.0)' \
           'def _e8m0_scale_bytes(scale)' \
           'WEIGHT_BLOCK_SIZE = (1, 32)'; do
  grep -qF "$tok" "$ROOT/s2-01-dequant-mxfp8.py" \
    || { echo "FAIL: s2-01-dequant-mxfp8.py에 '$tok' 없음"; fail=1; }
done
grep -q 'converter=MXFP8BlockDequantizer(' "$ROOT/s2-01-dequant-mxfp8.py" \
  || { echo "FAIL: convert_checkpoint가 MXFP8BlockDequantizer를 쓰지 않음 (FP8 base면 폭주)"; fail=1; }
grep -q 'FP8BlockDequantizer' "$ROOT/s2-01-dequant-mxfp8.py" \
  || { echo "FAIL: FP8BlockDequantizer 상속 자체가 없음"; fail=1; }
echo "OK: MXFP8BlockDequantizer (2**(byte-127)) 정의 + 사용"

# 2. shim 치환 + commit_dir 이전 검증
grep -q 'return Raon1MiniRoutedLinear(in_features, out_features, bias=False)' "$ROOT/s2-01-dequant-mxfp8.py" \
  || { echo "FAIL: _routed_projection 치환 원본 문자열 없음"; fail=1; }
grep -q 'return nn.Linear(in_features, out_features, bias=False)' "$ROOT/s2-01-dequant-mxfp8.py" \
  || { echo "FAIL: _routed_projection 치환 대상 문자열 없음"; fail=1; }
grep -q '^def verify_dequantized_bf16(' "$ROOT/s2-01-dequant-mxfp8.py" \
  || { echo "FAIL: verify_dequantized_bf16 정의 없음"; fail=1; }
vline=$(grep -n 'verify_dequantized_bf16(tmp, snap)' "$ROOT/s2-01-dequant-mxfp8.py" || true)
cline=$(grep -n 'commit_dir(tmp, OUT)' "$ROOT/s2-01-dequant-mxfp8.py" || true)
if [ -z "$vline" ] || [ -z "$cline" ] || [ "${vline%%:*}" -ge "${cline%%:*}" ]; then
  echo "FAIL: verify_dequantized_bf16(line ${vline:-none}) 가 commit_dir(line ${cline%%:*}) 이후"; fail=1
else
  echo "OK: verify_dequantized_bf16(line ${vline%%:*}) < commit_dir(line ${cline%%:*})"
fi
for tok in 'weight_scale_inv' 'F8_E4M3' 'max_abs' 'isnan\|isfinite'; do
  grep -q "$tok" "$ROOT/s2-01-dequant-mxfp8.py" || { echo "FAIL: verify_dequantized_bf16에 '$tok' 검사 없음"; fail=1; }
done
echo "OK: _apply_tracing_shim 치환 + verify_dequantized_bf16 계약"

# 3. 단위 테스트
if "$PY" "$ROOT/tests/test_dequant_mxfp8.py" >/tmp/verify-issue-2.log 2>&1; then
  echo "OK: tests/test_dequant_mxfp8.py ($(grep -c '^PASS' /tmp/verify-issue-2.log) passed)"
else
  echo "FAIL: 단위 테스트 실패"; grep -E '^(FAIL|Traceback)' /tmp/verify-issue-2.log || true; fail=1
fi

# 4. s2-01-dequant-mxfp8.sh은 shim 유무까지 확인하고 문법 검사 통과
grep -q 'return nn.Linear(in_features, out_features, bias=False)' "$ROOT/s2-01-dequant-mxfp8.sh" \
  || { echo "FAIL: s2-01-dequant-mxfp8.sh fresh-skip이 nn.Linear shim 유무를 확인하지 않음"; fail=1; }
bash -n "$ROOT/s2-01-dequant-mxfp8.sh" || { echo "FAIL: s2-01-dequant-mxfp8.sh bash -n"; fail=1; }

exit $fail