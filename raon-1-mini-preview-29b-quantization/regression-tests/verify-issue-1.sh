#!/usr/bin/env bash
# verify-issue-1.sh — issue-1 승인 기준 기계 검증
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0

# 1. s7-30-chat.sh: awq-w5-transformers/awq-w3-transformers/nvfp4-* 모두 QUANT="compressed-tensors"
for t in awq-w5-transformers awq-w3-transformers nvfp4-awq-transformers nvfp4-gptq-transformers; do
  line=$(grep -E "^[[:space:]]*${t}\)" "$ROOT/s7-30-chat.sh" || true)
  if ! echo "$line" | grep -q 'QUANT="compressed-tensors"'; then
    echo "FAIL: s7-30-chat.sh $t not compressed-tensors: $line"; fail=1
  else
    echo "OK: s7-30-chat.sh $t -> compressed-tensors"
  fi
done

# 2. QUANT="gptq" 잔존 금지
if grep -q 'QUANT="gptq"' "$ROOT/s7-30-chat.sh"; then
  echo 'FAIL: s7-30-chat.sh still contains QUANT="gptq"'; fail=1
else
  echo 'OK: no QUANT="gptq" in s7-30-chat.sh'
fi

# 3. smoke → -smoke repo 규칙 (rq_hf_parse 공용 헬퍼, s7-13이 사용)
grep -q 'rq_hf_parse "${2:-nvfp4-awq-transformers}"' "$ROOT/s7-30-chat.sh" \
  || { echo "FAIL: s7-30-chat.sh가 rq_hf_parse를 쓰지 않음"; fail=1; }
cnt=$(grep -c 'RQ_HF_REPO="$RQ_HF_REPO-smoke"' "$ROOT/config/common.env" || true)
if [ "$cnt" != 1 ]; then
  echo "FAIL: smoke 접미사 규칙 count=$cnt (expected 1)"; fail=1
else
  echo "OK: smoke → -smoke repo 규칙 exactly once"
fi

# 4. s4-eval-backend__transformers-only.py QUANT_OF 매핑 검증 (AST)
if ! python3 - "$ROOT/s4-eval-backend__transformers-only.py" <<'EOF'; then
import ast, sys
src = open(sys.argv[1], encoding="utf-8").read()
tree = ast.parse(src)
found = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "QUANT_OF" for t in node.targets):
        found = ast.literal_eval(node.value)
exp = {"nvfp4-awq-transformers": "compressed-tensors", "nvfp4-gptq-transformers": "compressed-tensors",
       "awq-w5-transformers": "compressed-tensors", "awq-w3-transformers": "compressed-tensors"}
assert found == exp, f"QUANT_OF mismatch: {found}"
print(f"OK: s4-eval-backend__transformers-only.py QUANT_OF={found}")
EOF
  echo "FAIL: s4-eval-backend__transformers-only.py QUANT_OF mapping wrong"; fail=1
fi

# 5. s4-15-answer-benchmarks-quantized__transformers-only.sh 콤마 분리 + 문법 검사 (base는 s4-07 단일 타깃이라 해당 없음)
if grep -q "tr ',' ' '" "$ROOT/s4-15-answer-benchmarks-quantized__transformers-only.sh"; then
  echo "OK: s4-15-answer-benchmarks-quantized__transformers-only.sh splits comma TARGETS"
else
  echo "FAIL: s4-15-answer-benchmarks-quantized__transformers-only.sh missing tr ',' ' ' split"; fail=1
fi
bash -n "$ROOT/s4-10-answer-benchmarks-base__transformers-only.sh" && echo "OK: s4-10-answer-benchmarks-base__transformers-only.sh bash -n" || { echo "FAIL: s4-10-answer-benchmarks-base__transformers-only.sh bash -n"; fail=1; }
bash -n "$ROOT/s4-15-answer-benchmarks-quantized__transformers-only.sh" && echo "OK: s4-15-answer-benchmarks-quantized__transformers-only.sh bash -n" || { echo "FAIL: s4-15-answer-benchmarks-quantized__transformers-only.sh bash -n"; fail=1; }
bash -n "$ROOT/s7-30-chat.sh" && echo "OK: s7-30-chat.sh bash -n" || { echo "FAIL: s7-30-chat.sh bash -n"; fail=1; }

exit $fail
