#!/usr/bin/env bash
# verify-issue-5.sh — issue-5 승인 기준 기계 검증
#   1. s4-65-update-readme.py/.sh 존재(+실행 권한), 검증/보충/카드 생성 계약
#   2. README 필수 섹션(frontmatter/사양/벤치표/사용법/라이선스) + extra_gated 금지
#   3. s5-01-upload-to-hf.py에 README 생성 로직이 없고 사전 검증만 한다 (+--dry-run)
#   4. build-ppl-eval.inc에 s4-31-prep 단계 (구 run-{smoke,full}.sh는 archive/)
#   5. 단위 테스트 (tests/test_prep_and_upload.py)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

# 0. 파일 존재 + 실행 권한
for f in s4-65-update-readme.py s4-65-update-readme.sh; do
  [ -f "$ROOT/$f" ] || { echo "FAIL: $f 없음"; fail=1; }
done
[ -x "$ROOT/s4-65-update-readme.sh" ] || { echo "FAIL: s4-65-update-readme.sh에 실행 권한 없음"; fail=1; }
bash -n "$ROOT/s4-65-update-readme.sh" || { echo "FAIL: s4-65-update-readme.sh bash -n"; fail=1; }
bash -n "$ROOT/s5-01-upload-to-hf.sh" || { echo "FAIL: s5-01-upload-to-hf.sh bash -n"; fail=1; }
grep -q ': "${ALLOW_MISSING_METRICS:=0}"' "$ROOT/s4-65-update-readme.sh" \
  || { echo "FAIL: s4-65-update-readme.sh에 ALLOW_MISSING_METRICS 기본값 초기화 없음 (set -u 크래시)"; fail=1; }
if grep -q '"$RQ_MODELS/raon-\*-$MODE/README.md"' "$ROOT/s4-65-update-readme.sh"; then
  echo "FAIL: s4-65-update-readme.sh에 따옴표 안 글로브(*) 버그 잔존"; fail=1
fi
echo "OK: s4-31-prep-upload.{py,sh} 존재 + 실행 권한 + bash -n + set -u 안전"

# 1. s4-65-update-readme.py 계약
for tok in 'def prepare_target(' 'def render_readme(' 'def run(' \
           'verify_compressed_output(' 'chat_template.jinja' \
           'configuration_raon1_mini.py' 'modeling_raon1_mini.py' \
           'tokenizer.json' 'tokenizer_config.json' 'config.json' \
           'prep-upload-' '"--allow-missing-metrics"' '"--targets"' '"--mode"'; do
  grep -qF -- "$tok" "$ROOT/s4-65-update-readme.py" || { echo "FAIL: s4-65-update-readme.py에 '$tok' 없음"; fail=1; }
done
grep -q 'ppl-' "$ROOT/s4-65-update-readme.py" || { echo "FAIL: summary.json(PPL) 미참조"; fail=1; }
grep -q 'score.json' "$ROOT/s4-65-update-readme.py" || { echo "FAIL: score.json(eval) 미참조"; fail=1; }
echo "OK: s4-65-update-readme.py 검증/보충/카드 계약"

# 2. README 카드 필수 내용 + 금지 필드 (실행 예제는 snippets/ 단일 소스)
for tok in 'license: apache-2.0' 'base_model: ' 'base_model_relation: quantized' \
           'library_name: transformers' 'pipeline_tag: text-generation' \
           'da4cf162c8613993f2189ebb438a5f9a23f3e5ca' 'Apache' \
           'FORBIDDEN_FRONTMATTER' 'extra_gated_'; do
  grep -qF -- "$tok" "$ROOT/s4-65-update-readme.py" || { echo "FAIL: 카드 템플릿에 '$tok' 없음"; fail=1; }
done
for tok in 'trust_remote_code=True'; do
  grep -qF -- "$tok" "$ROOT/snippets/nvfp4-awq-transformers/app-transformers.py" \
    || { echo "FAIL: 스니펫에 '$tok' 없음"; fail=1; }
done
for _t in awq-w3-transformers awq-w5-transformers nvfp4-awq-vllm nvfp4-awq-transformers nvfp4-gptq-transformers; do
  grep -qF -- '--trust-remote-code' "$ROOT/snippets/$_t/serve-vllm.sh" \
    || { echo "FAIL: snippets/$_t/serve-vllm.sh에 '--trust-remote-code' 없음"; fail=1; }
done
grep -qF -- '--quantization compressed-tensors' "$ROOT/snippets/nvfp4-awq-transformers/serve-vllm.sh" \
  || { echo "FAIL: 스니펫에 '--quantization compressed-tensors' 없음"; fail=1; }
grep -qF -- 'compressed-tensors' "$ROOT/snippets/nvfp4-awq-transformers/usage-transformers.sh" \
  || { echo "FAIL: 스니펫에 'compressed-tensors' 없음"; fail=1; }
grep -q 'FORBIDDEN_FRONTMATTER' "$ROOT/s4-65-update-readme.py" \
  || { echo "FAIL: frontmatter 금지 필드 검사 없음"; fail=1; }
echo "OK: README 카드 섹션 + extra_gated_* 금지"

# 3. s5-01-upload-to-hf.py는 업로드 전담
for tok in 'def check_ready(' 's4-65-update-readme.sh first' '"--dry-run"' 'upload_folder' 'create_repo'; do
  grep -qF -- "$tok" "$ROOT/s5-01-upload-to-hf.py" || { echo "FAIL: s5-01-upload-to-hf.py에 '$tok' 없음"; fail=1; }
done
for tok in 'render_readme' 'MODEL_CARD_TMPL' 'ppl-' 'score.json'; do
  if grep -qF -- "$tok" "$ROOT/s5-01-upload-to-hf.py"; then
    echo "FAIL: s5-01-upload-to-hf.py에 메타데이터 생성 로직 잔존: '$tok'"; fail=1
  fi
done
if grep -qE 'write_text\(|\.write\(|open\([^)]*[\x27"]w[\x27"]' "$ROOT/s5-01-upload-to-hf.py"; then
  echo "FAIL: s5-01-upload-to-hf.py가 파일을 씁니다 (업로드 전담이어야 함)"; fail=1
fi
echo "OK: s5-01-upload-to-hf.py 업로드 전담 (README 생성 로직 제거)"

# 4. 파이프라인 연동 (단일 타깃 빌드 엔진; 구 run-{smoke,full}.sh는 archive/)
grep -q '"s4-65-prep" ./s4-65-update-readme.sh "\$mode"' "$ROOT/build-ppl-eval.inc" \
  || { echo "FAIL: build-ppl-eval.inc에 s4-65-prep 단계 없음"; fail=1; }
bash -n "$ROOT/build-ppl-eval.inc" || { echo "FAIL: build-ppl-eval.inc bash -n"; fail=1; }
for w in "$ROOT"/run-s1-to-s4-*.sh; do
  bash -n "$w" || { echo "FAIL: $(basename "$w") bash -n"; fail=1; }
done
echo "OK: build-ppl-eval.inc s4-65-prep 단계 + wrapper 10종"

# 5. 단위 테스트
if "$PY" -c "import pytest" >/dev/null 2>&1; then
  if "$PY" -m pytest "$ROOT/tests/test_prep_and_upload.py" -q >/tmp/verify-issue-5.log 2>&1; then
    echo "OK: tests/test_prep_and_upload.py ($(grep -oE '[0-9]+ passed' /tmp/verify-issue-5.log | head -1))"
  else
    echo "FAIL: 단위 테스트 실패"; tail -20 /tmp/verify-issue-5.log; fail=1
  fi
else
  echo "WARN: pytest 없음 — 단독 러너로 대체"
  if "$PY" "$ROOT/tests/test_prep_and_upload.py" >/tmp/verify-issue-5.log 2>&1; then
    echo "OK: tests/test_prep_and_upload.py ($(grep -c '^PASS' /tmp/verify-issue-5.log) passed)"
  else
    echo "FAIL: 단위 테스트 실패"; grep -E '^FAIL' /tmp/verify-issue-5.log || true; fail=1
  fi
fi

exit $fail