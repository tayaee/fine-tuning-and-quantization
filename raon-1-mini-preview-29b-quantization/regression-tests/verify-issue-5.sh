#!/usr/bin/env bash
# verify-issue-5.sh — issue-5 승인 기준 기계 검증
#   1. 56-prep-upload.py/.sh 존재(+실행 권한), 검증/보충/카드 생성 계약
#   2. README 필수 섹션(frontmatter/사양/벤치표/사용법/라이선스) + extra_gated 금지
#   3. 58-upload-hf.py에 README 생성 로직이 없고 사전 검증만 한다 (+--dry-run)
#   4. run-{smoke,full}.sh에 56-prep 단계
#   5. 단위 테스트 (tests/test_prep_and_upload.py)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

# 0. 파일 존재 + 실행 권한
for f in 56-prep-upload.py 56-prep-upload.sh; do
  [ -f "$ROOT/$f" ] || { echo "FAIL: $f 없음"; fail=1; }
done
[ -x "$ROOT/56-prep-upload.sh" ] || { echo "FAIL: 56-prep-upload.sh에 실행 권한 없음"; fail=1; }
bash -n "$ROOT/56-prep-upload.sh" || { echo "FAIL: 56-prep-upload.sh bash -n"; fail=1; }
bash -n "$ROOT/58-upload-hf.sh" || { echo "FAIL: 58-upload-hf.sh bash -n"; fail=1; }
grep -q ': "${ALLOW_MISSING_METRICS:=0}"' "$ROOT/56-prep-upload.sh" \
  || { echo "FAIL: 56-prep-upload.sh에 ALLOW_MISSING_METRICS 기본값 초기화 없음 (set -u 크래시)"; fail=1; }
if grep -q '"$RQ_MODELS/raon-\*-$MODE/README.md"' "$ROOT/56-prep-upload.sh"; then
  echo "FAIL: 56-prep-upload.sh에 따옴표 안 글로브(*) 버그 잔존"; fail=1
fi
echo "OK: 56-prep-upload.{py,sh} 존재 + 실행 권한 + bash -n + set -u 안전"

# 1. 56-prep-upload.py 계약
for tok in 'def prepare_target(' 'def render_readme(' 'def run(' \
           'verify_compressed_output(' 'chat_template.jinja' \
           'configuration_raon1_mini.py' 'modeling_raon1_mini.py' \
           'tokenizer.json' 'tokenizer_config.json' 'config.json' \
           'prep-upload-' '"--allow-missing-metrics"' '"--targets"' '"--mode"'; do
  grep -qF -- "$tok" "$ROOT/56-prep-upload.py" || { echo "FAIL: 56-prep-upload.py에 '$tok' 없음"; fail=1; }
done
grep -q 'ppl-' "$ROOT/56-prep-upload.py" || { echo "FAIL: summary.json(PPL) 미참조"; fail=1; }
grep -q 'score.json' "$ROOT/56-prep-upload.py" || { echo "FAIL: score.json(eval) 미참조"; fail=1; }
echo "OK: 56-prep-upload.py 검증/보충/카드 계약"

# 2. README 카드 필수 내용 + 금지 필드
for tok in 'license: apache-2.0' 'base_model: ' 'base_model_relation: quantized' \
           'library_name: transformers' 'pipeline_tag: text-generation' \
           'da4cf162c8613993f2189ebb438a5f9a23f3e5ca' 'Apache' \
           'trust_remote_code=True' '--trust-remote-code' '--quantization compressed-tensors' \
           'FORBIDDEN_FRONTMATTER' 'extra_gated_'; do
  grep -qF -- "$tok" "$ROOT/56-prep-upload.py" || { echo "FAIL: 카드 템플릿에 '$tok' 없음"; fail=1; }
done
grep -q 'FORBIDDEN_FRONTMATTER' "$ROOT/56-prep-upload.py" \
  || { echo "FAIL: frontmatter 금지 필드 검사 없음"; fail=1; }
echo "OK: README 카드 섹션 + extra_gated_* 금지"

# 3. 58-upload-hf.py는 업로드 전담
for tok in 'def check_ready(' '56-prep-upload.sh first' '"--dry-run"' 'upload_folder' 'create_repo'; do
  grep -qF -- "$tok" "$ROOT/58-upload-hf.py" || { echo "FAIL: 58-upload-hf.py에 '$tok' 없음"; fail=1; }
done
for tok in 'render_readme' 'MODEL_CARD_TMPL' 'ppl-' 'score.json'; do
  if grep -qF -- "$tok" "$ROOT/58-upload-hf.py"; then
    echo "FAIL: 58-upload-hf.py에 메타데이터 생성 로직 잔존: '$tok'"; fail=1
  fi
done
if grep -qE 'write_text\(|\.write\(|open\([^)]*[\x27"]w[\x27"]' "$ROOT/58-upload-hf.py"; then
  echo "FAIL: 58-upload-hf.py가 파일을 씁니다 (업로드 전담이어야 함)"; fail=1
fi
echo "OK: 58-upload-hf.py 업로드 전담 (README 생성 로직 제거)"

# 4. 파이프라인 연동
for f in run-smoke.sh run-full.sh; do
  grep -q '"56-prep" ./56-prep-upload.sh "\$MODE"' "$ROOT/$f" \
    || { echo "FAIL: $f 에 56-prep 단계 없음"; fail=1; }
  bash -n "$ROOT/$f" || { echo "FAIL: $f bash -n"; fail=1; }
done
grep -q '56-prep-upload' "$ROOT/run-smoke.sh" || { echo "FAIL: run-smoke.sh validate 목록에 56-prep-upload 없음"; fail=1; }
echo "OK: run-{smoke,full}.sh 56-prep 단계"

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