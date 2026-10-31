#!/usr/bin/env bash
# verify-targets-guard.sh — 타깃 명시 강제(rq_require_targets) 기계 검증
#   1. 타깃을 받는 전 스크립트: $2·TARGETS 미지정 → "TARGETS is not specified" + exit 2 (작업 0건)
#   2. rq_require_targets 단위 동작 (명시값 통과 / 미지정 2 / TARGETS=all 통과)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0

GUARDED="s4-00-measure-ppl__transformers-only.sh s4-15-answer-benchmarks-quantized__transformers-only.sh s4-50-answer-via-server.sh
  s4-65-measure-speed__transformers-only.sh s4-70-update-readme.sh s4-75-verify-upload.sh
  s5-00-upload-to-hf.sh"

for f in config/common.env build-ppl-eval.inc $GUARDED; do
  bash -n "$ROOT/$f" || { echo "FAIL: $f bash -n"; fail=1; }
done
echo "OK: bash -n"

# 1. 미지정 → exit 2 + 사용법
for f in $GUARDED; do
  out="$(cd "$ROOT" && env -u TARGETS ./$f smoke 2>&1)" && rc=0 || rc=$?
  [ "$rc" = 2 ] || { echo "FAIL: $f 미지정 시 exit=$rc (expected 2)"; fail=1; continue; }
  echo "$out" | grep -q "TARGETS is not specified" || { echo "FAIL: $f 사용법 안내 없음"; fail=1; continue; }
  echo "$out" | grep -q "Available values" || { echo "FAIL: $f 가능 값 안내 없음"; fail=1; continue; }
  echo "$out" | grep -q "step start:" && { echo "FAIL: $f 미지정인데 스텝 실행됨"; fail=1; continue; }
  echo "OK: $f 미지정 → exit 2"
done

# 2. helper 단위 동작 (가벼움 — common.env만 source)
rc=0
( source "$ROOT/config/common.env" smoke >/dev/null 2>&1
  rq_require_targets "./x smoke" "" awq-w3-transformers nvfp4-awq-transformers >/dev/null 2>&1 ) || rc=$?
[ "$rc" = 2 ] || { echo "FAIL: helper 미지정 status=$rc (expected 2)"; fail=1; }
echo "OK: helper 미지정 → 2"

v="$(source "$ROOT/config/common.env" smoke >/dev/null 2>&1; rq_require_targets "./x smoke" "nvfp4-awq-transformers" awq-w3-transformers nvfp4-awq-transformers)"
[ "$v" = "nvfp4-awq-transformers" ] || { echo "FAIL: helper 명시값 반환='$v'"; fail=1; }
v="$(cd "$ROOT" && TARGETS=all bash -c 'source config/common.env smoke >/dev/null 2>&1; rq_require_targets "./x smoke" "" awq-w3-transformers nvfp4-awq-transformers')"
[ "$v" = "all" ] || { echo "FAIL: helper TARGETS=all 반환='$v'"; fail=1; }
echo "OK: helper 명시값/all 통과"

# 3. Hub repo id 규칙 단일 소스 — 카드(s4-70)와 업로드(s5-00)가 같아야 한다
#    (어긋나면 카드만 A 저장소, 가중치는 B 저장소에 올라간다 — 실제로 그렇게 됐었다)
grep -q "from sdlc import hf_repo_id" "$ROOT/s5-00-upload-to-hf.py" \
  || { echo "FAIL: s5-00-upload-to-hf.py 가 sdlc.hf_repo_id 를 쓰지 않는다"; fail=1; }
grep -q "from sdlc import hf_repo_id as _hf_repo_id" "$ROOT/s4-70-update-readme.py" \
  || { echo "FAIL: s4-70-update-readme.py 가 sdlc.hf_repo_id 를 쓰지 않는다"; fail=1; }
grep -q "Raon-1-mini-preview-NVFP4-AWQ-GB10-vLLM" "$ROOT/sdlc.py" \
  || { echo "FAIL: sdlc.py 에 vLLM 전용 타깃의 정식 repo slug 가 없다"; fail=1; }
PYBIN="${PY:-$ROOT/.venv/bin/python}"
if [ -x "$PYBIN" ]; then
  out="$(cd "$ROOT" && "$PYBIN" - <<'PY'
import importlib.util, sys
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_').removesuffix('.py'), ROOT / name)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


up = load('s5-00-upload-to-hf.py')
prep = load('s4-70-update-readme.py')
bad = [f"{t}/{m}" for t in up.ALL_TARGETS for m in ('full', 'smoke')
       if up.repo_name(t, m) != prep.hf_repo_id(t, m)]
print('MISMATCH ' + ','.join(bad) if bad else 'OK')
PY
)"
  [ "$out" = "OK" ] || { echo "FAIL: 카드/업로드 repo id 불일치 — $out"; fail=1; }
else
  echo "WARN: python 없음 — 정적 검사만 수행"
fi
echo "OK: 카드/업로드 repo id 단일 소스"

exit $fail
