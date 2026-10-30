#!/usr/bin/env bash
# s1-10-prep-calibration.sh [smoke|full] [n] [seed] — 한국어 혼합 → raon-calibration-<n>.jsonl.
# smoke 기본 8 (10분 목표), full 기본 256.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
N="${2:-$CALIB_N}"
SEED="${3:-42}"
rq_repro_echo
rq_banner "s1-10-prep-calibration" "Stage 1 - build the calibration set (Korean HF datasets -> datasets/raon-calibration-<N>.jsonl)"
OUT="$RQ_DATASETS/raon-calibration-$N.jsonl"
META="$OUT.meta"
# 동시 build(8종 wrapper 병렬 실행)가 같은 calib 파일을 동시에 만들지 않게 한다.
# 먼저 들어온 쪽이 생성하고, 기다린 쪽은 freshness로 즉시 skip한다.
rq_lock "$RQ_DATASETS"
# 존재하면 재빌드하지 않는다 (mtime 비교 안 함 — 스크립트 체크아웃 등으로
# mtime만 바뀌어도 산출물은 그대로 유효하므로. 강제 재생성은 FORCE=1).
if [ -f "$OUT" ] && [ "${FORCE:-0}" != 1 ]; then
  rq_log "skip: exists $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -lh "$OUT")
  (set -x; wc -l "$OUT")
  [ -f "$META" ] && (set -x; cat "$META")
  (set -x; sha256sum "$OUT" | cut -c1-64)
  exit 0
fi
# mtime freshness 실패해도 내용이 유효하면 재다운로드하지 않는다.
# (py 주석/리팩터 수준 변경 시 HF streaming 256 재생성 방지. 강제 재생성은 FORCE=1)
if [ "${FORCE:-0}" != 1 ] && [ -f "$OUT" ] && [ -f "$META" ] \
  && grep -qx "n=$N" "$META" 2>/dev/null && grep -qx "seed=$SEED" "$META" 2>/dev/null \
  && [ "$(wc -l <"$OUT")" = "$N" ] \
  && "$RQ_PY" - "$OUT" <<'PYEOF' 2>/dev/null; then
import json, sys
p = sys.argv[1]
with open(p, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        assert isinstance(r.get("text"), str) and r["text"].strip(), "empty text"
print("cache-valid")
PYEOF
  rq_log "skip: valid-cache $OUT (n=$N seed=$SEED, meta match; FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -lh "$OUT")
  (set -x; wc -l "$OUT")
  (set -x; cat "$META")
  (set -x; sha256sum "$OUT" | cut -c1-64)
  exit 0
fi
export UV_NO_SYNC=1; unset VIRTUAL_ENV
(set -x; uv run "$RQ_ROOT/s1-10-prep-calibration.py" --n "$N" --seed "$SEED")

echo ---- result ----
(set -x; ls -lh "$OUT")
(set -x; wc -l "$OUT")
[ -f "$META" ] && (set -x; cat "$META")
(set -x; sha256sum "$OUT" | cut -c1-64)
