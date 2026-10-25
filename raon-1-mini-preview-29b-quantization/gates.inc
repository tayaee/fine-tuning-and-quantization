# gates.inc — 9단계 SDLC 게이트 공용 모음 (bash).
#
# 사용처: run-{smoke,full}.sh, build-ppl-eval.inc
#   (두 곳 모두 config/common.env를 먼저 source한 뒤 이 파일을 source한다.)
#
# 계약: 모든 게이트는 "통과=0, 실패=1"을 반환하고, 실패 사유를 stderr에 남긴다.
# 게이트는 산출물을 *먼저* 검사한다 — 즉 다음 단계로 넘어가기 전에
# "패킹 안 된 dense 체크포인트 / NaN PPL / 응답 0건"을 여기서 막는다.

# --- 공통 파이썬 실행 헬퍼 (sdlc 모듈 import를 위해 RQ_ROOT를 sys.path에 넣는다) ---
# common.env 없이 단독 source되어도 동작하도록 인터프리터를 보장한다.
if [ -z "${RQ_PY:-}" ]; then
  RQ_PY="${RQ_ROOT:-.}/.venv/bin/python"
  [ -x "$RQ_PY" ] || RQ_PY="$(command -v python3 || true)"
fi
rq_py() {
  PYTHONPATH="${RQ_ROOT:-.}${PYTHONPATH:+:$PYTHONPATH}" "$RQ_PY" "$@"
}

# --- 게이트 0: Stage 1 계약(acceptance) 유효성 ------------------------------
rq_gate_acceptance() {
  local mode="${1:-$MODE}"
  local f="$RQ_OUTPUTS/acceptance-$mode.json"
  if [ ! -s "$f" ]; then
    echo "FAIL: $f 없음 — Stage 1(수용 기준) 미수행 (run ./s1-07-acceptance.sh $mode)" >&2
    return 1
  fi
  ( set -x; rq_py - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("version"), "version 누락"
assert d.get("targets"), "targets 누락"
for t in ("awq-w3", "awq-w5", "nvfp4-awq", "nvfp4-gptq"):
    spec = d["targets"].get(t)
    assert spec, f"targets.{t} 누락"
    assert spec.get("quant_format"), f"targets.{t}.quant_format 누락"
    assert spec.get("ppl_delta_max") is not None, f"targets.{t}.ppl_delta_max 누락"
print("acceptance OK", sys.argv[1])
PY
  )
}

# --- 게이트 1: 양자화 산출물 계약 (dense 폴백·미보정 global_scale 차단) --------
# quant_common.verify_compressed_output는 양자화 스크립트 내부에서도 호출되지만,
# 그 검사는 "새로 빌드할 때"만 걸린다. freshness로 건너뛴 기존 산출물이
# 오염돼 있으면(=이번에 재빌드하지 않았어도) 여기서 잡아낸다.
rq_gate_quant() {
  local model_dir="$1" fmt="$2"
  [ -d "$model_dir" ] || { echo "FAIL: 모델 디렉터리 없음: $model_dir" >&2; return 1; }
  ( set -x; cd "$RQ_ROOT" && rq_py - "$model_dir" "$fmt" <<'PY'
import sys
from quant_common import verify_compressed_output
try:
    verify_compressed_output(sys.argv[1], sys.argv[2])
except Exception as e:
    print("FAIL: %s" % e, file=sys.stderr)
    raise SystemExit(1)
print("quant OK", sys.argv[1], sys.argv[2])
PY
  )
}

# --- 게이트 2: PPL 유효성 + 수용 기준 일치 ----------------------------------
# NaN/미측정이면 '오염된 모델'을 eval에 흘려보내지 않는다.
# (s6-34-measure-ppl.py는 NaN이면 <target>.json을 아예 쓰지 않는다.)
rq_gate_ppl() {
  local f="$1" target="${2:-}"
  if [ ! -f "$f" ]; then
    echo "FAIL: $f 없음 — PPL 미측정(유한 토큰 0/NaN). 양자화 산출물이 오염됐을 가능성" >&2
    return 1
  fi
  ( set -x; rq_py - "$f" "$target" <<'PY'
import json, math, os, sys
from sdlc import load_acceptance, target_spec, verdict_for_delta

path, target = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
d = json.load(open(path))
ppl = d.get("ppl")
if ppl is None or not math.isfinite(float(ppl)):
    raise SystemExit("FAIL: %s ppl=%r (비유한 값)" % (path, ppl))
if target:
    # Δ/verdict은 summary.json(집계본)에 있다 — 타깃별 json은 측정값만 가진다.
    summary_path = os.path.join(os.path.dirname(path) or ".", "summary.json")
    if not os.path.isfile(summary_path):
        print("WARN: %s 없음 — 수용 기준 판정 대조 생략" % summary_path)
    else:
        summary = json.load(open(summary_path))
        row = summary.get(target) or {}
        delta = row.get("delta")
        if delta is None:
            print("WARN: %s에 %s의 delta 없음 — 판정 대조 생략" % (summary_path, target))
        else:
            spec = target_spec(load_acceptance(), target)
            expect = verdict_for_delta(float(delta), spec)
            got = row.get("verdict")
            if got and got != expect:
                raise SystemExit(
                    "FAIL: %s verdict=%s != 수용 기준 판정 %s (Δ=%.4f)"
                    % (summary_path, got, expect, float(delta))
                )
print("PPL OK %s ppl=%s n=%s seq_len=%s" % (path, ppl, d.get("n"), d.get("seq_len")))
PY
  )
}

# --- 게이트 3: eval 응답 생성 여부 + 표준 GSM8K 정확도 계약 대조 ---------------
rq_gate_eval() {
  local f="$1" target="${2:-}" mode="${3:-$MODE}"
  if [ ! -s "$f" ]; then
    echo "FAIL: $f 없음/빈 파일 — eval 응답 미생성" >&2
    return 1
  fi
  local n
  n=$(wc -l <"$f")
  if [ "${n:-0}" -lt 1 ]; then
    echo "FAIL: $f 라인 0 — eval 응답 미생성" >&2
    return 1
  fi
  echo "eval OK $f lines=$n"
  # 표준 GSM8K 정확도: base 대비 하락이 수용 기준 허용폭 이내인가.
  [ -n "$target" ] || return 0
  local score="$RQ_OUTPUTS/eval-$mode/score.json"
  [ -f "$score" ] || { echo "SKIP: $score 없음 — GSM8K 정확도 대조 생략"; return 0; }
  ( set -x; rq_py - "$score" "$target" <<'PY'
import json, sys
from sdlc import load_acceptance, target_spec

score = json.load(open(sys.argv[1]))
target = sys.argv[2]
base = ((score.get("base") or {}).get("benchmarks") or {}).get("gsm8k") or {}
cur = ((score.get(target) or {}).get("benchmarks") or {}).get("gsm8k") or {}
if base.get("acc") is None or cur.get("acc") is None:
    print("SKIP: GSM8K 정확도 비교 데이터 없음 (base=%s %s=%s)" % (base.get("acc"), target, cur.get("acc")))
    raise SystemExit(0)
spec = target_spec(load_acceptance(), target)
drop_max = float(spec.get("gsm8k_acc_drop_max", 0.02))
drop = float(base["acc"]) - float(cur["acc"])
if drop > drop_max:
    raise SystemExit(
        "FAIL: GSM8K 정확도 하락 %.3f > 허용폭 %.3f (base=%.3f %s=%.3f, n=%s)"
        % (drop, drop_max, float(base["acc"]), target, float(cur["acc"]), cur.get("n"))
    )
print("GSM8K OK %s acc=%.3f (base=%.3f, 하락 %.3f <= %.3f, n=%s)"
      % (target, float(cur["acc"]), float(base["acc"]), drop, drop_max, cur.get("n")))
PY
  )
}

# --- 게이트 4: Stage 8 릴리스 준비 (모델카드 + 지표 + 업로드 전 감사) ---------
# s8-43-prep-upload(모델카드/디렉터리 보충) + s8-46-verify-upload(전수 감사)의 산출물을 대조한다.
rq_gate_release() {
  local mode="${1:-$MODE}"
  local prep="$RQ_OUTPUTS/prep-upload-$mode.json"
  local ver="$RQ_OUTPUTS/verify-upload-$mode.json"
  local rc=0
  [ -s "$prep" ] || { echo "FAIL: $prep 없음 — s8-43-prep-upload 미수행" >&2; rc=1; }
  [ -s "$ver" ] || { echo "FAIL: $ver 없음 — s8-46-verify-upload 미수행" >&2; rc=1; }
  [ "$rc" -eq 0 ] || return 1
  ( set -x; rq_py - "$prep" "$ver" <<'PY'
import json, sys

prep = json.load(open(sys.argv[1]))
ver = json.load(open(sys.argv[2]))
bad = []
if not prep.get("ok"):
    bad.append(f"prep-upload ok=false: {prep.get('errors') or prep.get('targets')}")
if not ver.get("ok"):
    bad.append(f"verify-upload ok=false: {ver.get('bf16_errors') or ''}")
if bad:
    for b in bad:
        print("FAIL: " + b, file=sys.stderr)
    raise SystemExit(1)
print("release OK", sys.argv[1], sys.argv[2])
PY
  )
}

# --- 게이트 4b: 표준 한국어 KMMLU 정확도 계약 대조 ---------------------------
rq_gate_kmmluko() {
  local mode="${1:-$MODE}" target="${2:-}"
  local f="$RQ_OUTPUTS/eval-$mode/checks.json"
  [ -f "$f" ] || { echo "SKIP: $f 없음 — 한국어 검사 미수행 (s7-46-kmmluko.sh)"; return 0; }
  [ -n "$target" ] || return 0
  ( set -x; rq_py - "$f" "$target" <<'PY'
import json, sys
from sdlc import load_acceptance, target_spec

data = json.load(open(sys.argv[1]))
target = sys.argv[2]
sec = (data.get("kmmluko") or {})
targets = sec.get("targets") or {}
base = (targets.get("base") or {}).get("acc")
cur = (targets.get(target) or {}).get("acc")
if base is None or cur is None:
    print("SKIP: KMMLU 정확도 비교 데이터 없음 (base=%s %s=%s)" % (base, target, cur))
    raise SystemExit(0)
spec = target_spec(load_acceptance(), target)
drop_max = float(spec.get("kmmluko_acc_drop_max", 0.03))
drop = float(base) - float(cur)
if drop > drop_max:
    raise SystemExit("FAIL: KMMLU 정확도 하락 %.3f > 허용폭 %.3f (base=%.3f %s=%.3f)"
                     % (drop, drop_max, float(base), target, float(cur)))
print("KMMLU OK %s acc=%.3f (base=%.3f, 하락 %.3f <= %.3f)" % (target, float(cur), float(base), drop, drop_max))
PY
  )
}

# --- 게이트 4c: 지시/형식 준수 스위트 ---------------------------------------
rq_gate_format() {
  local mode="${1:-$MODE}" target="${2:-}"
  local f="$RQ_OUTPUTS/eval-$mode/checks.json"
  [ -f "$f" ] || { echo "SKIP: $f 없음 — 형식 검사 미수행 (s7-49-format.sh)"; return 0; }
  [ -n "$target" ] || return 0
  ( set -x; rq_py - "$f" "$target" <<'PY'
import json, sys
from sdlc import load_acceptance, target_spec

data = json.load(open(sys.argv[1]))
target = sys.argv[2]
sec = (data.get("format") or {})
targets = sec.get("targets") or {}
row = targets.get(target) or {}
rate = row.get("pass_rate")
if rate is None:
    print("SKIP: 형식 검사 결과 없음 (%s)" % target)
    raise SystemExit(0)
spec = target_spec(load_acceptance(), target)
min_rate = float(spec.get("format_pass_rate_min", 0.9))
if float(rate) < min_rate:
    raise SystemExit("FAIL: 형식/지시 준수 통과율 %.3f < 하한 %.3f (실패 케이스: %s)"
                     % (float(rate), min_rate, row.get("failed")))
print("FORMAT OK %s pass_rate=%.3f >= %.3f" % (target, float(rate), min_rate))
PY
  )
}

# --- 게이트 5: 배포 후 서빙 검증 ---------------------------------------------
# s8-64-verify-serving.py가 만든 outputs/serve-<mode>.json을 대조한다.
# 서빙을 아직 하지 않았다면 "미수행"은 실패가 아니라 skip으로 본다(배포는 수동 단계).
rq_gate_serving() {
  local mode="${1:-$MODE}"
  local f="$RQ_OUTPUTS/serve-$mode.json"
  if [ ! -s "$f" ]; then
    echo "SKIP: $f 없음 — 서빙 검증 미수행 (s8-58-serve-infer-all.sh 후 s8-64-verify-serving.sh)"
    return 0
  fi
  ( set -x; rq_py - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("ok") is False:
    raise SystemExit("FAIL: 서빙 검증 실패: %s" % (d.get("errors") or d.get("targets")))
print("serving OK", sys.argv[1])
PY
  )
}
