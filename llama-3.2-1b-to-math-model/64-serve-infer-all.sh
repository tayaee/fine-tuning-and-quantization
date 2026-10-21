#!/bin/bash
# 64-serve-infer-all.sh — 전 모델 순회 서빙+추론 (단일 터미널, 구 61+63 통합).
# 사용: ./64-serve-infer-all.sh [n]
# 모델마다: serve 기동 → 로딩 대기 → 62-infer-examples.sh n개(+한국어 1) 직접 실행
#   → logs/62.<slug>.log 저장 → 서버 kill → 다음 모델.
# - 로그가 이미 있으면 serve 생략 (재실행 resume).
# - 로딩 실패(프로세스 사망/타임아웃) → [SERVE-FAIL] 로그.
#   (GGUF는 61-serve-gguf.sh llama-server로 서빙 — vLLM GGUF 미지원.)
# - 추론 실패/타임아웃 → [INFER-FAIL] 로그 (둘 다 기록으로 인정, 재시도는 로그 삭제 후).
# - kill은 포트 기준 TERM→KILL escalation: vLLM이 SIGTERM을 오래 무시하고
#   고아(PPID=1)로 남아 GPU를 잡는 실측에 대응.
# 환경: PORT(기본 8000), BASE_URL(기본 http://localhost:$PORT/v1),
#   STRAT(기본 single), LOAD_TIMEOUT(기본 180), INFER_TIMEOUT(기본 600), N(기본 3).
set -uo pipefail
cd "$(dirname "$0")"
source "$(dirname "$0")/config/common.env" mini
STRAT="${STRAT:-single}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://localhost:${PORT}/v1}"
export PORT BASE_URL
LOAD_TIMEOUT="${LOAD_TIMEOUT:-180}"
INFER_TIMEOUT="${INFER_TIMEOUT:-600}"
N="${1:-3}"
mkdir -p logs

MODELS=(
    "full fft" "full fft-awq" "full fft-fp8" "full fft-gptq"
    "full qlora" "full qlora-gptq" "full qlora-fp8" "full qlora-awq"
    "mini fft" "mini fft-fp8" "mini fft-gptq" "mini fft-awq"
    "mini qlora" "mini qlora-fp8" "mini qlora-gptq" "mini qlora-awq"
    "full fft-gguf" "full qlora-gguf" "mini fft-gguf" "mini qlora-gguf"
)

slugify() { sed -e 's;/;__;g' | tr 'A-Z' 'a-z'; }

norm_target() {
    case "$1" in
        gptq) echo "fft-gptq" ;;
        awq)  echo "fft-awq" ;;
        fp8)  echo "fft-fp8" ;;
        gguf) echo "fft-gguf" ;;
        *)    echo "$1" ;;
    esac
}

# 60-serve-vllm.sh의 REPO 규칙과 동일하게 기대 slug 계산 (single 동결, full 무표기).
expected_slug() {
    local mode="$1" target="$2"
    target="$(norm_target "$target")"
    local base="${BASE_MODEL##*/}"
    local repo="tayaee/${base}-math-${target}"
    [ "${STRAT:-single}" != "single" ] && repo="${repo}-${STRAT}"
    [ "$mode" != "full" ] && repo="${repo}-${mode}"
    printf '%s' "$repo" | slugify
}

pids_on_port() {
    ss -ltnp 2>/dev/null | grep ":${PORT} " | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u
}

CHILD=""
kill_server() {
    if [ -n "${CHILD:-}" ] && kill -0 "$CHILD" 2>/dev/null; then
        kill "$CHILD" 2>/dev/null
    fi
    local i pids
    for i in $(seq 1 15); do
        pids="$(pids_on_port)"
        [ -z "$pids" ] && break
        for p in $pids; do kill "$p" 2>/dev/null; done
        sleep 1
    done
    pids="$(pids_on_port)"
    if [ -n "$pids" ]; then
        echo "[warn] port ${PORT} 점유 잔존 → KILL: $pids"
        for p in $pids; do kill -9 "$p" 2>/dev/null; done
        sleep 2
    fi
    if [ -n "${CHILD:-}" ]; then wait "$CHILD" 2>/dev/null; fi
    CHILD=""
}

wait_for_port_free() {
    local i=0
    while [ "$i" -lt 60 ]; do
        ss -ltn 2>/dev/null | grep -q ":${PORT} " || { echo "Port ${PORT} is free."; return 0; }
        sleep 1; i=$((i + 1))
    done
    echo "[warn] port ${PORT} still busy" >&2
    return 1
}
trap 'kill_server; exit 130' INT TERM

run_one() {
    local mode="$1" target="$2"
    local expected=""; expected="$(expected_slug "$mode" "$target")"
    local logf="logs/62.${expected}.log"
    if [ -f "$logf" ]; then
        echo "[skip] $mode $target — $logf 이미 있음"
        return 0
    fi
    rm -f "$logf.tmp"
    echo "=========================================="
    echo "Model: mode=$mode target=$target (expect $expected, n=$N)"
    echo "=========================================="

    if [[ "$target" == *-gguf ]]; then
        eval "./61-serve-gguf.sh $mode $target" &
    else
        eval "./60-serve-vllm.sh $mode $target" &
    fi
    CHILD=$!

    local elapsed=0 loaded=0 slug=""
    while [ "$elapsed" -lt "$LOAD_TIMEOUT" ]; do
        if ! kill -0 "$CHILD" 2>/dev/null; then
            echo "[SERVE-FAIL] $mode $target — 서버 프로세스가 로딩 중 사망"
            CHILD=""
            kill_server
            wait_for_port_free || true
            { echo "[SERVE-FAIL] $(date --iso-8601=s) mode=$mode target=$target"
              echo "reason: 서버 프로세스가 로딩 중 사망 (60-serve-vllm.sh 출력 참조)"; } > "$logf.tmp" && mv "$logf.tmp" "$logf"
            return 1
        fi
        slug="$(./model-slug.sh 2>/dev/null | tr -d '[:space:]')"
        if [ -n "$slug" ] && [ "$slug" != "null" ]; then
            loaded=1
            echo "Model loaded: $slug (${elapsed}s)."
            break
        fi
        sleep 2; elapsed=$((elapsed + 2))
    done
    if [ "$loaded" -eq 0 ]; then
        echo "[SERVE-FAIL] $mode $target — 로딩 타임아웃 (${LOAD_TIMEOUT}s)"
        kill_server
        wait_for_port_free || true
        { echo "[SERVE-FAIL] $(date --iso-8601=s) mode=$mode target=$target"
          echo "reason: 로딩 타임아웃 ${LOAD_TIMEOUT}s"; } > "$logf.tmp" && mv "$logf.tmp" "$logf"
        return 1
    fi

    slug="$(./model-slug.sh 2>/dev/null | tr -d '[:space:]')"
    if [ "$slug" != "$expected" ]; then
        echo "[warn] slug 불일치: expect=$expected actual=$slug — actual 우선"
        logf="logs/62.${slug}.log"
        if [ -f "$logf" ]; then
            echo "[skip] $slug — 로그 이미 있음"
            kill_server
            wait_for_port_free || true
            return 0
        fi
        rm -f "$logf.tmp"
    fi

    echo "[infer] $slug (n=$N, timeout ${INFER_TIMEOUT}s) → $logf"
    {
        echo "[model] $slug $(date --iso-8601=s) mode=$mode target=$target"
        if command -v timeout >/dev/null 2>&1; then
            timeout "$INFER_TIMEOUT" ./62-infer-examples.sh "$N"
        else
            ./62-infer-examples.sh "$N"
        fi
    } > "$logf.tmp" 2>&1 && rc=0 || rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "[OK] $slug" >> "$logf.tmp"
    elif [ "$rc" -eq 124 ]; then
        echo "[INFER-FAIL] timeout ${INFER_TIMEOUT}s $(date --iso-8601=s)" >> "$logf.tmp"
    else
        echo "[INFER-FAIL] rc=$rc $(date --iso-8601=s)" >> "$logf.tmp"
    fi
    mv "$logf.tmp" "$logf"
    tail -n 3 "$logf"

    kill_server
    wait_for_port_free || true
    echo "[done] $mode $target"
    return 0
}

if [ -n "$(pids_on_port)" ]; then
    echo "Port ${PORT}가 이미 사용 중($(pids_on_port)) — 기존 서버를 먼저 정리하라." >&2
    exit 1
fi

for spec in "${MODELS[@]}"; do
    read -r m t <<< "$spec"
    run_one "$m" "$t" || true
done

echo "########## 결과 ##########"
ok=0; fail=0
for spec in "${MODELS[@]}"; do
    read -r m t <<< "$spec"
    s="$(expected_slug "$m" "$t")"
    f="logs/62.${s}.log"
    if [ ! -f "$f" ]; then echo "[MISSING] $m $t"; fail=$((fail + 1)); continue; fi
    if grep -q "SERVE-FAIL\|INFER-FAIL" "$f"; then echo "[FAIL] $m $t ($f)"; fail=$((fail + 1)); else echo "[OK] $m $t"; ok=$((ok + 1)); fi
done
echo "OK=$ok FAIL/MISSING=$fail"
[ "$fail" -eq 0 ]
