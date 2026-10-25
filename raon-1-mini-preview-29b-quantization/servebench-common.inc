# servebench-common.inc — s7-53(베이스)/s7-54(양자화) 서빙벤치 공용 함수.
# GB10은 nvidia-smi 메모리 리포팅 미지원 → 시스템 가용 메모리(/proc/meminfo) +
# 서버 프로세스 RSS로 기록한다.
# 사용: source "$(dirname "$0")/servebench-common.inc" (common.env 로드 후)

pids_on_port() {
  ss -ltnp 2>/dev/null | grep ":${PORT} " | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u
}

CHILD=""
kill_server() {
  if [ -n "${CHILD:-}" ] && kill -0 "$CHILD" 2>/dev/null; then kill "$CHILD" 2>/dev/null; fi
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
  return 1
}

# mem_snapshot <phase> <server_pid> — 가용 메모리 + 서버 RSS를 한 줄 JSON으로.
mem_snapshot() {
  local phase="$1" pid="${2:-}"
  local avail_kb rss_kb
  avail_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  rss_kb=0
  if [ -n "$pid" ] && [ -r "/proc/$pid/status" ]; then
    rss_kb="$(awk '/^VmRSS:/ {print $2}' "/proc/$pid/status")"
  fi
  printf '{"ts":"%s","phase":"%s","mem_avail_gb":%.1f,"server_rss_gb":%.1f}\n' \
    "$(date --iso-8601=s)" "$phase" \
    "$(python3 -c "print(${avail_kb:-0}/1024/1024)")" \
    "$(python3 -c "print(${rss_kb:-0}/1024/1024)")"
}

# serve_and_sweep <mode> <target> <quant> <logf> — 서빙→4지점 메모리→스위프→종료.
# 메모리 4지점: before_load / after_warmup / before_teardown / after_teardown.
serve_and_sweep() {
  local mode="$1" target="$2" quant="$3" logf="$4"
  local memf="${logf%.log}.mem.jsonl"
  rm -f "$memf"
  echo "=========================================="
  echo "ServeBench: mode=$mode target=$target port=$PORT sweep=[$SWEEP] n=$SERVE_BENCH_N"
  echo "=========================================="
  mem_snapshot "before_load" "" | tee -a "$memf"
  eval "./s8-52-serve-vllm.sh $mode $target" > "${logf}.serve" 2>&1 &
  CHILD=$!
  local server_pid="$CHILD" elapsed=0 loaded=0 slug=""
  while [ "$elapsed" -lt "$LOAD_TIMEOUT" ]; do
    if ! kill -0 "$CHILD" 2>/dev/null; then
      echo "[SERVE-FAIL] $mode $target — 서버 프로세스가 로딩 중 사망 (see ${logf}.serve)"
      CHILD=""; mem_snapshot "load_failed" "" | tee -a "$memf"
      kill_server; wait_for_port_free || true
      return 1
    fi
    slug="$(./model-slug.sh 2>/dev/null | tr -d '[:space:]')"
    if [ -n "$slug" ] && [ "$slug" != "null" ]; then loaded=1; echo "Model loaded: $slug (${elapsed}s)."; break; fi
    sleep 10; elapsed=$((elapsed + 10))
  done
  if [ "$loaded" -eq 0 ]; then
    echo "[SERVE-FAIL] $mode $target — 로딩 타임아웃 (${LOAD_TIMEOUT}s)"
    kill_server; wait_for_port_free || true
    return 1
  fi
  mem_snapshot "after_warmup" "$server_pid" | tee -a "$memf"
  local c rc=0
  for c in $(echo "$SWEEP" | tr ',' ' '); do
    echo "[bench] $slug C=$c n=$SERVE_BENCH_N → $logf"
    if command -v timeout >/dev/null 2>&1; then
      SERVE_BENCH_C="$c" SERVE_BENCH_TARGET="$target" SERVE_BENCH_QUANT="$quant" \
        timeout "$BENCH_TIMEOUT" ./s7-52-servebench.sh "$mode" >> "$logf" 2>&1 || rc=$?
    else
      SERVE_BENCH_C="$c" SERVE_BENCH_TARGET="$target" SERVE_BENCH_QUANT="$quant" \
        ./s7-52-servebench.sh "$mode" >> "$logf" 2>&1 || rc=$?
    fi
    [ "$rc" -eq 0 ] || { echo "[BENCH-FAIL] $mode $target C=$c rc=$rc"; break; }
  done
  mem_snapshot "before_teardown" "$server_pid" | tee -a "$memf"
  kill_server; wait_for_port_free || true
  mem_snapshot "after_teardown" "" | tee -a "$memf"
  echo "[done] $mode $target rc=$rc"
  return "$rc"
}

# merge_memory <mode> <target> <memf> — 4지점 메모리를 checks.json history에 병합.
merge_memory() {
  local mode="$1" target="$2" memf="$3"
  MOD="$mode" TGT="$target" MEMF="$memf" python3 -c "
import json, os
mode, tgt, memf = os.environ['MOD'], os.environ['TGT'], os.environ['MEMF']
mem = {}
for line in open(memf, encoding='utf-8'):
    line = line.strip()
    if line:
        r = json.loads(line)
        mem[r['phase']] = {k: r[k] for k in ('ts', 'mem_avail_gb', 'server_rss_gb')}
shared = os.environ['RQ_SHARED']
for p in (f'{shared}/outputs/eval-{mode}/checks.json',
          f'{shared}/outputs/eval-{mode}/servebench-{mode}.json'):
    try:
        d = json.load(open(p, encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        continue
    sb = d.get('servebench', d)
    tg = sb.setdefault('targets', {}).setdefault(tgt, {})
    tg['memory'] = mem
    json.dump(d, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'memory merged: {p} target={tgt} phases={sorted(mem)}')
"
}
