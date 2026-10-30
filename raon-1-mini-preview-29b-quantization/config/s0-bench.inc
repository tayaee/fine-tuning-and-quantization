# config/s0-bench.inc — s0/s6 벤치 실행 공용 헬퍼 (endpoint 대상 측정).
# s0-15/s0-45(베이스), s6-13은 각자 로딩하고, 본 측정은 여기서 endpoint로 수행한다.
# 산출물 변경 없음(bench 디렉터리 아래 results.json + 최초 1회 baseline.json 충전).

# s0_bench_run <base_url> <model> <dir> — <dir>/prompts.jsonl을 endpoint로 측정해
# <dir>/results.json에 기록한다. baseline.json이 미측정이면 이번 결과로 충전한다.
s0_bench_run() {
  local base_url="$1" model="$2" dir="$3"
  RQ_PY="${RQ_PY:-python3}"
  "$RQ_PY" - "$base_url" "$model" "$dir" <<'PY'
import json, sys, time, urllib.request
base_url, model, d = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = [json.loads(l) for l in open(f"{d}/prompts.jsonl", encoding="utf-8")]
rows = []
for p in prompts:
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": p["prompt"]}],
                       "max_tokens": 256, "temperature": 0}).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
        text = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        rows.append({"id": p["id"], "ok": bool((text or "").strip()),
                     "latency_s": round(time.time() - t0, 3),
                     "response_head": (text or "")[:300]})
    except Exception as e:  # noqa: BLE001 — 벤치 기록용이므로 실패도 행으로 남긴다
        rows.append({"id": p["id"], "ok": False, "latency_s": round(time.time() - t0, 3),
                     "error": f"{type(e).__name__}: {e}", "response_head": ""})
    print(f"bench {p['id']}: {'OK' if rows[-1]['ok'] else 'FAIL'} {rows[-1]['latency_s']}s", flush=True)
ok_n = sum(1 for r in rows if r["ok"])
mean_lat = round(sum(r["latency_s"] for r in rows) / max(len(rows), 1), 3)
with open(f"{d}/results.json", "w", encoding="utf-8") as f:
    json.dump({"model": model, "n": len(rows), "ok": ok_n,
               "mean_latency_s": mean_lat, "rows": rows},
              f, ensure_ascii=False, indent=2)
base = f"{d}/baseline.json"
try:
    cur = json.load(open(base, encoding="utf-8"))
except (OSError, ValueError):
    cur = {}
if not cur.get("measured"):
    with open(base, "w", encoding="utf-8") as f:
        json.dump({"measured": True, "n": len(rows), "ok": ok_n,
                   "mean_latency_s": mean_lat}, f, ensure_ascii=False, indent=2)
    print(f"baseline charged {base}")
print(f"bench done ok={ok_n}/{len(rows)} mean_lat={mean_lat}s")
PY
}
