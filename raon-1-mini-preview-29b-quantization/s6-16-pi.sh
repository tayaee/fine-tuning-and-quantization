#!/usr/bin/env bash
# s6-16-pi.sh [mode] [model] — 고객 transformers 로딩본으로 pi 툴콜 검증 (Stage 6).
# model: HF repo id | short target | all (기본 nvfp4-awq).
# 모델마다 transformers 기반 최소 OpenAI 서버(:8001)를 띄워 pi 고정 질문을 던지고 내린다.
# 서버는 tools 왕복(chat→tool_calls→tool 결과→최종)을 지원한다. 산출물 변경 없음.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
source "$RQ_ROOT/config/s6-customer.inc"
source "$RQ_ROOT/config/s0-pi.inc"
rq_repro_echo
rq_banner "s6-16-pi" "Stage 6 - pi tool-call check against customer-loaded models (temp server)"
export UV_NO_SYNC=1; unset VIRTUAL_ENV
rq_hf_parse "${2:-nvfp4-awq}" "$MODE"
PORT="${PORT:-8001}"
FAIL=0
for _e in $RQ_HF_TARGETS_RUN; do
  case "$_e" in
    */*) REPO="$_e"; TAG="$(printf '%s' "$_e" | tr '/:' '__')" ;;
    *) REPO="$(s6_repo "$_e" "$MODE")"; TAG="$_e" ;;
  esac
  t="$TAG"
  LOGF="$RQ_ROOT/logs/s6-16-$t.log"
  SERVLOG="$RQ_ROOT/logs/s6-16-server-$t.log"
  OUTDIR="$RQ_OUTPUTS/s6-$MODE"
  mkdir -p "$OUTDIR"
  echo "=========================================="
  echo "Model: mode=$MODE target=$t (repo $REPO, port $PORT)"
  echo "=========================================="
  if ! LOCAL="$(s6_download "$REPO")"; then
    { echo "[SERVE-FAIL] $(date --iso-8601=s) mode=$MODE target=$t repo=$REPO (download 실패)"; } > "$LOGF"
    rq_stage_record 6 cat-tf fail "customer-pi:$t"
    FAIL=1; continue
  fi
  cat > "$OUTDIR/server-$t.py" <<'PYEOF'
"""s6-16 임시 OpenAI 호환 서버 (transformers 직결, stdlib만 사용)."""
import argparse, json, re, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOOL_JSON_RE = [
    re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL),
    re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL),
]
NATIVE_CALL_RE = re.compile(
    r"(?:<tool_call>\s*)?<function=([^>\s]+)>(.*?)</function>\s*(?:</tool_call>)?",
    re.DOTALL)
NATIVE_PARAM_RE = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def to_template_messages(messages):
    """OpenAI 메시지 → Raon chat template 메시지 (tool_calls 인자는 dict로)."""
    out = []
    for m in messages:
        role = m.get("role", "user")
        if role == "developer":
            role = "system"
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = c.get("function") or {}
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {"_raw": args}
                calls.append({"id": c.get("id"), "type": "function",
                              "function": {"name": fn.get("name"), "arguments": args}})
            out.append({"role": "assistant", "content": m.get("content") or "",
                        "tool_calls": calls})
        else:
            out.append({"role": role, "content": m.get("content") or ""})
    merged = []
    for m in out:
        if m["role"] == "system" and merged and merged[-1]["role"] == "system":
            merged[-1]["content"] += "\n" + m["content"]
        elif m["role"] == "system" and any(x["role"] != "system" for x in merged):
            merged.append({"role": "user", "content": m["content"]})
        else:
            merged.append(m)
    return merged


def parse_tool_calls(text, wanted):
    """네이티브 <tool_call> 블록 + JSON 폴백 파싱 → OpenAI tool_calls (요청된 이름만)."""
    found = []
    for m in NATIVE_CALL_RE.finditer(text):
        name = m.group(1).strip()
        args = {pm.group(1).strip(): pm.group(2).strip()
                for pm in NATIVE_PARAM_RE.finditer(m.group(2))}
        found.append((name, args))
    if not found:
        for rx in TOOL_JSON_RE:
            mm = rx.search(text)
            if not mm:
                continue
            try:
                obj = json.loads(mm.group(1))
            except ValueError:
                continue
            if obj.get("name"):
                found.append((obj["name"], obj.get("arguments", {}) or {}))
    calls = []
    for name, args in found:
        if wanted and name not in wanted:
            continue
        calls.append({"id": f"call_{int(time.time()*1000)}-{len(calls)}",
                      "type": "function",
                      "function": {"name": name,
                                   "arguments": args if isinstance(args, str)
                                   else json.dumps(args, ensure_ascii=False)}})
    return calls


def clean_text(text):
    text = NATIVE_CALL_RE.sub("", text)
    text = THINK_RE.sub("", text)
    return text.strip()


class Handler(BaseHTTPRequestHandler):
    model = None
    tok = None
    model_id = "s6-customer"

    def log_message(self, *a):
        pass

    def _send(self, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            self._send({"object": "list",
                        "data": [{"id": self.model_id, "object": "model"}]})
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/v1/chat/completions"):
            self.send_response(404); self.end_headers(); return
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
        messages = req.get("messages", [])
        tools = req.get("tools")
        wanted = {t.get("function", {}).get("name") for t in (tools or [])}
        wanted.discard(None)
        max_tokens = int(req.get("max_tokens", 256) or 256)
        inputs = self.tok.apply_chat_template(
            to_template_messages(messages), tools=tools or None,
            add_generation_prompt=True, return_tensors="pt",
            enable_thinking=True).to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=max(1, min(max_tokens + 256, 768)),
                                      do_sample=False)
        text = self.tok.decode(gen[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        calls = parse_tool_calls(text, wanted) if tools else []
        if calls:
            for c in calls:
                print(f"TOOL-CALL name={c['function']['name']}", flush=True)
            msg = {"role": "assistant", "content": "", "tool_calls": calls}
            finish = "tool_calls"
        else:
            msg = {"role": "assistant", "content": clean_text(text)}
            finish = "stop"
        if req.get("stream"):
            self._send_stream(msg, finish)
        else:
            self._send({"id": "chatcmpl-s6", "object": "chat.completion",
                        "created": int(time.time()), "model": self.model_id,
                        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                        "usage": {}})

    def _send_stream(self, msg, finish):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def chunk(delta, finish_reason=None):
            self.wfile.write(("data: " + json.dumps(
                {"id": "chatcmpl-s6", "object": "chat.completion.chunk",
                 "created": int(time.time()), "model": self.model_id,
                 "choices": [{"index": 0, "delta": delta,
                              "finish_reason": finish_reason}]},
                ensure_ascii=False) + "\n\n").encode("utf-8"))
            self.wfile.flush()

        chunk({"role": "assistant"})
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            chunk({"tool_calls": [{"index": 0, "id": call.get("id"),
                                   "type": "function",
                                   "function": {"name": fn.get("name", ""),
                                                "arguments": fn.get("arguments", "")}}]})
        if msg.get("content"):
            chunk({"content": msg["content"]})
        chunk({}, finish)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    Handler.tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    Handler.model = AutoModelForCausalLM.from_pretrained(
        a.model_dir, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    Handler.model_id = a.model_id
    Handler.model.eval()
    print(f"s6-server ready model={a.model_id} port={a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
PYEOF
  rm -f "$SERVLOG"
  nohup uv run python "$OUTDIR/server-$t.py" --model-dir "$LOCAL" --model-id "$REPO" --port "$PORT" >"$SERVLOG" 2>&1 &
  SRVPID=$!
  rq_log "server start pid=$SRVPID (target=$t)"
  ok=0
  for _i in $(seq 1 120); do
    if curl -s --max-time 3 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q '"id"'; then ok=1; break; fi
    kill -0 "$SRVPID" 2>/dev/null || break
    sleep 5
  done
  (
    echo "[model] $REPO $(date --iso-8601=s) mode=$MODE target=$t port=$PORT"
    if [ "$ok" = 1 ]; then
      s0_pi_register "raon-cust-$PORT" "http://localhost:${PORT}/v1" "$REPO" "EMPTY"
      PROBE="$(s0_pi_probe_dir)"
      EXPECT="$(s0_pi_expect "$PROBE")"
      (cd "$PROBE" && pi --provider "raon-cust-$PORT" --model "$REPO" --no-session \
        -p "$S0_PI_QUESTION" >"$OUTDIR/pi-answer-$t.txt" 2>&1) || rc=$?
      cat "$OUTDIR/pi-answer-$t.txt"
      if [ "${rc:-0}" -eq 0 ] && grep -q "TOOL-CALL" "$SERVLOG" 2>/dev/null \
        && grep -Eq "(^|[^0-9])$EXPECT([^0-9]|$)" "$OUTDIR/pi-answer-$t.txt"; then
        echo "pi tool-call OK (tool invoked, answer contains $EXPECT)"
        rc=0
      else
        echo "pi tool-call NOT verified (need: server TOOL-CALL + standalone $EXPECT in answer)"
        rc=1
      fi
    else
      echo "server did not come up (see $SERVLOG)"
      rc=1
    fi
    exit "${rc:-0}"
  ) > "$LOGF" 2>&1; rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[OK] $t" >> "$LOGF"; rq_stage_record 6 cat-tf pass "$LOGF"
  else
    echo "[PI-FAIL] rc=$rc $(date --iso-8601=s)" >> "$LOGF"
    rq_stage_record 6 cat-tf fail "customer-pi:$t"; FAIL=1
  fi
  tail -n 2 "$LOGF"
  kill "$SRVPID" 2>/dev/null || true
  wait "$SRVPID" 2>/dev/null || true
  rq_log "server stopped (target=$t)"
done
[ "$FAIL" -eq 0 ]
