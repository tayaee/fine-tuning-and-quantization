"""config/transformers-server.py — transformers 직결 최소 OpenAI 호환 서버 (stdlib만 사용).

s0-01(베이스 서빙)과 s4-40(로컬 가중치 서빙)이 공유한다.
chat template 네이티브 툴콜(<tool_call>/<function=>)을 OpenAI tool_calls로 번역하고
SSE 스트리밍을 지원한다. tool 호출이 일어나면 stderr에 TOOL-CALL을 남긴다.

  uv run python config/transformers-server.py --model-dir <dir> --model-id <id> --port <port>
"""
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
            # 상한은 요청 max_tokens 그대로 (벤치 공정 비교용 — 버퍼를 얹지 않는다).
            gen = self.model.generate(**inputs, max_new_tokens=max(1, min(max_tokens, 512)),
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