# config/s0-pi.inc — s0 pi 툴콜 검증 공용 헬퍼 (s0-07/s0-19가 source).
# 고정 질문으로 pi의 도구 호출 왕복을 검증한다:
#   "현재 디렉토리 파일이 몇 개니? 숫자 하나로 답해."
# 기대값 = probe 디렉터리 바로 아래 일반 파일 수 (실행 시점에 계산).
S0_PI_QUESTION='현재 디렉토리 파일이 몇 개니? 숫자 하나로 답해.'
S0_PI_TIMEOUT="${S0_PI_TIMEOUT:-300}"

# s0_pi_probe_dir — 고정 probe 디렉터리(5개 파일)를 준비하고 경로를 출력한다.
s0_pi_probe_dir() {
  local d="$RQ_SHARED/outputs/s0-pi-probe"
  mkdir -p "$d"
  local i
  for i in 1 2 3 4 5; do
    [ -f "$d/file$i.txt" ] || printf 'probe file %s\n' "$i" >"$d/file$i.txt"
  done
  printf '%s' "$d"
}

# s0_pi_expect <dir> — 기대 파일 수 출력.
s0_pi_expect() {
  find "$1" -maxdepth 1 -type f | wc -l | tr -d ' '
}

# s0_pi_register <provider> <base_url> <model_id> <api_key> — ~/.pi/agent/models.json에
# provider 항목을 병합한다(기존 파일은 .bak 백업, 기존 provider 키는 덮어쓰지 않고 models만 병합).
s0_pi_register() {
  local provider="$1" base_url="$2" model_id="$3" api_key="$4"
  RQ_PY="${RQ_PY:-python3}"
  "$RQ_PY" - "$provider" "$base_url" "$model_id" "$api_key" <<'PY'
import json, os, sys
provider, base_url, model_id, api_key = sys.argv[1:5]
home = os.path.expanduser("~")
path = os.path.join(home, ".pi", "agent", "models.json")
with open(path, encoding="utf-8") as f:
    data = json.load(f)
providers = data.setdefault("providers", {})
entry = providers.get(provider) or {}
entry["baseUrl"] = base_url
entry["api"] = "openai-completions"
entry["apiKey"] = api_key or "EMPTY"
models = entry.setdefault("models", [])
if not any(m.get("id") == model_id for m in models):
    models.append({"id": model_id, "name": f"{model_id} (s0 pi probe)",
                   "reasoning": True, "input": ["text"],
                   "contextWindow": 262144, "maxTokens": 8192,
                   "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}})
providers[provider] = entry
bak = path + ".bak"
if not os.path.exists(bak):
    with open(bak, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"registered provider={provider} model={model_id}")
PY
}

# s0_pi_verify <provider> <model> <dir> [servlog] — pi 고정 질문 검증 (관대 판정).
# 통과 조건: (1) 답에 기대 파일 수(단독 숫자)가 들어 있고,
# (2) servlog를 줬으면 서버에 TOOL-CALL 호출 증거가 있어야 한다.
# Raon처럼 산문형으로 답해도 도구 왕복이 입증되면 통과. 엄격 판정은 s0_pi_ask.
s0_pi_verify() {
  local provider="$1" model="$2" dir="$3" servlog="${4:-}"
  local expect out rc=0 ansf
  expect="$(s0_pi_expect "$dir")"
  command -v pi >/dev/null 2>&1 || { echo "SKIP: pi CLI 없음"; return 2; }
  ansf="$(mktemp)"
  if ! (cd "$dir" && S0_PI_TIMEOUT="${S0_PI_TIMEOUT:-300}" \
    pi --provider "$provider" --model "$model" --no-session \
    -p "$S0_PI_QUESTION" >"$ansf" 2>&1); then
    echo "FAIL: pi 실행 실패"; tail -n 3 "$ansf"; rm -f "$ansf"; return 1
  fi
  cat "$ansf"
  if [ -n "$servlog" ] && ! grep -q "TOOL-CALL" "$servlog" 2>/dev/null; then
    echo "FAIL: 서버에 도구 호출 증거(TOOL-CALL) 없음"; rm -f "$ansf"; return 1
  fi
  if grep -Eq "(^|[^0-9])$expect([^0-9]|$)" "$ansf"; then
    echo "pi tool-call OK (answer contains $expect)"
    rm -f "$ansf"; return 0
  fi
  echo "FAIL: pi 답에 기대 수($expect) 없음"; rm -f "$ansf"; return 1
}

# s0_pi_ask <provider> <model> <dir> — probe 디렉터리를 cwd로 pi 고정 질문을 던지고
# 단일 숫자 == 기대 파일 수인지 판정한다. 통과=0 (엄격 판정, minimax 등 준수 모델용).
s0_pi_ask() {
  local provider="$1" model="$2" dir="$3"
  local expect out rc=0
  expect="$(s0_pi_expect "$dir")"
  command -v pi >/dev/null 2>&1 || { echo "SKIP: pi CLI 없음"; return 2; }
  command -v timeout >/dev/null 2>&1 && _to="timeout $S0_PI_TIMEOUT" || _to=""
  out=$(cd "$dir" && $_to pi --provider "$provider" --model "$model" --no-session -p "$S0_PI_QUESTION" 2>&1) || rc=$?
  if [ "$rc" -eq 2 ]; then echo "SKIP: pi CLI 없음"; return 2; fi
  if [ "$rc" -ne 0 ]; then echo "FAIL: pi 실행 실패 rc=$rc"; printf '%s\n' "$out" | tail -n 5; return 1; fi
  out="$(printf '%s' "$out" | tr -d '[:space:]')"
  if [ "$out" = "$expect" ]; then
    echo "pi tool-call OK provider=$provider model=$model answer=$out (expect=$expect)"
    return 0
  fi
  echo "FAIL: pi 답='$out' 기대='$expect' (단일 숫자 불일치 — 도구 호출 미동작 가능)"
  return 1
}
