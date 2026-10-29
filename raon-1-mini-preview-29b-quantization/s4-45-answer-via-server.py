"""s4-45-answer-via-server.py — 실행 중인 OpenAI-호환 서버로 벤치마크 응답 생성.

in-process(s4-07/s4-10)가 읽을 수 없는 가중치(ModelOpt FP4 등)의 정확도를
실제 서빙 스택에서 측정한다. 출력 스키마는 s4-07/s4-10과 동일하므로
s4-20-aggregate 이후 단계가 그대로 동작한다.

  uv run s4-45-answer-via-server.py --mode full --target nvfp4-awq-vllm --benchmarks gsm8k,kmmluko,format
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def _load_eval_backend():
    spec = importlib.util.spec_from_file_location(
        "s4_eval_backend", os.path.join(ROOT, "s4-eval-backend__transformers-only.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["s4_eval_backend"] = mod
    spec.loader.exec_module(mod)
    return mod


ev = _load_eval_backend()

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")


def server_models(base_url: str, port: int) -> list[str]:
    """서버가 서빙 중인 모델 id 목록. 닿지 않으면 빈 목록."""
    url = (base_url or f"http://localhost:{port}/v1").rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [m.get("id", "") for m in data.get("data", []) if m.get("id")]
    except Exception:
        return []


def complete(base_url: str, port: int, model: str, prompt: str, max_tokens: int) -> str:
    """completions greedy 1건 (프롬프트 원문 전달 — 서버 템플릿 재적용 없음)."""
    from openai import OpenAI

    url = (base_url or f"http://localhost:{port}/v1").rstrip("/")
    client = OpenAI(base_url=url, api_key="dummy", timeout=600)
    r = client.completions.create(model=model, prompt=prompt,
                                  max_tokens=max_tokens, temperature=0.0)
    return r.choices[0].text or ""


def main(mode: str, target: str, n: int, benchmarks: list[str], base_url: str, port: int,
         model: str | None, max_tokens: int | None) -> dict:
    served = server_models(base_url, port)
    if not served:
        print(f"s4-45 SKIP: no server at {base_url or port} — in-process rows preserved")
        return {"ok": False, "status": "skipped"}
    model_id = model or served[0]
    if model is None and len(served) == 1 and served[0] != target:
        pass  # served-model-name이 타깃 slug와 다르면 아래에서 대조
    if model_id != target and not model:
        # 오귀인 방지: 서버가 다른 모델을 서빙하면 중단한다 (조용히 덮어쓰지 않음).
        print(f"s4-45 REFUSE: server serves '{model_id}', not '{target}'"
              f" (--model로 명시하거나 올바른 서버를 띄워라)")
        raise SystemExit(1)
    prompts = ev.load_prompts(mode, n, None, benchmarks)
    max_tokens = max_tokens or ev._default_max_tokens()
    outdir = ev.OUTDIR(mode)
    os.makedirs(outdir, exist_ok=True)
    for b in {p.get("bench", "smoke") for p in prompts}:
        os.makedirs(os.path.join(outdir, "bench", b), exist_ok=True)
    main_path = os.path.join(outdir, f"{target}.jsonl")
    # 이번 실행 벤치가 아닌 기존 행은 보존하되, 출력 없는 행(스킵 마커)은 버린다.
    kept = [r for r in ev.preserved_rows(main_path, {p.get("bench", "smoke") for p in prompts})
            if "output" in r]
    detail: dict[str, list] = {}
    with open(main_path, "w", encoding="utf-8") as f:
        for p in prompts:
            o = complete(base_url, port, model_id, p["prompt"], max_tokens)
            b = p.get("bench", "smoke")
            row = ev.build_row(p, o)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            detail.setdefault(b, []).append(row)
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for b, rows in detail.items():
        with open(os.path.join(outdir, "bench", b, f"{target}.jsonl"), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[{mode}/{target}] n={len(prompts)} backend=server({model_id}) -> {main_path}")
    return {"ok": True, "status": "ok", "n": len(prompts), "model": model_id}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="실행 중인 서버로 벤치마크 응답 생성 (in-process 불가 포맷용)")
    ap.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    ap.add_argument("--target", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--benchmarks", default=os.environ.get("EVAL_BENCHMARKS", "gsm8k"),
                    help="콤마 구분 (기본: $EVAL_BENCHMARKS 또는 gsm8k)")
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", ""))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--model", default=None, help="served-model-name (기본: 서버 조회)")
    ap.add_argument("--max-tokens", type=int, default=None)
    a = ap.parse_args()
    n = a.n if a.n is not None else int(os.environ.get("EVAL_N", "20" if a.mode == "smoke" else "200"))
    benches = [b for b in a.benchmarks.replace(" ", "").split(",") if b] or ["gsm8k"]
    out = main(a.mode, a.target, n, benches, a.base_url, a.port, a.model, a.max_tokens)
    print(f"s4-45 done: {out['status']}")
    raise SystemExit(0)
