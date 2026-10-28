"""s4-25-servebench.py — 서빙 처리량 측정 (Tier 3: 배포 성능 증거).

양자화의 이유(속도·메모리)는 품질 검사로 증명되지 않는다. KRAFTON 베이스 카드에는
처리량 표(AR/DFlash2 × C=1/8/32)가 공개돼 있고, 그 값은 **다른 스택·하드웨어**의
수치라 직접 비교는 되지 않지만 "우리가 무엇을 얻었는지"를 같은 축에서 보여줄 수 있다.

이 스크립트는 이미 떠 있는 OpenAI-호환 서버(vLLM, s7-13-chat.sh가 띄운 것)에 요청을 보내
TTFT(latency)와 출력 토큰/초를 측정한다. 서버가 없으면 측정하지 않고 사유를 남긴다.

  uv run s4-25-servebench.py --mode full --n 8 --concurrency 1
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sdlc import checks_path, merge_check

SHARED = Path(os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared"))

#: concurrency 스위프 표준값 — KRAFTON 베이스 카드의 처리량 표와 같은 축(C=1/8/32).
SWEEP_DEFAULT = (1, 8, 32)

#: 서빙 벤치 고정 워크로드 (타깃 간 비교 공정성용).
BENCH_MAX_TOKENS = 256
BENCH_TEMPERATURE = 0.0
BENCH_WARMUP = 2

PROMPTS = [
    "1부터 100까지의 합을 구하고, 계산 과정과 최종 답을 보여 주세요.",
    "세종대왕이 훈민정음을 만든 이유를 3문장으로 설명해 주세요.",
    "피보나치 수열의 10번째 항을 구하세요.",
    "대한민국의 수도와 인구가 가장 많은 도시를 각각 한 줄로 답해 주세요.",
    "양의 제곱근과 음의 제곱근의 차이를 예시로 설명해 주세요.",
    "리눅스에서 프로세스를 종료하는 명령어와 그 옵션 두 가지를 알려 주세요.",
    "삼각형의 내각의 합이 180도인 이유를 직관적으로 설명해 주세요.",
    "원주율이 pi인 이유를 설명하고, 아르키메데스의 방법으로 근삿값을 구해 보세요.",
]

#: KRAFTON Raon-1-mini-preview 카드에 공개된 처리량(출력 tok/s, AR/DFlash2).
#: 하드웨어·런타임이 달라 직접 비교는 불가 — 방향성参照용으로만 기록한다.
PUBLISHED_REFERENCE = {
    "source": "KRAFTON/Raon-1-mini-preview model card (throughput table)",
    "hardware": "different stack (vLLM 0.28.0 + MXFP8 target)",
    "gsm8k": {"ar_c1": 296.3, "dflash2_c1": 646.3, "ar_c8": 1110.4, "dflash2_c8": 1836.3, "ar_c32": 1528.9, "dflash2_c32": 2619.3},
    "note": "output tokens per second, end-to-end wall time; directional reference only",
}


def outdir(mode: str) -> Path:
    return SHARED / "outputs" / f"eval-{mode}"


def infer_target(model_id: str) -> str:
    """served-model-name(tayaee/Raon-1-mini-preview[-smoke][-suffix]) → 타깃 slug."""
    mid = (model_id or "").lower()
    for t in ("nvfp4-awq", "nvfp4-gptq", "awq-w3", "awq-w5"):
        if t in mid:
            return t
    return "base"


def hardware_info(quantization: str = "") -> dict:
    """서빙 측정 하드웨어·런타임 조건 (README 명시용 — 타깃 간 비교의 공정 조건)."""
    import subprocess

    gpu, driver = "unknown", ""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            gpu, _, driver = r.stdout.strip().splitlines()[0].partition(", ")
            gpu, driver = gpu.strip(), driver.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    mem_gb = 0.0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_gb = round(int(line.split()[1]) / 1024 / 1024, 1)
                    break
    except OSError:
        pass
    try:
        import vllm  # noqa: PLC0415
        vllm_ver = getattr(vllm, "__version__", "unknown")
    except ImportError:
        vllm_ver = "unknown"
    return {
        "gpu": gpu,
        "gpu_driver": driver,
        "unified_mem_gb": mem_gb,
        "vllm_version": vllm_ver,
        "tensor_parallel": int(os.environ.get("TP", "1")),
        "gpu_memory_utilization": float(os.environ.get("GPU_UTIL", "0.9")),
        "max_model_len": int(os.environ.get("VLLM_MAX_LEN", "8192")),
        "quantization": quantization,
    }


def _load_history(mode: str) -> dict:
    """checks.json의 기존 servebench 기록 (타깃별 누적 비교용)."""
    try:
        data = json.loads(checks_path(mode).read_text(encoding="utf-8"))
        sb = data.get("servebench")
        if isinstance(sb, dict) and isinstance(sb.get("targets"), dict):
            return {"targets": dict(sb["targets"]),
                    "hardware": dict(sb.get("hardware") or {})}
    except (OSError, json.JSONDecodeError):
        pass
    return {"targets": {}, "hardware": {}}


def _one(client, model_id: str, prompt: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    r = client.completions.create(model=model_id, prompt=prompt, temperature=BENCH_TEMPERATURE,
                                  max_tokens=max_tokens)
    wall = time.perf_counter() - t0
    text = r.choices[0].text or ""
    usage = getattr(r, "usage", None)
    out_tokens = int(getattr(usage, "completion_tokens", 0) or 0) or max(1, len(text) // 2)
    return {"wall_s": wall, "output_tokens": out_tokens,
            "tok_per_s": out_tokens / max(wall, 1e-6), "chars": len(text)}


def main(mode: str, n: int, concurrency: int, base_url: str, port: int, model: str | None,
         max_tokens: int, target: str | None = None,
         quantization: str = "") -> dict:
    from openai import OpenAI

    url = base_url or f"http://localhost:{port}/v1"
    payload: dict = {
        "mode": mode,
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n": n,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "temperature": BENCH_TEMPERATURE,
        "warmup": BENCH_WARMUP,
        "published_reference": PUBLISHED_REFERENCE,
    }
    try:
        client = OpenAI(base_url=url, api_key="dummy", timeout=600)
        model_id = model or client.models.list().data[0].id
    except Exception as e:  # noqa: BLE001 — 서버가 없으면 측정 불가
        payload["ok"] = False
        payload["status"] = "skipped"
        payload["reason"] = f"no server reachable at {url}: {type(e).__name__}: {e}"
        # skip도 기존 타깃 누적 기록은 보존한다 (다음 타깃 측정이 덮어쓰지 않도록).
        hist = _load_history(mode)
        if hist["targets"] or hist["hardware"]:
            payload["targets"] = hist["targets"]
            if hist["hardware"]:
                payload["hardware"] = hist["hardware"]
        merge_check(mode, "servebench", payload)
        out = outdir(mode) / f"servebench-{mode}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"servebench SKIP: {payload['reason']}")
        return payload

    slug = target or infer_target(model_id)
    hw = hardware_info(quantization or ("none" if slug == "base" else "compressed-tensors"))
    for _ in range(max(0, BENCH_WARMUP)):  # 웜업 — CUDA 그래프·KV 캐시 안정화 후 측정
        try:
            _one(client, model_id, PROMPTS[0], min(max_tokens, 64))
        except Exception:  # noqa: BLE001, S110 — 웜업 실패는 무시하고 본 측정 진행
            pass
    prompts = (PROMPTS * ((n // max(len(PROMPTS), 1)) + 1))[:n]
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        results = list(ex.map(lambda p: _one(client, model_id, p, max_tokens), prompts))
    ttft_like = [r["wall_s"] for r in results]
    tps = [r["tok_per_s"] for r in results]
    run = {
        "model": model_id,
        "server": url,
        "measured_at": payload["measured_at"],
        "n": n,
        "max_tokens": max_tokens,
        "wall_s_total": round(sum(ttft_like), 3),
        "output_tokens_total": sum(r["output_tokens"] for r in results),
        "tok_per_s_mean": round(statistics.fmean(tps), 1) if tps else 0.0,
        "tok_per_s_p50": round(statistics.median(tps), 1) if tps else 0.0,
        "req_latency_s_p50": round(statistics.median(ttft_like), 3) if ttft_like else 0.0,
        "req_latency_s_max": round(max(ttft_like), 3) if ttft_like else 0.0,
    }
    hist = _load_history(mode)
    hist["targets"].setdefault(slug, {})[str(concurrency)] = run
    hist["hardware"] = hw
    payload.update({
        "ok": True,
        "status": "ok",
        "target": slug,
        "hardware": hw,
        "targets": hist["targets"],
        # 구 카드 렌더러 호환: 최신 1회 측정값을 최상위에도 둔다.
        "model": model_id,
        "server": url,
        "wall_s_total": run["wall_s_total"],
        "output_tokens_total": run["output_tokens_total"],
        "tok_per_s_mean": run["tok_per_s_mean"],
        "tok_per_s_p50": run["tok_per_s_p50"],
        "req_latency_s_p50": run["req_latency_s_p50"],
        "req_latency_s_max": run["req_latency_s_max"],
    })
    merge_check(mode, "servebench", payload)
    out = outdir(mode) / f"servebench-{mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"servebench: model={model_id} c={concurrency} n={n} "
          f"tok/s(p50)={payload['tok_per_s_p50']} lat(p50)={payload['req_latency_s_p50']}s")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="서빙 처리량 측정 (Tier 3)")
    ap.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    ap.add_argument("--n", type=int, default=int(os.environ.get("SERVE_BENCH_N", "8")))
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("SERVE_BENCH_C", "1")))
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("SERVE_BENCH_MAXTOK", "256")))
    ap.add_argument("--model", default=None)
    ap.add_argument("--target", default=os.environ.get("SERVE_BENCH_TARGET", ""),
                    help="타깃 slug (base|nvfp4-awq|...). 비우면 served-model-name에서 추론")
    ap.add_argument("--quantization", default=os.environ.get("SERVE_BENCH_QUANT", ""))
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", ""))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    a = ap.parse_args()
    main(a.mode, a.n, a.concurrency, a.base_url, a.port, a.model, a.max_tokens,
         a.target or None, a.quantization)
