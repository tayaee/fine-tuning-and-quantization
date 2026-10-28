"""s4-28-bench-transformers.py — transformers 인프로세스 처리량 측정 (Tier 3: 배포 성능 증거).

배경: stock vLLM(0.31.0)은 Raon 커스텀 아키텍처를 서빙하지 못한다
(TransformersMoEForCausalLM 폴백 → `assert top_k is not None`으로 사망).
HF 카드의 정석 로딩은 transformers + trust_remote_code이므로,
서빙 처리량도 같은 스택에서 인프로세스로 잰다 — 서버가 필요 없다.

워크로드는 s4-25-servebench.py와 같은 고정 8프롬프트(타깃 간 비교 공정용).
C=1 순차 greedy 생성만 측정한다 (동시성은 vLLM 서버 전제라 해당 없음).

  uv run s4-28-bench-transformers.py --mode full --targets base,nvfp4-awq --n 8

산출물:
  outputs/eval-<mode>/bench-transformers-<mode>.json  (전체 기록)
  outputs/eval-<mode>/checks.json servebench.targets.<t>["1"] (카드 렌더용, backend 표기)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sdlc import checks_path, merge_check

SHARED = Path(os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared"))

ALL_TARGETS = ("base", "awq-w3", "awq-w5", "nvfp4-awq", "nvfp4-gptq")

#: s4-25-servebench.py와 동일한 고정 워크로드 (수치 비교 공정용).
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

WARMUP = 1


def resolve_target(target: str, mode: str) -> Path:
    """타깃 → 로컬 모델 디렉터리 (s4-07-eval.py resolve와 동일)."""
    if target == "base":
        return SHARED / "models" / "raon-bf16"  # 19-dequant 복원본
    return SHARED / "models" / f"raon-{target}-{mode}"


def parse_targets(s: str) -> list[str]:
    """콤마/공백 구분 타깃 목록. "all"이면 base + 4종."""
    tokens = [t.strip() for t in s.replace(",", " ").split() if t.strip()]
    if not tokens or "all" in tokens:
        return list(ALL_TARGETS)
    unknown = [t for t in tokens if t not in ALL_TARGETS]
    if unknown:
        raise SystemExit(f"unknown target(s): {unknown} (allowed: {list(ALL_TARGETS)} | all)")
    return tokens


def mem_snapshot() -> dict:
    """가용 메모리 + 벤치 프로세스 RSS (GB). 카드 렌더러가 읽는 server_rss_gb 키를 재사용한다."""
    avail_kb, rss_kb = 0, 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
                    break
    except OSError:
        pass
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    break
    except OSError:
        pass
    return {"mem_avail_gb": round(avail_kb / 1024 / 1024, 1),
            "server_rss_gb": round(rss_kb / 1024 / 1024, 1)}


def hardware_info() -> dict:
    """측정 하드웨어·런타임 조건 (README 명시용). s4-25-servebench.py와 같은 키 + backend 표기."""
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
        import transformers  # noqa: PLC0415
        tf_ver = getattr(transformers, "__version__", "unknown")
    except ImportError:
        tf_ver = "unknown"
    try:
        import torch  # noqa: PLC0415
        torch_ver = getattr(torch, "__version__", "unknown")
    except ImportError:
        torch_ver = "unknown"
    return {
        "gpu": gpu,
        "gpu_driver": driver,
        "unified_mem_gb": mem_gb,
        "backend": "transformers",
        "engine": f"transformers {tf_ver} / torch {torch_ver}",
        "tensor_parallel": 1,
        "gpu_memory_utilization": float(os.environ.get("GPU_UTIL", "0.9")),
        "max_model_len": int(os.environ.get("VLLM_MAX_LEN", "8192")),
        "quantization": "in-process (no server)",
    }


def summarize(per_req: list[dict]) -> dict:
    """요청별 기록 → 집계 지표 (s4-25 run dict와 같은 키)."""
    tps = [r["tok_per_s"] for r in per_req]
    lat = [r["wall_s"] for r in per_req]
    return {
        "wall_s_total": round(sum(lat), 3),
        "output_tokens_total": sum(r["output_tokens"] for r in per_req),
        "tok_per_s_mean": round(statistics.fmean(tps), 1) if tps else 0.0,
        "tok_per_s_p50": round(statistics.median(tps), 1) if tps else 0.0,
        "req_latency_s_p50": round(statistics.median(lat), 3) if lat else 0.0,
        "req_latency_s_max": round(max(lat), 3) if lat else 0.0,
    }


def release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except (ImportError, RuntimeError):
        pass


def measure_target(target: str, model_path: Path, n: int, max_tokens: int) -> dict:
    """타깃 1개를 인프로세스로 측정한다. 모델은 함수 안에서 로드·반납한다."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from quant_common import patch_unified_memory_info

    patch_unified_memory_info()
    mem_before = mem_snapshot()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model.eval()
    mem_warm: dict = {}
    try:
        for _ in range(max(0, WARMUP)):  # 웜업 — 커널·캐시 안정화 후 측정
            ids = tok(PROMPTS[0], return_tensors="pt")["input_ids"].to(model.device)
            with torch.no_grad():
                model.generate(ids, do_sample=False, max_new_tokens=min(max_tokens, 32),
                               use_cache=True)
        mem_warm = mem_snapshot()
        prompts = (PROMPTS * ((n // max(len(PROMPTS), 1)) + 1))[:n]
        per_req: list[dict] = []
        for p in prompts:
            ids = tok(p, return_tensors="pt", truncation=True, max_length=2048)["input_ids"]
            ids = ids.to(model.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(ids, do_sample=False, max_new_tokens=max_tokens,
                                     use_cache=True)
            wall = time.perf_counter() - t0
            text = tok.decode(out[0, ids.size(1):], skip_special_tokens=True)
            out_tokens = len(tok.encode(text)) or 1
            per_req.append({"wall_s": wall, "output_tokens": out_tokens,
                            "tok_per_s": out_tokens / max(wall, 1e-6)})
    finally:
        del model
        try:
            del tok
        except UnboundLocalError:
            pass
        release_memory()
    run = summarize(per_req)
    run.update({
        "backend": "transformers",
        "model": str(model_path),
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n": n,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    })
    mem_done = mem_snapshot()
    memory = {"before_load": mem_before, "after_warmup": mem_warm or mem_done,
              "before_teardown": mem_done, "after_teardown": mem_done}
    return {"run": run, "memory": memory}


def load_history(mode: str) -> dict:
    """checks.json의 기존 servebench 기록 (타깃 누적 비교용)."""
    try:
        data = json.loads((SHARED / "outputs" / f"eval-{mode}" / "checks.json").read_text(
            encoding="utf-8"))
        sb = data.get("servebench")
        if isinstance(sb, dict) and isinstance(sb.get("targets"), dict):
            return {"targets": dict(sb["targets"]),
                    "hardware": dict(sb.get("hardware") or {})}
    except (OSError, json.JSONDecodeError):
        pass
    return {"targets": {}, "hardware": {}}


def merge_target(mode: str, target: str, measured: dict, hardware: dict) -> dict:
    """타깃 1개 결과를 checks.json servebench에 누적 + 벤치 파일에 기록. 기존 타깃을 보존한다."""
    hist = load_history(mode)
    hist["targets"].setdefault(target, {})["1"] = measured["run"]
    hist["targets"][target]["memory"] = measured["memory"]
    hist["targets"][target]["backend"] = "transformers"
    hist["hardware"] = hardware
    payload = {
        "mode": mode,
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ok": True,
        "status": "ok",
        "backend": "transformers",
        "concurrency": 1,
        "n": measured["run"]["n"],
        "max_tokens": measured["run"]["max_tokens"],
        "temperature": 0.0,
        "hardware": hardware,
        "targets": hist["targets"],
    }
    merge_check(mode, "servebench", payload)
    out = SHARED / "outputs" / f"eval-{mode}" / f"bench-transformers-{mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def base_recorded(mode: str) -> bool:
    """base 측정값이 이미 checks.json에 있는지 (카드 2줄의 기준행)."""
    v = load_history(mode).get("targets", {}).get("base", {}).get("1", {}).get("tok_per_s_p50")
    return isinstance(v, (int, float))


def ensure_base_first(mode: str, targets: list[str]) -> list[str]:
    """양자화 타깃 요청에 base를 선행시킨다 — 모델 카드가 base+양자화 2줄이 되도록.

    base 기록이 이미 있으면 생략한다 (29B 재로드를 피하기 위함).
    """
    rest = [t for t in targets if t != "base"]
    if not rest or "base" in targets or base_recorded(mode):
        return list(targets)
    print("[auto] base 기록 없음 → base를 먼저 측정한다 (카드 비교행)")
    return ["base", *targets]


def main(mode: str, targets: list[str], n: int, max_tokens: int) -> dict:
    """타깃별 순차 측정. 없는 모델은 skipped로 기록하고 계속한다."""
    targets = ensure_base_first(mode, targets)
    hardware = hardware_info()
    results: dict = {"mode": mode, "backend": "transformers", "targets": {},
                     "skipped": {}, "errors": {}}
    for t in targets:
        path = resolve_target(t, mode)
        if not (path / "config.json").is_file():
            results["skipped"][t] = f"missing model: {path} (양자화/dequant를 먼저 실행하라)"
            print(f"[skip] {t}: {results['skipped'][t]}")
            continue
        print(f"[bench] {t} ← {path} (n={n}, max_tokens={max_tokens}) ...")
        try:
            measured = measure_target(t, path, n, max_tokens)
        except Exception as e:  # noqa: BLE001 — 한 타깃 실패가 나머지를 막지 않는다
            results["errors"][t] = f"{type(e).__name__}: {e}"
            print(f"[FAIL] {t}: {results['errors'][t]}")
            continue
        merge_target(mode, t, measured, hardware)
        results["targets"][t] = measured["run"]
        print(f"[done] {t} tok/s(p50)={measured['run']['tok_per_s_p50']} "
              f"lat(p50)={measured['run']['req_latency_s_p50']}s")
    results["hardware"] = hardware
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="transformers 인프로세스 처리량 측정 (서버 불필요)")
    ap.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    ap.add_argument("--targets", default=os.environ.get("TARGETS", "base,nvfp4-awq"),
                    help="콤마/공백 구분 (base|awq-w3|awq-w5|nvfp4-awq-modelopt|nvfp4-awq|nvfp4-gptq|all)")
    ap.add_argument("--n", type=int, default=int(os.environ.get("TRANSFORMERS_BENCH_N", "8")))
    ap.add_argument("--max-tokens", type=int,
                    default=int(os.environ.get("TRANSFORMERS_BENCH_MAXTOK", "256")))
    a = ap.parse_args()
    out = main(a.mode, parse_targets(a.targets), a.n, a.max_tokens)
    print(f"measured={sorted(out['targets'])} skipped={sorted(out['skipped'])} "
          f"errors={sorted(out['errors'])}")
