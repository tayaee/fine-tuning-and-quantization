"""s4-01-measure-ppl.py — base + 4종 in-domain PPL + Δ판정.
텍스트: calib 파일에서 앞 N개 (학습-캘리브 동일 분포).
판정 (base 대비 Δ): <0.3 Accept / 0.3~1.0 Conditional / >=1.0 Discard.
출력: outputs/ppl-<mode>/<target>.json + 집계 outputs/ppl-<mode>/summary.json

메모리 격리(요구사항 1): 29B 모델 5개를 한 프로세스에서 순차 로드하면 GB10
(128GB 통합 메모리)에서 이전 모델의 CUDA 캐시/파편화가 반환되지 않아 2번째
로드부터 OOM이 난다. 그래서 타겟 1개를 측정하고 나서 gc.collect() +
torch.cuda.empty_cache()로 cuda 캐시를 비우고, s4-01-measure-ppl.sh는 타겟마다
별도 프로세스로 띄워 OS 메모리까지 100% 회수한다.

  uv run s4-01-measure-ppl.py --mode smoke|full [--n N] [--targets all|base,nvfp4-gptq-transformers,...]
  uv run s4-01-measure-ppl.py --mode smoke --summarize-only
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from typing import TYPE_CHECKING

from quant_common import load_calib_texts

if TYPE_CHECKING:
    from pathlib import Path

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")
# base = 19-dequant 복원본(공정 기준). 이슈는 bf16으로 부르지만 기존 산출물
# 파일명(base.json)과 s4-07-eval/s4-13-gsm8k 관례가 base라 별칭으로만 받는다.
ALL_TARGETS = ["base", "awq-w5-transformers", "awq-w3-transformers", "nvfp4-awq-transformers", "nvfp4-gptq-transformers"]
TARGET_ALIASES = {"bf16": "base", "raon-bf16": "base"}


def MODELS_DIR(mode: str) -> str:
    return f"{SHARED}/models"


def OUTDIR(mode: str) -> str:
    return f"{SHARED}/outputs/ppl-{mode}"


def parse_targets(spec: str | None = None) -> list[str]:
    """`--targets` 문자열 → 타겟 리스트. 콤마/공백 모두 구분자로 받는다.
    기본값: 환경변수 TARGETS (없으면 전 타겟). `all`은 전 타겟.
    bf16은 base로 정규화하고 중복은 첫 등장 순서로 제거한다."""
    if spec is None:
        spec = os.environ.get("TARGETS") or "all"
    spec = spec.strip()
    if spec == "all":
        return list(ALL_TARGETS)
    out: list[str] = []
    for raw in spec.replace(",", " ").split():
        t = TARGET_ALIASES.get(raw, raw)
        if t not in out:
            out.append(t)
    if not out:
        raise SystemExit(f"--targets가 비었다: {spec!r}")
    return out


def resolve(target: str, mode: str) -> str:
    if target == "base":
        return f"{MODELS_DIR(mode)}/raon-bf16"  # 19-dequant 복원본 (공정 기준)
    return f"{MODELS_DIR(mode)}/raon-{target}-{mode}"


def model_exists(model_name: str) -> bool:
    return os.path.exists(f"{model_name}/config.json")


def release_memory() -> None:
    """측정 직후 CUDA 캐시/파이썬 힙 반환 (5종 순차 로드 OOM 방지)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except (ImportError, RuntimeError):  # pragma: no cover - CPU 전용 환경
        pass


def ppl_of(model_name: str, texts: list[str], seq: int) -> float:
    """모델 하나를 로드해 in-domain PPL 계산 후 메모리 반환."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from quant_common import patch_unified_memory_info

    patch_unified_memory_info()

    @torch.no_grad()
    def _run() -> float:
        kw = {"trust_remote_code": True}
        tok = AutoTokenizer.from_pretrained(model_name, **kw)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="auto", **kw
        )
        model.eval()
        nll, ntok = 0.0, 0
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=seq)["input_ids"].to(model.device)
            if ids.size(1) < 8:
                continue
            out = model(ids, labels=ids)
            nll += out.loss.item() * ids.size(1)
            ntok += ids.size(1)
        return math.exp(nll / ntok) if ntok else float("nan")

    try:
        return _run()
    finally:
        release_memory()


def acceptance_spec(target: str) -> dict:
    """타깃별 수용 기준(Stage 1 산출물). 파일이 없으면 기존 기본값(0.3/1.0)."""
    try:
        from sdlc import load_acceptance, target_spec

        return target_spec(load_acceptance(), target)
    except Exception:  # noqa: BLE001 — 기준 파일이 없으면 하드코딩 기본값으로 폴백
        return {"ppl_delta_warn": 0.3, "ppl_delta_max": 1.0}


def verdict(delta: float, spec: dict | None = None) -> str:
    """base 대비 Δ 판정: <warn Accept / <max Conditional / 그 이상 Discard.

    임계값은 config/acceptance.toml(Stage 1)이 단일 소스다.
    """
    spec = spec or {}
    warn = float(spec.get("ppl_delta_warn", 0.3))
    mx = float(spec.get("ppl_delta_max", 1.0))
    if delta < warn:
        return "Accept"
    if delta < mx:
        return "Conditional"
    return "Discard"


def summarize(outdir: str | Path, measured: set[str] | tuple[str, ...] = ()) -> dict:
    """디렉터리의 <target>.json을 취합해 summary.json 갱신. 모델을 로드하지 않는다.

    base(캐시 가능)가 없으면 'no-ref', base가 이번 실행에 측정된 건 'measured',
    이전 실행 캐시를 쓴 건 'cached'로 delta 판정의 출처를 명시한다.
    """
    outdir = str(outdir)
    rows: dict[str, dict] = {}
    for name in sorted(os.listdir(outdir)):
        if not name.endswith(".json") or name == "summary.json":
            continue
        with open(os.path.join(outdir, name), encoding="utf-8") as f:
            row = json.load(f)
        rows[row.get("target") or name[:-5]] = dict(row)
    base = rows.get("base", {}).get("ppl")
    ref_src = "measured" if "base" in measured else "cached"
    for t, r in rows.items():
        if t == "base":
            r["verdict"] = "ref"
            continue
        if r.get("skipped"):
            r["verdict"] = "skipped"
            continue
        if base is None:
            r["verdict"] = "no-ref"
            continue
        r["delta"] = r["ppl"] - base
        r["verdict"] = verdict(r["delta"], acceptance_spec(t))
        r["ref_source"] = ref_src
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return rows


def measure_targets(mode: str, n: int, targets: str | None, seq: int, outdir: str | Path) -> dict:
    """타겟별 PPL 측정 → <target>.json 저장 → summary 재집계.

    targets에 base가 없더라도 디렉터리에 캐시된 base.json이 있으면 판정 기준으로 쓴다.
    """
    os.makedirs(outdir, exist_ok=True)
    calib_file = os.environ.get("CALIB_FILE") or f"{SHARED}/datasets/raon-calibration-{8 if mode == 'smoke' else 256}.jsonl"
    texts = load_calib_texts(n, calib_file)
    measured: set[str] = set()
    for t in parse_targets(targets):
        name = resolve(t, mode)
        if not model_exists(name):
            hint = " (run s2-01-dequant-mxfp8.sh first)" if t == "base" else ""
            print(f"SKIP {t}: missing {name}{hint}")
            continue
        value = ppl_of(name, texts, seq)
        if math.isnan(value):  # 유효 토큰 0개 — 오염된 json을 남기지 않는다
            print(f"SKIP {t}: ppl=NaN (유효 토큰 없음)")
            continue
        with open(os.path.join(outdir, f"{t}.json"), "w", encoding="utf-8") as f:
            json.dump({"target": t, "ppl": value, "n": n, "seq_len": seq}, f, ensure_ascii=False, indent=2)
        measured.add(t)
        print(f"{t}: ppl={value:.3f}")
        release_memory()
    return summarize(outdir, measured=measured)


def main(mode: str, n: int, targets: str | None, summarize_only: bool = False):
    outdir = OUTDIR(mode)
    os.makedirs(outdir, exist_ok=True)
    if summarize_only:
        return summarize(outdir)
    seq = int(os.environ.get("SEQ_LEN", "512" if mode == "smoke" else "2048"))
    return measure_targets(mode, n, targets, seq, outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="in-domain PPL 측정 + base 대비 Δ판정")
    ap.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--targets", default=None, help="콤마/공백 구분 (기본: $TARGETS 또는 all)")
    ap.add_argument("--summarize-only", action="store_true", help="모델 로드 없이 캐시 json만 집계")
    a = ap.parse_args()
    d_n = int(os.environ.get("PPL_N", "8" if a.mode == "smoke" else "32"))
    main(a.mode, a.n or d_n, a.targets, a.summarize_only)