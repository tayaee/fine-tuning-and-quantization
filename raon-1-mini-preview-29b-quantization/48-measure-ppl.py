"""48-measure-ppl.py — base + 5종 in-domain PPL + Δ판정.
텍스트: calib 파일에서 앞 N개 (학습-캘리브 동일 분포).
판정 (base 대비 Δ): <0.3 Accept / 0.3~1.0 Conditional / >=1.0 Discard.
출력: outputs/ppl-<mode>/summary.json + 타깃별 json.

  uv run 48-measure-ppl.py --mode smoke|full [--n N] [--targets all|base,nvfp4-gptq,...]
"""

import argparse
import json
import math
import os

import torch
from quant_common import load_calib_texts

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")
ALL_TARGETS = ["base", "awq-w3", "awq-w5", "nvfp4-awq", "nvfp4-gptq"]


def resolve(target: str, mode: str) -> str:
    if target == "base":
        return f"{SHARED}/models/raon-bf16"  # 30-dequant 복원본 (공정 기준)
    return f"{SHARED}/models/raon-{target}-{mode}"


@torch.no_grad()
def ppl_of(model_name: str, texts: list[str], seq: int) -> float:
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    del model
    torch.cuda.empty_cache()
    return math.exp(nll / ntok)


def verdict(delta: float) -> str:
    if delta < 0.3:
        return "Accept"
    if delta < 1.0:
        return "Conditional"
    return "Discard"


def main(mode: str, n: int, targets: str):
    ts = ALL_TARGETS if targets == "all" else targets.split(",")
    texts = load_calib_texts(n)
    seq = int(os.environ.get("SEQ_LEN", "512" if mode == "smoke" else "2048"))
    outdir = f"{SHARED}/outputs/ppl-{mode}"
    os.makedirs(outdir, exist_ok=True)
    res: dict[str, dict] = {}
    for t in ts:
        name = resolve(t, mode)
        if not os.path.exists(f"{name}/config.json"):
            print(f"SKIP {t}: missing {name} (run 30-dequant-mxfp8.sh first)" if t == "base" else f"SKIP {t}: missing {name}")
            continue
        p = ppl_of(name, texts, seq)
        res[t] = {"ppl": p}
        print(f"{t}: ppl={p:.3f}")
        with open(f"{outdir}/{t}.json", "w", encoding="utf-8") as f:
            json.dump({"target": t, "ppl": p, "n": n}, f, ensure_ascii=False, indent=2)
    base = res.get("base", {}).get("ppl")
    for t, r in res.items():
        if t == "base" or base is None:
            r["verdict"] = "ref" if t == "base" else "no-ref"
        else:
            r["delta"] = r["ppl"] - base
            r["verdict"] = verdict(r["delta"])
    with open(f"{outdir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--targets", default="all")
    a = ap.parse_args()
    d_n = int(os.environ.get("PPL_N", "8" if a.mode == "smoke" else "32"))
    main(a.mode, a.n or d_n, a.targets)
