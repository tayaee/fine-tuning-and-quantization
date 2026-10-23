"""50-eval.py — vLLM offline 한국어 스모크 생성.
입력: calib 파일 뒤 EVAL_N개 (앞은 PPL/calib용, 겹침 방지) — 전용 eval 파일이 있으면 우선.
출력: outputs/eval-<mode>/<target>.jsonl ({prompt, output}).
양자화 ckpt는 타깃별 quantization 플래그 필요 (60-serve-vllm.sh과 동일 매핑).

  uv run 50-eval.py --mode smoke --target nvfp4-gptq --n 8 --tp 2
"""

import argparse
import json
import os

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")
ALL_TARGETS = ["base", "awq-w3", "awq-w5", "nvfp4-awq", "nvfp4-gptq"]
QUANT_OF = {"nvfp4-awq": "compressed-tensors",
            "nvfp4-gptq": "compressed-tensors", "awq-w5": "compressed-tensors", "awq-w3": "compressed-tensors"}


def resolve(target: str, mode: str) -> str:
    if target == "base":
        return f"{SHARED}/models/raon-bf16"  # 30-dequant 복원본
    return f"{SHARED}/models/raon-{target}-{mode}"


def load_prompts(n: int, calib_file: str | None) -> list[str]:
    from quant_common import load_calib_texts

    eval_file = os.environ.get("EVAL_FILE", f"{SHARED}/datasets/raon-eval.jsonl")
    if os.path.exists(eval_file):
        texts = load_calib_texts(10**9, eval_file)
        print(f"eval_src={eval_file}")
        return texts[:n] if n else texts
    all_texts = load_calib_texts(10**9, calib_file)
    print("eval_src=calib-tail (전용 eval 파일 없음, 12 뒤쪽 슬라이스)")
    return all_texts[-n:] if n else all_texts


def main(mode: str, target: str, n: int, tp: int, calib_file: str | None):
    from vllm import LLM, SamplingParams

    name = resolve(target, mode)
    if not os.path.exists(f"{name}/config.json"):
        raise SystemExit(f"missing model: {name} (run 30-dequant-mxfp8.sh first)" if target == "base" else f"missing model: {name}")
    prompts = load_prompts(n, calib_file)
    kw: dict = {"model": name, "tensor_parallel_size": tp, "trust_remote_code": True,
                "gpu_memory_utilization": float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.9")),
                "max_model_len": int(os.environ.get("VLLM_MAX_LEN", "8192"))}
    if target in QUANT_OF:
        kw["quantization"] = QUANT_OF[target]
    llm = LLM(**kw)
    outs = llm.generate(prompts, SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=1024))
    outdir = f"{SHARED}/outputs/eval-{mode}"
    os.makedirs(outdir, exist_ok=True)
    outp = f"{outdir}/{target}.jsonl"
    with open(outp, "w", encoding="utf-8") as f:
        for p, o in zip(prompts, outs):
            f.write(json.dumps({"prompt": p, "output": o.outputs[0].text}, ensure_ascii=False) + "\n")
    print(f"[{mode}/{target}] n={len(prompts)} -> {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--target", default="nvfp4-gptq",
                    choices=["all"] + ALL_TARGETS)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--calib-file", default=None)
    a = ap.parse_args()
    d_n = int(os.environ.get("EVAL_N", "8" if a.mode == "smoke" else "200"))
    d_tp = int(os.environ.get("TP", "2"))
    targets = ALL_TARGETS if a.target == "all" else [a.target]
    for t in targets:
        main(a.mode, t, a.n or d_n, a.tp or d_tp, a.calib_file)
