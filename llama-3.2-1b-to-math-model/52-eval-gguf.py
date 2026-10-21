"""52-eval-gguf.py — Stage 7b-gguf. llama-completion greedy GSM8K 추론 (GGUF 10종).
50-eval.py(vLLM)는 GGUF 미지원 → llama.cpp로 별도 평가한다.
조건 동일 강제: PROMPT_NO_INPUT 포맷, greedy(temp 0), max 512, gsm8k-test 앞 N개.
출력: $M1B_SHARED/outputs/eval-<mode>-<strat>/<target>.jsonl (원본행 + kd_data, 54-score 호환).

  uv run 52-eval-gguf.py --mode mini|full --target fft-gguf-q4_k_m [--n N]
  타깃: fft-gguf-{q8_0,q6_k,q5_k_m,q4_k_m,q3_k_m}, qlora-gguf-{...}
전제: llama-completion 빌드 ($LLAMACPP/build/bin/llama-completion). 없으면 FileNotFoundError.
(llama-cli는 신버전에서 대화형으로 진입하므로 사용 금지. --no-conversation도
llama-cli에서 미지원이라 llama-completion이 정답.)
"""

import argparse
import json
import os
import subprocess

from io_common import commit_file, tmp_path

SHARED = os.environ.get("M1B_SHARED", os.environ.get("P5_SHARED", "/rosenas/data/AIML/math-1b-shared"))
LLAMACPP = os.environ.get("LLAMACPP", os.path.expanduser("~/git/llama.cpp"))
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)
GQUANTS = ["q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m"]
TARGETS = [f"fft-gguf-{q}" for q in GQUANTS] + [f"qlora-gguf-{q}" for q in GQUANTS]


def resolve(target: str, mode: str, strat: str) -> str:
    if target.startswith("fft-gguf-"):
        q = target[len("fft-gguf-") :]
        return f"{SHARED}/models/synthetic-fft-{mode}-{strat}-gguf/model-{q}.gguf"
    if target.startswith("qlora-gguf-"):
        q = target[len("qlora-gguf-") :]
        return (
            f"{SHARED}/models/synthetic-qlora-{mode}-{strat}-merged-gguf/model-{q}.gguf"
        )
    raise SystemExit(f"unknown target: {target} (gguf 변종 10종만 지원)")


def gen_one(cli: str, model: str, prompt: str, n_predict: int, ctx: str) -> str:
    proc = subprocess.run(
        [
            cli,
            "-m",
            model,
            "-p",
            prompt,
            "-n",
            str(n_predict),
            "-c",
            ctx,
            "--temp",
            "0",
            "--top-k",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=1200,
        stdin=subprocess.DEVNULL,
    )  # 대화형 진입 원천 차단
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-completion rc={proc.returncode}: {(proc.stderr or '')[-300:]}"
        )
    out = proc.stdout or ""
    if out.startswith(prompt):  # 프롬프트 에코 제거 (버전 무관)
        out = out[len(prompt) :]
    return out.strip()


def main(mode: str, strat: str, target: str, n: int):
    cli = os.environ.get("LLAMACPP_CLI", f"{LLAMACPP}/build/bin/llama-completion")
    if not (os.path.isfile(cli) and os.access(cli, os.X_OK)):
        raise FileNotFoundError(
            f"llama-completion 없음: {cli} (llama.cpp에서 cmake --build로 빌드)"
        )
    rows = []
    with open(f"{SHARED}/datasets/gsm8k-test.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if n > 0:
        rows = rows[:n]
    model = resolve(target, mode, strat)
    if not os.path.exists(model):
        raise FileNotFoundError(f"산출물 없음: {model} (42/52 먼저 실행)")
    ctx = os.environ.get("LLAMACPP_CTX", "2048")

    def q_of(r):
        s = r["source"]
        return s[0] if isinstance(s, list) else s

    outdir = f"{SHARED}/outputs/eval-{mode}-{strat}"
    os.makedirs(outdir, exist_ok=True)
    outp = f"{outdir}/{target}.jsonl"
    tmp = tmp_path(outp)
    with open(tmp, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            t = gen_one(
                cli, model, PROMPT_NO_INPUT.format(instruction=q_of(r)), 512, ctx
            )
            r = dict(r)
            r["kd_data"] = [t]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{mode}][{target}] {i + 1}/{len(rows)} chars={len(t)}", flush=True)
    commit_file(tmp, outp)
    print(f"[{mode}][{target}] n={len(rows)} model={model} -> {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument(
        "--strat",
        default=os.environ.get("STRAT", "single"),
        choices=["single", "ddp", "fsdp"],
    )
    ap.add_argument("--target", required=True, choices=TARGETS)
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    n = (
        a.n
        if a.n is not None
        else int(os.environ.get("EVAL_N", "10" if a.mode == "mini" else "0"))
    )
    main(a.mode, a.strat, a.target, n)
