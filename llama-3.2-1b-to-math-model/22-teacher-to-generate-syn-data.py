"""22-generate.py — teacher로 합성 생성 + invalid 재생성 루프.
원본: data_gen_vllm.py (apply_chat_template, 마커 3종 검사).
마커 미포함 행은 invalid로 분리해 재생성한다 (valid가 안정될 때까지, 최대 3회).
"""

import argparse
import os

from io_common import commit_file, tmp_path

SHARED = os.environ.get("M1B_SHARED", os.environ.get("P5_SHARED", "/rosenas/data/AIML/math-1b-shared"))
MARKERS = ("Transformed Domain Question", "Transformed Domain Answer", "The answer is")
MAX_ITERS = 3


def is_valid(t: str) -> bool:
    return all(m in (t or "") for m in MARKERS)


def main(
    mode: str,
    tp: int,
    maxlen: int,
    teacher: str,
    n: int,
    gpu_mem_util: float,
    max_num_seqs: int,
    revision: str = "",
):
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(f"{SHARED}/datasets/prompts-{mode}.parquet").head(n)
    df = df.reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(teacher, revision=revision or None)

    def to_chat(p):
        return tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )

    llm = LLM(
        model=teacher,  # teacher 8B, student 1B
        revision=revision or None,
        tensor_parallel_size=tp,  # tp1
        max_model_len=maxlen,  # 8192 (mini/full 통일)
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=max_num_seqs,  # 8 (mini/full 통일)
        dtype="auto",
        enforce_eager=True,
    )
    sp = SamplingParams(
        temperature=0.5, top_p=0.8, top_k=5, repetition_penalty=1.05, max_tokens=2048
    )
    print("model ready", flush=True)

    df["generated"] = None
    pending = df.index.tolist()
    for it in range(1, MAX_ITERS + 1):
        chats = [to_chat(df.loc[i, "prompt"]) for i in pending]
        outs = llm.generate(chats, sp)
        nxt = []
        for i, o in zip(pending, outs):
            t = o.outputs[0].text
            df.at[i, "generated"] = t
            if not is_valid(t):
                nxt.append(i)
        print(
            f"[{mode}] iter{it}: valid={len(pending) - len(nxt)}/{len(pending)}",
            flush=True,
        )
        pending = nxt
        if not pending:
            break
    out = f"{SHARED}/datasets/generated-{mode}.csv"
    tmp = tmp_path(out)
    df.to_csv(tmp, index=False)
    commit_file(tmp, out)
    print(f"[{mode}] invalid_remain={len(pending)} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--maxlen", type=int, default=8192)
    ap.add_argument("--teacher", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument(
        "--revision",
        default=os.environ.get("TEACHER_REVISION", ""),
        help="HF 리비전 고정 (기본 $TEACHER_REVISION, 1년 후 재현용).",
    )
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument(
        "--gpu-mem-util",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.8")),
        help="vLLM gpu_memory_utilization (기본 0.8: DGX Spark 통합메모리에서 "
        "OS/Xorg 점유분(~13GB)을 피하려고 0.9에서 낮춤)",
    )
    ap.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="vLLM 동시 처리 시퀀스 수 (기본 8: mini/full 통일. "
        "VLLM_MAX_NUM_SEQS로 오버라이드 가능)",
    )
    a = ap.parse_args()
    n = (
        a.n
        if a.n is not None
        else int(os.environ.get("SYNTH_N", "64" if a.mode == "mini" else "10000"))
    )
    max_num_seqs = (
        a.max_num_seqs
        if a.max_num_seqs is not None
        else int(os.environ.get("VLLM_MAX_NUM_SEQS", "8"))
    )
    main(a.mode, a.tp, a.maxlen, a.teacher, n, a.gpu_mem_util, max_num_seqs, a.revision)
