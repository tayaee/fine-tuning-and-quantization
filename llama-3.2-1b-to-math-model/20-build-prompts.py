"""20-build-prompts.py — Stage 2b. 후보 2개씩 template에 채워 프롬프트 생성.
원본: template.txt({dg_instruct}/{dg_output}) + notebook 후반부
(강의는 나중에 instruction+input 합침 → dg_instruct에 input 포함).

  uv run 20-build-prompts.py --mode mini|full [--push]
출력: $M1B_SHARED/datasets/prompts-<mode>.parquet [prompt]
  --push 시 tayaee/alpaca_syntheticdatagen_prompt-<mode> 로 Hub 업로드
"""

import argparse
import os
import random

from io_common import commit_file, tmp_path

SHARED = os.environ.get("M1B_SHARED", os.environ.get("P5_SHARED", "/rosenas/data/AIML/math-1b-shared"))
HF_ID = os.environ.get("HF_ID", "tayaee")


def target_n(mode: str) -> int:
    return int(os.environ.get("SYNTH_N", "64" if mode == "mini" else "10000"))


def main(mode: str, seed: int, push: bool):
    import pandas as pd

    random.seed(seed)
    with open(
        os.path.join(os.path.dirname(__file__), "assets", "template.txt"),
        encoding="utf-8",
    ) as f:
        template = f.read()
    cand: pd.DataFrame = pd.read_parquet(f"{SHARED}/datasets/candidates-{mode}.parquet")
    cand = cand.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    prompts = []
    for _, r in cand.head(target_n(mode)).iterrows():
        # 강의 후반 수정 반영: instruction에 input을 합친다
        inst = str(r["dg_instruction"]) + "\n" + str(r["dg_input"])
        prompts.append(template.format(dg_instruct=inst, dg_output=str(r["dg_output"])))
    df = pd.DataFrame({"prompt": prompts})
    out = f"{SHARED}/datasets/prompts-{mode}.parquet"
    tmp = tmp_path(out)
    df.to_parquet(tmp)
    commit_file(tmp, out)
    print(f"[{mode}] n={len(df)} -> {out}")
    if push:
        from datasets import Dataset

        Dataset.from_pandas(df[["prompt"]]).push_to_hub(
            f"{HF_ID}/alpaca_syntheticdatagen_prompt-{mode}"
        )
        print(f"[{mode}] pushed: {HF_ID}/alpaca_syntheticdatagen_prompt-{mode}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    main(a.mode, a.seed, a.push)
