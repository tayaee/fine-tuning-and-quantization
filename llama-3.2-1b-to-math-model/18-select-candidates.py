"""18-select-candidates.py — Stage 2a. Alpaca에서 GSM8K 유사 후보 선별.
원본: datagen.ipynb 전반 (all-mpnet-base-v2, 500개 배치, cosine, top1000→100).

  uv run 18-select-candidates.py --mode mini|full [--seed 1]
출력:
  $M1B_SHARED/datasets/alpaca-embeddings.parquet (52k, 캐시 — 모드 공용)
  $M1B_SHARED/datasets/candidates-<mode>.parquet (gsm_idx, dg_instruction, dg_input, dg_output)
"""

import argparse
import os
import random

from io_common import commit_file, tmp_path

SHARED = os.environ.get("M1B_SHARED", os.environ.get("P5_SHARED", "/rosenas/data/AIML/math-1b-shared"))
EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"
BATCH = 500
TOPK, SAMPLEK = 1000, 100


def main(mode: str, seed: int):
    import numpy as np
    import pandas as pd
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    random.seed(seed)
    np.random.seed(seed)

    emb_path = f"{SHARED}/datasets/alpaca-embeddings.parquet"
    dg = None

    # 1. 파일이 없으면 빌드 후 저장
    if not os.path.exists(emb_path):
        raw_df = pd.DataFrame(load_dataset("tatsu-lab/alpaca")["train"])
        raw_df["cri"] = raw_df["instruction"] + raw_df["output"]
        st = SentenceTransformer(EMB_MODEL)

        vecs = []
        for s in range(0, len(raw_df), BATCH):
            batch_texts = raw_df["cri"].tolist()[s : s + BATCH]
            vecs.extend(st.encode(batch_texts, show_progress_bar=False).tolist())

        raw_df["embedding"] = vecs
        tmp = tmp_path(emb_path)
        raw_df[["instruction", "input", "output", "embedding"]].to_parquet(tmp)
        commit_file(tmp, emb_path)
        print(f"[{mode}] embeddings built and cached: {len(raw_df)}")

    # 2. 파일이 존재하고 dg가 아직 None이면 로드
    if os.path.exists(emb_path) and dg is None:
        dg = pd.read_parquet(emb_path)
        print(f"[{mode}] embeddings cache hit/loaded: {len(dg)}")

    st = SentenceTransformer(EMB_MODEL)
    gsm_texts, gsm_idx = [], []
    with open(f"{SHARED}/datasets/gsm8k-train.jsonl", encoding="utf-8") as f:
        import json

        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            src = r["source"][0] if isinstance(r["source"], list) else r["source"]
            tgt = r["target"][0] if isinstance(r["target"], list) else r["target"]
            gsm_texts.append(src + tgt)
            gsm_idx.append(i)
    gsm_rows = int(os.environ.get("GSM_ROWS", "256" if mode == "mini" else "0"))
    if gsm_rows > 0:
        gsm_texts, gsm_idx = gsm_texts[:gsm_rows], gsm_idx[:gsm_rows]
    gsm_vecs = np.array(
        st.encode(gsm_texts, batch_size=32, show_progress_bar=False), dtype=np.float32
    )
    dg_vecs = np.array(dg["embedding"].tolist(), dtype=np.float32)

    sim = cosine_similarity(gsm_vecs, dg_vecs)  # [G, 52k]
    rows = []
    for gi, grow in zip(gsm_idx, sim):
        top = np.argsort(grow)[-TOPK:]  # 상위 1000
        for j in random.sample(top.tolist(), SAMPLEK):
            rows.append(
                (
                    gi,
                    dg.iloc[j]["instruction"],
                    dg.iloc[j]["input"],
                    dg.iloc[j]["output"],
                )
            )
    out = pd.DataFrame(
        rows, columns=["gsm_idx", "dg_instruction", "dg_input", "dg_output"]
    )
    out_path = f"{SHARED}/datasets/candidates-{mode}.parquet"
    tmp = tmp_path(out_path)
    out.to_parquet(tmp)
    commit_file(tmp, out_path)
    print(f"[{mode}] gsm={len(gsm_idx)} candidates={len(out)} -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    main(a.mode, a.seed)
