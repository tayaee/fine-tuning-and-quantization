"""12-prep-calibration.py — AWQ/GPTQ/FP8 캘리브레이션용 GSM8K 슬라이스 생성.
gsm8k-train.jsonl → seed 고정 shuffle → 앞 N개 → gsm8k-calibration-<N>.jsonl.
순서 편향을 피하려고 shuffle 필수. train에서만 자르고 test는 절대 제외.

  uv run 12-prep-calibration.py [--n 256] [--seed 42]
"""

import argparse
import os
import random

from io_common import commit_file, tmp_path

SHARED = os.environ.get("M1B_SHARED", os.environ.get("P5_SHARED", "/rosenas/data/AIML/math-1b-shared"))


def main(n: int, seed: int):
    src = f"{SHARED}/datasets/gsm8k-train.jsonl"
    out = f"{SHARED}/datasets/gsm8k-calibration-{n}.jsonl"
    rows = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(line)
    if n > len(rows):
        raise SystemExit(f"n={n} > train rows={len(rows)}")
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    tmp = tmp_path(out)
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(rows[i] + "\n" for i in idx[:n])
    commit_file(tmp, out)
    print(f"train={len(rows)} seed={seed} n={n} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.n, a.seed)
