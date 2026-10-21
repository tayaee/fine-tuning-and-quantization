"""24-postprocess.py — Stage 2c. 생성 CSV 후처리 → SFT/캘리브용 JSONL.
원본: data_check.ipynb (Q/A 마커 split, 케이스별 예외, 패턴 제거+rstrip).
규칙:
  - 'Transformed Domain Question'으로 split → 뒷부분을 질문 후보로.
  - 질문 후보에 'Transformed Domain Answer'가 있으면 그 앞=질문, 뒤=답변.
  - 둘 중 하나라도 없거나 비면 해당 행 drop.
  - 잔여 마커/장식 패턴 제거 + 양끝 strip.

  uv run 24-postprocess.py --mode mini|full
출력: $M1B_SHARED/datasets/synthetic-<mode>.jsonl ({"source":[q],"target":[a]})
"""

import argparse
import json
import os
import re

from io_common import commit_file, tmp_path

SHARED = os.environ.get("M1B_SHARED", os.environ.get("P5_SHARED", "/rosenas/data/AIML/math-1b-shared"))
QMARK = "Transformed Domain Question"
AMARK = "Transformed Domain Answer"
STRIP_PATTERNS = [
    r"^\*\*\s*",
    r"\*\*$",
    r"^:\s*",
    r"^-\s*",
    r"^\d+[\.\)]\s*",
    r"^(Transformed Domain Question|Transformed Domain Answer)\s*",
]


def clean(s: str) -> str:
    s = s.strip().strip("*").strip()
    for p in STRIP_PATTERNS:
        s = re.sub(p, "", s).strip()
    s = re.sub(r"\n-\s*$", "", s).strip()
    return s.strip().rstrip().strip('"').strip()


def split_row(t: str):
    if not t or QMARK not in t or AMARK not in t:
        return None
    if t.count(QMARK) > 1 or t.count(AMARK) > 1:
        return None
    _, _, after_q = t.partition(QMARK)
    q, _, a = after_q.partition(AMARK)
    q, a = clean(q), clean(a)
    if len(q) < 10 or len(a) < 10:
        return None
    return q, a


def main(mode: str):
    import pandas as pd

    df = pd.read_csv(f"{SHARED}/datasets/generated-{mode}.csv")
    rows, dropped = [], 0
    for t in df["generated"].tolist():
        r = split_row(str(t))
        if r is None:
            dropped += 1
            continue
        rows.append({"source": [r[0]], "target": [r[1]]})
    out = f"{SHARED}/datasets/synthetic-{mode}.jsonl"
    tmp = tmp_path(out)
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    commit_file(tmp, out)
    print(f"[{mode}] kept={len(rows)} dropped={dropped} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    main(ap.parse_args().mode)
