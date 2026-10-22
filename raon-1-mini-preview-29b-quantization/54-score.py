"""54-score.py — eval 스모크 집계 (정답지 없는 스모크용).
지표: 완결율(빈 출력 제외), 평균 출력 길이, thinking 포함률(<think> 흔적).
full 한국어 벤치(KoBEST 등) 정식 평가는 별도 스코프.
출력: outputs/eval-<mode>/score.json (stdout 표).

  uv run 54-score.py --mode smoke
"""

import argparse
import glob
import json
import os

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")


def main(mode: str):
    outdir = f"{SHARED}/outputs/eval-{mode}"
    rows = {}
    for fp in sorted(glob.glob(f"{outdir}/*.jsonl")):
        t = os.path.basename(fp)[:-6]
        outs = [json.loads(l)["output"] for l in open(fp, encoding="utf-8") if l.strip()]
        n = len(outs)
        done = sum(1 for o in outs if o.strip())
        think = sum(1 for o in outs if "<think>" in o.lower() or "생각" in o or "추론" in o)
        avglen = sum(len(o) for o in outs) / max(n, 1)
        rows[t] = {"n": n, "done_rate": round(done / max(n, 1), 3),
                   "avg_len": round(avglen, 1), "think_rate": round(think / max(n, 1), 3)}
    with open(f"{outdir}/score.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"{'target':12} {'n':>4} {'done':>6} {'avglen':>8} {'think':>6}")
    for t, r in rows.items():
        print(f"{t:12} {r['n']:>4} {r['done_rate']:>6} {r['avg_len']:>8} {r['think_rate']:>6}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    a = ap.parse_args()
    main(a.mode)
