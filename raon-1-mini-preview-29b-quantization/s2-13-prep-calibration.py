"""s2-13-prep-calibration.py — 한국어 calib 혼합 생성 (chat template 적용済み text).
출싯: datasets/raon-calibration-<N>.jsonl (`{"text"}`).
혼합 (full 기본): KoAlpaca-RealQA 128 + ultrachat_ko 64 + ultrachat_en 64.
smoke는 KoAlpaca만 앞에서 N개 (스모크).

  uv run s2-13-prep-calibration.py --n 8 --seed 42 [--no-template]
"""

import argparse
import os
import random

from io_common import commit_file, tmp_path

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")
BASE_MODEL = os.environ.get("BASE_MODEL", "KRAFTON/Raon-1-mini-preview")

SOURCES = [
    ("beomi/KoAlpaca-RealQA", "train", 128),
    ("ChuGyouk/HFH4_ultrachat_200k_ko", "train", 64),
    ("HuggingFaceH4/ultrachat_200k", "train_sft", 64),
]


def to_messages(rec: dict) -> list[dict]:
    if "messages" in rec and isinstance(rec["messages"], list):
        return [m for m in rec["messages"] if m.get("content")][:4]
    user = rec.get("instruction") or rec.get("input") or rec.get("prompt") or rec.get("question") or ""
    asst = rec.get("output") or rec.get("response") or rec.get("answer") or ""
    msgs = []
    if user:
        msgs.append({"role": "user", "content": str(user)})
    if asst:
        msgs.append({"role": "assistant", "content": str(asst)})
    return msgs


def cache_valid(path: str, n: int, seed: int) -> bool:
    """이미 저장된 산출물이 n/seed/형식 조건을 만족하면 True (HF 재다운로드 방지)."""
    import json

    if not os.path.exists(path) or not os.path.exists(path + ".meta"):
        return False
    try:
        meta = {}
        with open(path + ".meta", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    meta[k] = v
        if meta.get("n") != str(n) or meta.get("seed") != str(seed):
            return False
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                assert isinstance(r.get("text"), str) and r["text"].strip()
                count += 1
        return count == n
    except Exception:
        return False


def write_meta(path: str, n: int, seed: int):
    content = (
        f"n={n}\nseed={seed}\nbase_model={BASE_MODEL}\n"
        f"sources={';'.join(s[0] for s in SOURCES)}\n"
    )
    tmp = tmp_path(path + ".meta")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    commit_file(tmp, path + ".meta")


def main(n: int, seed: int, use_template: bool):
    out = f"{SHARED}/datasets/raon-calibration-{n}.jsonl"
    if os.environ.get("FORCE", "0") != "1" and use_template and cache_valid(out, n, seed):
        print(f"reuse valid-cache -> {out} (n={n} seed={seed})")
        return
    from datasets import load_dataset

    tok = None
    if use_template:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    per_src = max(1, (n + len(SOURCES) - 1) // len(SOURCES))
    pool: list[str] = []
    for ds_id, split, want in SOURCES:
        take = min(want, per_src) if n <= 32 else want
        try:
            ds = load_dataset(ds_id, split=split, streaming=True)
        except Exception as e:  # noqa: BLE001 — split 개명 등에도 다음 소스로 계속
            print(f"SKIP source {ds_id}:{split} ({e})")
            continue
        got = 0
        for rec in ds:
            msgs = to_messages(dict(rec))
            if not msgs:
                continue
            if tok is not None:
                try:
                    text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=True)
                except TypeError:
                    text = tok.apply_chat_template(msgs, tokenize=False)
            else:
                text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
            pool.append(text)
            got += 1
            if got >= take or len(pool) >= n:
                break
            if len(pool) >= n:
                break
        if len(pool) >= n:
            break
    if len(pool) < n:
        raise SystemExit(f"collected {len(pool)} < n={n}")
    rng = random.Random(seed)
    idx = list(range(len(pool)))
    rng.shuffle(idx)
    out = f"{SHARED}/datasets/raon-calibration-{n}.jsonl"
    import json

    tmp = tmp_path(out)
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"text": pool[i]}, ensure_ascii=False) + "\n" for i in idx[:n])
    commit_file(tmp, out)
    write_meta(out, n, seed)
    print(f"seed={seed} n={n} template={use_template} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-template", action="store_true")
    a = ap.parse_args()
    main(a.n, a.seed, not a.no_template)
