"""32-quant-fft-awq.py — Stage 4a. FFT → AWQ 4bit (llmcompressor oneshot + AWQModifier).
원본: quantizaton_math.ipynb 후반 (autoawq) → llmcompressor로 교체.
입력: synthetic-fft-<mode>-<strat>. 출력에 tokenizer 동봉 필수.
캘리브레이션: gsm8k-calibration-256.jsonl (12-prep-calibration.sh 생성,
gsm8k-train에서 seed 고정 추출). AWQ 예제 정석 256.

  uv run 32-quant-fft-awq.py --mode mini|full [--calib N] [--calib-file PATH]
"""

import argparse
import os

from io_common import clean_tmp, commit_dir, tmp_path
from quant_common import load_calib_texts

SHARED = os.environ.get("M1B_SHARED", os.environ.get("P5_SHARED", "/rosenas/data/AIML/math-1b-shared"))


def main(mode: str, strat: str, calib: int, calib_file: str | None):
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier
    from transformers import AutoTokenizer

    src = f"{SHARED}/models/synthetic-fft-{mode}-{strat}"
    out = f"{SHARED}/models/synthetic-fft-{mode}-{strat}-awq"
    texts = load_calib_texts(calib, calib_file)
    ds = Dataset.from_list([{"text": t} for t in texts])
    recipe = AWQModifier(ignore=["lm_head"], scheme="W4A16", targets=["Linear"])
    tmp = clean_tmp(tmp_path(out))
    oneshot(
        model=src,
        dataset=ds,
        recipe=recipe,
        output_dir=tmp,
        max_seq_length=1024,
        num_calibration_samples=calib,
    )
    AutoTokenizer.from_pretrained(src).save_pretrained(tmp)
    commit_dir(tmp, out)
    print(f"[{mode}] calib={calib} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument(
        "--strat",
        default=os.environ.get("STRAT", "single"),
        choices=["single", "ddp", "fsdp"],
    )
    ap.add_argument("--calib", type=int, default=None)
    ap.add_argument(
        "--calib-file",
        default=None,
        help="캘리브레이션 jsonl (기본 gsm8k-calibration-256.jsonl, "
        "CALIB_FILE로 오버라이드 가능)",
    )
    a = ap.parse_args()
    default_calib = int(os.environ.get("CALIB_N", "32" if a.mode == "mini" else "256"))
    main(a.mode, a.strat, a.calib or default_calib, a.calib_file)
