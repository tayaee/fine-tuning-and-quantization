"""38-quant-fft-gptq.py — Stage 4d. FFT → GPTQ 4bit-g128 (gptqmodel 7.x).
원본: quantizaton_math.ipynb 전반 (auto-gptq) → GPTQModel.load/quantize/save.
캘리브: gsm8k-calibration-256.jsonl (12-prep-calibration.sh 생성), 학습 프롬프트 템플릿 적용.
입력: synthetic-fft-<mode>-<strat>.

  uv run 38-quant-fft-gptq.py --mode mini|full [--calib N] [--calib-file PATH]
"""

import argparse
import os

from io_common import clean_tmp, commit_dir, tmp_path
from quant_common import load_calib_texts

SHARED = os.environ.get("M1B_SHARED", os.environ.get("P5_SHARED", "/rosenas/data/AIML/math-1b-shared"))


def main(mode: str, strat: str, calib: int, calib_file: str | None):
    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer

    src = f"{SHARED}/models/synthetic-fft-{mode}-{strat}"
    out = f"{SHARED}/models/synthetic-fft-{mode}-{strat}-gptq"
    texts = load_calib_texts(calib, calib_file)
    tok = AutoTokenizer.from_pretrained(src)
    qc = QuantizeConfig(bits=4, group_size=128)
    model = GPTQModel.load(src, qc)
    model.quantize(calibration=texts[:calib], tokenizer=tok)
    tmp = clean_tmp(tmp_path(out))
    model.save(tmp)
    tok.save_pretrained(tmp)
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
