"""36-quant-nvfp4-gptq.py — 원본 → GPTQ NVFP4 (llmcompressor GPTQModifier).
NVFP4 스킬 정석: actorder=static, dampening 0.01, block 128, MoE는 calib 확보가 중요.
full 권장 calib>=256 (기본값 따름). smoke는 정확도보다 파이프 검증 목적.

  uv run 36-quant-nvfp4-gptq.py --mode smoke|full [--calib N] [--calib-file PATH]
"""

import argparse
import os

from io_common import clean_tmp, commit_dir, tmp_path
from quant_common import (
    DEFAULT_IGNORE,
    NVFP4_PACK_QUANTIZED,
    allow_trusted_remote_code,
    load_calib_texts,
    resolve_targets,
    save_custom_code,
    verify_compressed_output,
)

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")
BASE_MODEL = os.environ.get("BASE_MODEL", "KRAFTON/Raon-1-mini-preview")
BASE_REVISION = os.environ.get("BASE_REVISION") or "main"
SRC = os.environ.get("RQ_SRC", f"{SHARED}/models/raon-bf16")
TARGET = "nvfp4-gptq"


def main(mode: str, calib: int, calib_file: str | None, seq_len: int):
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from transformers import AutoTokenizer

    out = f"{SHARED}/models/raon-{TARGET}-{mode}"
    allow_trusted_remote_code()
    texts = load_calib_texts(calib, calib_file)
    ds = Dataset.from_list([{"text": t} for t in texts])
    recipe = GPTQModifier(
        targets=resolve_targets(),
        scheme="NVFP4",
        ignore=DEFAULT_IGNORE,
        actorder="static",
        dampening_frac=0.01,
        block_size=128,
    )
    tmp = clean_tmp(tmp_path(out))
    oneshot(
        model=SRC,
        trust_remote_code_model=True,
        model_revision=BASE_REVISION,
        dataset=ds,
        recipe=recipe,
        output_dir=tmp,
        max_seq_length=seq_len,
        num_calibration_samples=calib,
    )
    AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, revision=BASE_REVISION).save_pretrained(tmp)
    save_custom_code(tmp, SRC)
    # compressor가 can_compress=False로 dense 폴백하면 미패킹 BF16가 그대로 커밋된다
    # → commit_dir 전에 포맷 계약 검증으로 조기 실패시킨다.
    verify_compressed_output(tmp, NVFP4_PACK_QUANTIZED)
    commit_dir(tmp, out)
    print(f"[{mode}] {TARGET} calib={calib} seq={seq_len} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--calib", type=int, default=None)
    ap.add_argument("--calib-file", default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    a = ap.parse_args()
    d_calib = int(os.environ.get("CALIB_N", "8" if a.mode == "smoke" else "256"))
    d_seq = int(os.environ.get("SEQ_LEN", "512" if a.mode == "smoke" else "2048"))
    main(a.mode, a.calib or d_calib, a.calib_file, a.seq_len or d_seq)
