"""32-quant-awq-w5.py — 원본 → (AWQ smoothing) + GPTQ W5A16-g128 (llmcompressor oneshot).
비표준 5bit: AWQ smoothing 후 GPTQ W5로 양자화.
custom arch에서 AWQ replay가 RoutedLinear와 호환되지 않아 기본 OFF
(RQ_AWQ_FIRST=1로 명시해야 시도, 실패 시 pure-GPTQ 폴백).

  uv run 32-quant-awq-w5.py --mode smoke|full [--calib N] [--calib-file PATH]
"""

import argparse
import os

from io_common import clean_tmp, commit_dir, tmp_path
from quant_common import (
    DEFAULT_IGNORE,
    allow_trusted_remote_code,
    build_awq_smoothing,
    load_calib_texts,
    resolve_targets,
    save_custom_code,
)

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")
BASE_MODEL = os.environ.get("BASE_MODEL", "KRAFTON/Raon-1-mini-preview")
BASE_REVISION = os.environ.get("BASE_REVISION") or "main"
SRC = os.environ.get("RQ_SRC", f"{SHARED}/models/raon-bf16")
TARGET = "awq-w5"


def build_recipes():
    from llmcompressor.modifiers.gptq import GPTQModifier

    def build_gptq5():
        # 매 시도 fresh 인스턴스 (실패한 modifier 재사용 시 init 오류)
        return GPTQModifier(
            targets=resolve_targets(),
            ignore=DEFAULT_IGNORE,
            config_groups={"group_0": {"targets": resolve_targets(), "weights": {"num_bits": 5, "type": "int", "symmetric": False, "strategy": "group", "group_size": 64}}},
        )

    # custom arch: AWQ replay가 RoutedLinear와 호환되지 않아 결정적 실패하므로
    # 기본 OFF (RQ_AWQ_FIRST=1로 명시해야 시도, 실패 시 pure 폴백).
    # OFF = 폴백 결과와 동일, doomed 시도 시간(~7분/모델) + 2중 모델 메모리 방지.
    if os.environ.get("RQ_AWQ_FIRST", "0") == "1":
        return [lambda: [build_awq_smoothing(), build_gptq5()], build_gptq5]
    return [build_gptq5]


def main(mode: str, calib: int, calib_file: str | None, seq_len: int):
    from datasets import Dataset
    from llmcompressor import oneshot
    from transformers import AutoTokenizer

    out = f"{SHARED}/models/raon-{TARGET}-{mode}"
    allow_trusted_remote_code()
    texts = load_calib_texts(calib, calib_file)
    ds = Dataset.from_list([{"text": t} for t in texts])
    recipes = build_recipes()
    last_err = None
    for build in recipes:
        recipe = build()  # 매 시도 fresh 인스턴스 (실패작의 모델 참조를 붙잡지 않음)
        tmp = clean_tmp(tmp_path(out))
        try:
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
            commit_dir(tmp, out)
            print(f"[{mode}] {TARGET} calib={calib} seq={seq_len} -> {out}")
            return
        except Exception as e:  # noqa: BLE001 — 폴백 체인
            import gc
            import traceback

            traceback.print_exc()
            # str만 보관: exception 객체는 traceback→프레임→58GB 모델을 붙잡아
            # 다음 시도의 모델과 2중 적재(OOM, rc=137)된다.
            last_err = f"{type(e).__name__}: {e}"
            print(f"recipe={[type(r).__name__ for r in recipe]} failed: {last_err}")
            del recipe
            gc.collect()
    raise SystemExit(f"all recipes failed, last: {last_err}")


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
