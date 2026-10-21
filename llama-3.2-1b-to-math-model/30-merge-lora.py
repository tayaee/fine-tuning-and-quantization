"""30-merge-lora.py — Stage 3c. QLoRA 어댑터 병합 (31 직후, 5* PTQ·7* 평가 전 필수).
원본 tried: base 로드 → 특수토큰(<mask> 등) 동일 추가 → resize_token_embeddings
→ merge_and_unload → 저장. resize 생략 시 size-mismatch 에러.
base는 학습 때와 동일한 모델(unsloth 미러/smoke, meta-llama/full)을 쓴다.

  uv run 30-merge-lora.py --mode mini|full [--base <model-id>]
"""

import argparse
import os

from io_common import clean_tmp, commit_dir, tmp_path

SHARED = os.environ.get("M1B_SHARED", os.environ.get("P5_SHARED", "/rosenas/data/AIML/math-1b-shared"))
BASE = os.environ.get("BASE_MODEL", "unsloth/Llama-3.2-1B")
MASK_TOKEN = "<mask>"
PAD_TOKEN = "[PAD]"


def main(mode: str, base: str, strat: str, revision: str = ""):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from transformers import BloomPreTrainedModel  # noqa
    except ImportError:  # peft<=0.19 + transformers 5.x 호환 shim (10번과 동일)
        import transformers as _tf

        class _BloomStub:
            pass

        _tf.BloomPreTrainedModel = _BloomStub
    # Compat shim 2: peft의 AWQ 디스패처가 gptqmodel 구 심볼
    # (AwqGEMMQuantLinear)을 임포트한다. gptqmodel 7.5는 AwqGEMMLinear로
    # 개명됨. isinstance 체크용이므로 alias로 충분 (fp 레이어는 매칭 안 됨).
    try:
        from gptqmodel.nn_modules.qlinear import gemm_awq as _ga

        if not hasattr(_ga, "AwqGEMMQuantLinear") and hasattr(_ga, "AwqGEMMLinear"):
            _ga.AwqGEMMQuantLinear = _ga.AwqGEMMLinear
    except ImportError:
        pass
    from peft import PeftModel

    adapter = f"{SHARED}/models/synthetic-qlora-{mode}-{strat}"
    out = f"{SHARED}/models/synthetic-qlora-{mode}-{strat}-merged"
    tok = AutoTokenizer.from_pretrained(adapter)
    n0 = len(tok)
    added = []
    if tok.pad_token is None:
        added.append(PAD_TOKEN)
    if MASK_TOKEN not in tok.get_vocab():
        added.append(MASK_TOKEN)
    if added:
        tok.add_special_tokens({"additional_special_tokens": added})
    if tok.pad_token is None:
        tok.pad_token = PAD_TOKEN
    print(f"[{mode}] vocab {n0} -> {len(tok)}")

    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, revision=revision or None
    )
    model.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    tmp = clean_tmp(tmp_path(out))
    model.save_pretrained(tmp)
    tok.save_pretrained(tmp)
    commit_dir(tmp, out)
    print(f"[{mode}] merged -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument(
        "--strat",
        default=os.environ.get("STRAT", "single"),
        choices=["single", "ddp", "fsdp"],
    )
    ap.add_argument("--base", default=BASE)
    ap.add_argument(
        "--revision",
        default=os.environ.get("BASE_REVISION", ""),
        help="HF 리비전 고정 (기본 $BASE_REVISION, 1년 후 재현용).",
    )
    a = ap.parse_args()
    main(a.mode, a.base, a.strat, a.revision)
