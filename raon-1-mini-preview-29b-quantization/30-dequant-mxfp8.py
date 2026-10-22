"""30-dequant-mxfp8.py — MXFP8 베이스 → BF16 복원 (compressed_tensors convert_checkpoint).
llmcompressor oneshot은 mxfp8 기양자화 모델을 거부하므로, PTQ 전에 full-precision으로
되돌린다. 가중치 파일 단위 스트리밍 변환 (모델 전체 로드 불필요).
타깃은 expert linear만 (router/attn/mlp/lm_head는 BF16 그대로라 스케일 없음).
출력: models/raon-bf16 (mode 독립, 1회). config의 quantization_config는 제거한다.

  uv run 30-dequant-mxfp8.py
"""

import json
import os
import struct
from collections import Counter

from io_common import clean_tmp, commit_dir, tmp_path

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")
BASE_MODEL = os.environ.get("BASE_MODEL", "KRAFTON/Raon-1-mini-preview")
BASE_REVISION = os.environ.get("BASE_REVISION") or "main"
OUT = f"{SHARED}/models/raon-bf16"


EXPERT_TARGETS = ["re:.*\\.moe\\.experts\\.[0-9]+\\.(gate_proj|up_proj|down_proj)"]


def _apply_tracing_shim(tmp: str) -> None:
    """llmcompressor SequentialPipeline(fx trace) 호환 shim.
    trace 과정에서 ModelOutput이 plain dict로 바뀌어 CausalLM forward의
    `outputs.last_hidden_state` attribute 접근이 실패한다
    ('dict' object has no attribute 'last_hidden_state', 26/26 블록).
    Llama류는 outputs[0] getitem이라 무사. getitem은 런타임 동일 동작
    (ModelOutput도 getitem 지원)이라 추론·PPL·서빙에 영향 없음.
    """
    subs = [
        ("hidden = outputs.last_hidden_state", 'hidden = outputs["last_hidden_state"]'),
        ("past_key_values=outputs.past_key_values", 'past_key_values=outputs.get("past_key_values")'),
        ("hidden_states=outputs.hidden_states", 'hidden_states=outputs.get("hidden_states")'),
        # 이미 구-shim(getitem) 적용본 마이그레이션
        ('past_key_values=outputs["past_key_values"]', 'past_key_values=outputs.get("past_key_values")'),
        ('hidden_states=outputs["hidden_states"]', 'hidden_states=outputs.get("hidden_states")'),
    ]
    p = f"{tmp}/modeling_raon1_mini.py"
    with open(p, encoding="utf-8") as f:
        src = f.read()
    n = 0
    for old, new in subs:
        if old in src:
            src = src.replace(old, new)
            n += 1
    if n:
        with open(p, "w", encoding="utf-8") as f:
            f.write(src)
    print(f"tracing_shim applied={n}/{len(subs)}")


def main():
    import torch
    from compressed_tensors.entrypoints.convert import convert_checkpoint
    from compressed_tensors.entrypoints.convert.converters import FP8BlockDequantizer
    from huggingface_hub import snapshot_download

    snap = snapshot_download(
        repo_id=BASE_MODEL,
        revision=BASE_REVISION,
        allow_patterns=["*.safetensors", "*.json", "*.py", "*.jinja", "*.md"],
    )
    print(f"src_snapshot={snap}")
    tmp = clean_tmp(tmp_path(OUT))
    convert_checkpoint(
        model_stub=snap,
        save_directory=tmp,
        converter=FP8BlockDequantizer(
            targets=EXPERT_TARGETS,
            weight_block_size=(1, 32),
            dtype=torch.bfloat16,
        ),
    )
    # quantization_config 제거 확인 (남아 있으면 일반 BF16 로드가 mxfp8 경로로 빠짐)
    cfg_path = f"{tmp}/config.json"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    removed = cfg.pop("quantization_config", None)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"quantization_config removed={removed is not None}")
    _apply_tracing_shim(tmp)
    # 산출물 dtype 검증: F8/U8 잔류 금지
    import glob

    counts: Counter = Counter()
    n_tensors = 0
    for shard in sorted(glob.glob(f"{tmp}/*.safetensors")):
        with open(shard, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            counts[v["dtype"]] += 1
            n_tensors += 1
    print(f"tensors={n_tensors} dtype_counts={dict(counts)}")
    # U8/F8 잔류 = dequant 누락 (스케일이 남으면 안 됨)
    bad = [d for d in counts if d not in ("BF16", "F32", "I64", "BOOL")]
    if bad:
        raise SystemExit(f"dequant verification failed: {dict(counts)}")
    commit_dir(tmp, OUT)
    print(f"-> {OUT}")


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description="MXFP8 -> BF16 restore").parse_args()
    main()
