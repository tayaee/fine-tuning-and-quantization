"""config/nvfp4_to_bf16.py — modelopt_fp4(NVFP4) 체크포인트를 BF16 으로 변환한다.

왜 필요한가
------------
`nvfp4-awq-vllm` 는 ModelOpt 의 `modelopt_fp4` 포맷이다. expert 가중치가
NVFP4(E2M1, group 16) 로 패킹되어 있고 transformers 에는 이 포맷의 양자화기가
없다 (`Unknown quantization type, got modelopt_fp4`). 그래서 그대로
`AutoModelForCausalLM.from_pretrained` 를 하면 패킹된 U8 가중치가 raw 로 적재되어
shape mismatch 로 죽는다.

이 스크립트는 그 패킹을 BF16 으로 풀어 **일반 transformers 가 읽을 수 있는
체크포인트** 로 다시 저장한다. vLLM 포크로 서빙할 계획이면 이 변환은 필요 없다
(포크가 modelopt_fp4 로더를 구현한다) — transformers 로 써야 할 때만 쓴다.

수치 규칙 (compressed-tensors 의 NVFP4 와 동일 = vLLM 포크의 modelopt 로더와 동일):
    값 = e2m1(니블) * (블록스케일_F8_E4M3 * 전역스케일_F32)
    바이트의 low 니블 = 짝수 번째 원소, high 니블 = 홀수 번째 원소.

사용:
    python config/nvfp4_to_bf16.py <repo-or-dir> --out <dir>
    python config/nvfp4_to_bf16.py tayaee/Raon-1-mini-preview-nvfp4-awq-vllm --out ./bf16
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

#: NVFP4 E2M1 양수 테이블 (compressed-tensors / vLLM 포크와 동일).
E2M1_TO_FLOAT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

#: 그대로 복사할 파일 (토크나이저·커스텀 코드 등).
COPY_SUFFIXES = (".json", ".jinja", ".py", ".txt", ".model")


def dequantize_nvfp4(
    packed, scale, global_scale, dtype, group_size: int = 16
):
    """패킹된 NVFP4 가중치 + 스케일 -> BF16.

    packed:        U8 [out, in/2]      (바이트당 원소 2개)
    scale:         F8_E4M3 [out, in/group_size]
    global_scale:  F32 스칼라 (전역 스케일) 또는 None
    """
    import torch

    out_features, in_half = packed.shape
    in_features = in_half * 2
    n_groups = scale.shape[1]
    if in_features % n_groups:
        raise ValueError(f"in_features({in_features}) 가 그룹 수({n_groups})로 나누어지지 않음")
    group_size = in_features // n_groups

    table = torch.tensor(E2M1_TO_FLOAT, dtype=torch.float32, device=packed.device)
    low = (packed & 0x0F).to(torch.long)
    high = ((packed >> 4) & 0x0F).to(torch.long)
    lo_val = table[low & 0x07] * torch.where((low & 0x08) != 0, -1.0, 1.0)
    hi_val = table[high & 0x07] * torch.where((high & 0x08) != 0, -1.0, 1.0)

    s = scale.to(torch.float32)
    if global_scale is not None:
        s = s * global_scale.to(torch.float32).reshape(1, 1)
    # 블록 스케일 1개가 group_size(=16) 개 원소를 덮는다 → 바이트로는 group_size/2 개.
    s = s.repeat_interleave(group_size // 2, dim=1)
    if s.shape[1] != in_half:
        raise ValueError(f"스케일 반복 후 열 수 불일치: {s.shape[1]} != {in_half}")

    out = torch.empty((out_features, in_features), dtype=torch.float32, device=packed.device)
    out[:, 0::2] = lo_val * s
    out[:, 1::2] = hi_val * s
    return out.to(dtype)


def _is_nvfp4_packed(name: str, tensor) -> bool:
    """패킹된 NVFP4 expert 가중치인가 (U8 + .weight_scale 동봉)."""
    import torch

    return name.endswith(".weight") and tensor.dtype == torch.uint8


def convert(
    src: str | Path,
    out_dir: str | Path,
    dtype: str = "bfloat16",
    max_shards: int | None = None,
) -> dict:
    """modelopt_fp4 체크포인트를 BF16 체크포인트로 변환한다."""
    import torch
    from safetensors.torch import load_file, save_file

    src = Path(src)
    if not src.is_dir():
        from huggingface_hub import snapshot_download

        src = Path(
            snapshot_download(
                repo_id=str(src),
                allow_patterns=["*.json", "*.jinja", "*.py", "*.txt", "*.model", "*.safetensors"],
            )
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_dtype = getattr(torch, dtype)

    cfg_path = src / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    qc = cfg.get("quantization_config") or {}
    if qc.get("quant_method") != "modelopt_fp4":
        raise SystemExit(
            f"quant_method={qc.get('quant_method')!r} — modelopt_fp4 가 아니다 (변환 불필요)"
        )

    index_path = src / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = [p.name for p in sorted(src.glob("*.safetensors"))]
    if max_shards:
        shards = shards[:max_shards]

    # 설정/토크나이저/커스텀 코드 복사 (weights 는 제외)
    for p in src.iterdir():
        if p.is_file() and p.suffix in COPY_SUFFIXES and p.name != "model.safetensors.index.json":
            shutil.copy2(p, out_dir / p.name)

    # config: modelopt_fp4 표지를 떼어 일반 BF16 체크포인트로 만든다.
    cfg.pop("quantization_config", None)
    cfg["dtype"] = dtype
    (out_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    new_map: dict[str, str] = {}
    n_deq = n_kept = 0
    for shard in shards:
        tensors = load_file(str(src / shard))
        out_tensors: dict[str, torch.Tensor] = {}
        consumed: set[str] = set()
        for name, tensor in tensors.items():
            if _is_nvfp4_packed(name, tensor) and (name[: -len(".weight")] + ".weight_scale") in tensors:
                base = name[: -len(".weight")]
                out_tensors[name] = dequantize_nvfp4(
                    tensors[name],
                    tensors[base + ".weight_scale"],
                    tensors.get(base + ".weight_scale_2"),
                    target_dtype,
                )
                consumed.update({base + ".weight_scale", base + ".weight_scale_2", base + ".input_scale"})
                n_deq += 1
            elif name in consumed:
                continue
            else:
                out_tensors[name] = tensor
                n_kept += 1
        out_name = shard
        save_file(out_tensors, str(out_dir / out_name), metadata={"format": "pt"})
        for name in out_tensors:
            new_map[name] = out_name
        del tensors, out_tensors
        print(f"[{shard}] dequantized={n_deq} kept={n_kept}", flush=True)

    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": new_map}, indent=2) + "\n",
        encoding="utf-8",
    )
    total = sum(p.stat().st_size for p in out_dir.glob("*.safetensors"))
    print(f"변환 완료: {out_dir}  ({total / 1e9:.1f} GB, expert 가중치 {n_deq}개 역양자화)")
    return {"out_dir": str(out_dir), "dequantized": n_deq, "kept": n_kept, "bytes": total}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="Hub repo id 또는 로컬 체크포인트 디렉터리")
    ap.add_argument("--out", required=True, help="출력 디렉터리")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-shards", type=int, default=None, help="디버깅: 앞 N개 샤드만")
    a = ap.parse_args(argv[1:])
    convert(a.src, a.out, a.dtype, a.max_shards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
