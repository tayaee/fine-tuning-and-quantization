"""s2-01-dequant-mxfp8-into-bf16.py — MXFP8 베이스 → BF16 복원 (compressed_tensors convert_checkpoint).
llmcompressor oneshot은 mxfp8 기양자화 모델을 거부하므로, PTQ 전에 full-precision으로
되돌린다. 가중치 파일 단위 스트리밍 변환 (모델 전체 로드 불필요).
타깃은 expert linear만 (router/attn/mlp/lm_head는 BF16 그대로라 스케일 없음).
출력: models/raon-bf16 (mode 독립, 1회). config의 quantization_config는 제거한다.

  uv run s2-01-dequant-mxfp8-into-bf16.py
"""

import json
import os
import struct
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from io_common import clean_tmp, commit_dir, tmp_path

if TYPE_CHECKING:
    import torch

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")
BASE_MODEL = os.environ.get("BASE_MODEL", "KRAFTON/Raon-1-mini-preview")
BASE_REVISION = os.environ.get("BASE_REVISION") or "main"
OUT = f"{SHARED}/models/raon-bf16"


EXPERT_TARGETS = ["re:.*\\.moe\\.experts\\.[0-9]+\\.(gate_proj|up_proj|down_proj)"]
WEIGHT_BLOCK_SIZE = (1, 32)

# _routed_projection이 커스텀 nn.Linear 서브클래스를 반환하면 downstream이 다 죽는다:
#  - compressed-tensors 0.19 컴프레서는 `module_type in COMPRESSIBLE_MODULE_TYPES`의
#    정확 타입 멤버십을 검사하므로 can_compress=False → DenseCompressor 폴백 →
#    format:"dense" + 59GB BF16 체크포인트가 정상 산출물로 커밋된다.
#  - AWQModifier 스무딩이 hidden=... 키워드를 넘겨 TypeError.
# 가중치는 이미 BF16으로 복원했으므로 표준 nn.Linear로 바꿔도 수식·키·동작은 동일하다.
_ROUTED_PROJECTION_SUB = (
    "return Raon1MiniRoutedLinear(in_features, out_features, bias=False)",
    "return nn.Linear(in_features, out_features, bias=False)",
)
ROUTED_LINEAR_MARKER = _ROUTED_PROJECTION_SUB[1]

# llmcompressor의 moe_calibration_context(get_calibrate_all_experts_flag=True)는
# 내장 LinearExperts2D만 감싸고 trust_remote_code Raon1MiniMoE는 알지 못한다.
# hit>0 expert만 호출하면 미라우팅 expert의 input_global_scale이 미초기화(0/쓰레기값)로
# 남아 추론 시 NaN이 발생하므로, 캘리브레이션 중(hooks 활성 시) 전 expert에 flat을 통과시킨다.
_MOE_CALIB_SUB = (
    """        for expert_idx in hit.nonzero().flatten().tolist():
            rows, slot = torch.where(top == expert_idx)
            expert_out = self.experts[expert_idx](flat.index_select(0, rows))
            out.index_add_(
                0, rows, expert_out.float() * probs[rows, slot].unsqueeze(-1)
            )""",
    """        _cal_mod = __import__("sys").modules.get("llmcompressor.modeling.moe.context")
        _hk_mod = __import__("sys").modules.get("llmcompressor.modifiers.utils.hooks")
        _cal_all = bool(
            flat.shape[0]
            and _cal_mod is not None
            and getattr(_cal_mod, "get_calibrate_all_experts_flag", lambda: False)()
            and not getattr(getattr(_hk_mod, "HooksMixin", None), "_HOOKS_DISABLED", False)
        )
        for expert_idx in hit.nonzero().flatten().tolist():
            rows, slot = torch.where(top == expert_idx)
            expert_out = (
                self.experts[expert_idx](flat).index_select(0, rows)
                if _cal_all
                else self.experts[expert_idx](flat.index_select(0, rows))
            )
            out.index_add_(
                0, rows, expert_out.float() * probs[rows, slot].unsqueeze(-1)
            )
        if _cal_all:
            for expert_idx in (hit == 0).nonzero().flatten().tolist():
                self.experts[expert_idx](flat)""",
)
MOE_CALIB_MARKER = "get_calibrate_all_experts_flag"


def _apply_tracing_shim(model_dir) -> dict:
    """output 디렉터리의 modeling_raon1_mini.py에 런타임 shim을 영구 적용.

    1) llmcompressor SequentialPipeline(fx trace) 호환: trace 중 ModelOutput이 plain
       dict로 바뀌어 CausalLM forward의 `outputs.last_hidden_state` attribute 접근이
       실패한다('dict' object has no attribute 'last_hidden_state', 26/26 블록).
       Llama류는 outputs[0] getitem이라 무사. getitem은 런타임 동일 동작
       (ModelOutput도 getitem 지원)이라 추론·PPL·서빙에 영향 없음.
    2) _routed_projection → nn.Linear (압축 패킹/AWQ 스무딩 호환, 위 상수 설명).
    3) Raon1MiniMoE.forward → moe_calibration_context 활성 시 전 expert 캘리브레이션 보장.

    적용 결과를 dict로 반환(치환 개수). save_custom_code로 모든 하위 양자화
    모델에도 전파되므로 여기서 영구 보존되어야 한다.
    """
    subs = [
        ("hidden = outputs.last_hidden_state", 'hidden = outputs["last_hidden_state"]'),
        ("past_key_values=outputs.past_key_values", 'past_key_values=outputs.get("past_key_values")'),
        ("hidden_states=outputs.hidden_states", 'hidden_states=outputs.get("hidden_states")'),
        # 이미 구-shim(getitem) 적용본 마이그레이션
        ('past_key_values=outputs["past_key_values"]', 'past_key_values=outputs.get("past_key_values")'),
        ('hidden_states=outputs["hidden_states"]', 'hidden_states=outputs.get("hidden_states")'),
        _ROUTED_PROJECTION_SUB,
        _MOE_CALIB_SUB,
    ]
    p = Path(model_dir) / "modeling_raon1_mini.py"
    if not p.is_file():
        raise RuntimeError(f"{p}: modeling_raon1_mini.py 없음 — custom code 복사 실패")
    src = p.read_text(encoding="utf-8")
    n = 0
    for old, new in subs:
        if old in src:
            src = src.replace(old, new)
            n += 1
    if n:
        p.write_text(src, encoding="utf-8")
    print(f"tracing_shim applied={n}/{len(subs)} file={p}")
    if ROUTED_LINEAR_MARKER not in src:
        raise RuntimeError(
            f"{p}: _routed_projection shim 미적용 ('{ROUTED_LINEAR_MARKER}' 없음)"
            " — RoutedLinear 서브클래스가 남아 압축 패킹이 dense로 폴백한다"
        )
    return {"applied": n, "total": len(subs)}


def _e8m0_scale_bytes(scale) -> "torch.Tensor":
    """E8M0 스케일을 raw 지수 바이트(uint8) 뷰로 변환.

    벤더 modeling_raon1_mini._scale_bytes와 동일 규칙: uint8면 그대로,
    float8_e8m0fnu로 저장돼 있으면 바이트 뷰로 재해석한다.
    """
    import torch

    return scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)


def _make_mxfp8_dequantizer_cls():
    """MXFP8BlockDequantizer 팩토리 — compressed_tensors를 지연 import한다.

    모듈 최상위 import를 피해야 s2-01-dequant-mxfp8-into-bf16.py를 --help/테스트로 빠르게
    로드할 수 있다(compressed_tensors가 무거운 패키지).
    """
    import torch
    from compressed_tensors.entrypoints.convert.converters import FP8BlockDequantizer
    from compressed_tensors.quantization.utils.helpers import (
        maybe_pad_tensor_for_block_quant,
    )

    class MXFP8BlockDequantizer(FP8BlockDequantizer):
        """MXFP8(E4M3 + E8M0) 블록 디퀀타이저.

        base 구현은 `weight * weight_scale_inv`를 그대로 곱하지만, Raon 체크포인트의
        weight_scale_inv는 IEEE754 E8M0 **지수 바이트**(uint8)라 의미가 2^(byte-127)이다.
        base 그대로면 2^-12여야 할 스케일이 115.0으로 곱해져 expert 가중치가
        ~4.7e5배 폭주한다. 아래는 벤더 dequantize_mxfp8와 수치적으로 동일한 경로다.
        """

        def _create_dequantized_weight(self, weight, scale_inv, dtype=None):  # pyright: ignore[reportIncompatibleMethodOverride] — 이슈가 요구한 scale_inv 이름 유지(base는 weight_scale_inv라 부를 뿐, 실제 호출은 전부 위치 인자)
            """F8_E4M3 [N,K] + E8M0 바이트 [N,K/32] → value * 2**(byte-127) in dtype."""
            dtype = dtype or self.dtype
            original_shape = weight.shape
            # base의 weight_block_size 타입이 tuple[int]로 선언돼 있어 2-튜플로 좁힌다
            block_height, block_width = (int(v) for v in self.weight_block_size)
            block = (block_height, block_width)

            # 블록 크기 배수가 아니면 패딩 → 디퀀타 → 원래 모양으로 복원
            weight = maybe_pad_tensor_for_block_quant(weight, block)
            padded_shape = weight.shape
            rows_blocks = padded_shape[0] // block_height
            cols_blocks = padded_shape[1] // block_width

            weight_blocks = weight.reshape(rows_blocks, block_height, cols_blocks, block_width).transpose(1, 2)
            # E8M0 바이트 → 실제 스케일. float32로 올려 exp2가 정확히 계산되게 한다.
            true_scale = torch.exp2(_e8m0_scale_bytes(scale_inv).to(torch.float32) - 127.0)
            factor = true_scale.unsqueeze(-1).unsqueeze(-1)  # 브로드캐스팅용 (…,1,1)

            dequantized = (weight_blocks.to(torch.float32) * factor).to(dtype)
            dequantized = dequantized.transpose(1, 2).reshape(padded_shape)
            if original_shape != padded_shape:
                dequantized = dequantized[tuple([slice(v) for v in original_shape])]
            return dequantized

    return MXFP8BlockDequantizer


class _LazyMXFP8Dequantizer:
    """`MXFP8BlockDequantizer` 호출/속성 접근 시에만 실제 클래스를 만든다(지연 import)."""

    _cls = None

    def __call__(self, *args, **kwargs):
        cls = self._resolve()
        return cls(*args, **kwargs)

    @staticmethod
    def _resolve():
        if _LazyMXFP8Dequantizer._cls is None:
            _LazyMXFP8Dequantizer._cls = _make_mxfp8_dequantizer_cls()
        return _LazyMXFP8Dequantizer._cls

    def __getattr__(self, name):
        return getattr(self._resolve(), name)


MXFP8BlockDequantizer = _LazyMXFP8Dequantizer()


def _shard_map(src_dir: Path) -> dict:
    """원본 체크포인트의 텐서명 → shard 파일명."""
    idx = src_dir / "model.safetensors.index.json"
    if not idx.is_file():
        raise RuntimeError(f"{idx}: safetensors index 없음 — 원본 MXFP8 체크포인트 아님")
    with idx.open(encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def _pick_sample_expert_key(src_dir: Path, weight_map: dict) -> str:
    """검증용 샘플 expert 가중치 키.

    이슈 본문은 `model.layers.0...`을 예시했지만 이 체크포인트는 0층이 dense
    (`first_k_dense_replace=1`)라 MoE expert는 1층부터다. 인덱스에서 실제 expert
    weight를 찾아 첫 것을 고르고, 없으면 이슈 본문 키로 폴백한다.
    """
    prefix = ".moe.experts."
    suffix = "_proj.weight"
    for name in sorted(weight_map):
        if prefix in name and name.endswith(suffix) and name + "_scale_inv" in weight_map:
            return name
    return "model.layers.1.moe.experts.0.gate_proj.weight"


def _read_header(shard: Path) -> dict:
    with shard.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def _reference_expert_weight(src_dir: Path, key: str) -> "torch.Tensor":
    """원본 (weight, weight_scale_inv)에서 2**(byte-127) 기준값을 직접 계산."""
    import torch
    from safetensors import safe_open

    weight_map = _shard_map(src_dir)
    scale_key = key + "_scale_inv"
    for probe in (key, scale_key):
        if probe not in weight_map:
            raise RuntimeError(f"{probe}: 원본 체크포인트에 없음 — 검증 기준 계산 불가")
    with safe_open(src_dir / weight_map[key], framework="pt") as f:
        weight = f.get_tensor(key)
    with safe_open(src_dir / weight_map[scale_key], framework="pt") as f:
        scale = f.get_tensor(scale_key)

    rows, cols = weight.shape
    block_w = 32
    if cols % block_w:
        raise RuntimeError(f"{key}: cols={cols}가 블록(32) 배수 아님")
    blocks = weight.float().view(rows, cols // block_w, block_w)
    factor = torch.exp2(_e8m0_scale_bytes(scale).to(torch.float32) - 127.0).unsqueeze(-1)
    return (blocks * factor).view(rows, cols)


def verify_dequantized_bf16(tmp, src_dir, *, sample_key: str | None = None, atol=1e-3, rtol=1e-3):
    """commit_dir 전에 raon-bf16 산출물이 진짜 BF16 복원본인지 검증. 실패 시 RuntimeError.

    검사 항목(요구사항 3):
      1) F8_E4M3 dtype 텐서 / weight_scale_inv 키 잔존 금지
      2) 샘플 expert 가중치가 원본 (weight, weight_scale_inv)의 2**(byte-127) 값과
         allclose(atol, rtol), abs().max() < 10.0, NaN/Inf 0개
      3) modeling_raon1_mini.py에 _routed_projection → nn.Linear 치환이 있어야 함

    반환값: {"tensors", "dtypes", "max_abs", "sample"}
    """
    import torch
    from safetensors import safe_open

    tmp = Path(tmp)
    src_dir = Path(src_dir)
    if not tmp.is_dir():
        raise RuntimeError(f"{tmp}: 디렉터리 없음")

    counts: Counter = Counter()
    names: dict = {}
    shards = sorted(tmp.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"{tmp}: .safetensors 없음 — 디퀀타이즈 산출물 아님")
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118 — safetensors는 순회 API가 없다
                dtype = str(f.get_slice(name).get_dtype())
                counts[dtype] += 1
                names[name] = dtype

    residual_fp8 = sorted(n for n, d in names.items() if d in ("F8_E4M3", "F8_E4M3FN", "F8_E5M2"))
    if residual_fp8:
        raise RuntimeError(f"{tmp}: F8_E4M3 텐서 잔존 {len(residual_fp8)}개 (예: {residual_fp8[:3]}) — 디퀀타 누락")
    residual_scale = sorted(n for n in names if n.endswith("weight_scale_inv"))
    if residual_scale:
        raise RuntimeError(f"{tmp}: weight_scale_inv 잔존 {len(residual_scale)}개 (예: {residual_scale[:3]}) — 디퀀타 누락")

    key = sample_key or _pick_sample_expert_key(src_dir, _shard_map(src_dir))
    if key not in names:
        raise RuntimeError(f"{tmp}: 샘플 expert 텐서 {key} 없음 — 검증할 가중치가 비어 있음")
    shard_for_key = next(s for s in shards if key in _read_header(s))
    with safe_open(shard_for_key, framework="pt") as f:
        got = f.get_tensor(key).float()
    ref = _reference_expert_weight(src_dir, key)

    if got.shape != ref.shape:
        raise RuntimeError(f"{tmp}: {key} shape={tuple(got.shape)} != 원본 {tuple(ref.shape)}")
    finite = bool(torch.isfinite(got).all())
    max_abs = float(got.abs().max())
    if not finite:
        n_bad = int((~torch.isfinite(got)).sum())
        raise RuntimeError(f"{tmp}: {key} 에 NaN/Inf {n_bad}개 — 디퀀타이즈 결과 비정상")
    if max_abs >= 10.0:
        raise RuntimeError(f"{tmp}: {key} max_abs={max_abs:.4f} >= 10.0 — E8M0 스케일 미적용 폭주 의심")
    if not torch.allclose(got, ref, atol=atol, rtol=rtol):
        diff = float((got - ref).abs().max())
        raise RuntimeError(f"{tmp}: {key} 가 원본 2**(byte-127) 기준과 불일치 (max|diff|={diff:.6f})")

    modeling = tmp / "modeling_raon1_mini.py"
    if not modeling.is_file():
        raise RuntimeError(f"{tmp}: modeling_raon1_mini.py 없음 — trust_remote_code 재로드 불가")
    src_text = modeling.read_text(encoding="utf-8")
    if ROUTED_LINEAR_MARKER not in src_text:
        raise RuntimeError(f"{modeling}: '{ROUTED_LINEAR_MARKER}' 없음 — _routed_projection shim 미적용")

    info = {"tensors": sum(counts.values()), "dtypes": dict(counts), "max_abs": round(max_abs, 6), "sample": key}
    print(f"verify_dequantized_bf16 OK {tmp} {info}")
    return info


def main():
    import torch
    from compressed_tensors.entrypoints.convert import convert_checkpoint
    from huggingface_hub import snapshot_download

    from quant_common import patch_unified_memory_info

    patch_unified_memory_info()
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
        # FP8BlockDequantizer가 아니라 E8M0 지수 바이트를 해석하는 서브클래스.
        # (base는 uint8 바이트를 그대로 곱해 expert 가중치를 ~4.7e5배 폭주시킨다)
        converter=MXFP8BlockDequantizer(
            targets=EXPERT_TARGETS,
            weight_block_size=WEIGHT_BLOCK_SIZE,
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
    # 원본 2**(byte-127) 기준 대조 + shim 확인 — 실패하면 커밋하지 않는다
    verify_dequantized_bf16(tmp, snap)
    commit_dir(tmp, OUT)
    print(f"-> {OUT}")


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description="MXFP8 -> BF16 restore").parse_args()
    main()
