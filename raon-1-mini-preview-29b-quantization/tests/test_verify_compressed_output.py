"""tests/test_verify_compressed_output.py — quant_common.verify_compressed_output 단위 테스트.

pytest가 없는 환경에서도 단독 실행된다:
    uv run --no-sync python tests/test_verify_compressed_output.py
(regression-tests/verify-issue-3.sh가 이 경로를 쓴다)

픽스처 텐서 dtype은 compressed_tensors 컴프레서의 실제 계약에서 가져왔다:
  pack-quantized      → weight_packed I32, weight_scale, (asymmetric일 때만) weight_zero_point
  nvfp4-pack-quantized→ weight_packed U8, weight_scale F8_E4M3, weight_global_scale F32
"""

import contextlib
import json
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_common import verify_compressed_output

PACK = "pack-quantized"
NVFP4 = "nvfp4-pack-quantized"


def _qc(fmt, *, symmetric=True):
    return {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "float" if fmt == NVFP4 else "int",
                    "symmetric": symmetric,
                    "strategy": "block" if fmt == NVFP4 else "group",
                    "block_size": 16 if fmt == NVFP4 else 64,
                },
            }
        },
        "format": fmt,
        "ignore": ["lm_head"],
    }


def _nvfp4_tensors(prefix="model.layers.1.moe.experts.0.gate_proj"):
    return {
        f"{prefix}.weight_packed": torch.zeros(4, 8, dtype=torch.uint8),
        f"{prefix}.weight_scale": torch.zeros(4, 8, dtype=torch.float8_e4m3fn),
        f"{prefix}.weight_global_scale": torch.zeros(4, 1, dtype=torch.float32),
    }


def _pack_tensors(prefix="model.layers.1.moe.experts.0.gate_proj", *, zero_point=False):
    t = {
        f"{prefix}.weight_packed": torch.zeros(4, 16, dtype=torch.int32),
        f"{prefix}.weight_scale": torch.zeros(4, 16, dtype=torch.bfloat16),
        f"{prefix}.weight_shape": torch.tensor([8, 128], dtype=torch.int64),
    }
    if zero_point:
        t[f"{prefix}.weight_zero_point"] = torch.zeros(4, 1, dtype=torch.int32)
    return t


@contextlib.contextmanager
def _model(
    *,
    fmt=PACK,
    symmetric=True,
    tensors=None,
    shards=1,
    quantization_config=True,
    custom_code=True,
):
    """최소 체크포인트 픽스처를 임시 디렉터리에 만들고 그 Path를 준다."""
    root = tempfile.mkdtemp(prefix="vco-")
    d = Path(root)
    cfg = {"model_type": "raon1_mini", "architectures": ["Raon1MiniForCausalLM"]}
    if quantization_config:
        cfg["quantization_config"] = _qc(fmt, symmetric=symmetric)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    base = _nvfp4_tensors() if fmt == NVFP4 else _pack_tensors()
    if tensors is not None:  # tensors가 주어지면 그대로 사용 (키 삭제 케이스도 포함)
        base = tensors
    if shards == 1:
        save_file(base, str(d / "model-00001-of-00001.safetensors"))
    else:
        items = list(base.items())
        per = -(-len(items) // shards)
        for i in range(shards):
            chunk = dict(items[i * per : (i + 1) * per])
            if not chunk:
                chunk = {"model.layers.0.norm.weight": torch.zeros(2, dtype=torch.bfloat16)}
            save_file(chunk, str(d / f"model-{i + 1:05d}-of-{shards:05d}.safetensors"))
    if custom_code:
        (d / "configuration_raon1_mini.py").write_text("# config\n", encoding="utf-8")
        (d / "modeling_raon1_mini.py").write_text("# model\n", encoding="utf-8")
    try:
        yield d
    finally:
        for child in sorted(d.rglob("*"), reverse=True):
            child.unlink() if child.is_file() else child.rmdir()
        d.rmdir()


@contextlib.contextmanager
def _rejects(fragment):
    """RuntimeError가 발생하고 메시지에 fragment가 포함돼야 한다."""
    try:
        yield
    except RuntimeError as e:
        assert fragment in str(e), f"expected {fragment!r} in {e!s}"
        return
    raise AssertionError(f"expected RuntimeError containing {fragment!r}")


# --- config.json / format -------------------------------------------------


def test_rejects_dense_format_fallback():
    """압축기 폴백(가중치 미패킹 59GB BF16)을 즉시 실패시킨다."""
    with _model(fmt="dense", tensors=_pack_tensors()) as d, _rejects("dense"):
        verify_compressed_output(d, PACK)


def test_rejects_missing_quantization_config():
    """quantization_config 자체가 없으면 압축이 안 된 것이다."""
    with _model(quantization_config=False) as d, _rejects("quantization_config"):
        verify_compressed_output(d, PACK)


def test_rejects_format_mismatch():
    """nvfp4로 기대하는데 pack-quantized면 거부."""
    with _model(fmt=PACK, tensors=_pack_tensors()) as d, _rejects(NVFP4):
        verify_compressed_output(d, NVFP4)


def test_rejects_missing_config_file():
    with tempfile.TemporaryDirectory() as root, _rejects("config.json"):
        verify_compressed_output(root, PACK)


# --- 텐서 계약 ------------------------------------------------------------


def test_rejects_missing_weight_packed():
    """weight_packed가 하나도 없으면 미압축(dense weight 잔존)."""
    t = {k: v for k, v in _pack_tensors().items() if not k.endswith("weight_packed")}
    t["model.layers.1.moe.experts.0.gate_proj.weight"] = torch.zeros(8, 128, dtype=torch.bfloat16)
    with _model(tensors=t) as d, _rejects("weight_packed"):
        verify_compressed_output(d, PACK)


def test_rejects_nvfp4_with_int32_packed():
    """nvfp4는 U8 패킹이어야 한다 (I32면 int 패킹 결과)."""
    t = _nvfp4_tensors()
    t["model.layers.1.moe.experts.0.gate_proj.weight_packed"] = torch.zeros(4, 16, dtype=torch.int32)
    with _model(fmt=NVFP4, tensors=t) as d, _rejects("U8"):
        verify_compressed_output(d, NVFP4)


def test_rejects_pack_quantized_with_uint8_packed():
    """pack-quantized는 I32 패킹이어야 한다."""
    t = _pack_tensors()
    t["model.layers.1.moe.experts.0.gate_proj.weight_packed"] = torch.zeros(4, 8, dtype=torch.uint8)
    with _model(tensors=t) as d, _rejects("I32"):
        verify_compressed_output(d, PACK)


def test_rejects_nvfp4_without_global_scale():
    """nvfp4는 블록 스케일과 함께 F32 global_scale이 있어야 한다."""
    t = _nvfp4_tensors()
    t.pop("model.layers.1.moe.experts.0.gate_proj.weight_global_scale")
    with _model(fmt=NVFP4, tensors=t) as d, _rejects("weight_global_scale"):
        verify_compressed_output(d, NVFP4)


def test_rejects_nvfp4_with_bf16_scale():
    """nvfp4 블록 스케일은 F8_E4M3이어야 한다."""
    t = _nvfp4_tensors()
    t["model.layers.1.moe.experts.0.gate_proj.weight_scale"] = torch.zeros(4, 8, dtype=torch.bfloat16)
    with _model(fmt=NVFP4, tensors=t) as d, _rejects("F8_E4M3"):
        verify_compressed_output(d, NVFP4)


def test_rejects_nvfp4_with_zero_point():
    """nvfp4는 대칭 양자화라 weight_zero_point가 남아 있으면 안 된다."""
    t = _nvfp4_tensors()
    t["model.layers.1.moe.experts.0.gate_proj.weight_zero_point"] = torch.zeros(4, 1, dtype=torch.uint8)
    with _model(fmt=NVFP4, tensors=t) as d, _rejects("weight_zero_point"):
        verify_compressed_output(d, NVFP4)


def test_rejects_symmetric_pack_with_zero_point():
    """대칭 pack-quantized에 남은 weight_zero_point는 vLLM 로드 실패 원인."""
    t = _pack_tensors(zero_point=True)
    with _model(tensors=t) as d, _rejects("weight_zero_point"):
        verify_compressed_output(d, PACK)


def test_rejects_pack_without_scale():
    t = _pack_tensors()
    t.pop("model.layers.1.moe.experts.0.gate_proj.weight_scale")
    with _model(tensors=t) as d, _rejects("weight_scale"):
        verify_compressed_output(d, PACK)


def test_rejects_missing_safetensors():
    """safetensors shard 자체가 없으면 검증 불가."""
    with _model(tensors=_pack_tensors()) as d:
        (d / "model-00001-of-00001.safetensors").unlink()
        with _rejects(".safetensors"):
            verify_compressed_output(d, PACK)


# --- 커스텀 코드 ----------------------------------------------------------


def test_rejects_missing_custom_code():
    """trust_remote_code 재로드는 modeling/configuration .py가 있어야 한다."""
    with _model(custom_code=False) as d, _rejects("modeling_raon1_mini.py"):
        verify_compressed_output(d, PACK)


def test_rejects_missing_configuration_code():
    with _model(custom_code=False) as d:
        (d / "modeling_raon1_mini.py").write_text("# model\n", encoding="utf-8")
        with _rejects("configuration_raon1_mini.py"):
            verify_compressed_output(d, PACK)


# --- 통과 케이스 ----------------------------------------------------------


def test_accepts_nvfp4_fixture():
    with _model(fmt=NVFP4) as d:
        info = verify_compressed_output(d, NVFP4)
    assert info["format"] == NVFP4
    assert info["packed"] >= 1


def test_accepts_pack_quantized_symmetric_fixture():
    with _model(fmt=PACK) as d:
        info = verify_compressed_output(d, PACK)
    assert info["format"] == PACK


def test_accepts_pack_quantized_asymmetric_fixture():
    """W3/W5 레시피는 asymmetric(group)라 zero_point가 정상이다."""
    with _model(fmt=PACK, symmetric=False, tensors=_pack_tensors(zero_point=True)) as d:
        verify_compressed_output(d, PACK)


def test_accepts_multi_shard_nvfp4():
    """실제 산출물은 shard 여러 개 — 전부 스캔해야 한다."""
    t = _nvfp4_tensors()
    t.update(_nvfp4_tensors("model.layers.2.moe.experts.0.up_proj"))
    with _model(fmt=NVFP4, tensors=t, shards=2) as d:
        info = verify_compressed_output(d, NVFP4)
    assert info["packed"] == 2


def test_accepts_str_path():
    """tmp_path()가 str을 주므로 문자열 경로도 받아야 한다."""
    with _model(fmt=NVFP4) as d:
        verify_compressed_output(str(d), NVFP4)


def test_rejects_nvfp4_with_zero_or_uninitialized_input_global_scale():
    """MoE expert 미캘리브레이션으로 input_global_scale이 0.0이나 쓰레기값이면 거부."""
    t = _nvfp4_tensors()
    t["model.layers.1.moe.experts.0.gate_proj.weight_global_scale"] = torch.ones(1, dtype=torch.float32)
    t["model.layers.1.moe.experts.0.gate_proj.input_global_scale"] = torch.zeros(1, dtype=torch.float32)
    with _model(fmt=NVFP4, tensors=t) as d, _rejects("global_scale"):
        verify_compressed_output(d, NVFP4)
    t["model.layers.1.moe.experts.0.gate_proj.input_global_scale"] = torch.tensor([-9.58e21], dtype=torch.float32)
    with _model(fmt=NVFP4, tensors=t) as d, _rejects("global_scale"):
        verify_compressed_output(d, NVFP4)


def _main() -> int:
    failed = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — 단독 러너는 실패 목록만 보고
            failed.append((name, e))
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"PASS {name}")
    print(f"\n{len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())