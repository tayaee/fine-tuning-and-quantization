"""tests/test_dequant_mxfp8.py — s2-01-dequant-mxfp8-into-bf16.py 단위 테스트 (issue-2).

pytest가 없는 환경에서도 단독 실행된다:
    .venv/bin/python tests/test_dequant_mxfp8.py
(regression-tests/verify-issue-2.sh가 이 경로를 쓴다)

검증 대상(승인 기준):
  1. MXFP8BlockDequantizer — uint8 E8M0 weight_scale_inv에 2**(byte-127) 적용
     (벤더 modeling_raon1_mini.dequantize_mxfp8와 동일한 수치)
  2. _apply_tracing_shim — _routed_projection이 torch.nn.Linear를 반환
  3. verify_dequantized_bf16 — 정상 통과 / 폭주 가중치·잔존 scale_inv 거부
"""

import contextlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests._compat import pytest, run_standalone

EXPERT_KEY = "model.layers.1.moe.experts.0.gate_proj.weight"


def _load_script(name: str):
    """`s2-01-dequant-mxfp8-into-bf16.py`처럼 숫자로 시작하는 파일을 모듈로 로드."""
    mod_name = name.replace("-", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / name)
    assert spec is not None and spec.loader is not None, f"{name} 로드 실패"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


deq = _load_script("s2-01-dequant-mxfp8-into-bf16.py")


# ---------------------------------------------------------------- 픽스처/독립 기준값


def _mxfp8_fixture(rows: int = 4, cols: int = 64, exps=(115, 120, 130, 127)):
    """F8_E4M3 가중치 [rows, cols] + uint8 E8M0 스케일 [rows, cols//32]."""
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(rows, cols, generator=g) * 0.1).to(torch.float8_e4m3fn)
    # 블록 크기 (1, 32): 행마다 지수 바이트를 다르게 준다 (115 → 2^-12)
    ex = torch.tensor([exps[r % len(exps)] for r in range(rows)], dtype=torch.uint8)
    scale = ex.repeat_interleave(cols // 32).reshape(rows, cols // 32).contiguous()
    return w, scale


def _reference_dequant(w, scale, block=(1, 32)):
    """벤더 modeling_raon1_mini.dequantize_mxfp8 재현 (독립 구현 — 검증의 진실원천)."""
    rows, cols = w.shape
    bw = block[1]
    blocks = w.float().view(rows, cols // bw, bw)
    factor = torch.exp2(scale.view(torch.uint8).float() - 127.0).unsqueeze(-1)
    return (blocks * factor).view(rows, cols)


@contextlib.contextmanager
def _dir(*, tensors=None, modeling: str | None = None):
    """verify_dequantized_bf16 대상 디렉터리 픽스처."""
    root = tempfile.mkdtemp(prefix="vdb-")
    d = Path(root)
    save_file(tensors or {}, str(d / "model-00001-of-00001.safetensors"))
    if modeling is not None:
        (d / "modeling_raon1_mini.py").write_text(modeling, encoding="utf-8")
    try:
        yield d
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _src_dir(w, scale):
    """원본 MXFP8 체크포인트(참조 계산의 진실원천)."""
    root = tempfile.mkdtemp(prefix="vdb-src-")
    d = Path(root)
    shard = "model-00001-of-00001.safetensors"
    save_file(
        {
            EXPERT_KEY: w,
            EXPERT_KEY.replace(".weight", ".weight_scale_inv"): scale,
        },
        str(d / shard),
    )
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {EXPERT_KEY: shard, EXPERT_KEY.replace(".weight", ".weight_scale_inv"): shard}}),
        encoding="utf-8",
    )
    return d


ROUTED_SRC = (
    "import torch.nn as nn\n"
    "\n"
    "class Raon1MiniRoutedLinear(nn.Linear):\n"
    "    def forward(self, hidden):\n"
    "        return hidden\n"
    "\n"
    "def _routed_projection(config, in_features, out_features):\n"
    "    return Raon1MiniRoutedLinear(in_features, out_features, bias=False)\n"
)
SHIMMED_MARKER = "return nn.Linear(in_features, out_features, bias=False)"


def _shimmed_source() -> str:
    src = ROUTED_SRC
    r = deq._ROUTED_PROJECTION_SUB
    return src.replace(r[0], r[1])


# ---------------------------------------------------------------- MXFP8BlockDequantizer


def test_mxfp8_block_dequantizer_matches_vendor_reference():
    w, scale = _mxfp8_fixture()
    conv = deq.MXFP8BlockDequantizer(weight_block_size=(1, 32), dtype=torch.bfloat16)
    got = conv._create_dequantized_weight(w, scale)
    ref = _reference_dequant(w, scale)
    assert got.dtype is torch.bfloat16
    assert got.shape == w.shape
    assert torch.equal(got, ref.to(torch.bfloat16))


def test_mxfp8_block_dequantizer_applies_e8m0_exponent():
    """byte 115 → 2^-12. base 구현처럼 원시 115.0을 곱하면 4.7e5배 튄다."""
    w, scale = _mxfp8_fixture()
    conv = deq.MXFP8BlockDequantizer(weight_block_size=(1, 32), dtype=torch.bfloat16)
    got = conv._create_dequantized_weight(w, scale)[0].float()
    raw = w[0].float() * 115.0
    assert torch.allclose(got, w[0].float() * (2.0 ** (115 - 127)), atol=1e-6)
    assert raw.abs().max() / got.abs().max() > 1e5


def test_mxfp8_block_dequantizer_accepts_float8_e8m0_scale():
    """벤더 _scale_bytes와 동일하게 float8_e8m0fnu 저장형도 같은 지수 바이트로 본다."""
    w, scale = _mxfp8_fixture()
    conv = deq.MXFP8BlockDequantizer(weight_block_size=(1, 32), dtype=torch.bfloat16)
    got_u8 = conv._create_dequantized_weight(w, scale)
    got_f8 = conv._create_dequantized_weight(w, scale.view(torch.uint8).view(torch.float8_e8m0fnu))
    assert torch.equal(got_u8, got_f8)


def test_mxfp8_block_dequantizer_honors_explicit_dtype():
    w, scale = _mxfp8_fixture()
    conv = deq.MXFP8BlockDequantizer(weight_block_size=(1, 32), dtype=torch.bfloat16)
    assert conv._create_dequantized_weight(w, scale, torch.float32).dtype is torch.float32


def test_mxfp8_block_dequantizer_handles_non_block_aligned_cols():
    w, scale = _mxfp8_fixture(rows=2, cols=34, exps=(115, 127))
    conv = deq.MXFP8BlockDequantizer(weight_block_size=(1, 32), dtype=torch.bfloat16)
    assert conv._create_dequantized_weight(w, scale).shape == w.shape


def test_mxfp8_block_dequantizer_process_drops_scale_inv():
    """base process() 경로(2인자 호출)가 그대로 동작해야 한다."""
    w, scale = _mxfp8_fixture()
    conv = deq.MXFP8BlockDequantizer(weight_block_size=(1, 32), dtype=torch.bfloat16)
    out = conv.process({EXPERT_KEY: w, EXPERT_KEY.replace(".weight", ".weight_scale_inv"): scale})
    assert EXPERT_KEY.replace(".weight", ".weight_scale_inv") not in out
    assert torch.equal(out[EXPERT_KEY], _reference_dequant(w, scale).to(torch.bfloat16))


# ---------------------------------------------------------------- _apply_tracing_shim


def test_apply_tracing_shim_rewrites_routed_projection():
    with _dir(modeling=ROUTED_SRC) as d:
        deq._apply_tracing_shim(d)
        src = (d / "modeling_raon1_mini.py").read_text(encoding="utf-8")
    assert SHIMMED_MARKER in src
    assert "Raon1MiniRoutedLinear(in_features, out_features, bias=False)" not in src


def test_routed_projection_returns_plain_linear():
    """치환 후 실제 호출 결과가 정확히 torch.nn.Linear (서브클래스 아님)."""
    with _dir(modeling=ROUTED_SRC) as d:
        deq._apply_tracing_shim(d)
        spec = importlib.util.spec_from_file_location("shimmed_modeling", d / "modeling_raon1_mini.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        proj = mod._routed_projection(SimpleNamespace(), 8, 16)
    assert type(proj) is torch.nn.Linear


def test_apply_tracing_shim_keeps_attention_shims():
    """실제 modeling 파일처럼 attention forward + _routed_projection을 함께 가진 픽스처."""
    attn = (
        "class X:\n"
        "    def forward(self, outputs):\n"
        "        hidden = outputs.last_hidden_state\n"
        "        return dict(hidden=hidden, past_key_values=outputs.past_key_values,\n"
        "                    hidden_states=outputs.hidden_states)\n"
    )
    with _dir(modeling=ROUTED_SRC + attn) as d:
        deq._apply_tracing_shim(d)
        src = (d / "modeling_raon1_mini.py").read_text(encoding="utf-8")
    assert 'hidden = outputs["last_hidden_state"]' in src
    assert 'outputs.get("past_key_values")' in src
    assert 'outputs.get("hidden_states")' in src
    assert SHIMMED_MARKER in src


def test_apply_tracing_shim_raises_when_routed_projection_absent():
    """치환 대상이 없으면 조용히 no-op 되지 않고 실패해야 한다."""
    with _dir(modeling="class X: pass\n") as d, pytest.raises(RuntimeError, match="nn.Linear"):
        deq._apply_tracing_shim(d)


def _vendor_snapshot() -> Path:
    """벤더 스냅샷 캐시 위치 (없으면 이 테스트는 skip)."""
    import os

    from huggingface_hub import constants as hf_constants

    rev = os.environ.get("BASE_REVISION") or "da4cf162c8613993f2189ebb438a5f9a23f3e5ca"
    for hub_root in (Path(hf_constants.HF_HUB_CACHE), Path(deq.SHARED) / "hf-cache" / "hub"):
        cand = hub_root / "models--KRAFTON--Raon-1-mini-preview" / "snapshots" / rev
        if (cand / "modeling_raon1_mini.py").is_file():
            return cand
    return Path("")


def test_vendor_model_experts_become_plain_linear():
    """승인 기준 그대로: 벤더 파일 + 실제 Raon1MiniModel에서 expert가 nn.Linear.

    shim이 '문자열 치환'으로만 만족하는지 아니라 실제로 모델을 구성했을 때 expert
    모듈 타입이 정확히 torch.nn.Linear인지까지 확인한다.
    (이 체크포인트는 first_k_dense_replace=1이라 0층이 dense — MoE는 1층부터)
    """
    snap = _vendor_snapshot()
    if not snap:
        pytest.skip("vendor snapshot 미캐시 (BASE_REVISION 스냅샷 필요)")

    pkg_root = Path(tempfile.mkdtemp(prefix="tiny-raon-"))
    pkg = pkg_root / "tiny_raon"
    pkg.mkdir()
    try:
        for name in ("modeling_raon1_mini.py", "configuration_raon1_mini.py"):
            shutil.copy(snap / name, pkg / name)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        sys.path.insert(0, str(pkg_root))
        deq._apply_tracing_shim(pkg)

        # 임시 패키지라 정적 분석은 경로를 못 찾는다 (런타임엔 sys.path로 해결)
        import tiny_raon.configuration_raon1_mini as vcfg  # pyright: ignore[reportMissingImports]
        import tiny_raon.modeling_raon1_mini as vmod  # pyright: ignore[reportMissingImports]

        raw = json.loads((snap / "config.json").read_text(encoding="utf-8"))
        raw.update(
            hidden_size=64, num_hidden_layers=3, num_attention_heads=4, head_dim=16, vocab_size=128,
            intermediate_size=128, moe_intermediate_size=64, num_experts=4, num_experts_per_token=2,
            q_lora_rank=32, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
            raon_mhc_streams=1, max_position_embeddings=512, rope_parameters=None, pad_token_id=0,
        )
        raw["rda_config"] = {"num_heads": 4, "head_dim": 16, "conv_kernel_size": 2, "full_rank_gate": True,
                             "gate_lower_bound": -5.0, "mla_layers": [], "rda_layers": []}
        raw.pop("quantization_config", None)
        drop = {"architectures", "model_type", "auto_map", "transformers_version"}
        with torch.no_grad():
            model = vmod.Raon1MiniModel(vcfg.Raon1MiniConfig(**{k: v for k, v in raw.items() if k not in drop}))

        moe = model.layers[1].moe
        assert type(moe.experts[0].gate_proj) is torch.nn.Linear
        assert type(moe.experts[0].up_proj) is torch.nn.Linear
        assert type(moe.experts[0].down_proj) is torch.nn.Linear
        # 0층은 dense(MLP) — 라우팅 expert가 아니라 shim 대상이 아니다
        assert not hasattr(model.layers[0], "moe")
    finally:
        shutil.rmtree(pkg_root, ignore_errors=True)


def test_apply_tracing_shim_is_idempotent_and_accepts_str():
    with _dir(modeling=ROUTED_SRC) as d:
        deq._apply_tracing_shim(d)
        first = (d / "modeling_raon1_mini.py").read_text(encoding="utf-8")
        deq._apply_tracing_shim(str(d))
        assert (d / "modeling_raon1_mini.py").read_text(encoding="utf-8") == first


def test_apply_tracing_shim_missing_file_is_reported():
    with _dir() as d, pytest.raises(RuntimeError):
        deq._apply_tracing_shim(d)


# ---------------------------------------------------------------- verify_dequantized_bf16


def _verify_pair(tmp_dir: Path, src_dir: Path):
    return deq.verify_dequantized_bf16(tmp_dir, src_dir, sample_key=EXPERT_KEY)


def test_verify_dequantized_bf16_accepts_correct_output():
    w, scale = _mxfp8_fixture()
    ref = _reference_dequant(w, scale).to(torch.bfloat16)
    src = _src_dir(w, scale)
    with _dir(tensors={EXPERT_KEY: ref}, modeling=_shimmed_source()) as d:
        info = _verify_pair(d, src)
    assert info["max_abs"] < 10.0
    assert info["tensors"] == 1


def test_verify_dequantized_bf16_rejects_exploded_weights():
    w, scale = _mxfp8_fixture()
    exploded = (w.float() * 115.0).to(torch.bfloat16)
    src = _src_dir(w, scale)
    with _dir(tensors={EXPERT_KEY: exploded}, modeling=_shimmed_source()) as d, pytest.raises(RuntimeError, match="max_abs"):
        _verify_pair(d, src)


def test_verify_dequantized_bf16_rejects_mismatched_weights():
    """10 미만이지만 기준과 다른(조용히 잘못된) 가중치도 거부."""
    w, scale = _mxfp8_fixture()
    ref = _reference_dequant(w, scale).to(torch.bfloat16)
    src = _src_dir(w, scale)
    with _dir(tensors={EXPERT_KEY: (ref.float() * 1.5).to(torch.bfloat16)}, modeling=_shimmed_source()) as d, pytest.raises(RuntimeError):
        _verify_pair(d, src)


def test_verify_dequantized_bf16_rejects_residual_scale_inv():
    w, scale = _mxfp8_fixture()
    ref = _reference_dequant(w, scale).to(torch.bfloat16)
    src = _src_dir(w, scale)
    tensors = {EXPERT_KEY: ref, EXPERT_KEY.replace(".weight", ".weight_scale_inv"): scale}
    with _dir(tensors=tensors, modeling=_shimmed_source()) as d, pytest.raises(RuntimeError, match="weight_scale_inv"):
        _verify_pair(d, src)


def test_verify_dequantized_bf16_rejects_residual_fp8_dtype():
    w, scale = _mxfp8_fixture()
    src = _src_dir(w, scale)
    with _dir(tensors={EXPERT_KEY: w}, modeling=_shimmed_source()) as d, pytest.raises(RuntimeError, match="F8_E4M3"):
        _verify_pair(d, src)


def test_verify_dequantized_bf16_rejects_nan():
    w, scale = _mxfp8_fixture()
    ref = _reference_dequant(w, scale).to(torch.bfloat16)
    ref[0, 0] = float("nan")
    src = _src_dir(w, scale)
    with _dir(tensors={EXPERT_KEY: ref}, modeling=_shimmed_source()) as d, pytest.raises(RuntimeError, match="NaN|Inf"):
        _verify_pair(d, src)


def test_verify_dequantized_bf16_rejects_unshimmed_modeling():
    w, scale = _mxfp8_fixture()
    ref = _reference_dequant(w, scale).to(torch.bfloat16)
    src = _src_dir(w, scale)
    with _dir(tensors={EXPERT_KEY: ref}, modeling=ROUTED_SRC) as d, pytest.raises(RuntimeError, match="nn.Linear"):
        _verify_pair(d, src)


def test_verify_dequantized_bf16_rejects_missing_modeling():
    w, scale = _mxfp8_fixture()
    ref = _reference_dequant(w, scale).to(torch.bfloat16)
    src = _src_dir(w, scale)
    with _dir(tensors={EXPERT_KEY: ref}) as d, pytest.raises(RuntimeError, match="modeling_raon1_mini.py"):
        _verify_pair(d, src)


# ---------------------------------------------------------------- 단독 실행


def _main() -> int:
    return run_standalone(globals())


if __name__ == "__main__":
    raise SystemExit(_main())