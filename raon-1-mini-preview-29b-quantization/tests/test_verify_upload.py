"""tests/test_verify_upload.py — s7-57-verify-upload.py 단위 테스트.

pytest가 없는 환경에서도 단독 실행된다:
    .venv/bin/python tests/test_verify_upload.py
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests._compat import pytest, run_standalone


def _load_script(name: str):
    mod_name = name.replace("-", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / name)
    assert spec is not None and spec.loader is not None, f"{name} 로드 실패"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


prep = _load_script("s7-56-prep-upload.py")
ver = _load_script("s7-57-verify-upload.py")

CUSTOM_CFG_PY = "class Raon1MiniConfig:\n    pass\n"
CUSTOM_MOD_PY = (
    "import torch.nn as nn\n"
    "class Raon1MiniModel(nn.Module):\n    pass\n"
    "class Raon1MiniForCausalLM(nn.Module):\n    pass\n"
    "def _routed_projection(config, in_features, out_features):\n"
    "    return nn.Linear(in_features, out_features, bias=False)\n"
)
CHAT_JINJA = "<|im_start|>user\n{{ content }}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"


def _make_valid_workspace(root: Path, target: str = "nvfp4-awq", mode: str = "full") -> Path:
    bf16 = root / "models" / "raon-bf16"
    bf16.mkdir(parents=True, exist_ok=True)
    (bf16 / "config.json").write_text(json.dumps({"model_type": "raon1_mini"}), encoding="utf-8")
    (bf16 / "configuration_raon1_mini.py").write_text(CUSTOM_CFG_PY, encoding="utf-8")
    (bf16 / "modeling_raon1_mini.py").write_text(CUSTOM_MOD_PY, encoding="utf-8")
    (bf16 / "chat_template.jinja").write_text(CHAT_JINJA, encoding="utf-8")
    save_file(
        {"model.layers.1.moe.experts.0.gate_proj.weight": torch.ones((4, 4), dtype=torch.bfloat16)},
        str(bf16 / "model-00001-of-00001.safetensors"),
    )

    mdir = root / "models" / f"raon-{target}-{mode}"
    mdir.mkdir(parents=True, exist_ok=True)
    is_nvfp4 = target.startswith("nvfp4")
    num_bits = 4 if is_nvfp4 else (3 if target == "awq-w3" else 5)
    fmt = "nvfp4-pack-quantized" if is_nvfp4 else "pack-quantized"
    w_type = "float" if is_nvfp4 else "int"

    cfg = {
        "architectures": ["Raon1MiniForCausalLM"],
        "model_type": "raon1_mini",
        "auto_map": {
            "AutoConfig": "configuration_raon1_mini.Raon1MiniConfig",
            "AutoModel": "modeling_raon1_mini.Raon1MiniModel",
            "AutoModelForCausalLM": "modeling_raon1_mini.Raon1MiniForCausalLM",
        },
        "eos_token_id": [259202, 259200],
        "pad_token_id": 259200,
        "num_hidden_layers": 2,
        "num_experts": 4,
        "num_experts_per_token": 2,
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": fmt,
            "ignore": ["lm_head", "re:.*moe\\.gate$", "re:.*self_attn.*", "re:.*shared_experts.*"],
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": num_bits,
                        "type": w_type,
                        "symmetric": is_nvfp4,
                        "group_size": 16 if is_nvfp4 else 64,
                    }
                }
            },
        },
    }
    (mdir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (mdir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (mdir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (mdir / "chat_template.jinja").write_text(CHAT_JINJA, encoding="utf-8")
    (mdir / "configuration_raon1_mini.py").write_text(CUSTOM_CFG_PY, encoding="utf-8")
    (mdir / "modeling_raon1_mini.py").write_text(CUSTOM_MOD_PY, encoding="utf-8")

    p = "model.layers.1.moe.experts.0.gate_proj"
    if is_nvfp4:
        tensors = {
            f"{p}.weight_packed": torch.zeros((4, 4), dtype=torch.uint8),
            f"{p}.weight_scale": torch.ones((4, 1), dtype=torch.float8_e4m3fn),
            f"{p}.weight_global_scale": torch.ones((1,), dtype=torch.float32),
            f"{p}.input_global_scale": torch.ones((1,), dtype=torch.float32),
            "lm_head.weight": torch.ones((8, 4), dtype=torch.bfloat16),
        }
    else:
        tensors = {
            f"{p}.weight_packed": torch.zeros((4, 2), dtype=torch.int32),
            f"{p}.weight_scale": torch.ones((4, 1), dtype=torch.bfloat16),
            f"{p}.weight_zero_point": torch.zeros((4, 1), dtype=torch.int32),
            "lm_head.weight": torch.ones((8, 4), dtype=torch.bfloat16),
        }
    shard_name = "model-00001-of-00001.safetensors"
    save_file(tensors, str(mdir / shard_name))
    (mdir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: shard_name for k in tensors}}),
        encoding="utf-8",
    )

    ppl_dir = root / "outputs" / f"ppl-{mode}"
    ppl_dir.mkdir(parents=True, exist_ok=True)
    (ppl_dir / "summary.json").write_text(
        json.dumps(
            {
                "base": {"target": "base", "ppl": 6.50, "verdict": "ref"},
                target: {"target": target, "ppl": 6.65, "delta": 0.15, "verdict": "Accept"},
            }
        ),
        encoding="utf-8",
    )

    eval_dir = root / "outputs" / f"eval-{mode}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    benches = {b: {"n": 3, "acc": 0.667} for b in ("kmmlu_redux", "gsm8k", "humaneval")}
    (eval_dir / "score.json").write_text(
        json.dumps(
            {
                "base": {"n": 9, "done_rate": 1.0, "avg_len": 80.0, "think_rate": 0.0, "benchmarks": benches},
                target: {"n": 9, "done_rate": 1.0, "avg_len": 78.0, "think_rate": 0.0, "benchmarks": benches},
            }
        ),
        encoding="utf-8",
    )
    (eval_dir / f"{target}.jsonl").write_text(
        json.dumps({"bench": "gsm8k", "prompt": "1+1", "gold": "2", "output": "#### 2"}) + "\n",
        encoding="utf-8",
    )
    return mdir


@pytest.fixture
def ws(monkeypatch):
    d = Path(tempfile.mkdtemp(prefix="rq-ver-"))
    for mod in (prep, ver):
        monkeypatch.setattr(mod, "SHARED", str(d))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_valid_checkpoint_passes_all_checks(ws):
    _make_valid_workspace(ws, "nvfp4-awq", "full")
    prep.run("full", "nvfp4-awq")
    res = ver.run("full", "nvfp4-awq")
    assert res["ok"] is True
    assert res["bf16_ok"] is True
    t_rep = res["targets"]["nvfp4-awq"]
    assert t_rep["ok"] is True
    assert all(t_rep["checks"].values())
    assert (ws / "outputs" / "verify-upload-full.json").is_file()


def test_rejects_bf16_with_residual_quant_config_or_missing_shim(ws):
    _make_valid_workspace(ws, "nvfp4-awq", "full")
    prep.run("full", "nvfp4-awq")
    bf16 = ws / "models" / "raon-bf16"
    (bf16 / "config.json").write_text(json.dumps({"quantization_config": {"format": "mxfp8"}}), encoding="utf-8")
    (bf16 / "modeling_raon1_mini.py").write_text("class Raon1MiniModel: pass\n", encoding="utf-8")
    errs = ver.check_bf16_dir(bf16)
    assert any("quantization_config" in e for e in errs)
    assert any("트레이싱 심 누락" in e for e in errs)


def test_rejects_safetensors_index_mismatch(ws):
    mdir = _make_valid_workspace(ws, "nvfp4-awq", "full")
    prep.run("full", "nvfp4-awq")
    (mdir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"nonexistent.tensor": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    rep = ver.verify_target("full", "nvfp4-awq")
    assert rep["ok"] is False
    assert rep["checks"]["safetensors_index"] is False
    assert any("샤드 헤더에 없는 텐서 키" in e for e in rep["errors"])


def test_rejects_missing_input_global_scale_for_nvfp4(ws):
    mdir = _make_valid_workspace(ws, "nvfp4-awq", "full")
    prep.run("full", "nvfp4-awq")
    p = "model.layers.1.moe.experts.0.gate_proj"
    tensors = {
        f"{p}.weight_packed": torch.zeros((4, 4), dtype=torch.uint8),
        f"{p}.weight_scale": torch.ones((4, 1), dtype=torch.float8_e4m3fn),
        f"{p}.weight_global_scale": torch.ones((1,), dtype=torch.float32),
        "lm_head.weight": torch.ones((8, 4), dtype=torch.bfloat16),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, str(mdir / shard))
    (mdir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: shard for k in tensors}}),
        encoding="utf-8",
    )
    rep = ver.verify_target("full", "nvfp4-awq")
    assert rep["ok"] is False
    assert any("input_global_scale" in e for e in rep["errors"])


def test_rejects_out_of_range_size_and_tensor_count_on_full_29b_config(ws):
    mdir = _make_valid_workspace(ws, "nvfp4-awq", "full")
    prep.run("full", "nvfp4-awq")
    cfg = json.loads((mdir / "config.json").read_text(encoding="utf-8"))
    cfg["num_hidden_layers"] = 26
    cfg["num_experts"] = 128
    (mdir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    rep = ver.verify_target("full", "nvfp4-awq", allow_tiny=False)
    assert rep["ok"] is False
    assert rep["checks"]["model_size"] is False
    assert rep["checks"]["safetensors_index"] is False
    assert any("정상 범위" in e for e in rep["errors"])
    assert any("9216" in e for e in rep["errors"])


def test_rejects_missing_vllm_config_fields(ws):
    mdir = _make_valid_workspace(ws, "nvfp4-awq", "full")
    prep.run("full", "nvfp4-awq")
    cfg = json.loads((mdir / "config.json").read_text(encoding="utf-8"))
    cfg.pop("auto_map", None)
    cfg.pop("num_experts_per_token", None)
    (mdir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    rep = ver.verify_target("full", "nvfp4-awq")
    assert rep["ok"] is False
    assert rep["checks"]["vllm_ready"] is False


def test_rejects_discard_ppl_and_prompts_pollution_in_score(ws):
    _make_valid_workspace(ws, "nvfp4-awq", "full")
    prep.run("full", "nvfp4-awq")
    ppl_fp = ws / "outputs" / "ppl-full" / "summary.json"
    ppl_fp.write_text(
        json.dumps(
            {
                "base": {"target": "base", "ppl": 6.50, "verdict": "ref"},
                "nvfp4-awq": {"target": "nvfp4-awq", "ppl": 8.00, "delta": 1.50, "verdict": "Discard"},
            }
        ),
        encoding="utf-8",
    )
    score_fp = ws / "outputs" / "eval-full" / "score.json"
    sdata = json.loads(score_fp.read_text(encoding="utf-8"))
    sdata["prompts-gsm8k"] = {"n": 3, "done_rate": 0.0, "benchmarks": {}}
    score_fp.write_text(json.dumps(sdata), encoding="utf-8")

    rep = ver.verify_target("full", "nvfp4-awq")
    assert rep["ok"] is False
    assert rep["checks"]["metrics"] is False
    assert any("Discard" in e for e in rep["errors"])
    assert any("prompts-gsm8k" in e for e in rep["errors"])


def test_rejects_readme_with_missing_metric_dashes(ws):
    mdir = _make_valid_workspace(ws, "nvfp4-awq", "full")
    prep.run("full", "nvfp4-awq")
    readme = (mdir / "README.md").read_text(encoding="utf-8")
    # Model 컬럼 = 정식 HF repo id (full → 접미사 없음)
    readme = readme.replace(
        "| **tayaee/Raon-1-mini-preview-nvfp4-awq** | 6.650 | 0.150 |",
        "| **tayaee/Raon-1-mini-preview-nvfp4-awq** | — | — |",
    )
    (mdir / "README.md").write_text(readme, encoding="utf-8")
    rep = ver.verify_target("full", "nvfp4-awq")
    assert rep["ok"] is False
    assert rep["checks"]["readme"] is False
    assert any("결측치" in e for e in rep["errors"])


def test_rejects_zero_or_garbage_input_global_scale(ws):
    mdir = _make_valid_workspace(ws, "nvfp4-awq", "full")
    prep.run("full", "nvfp4-awq")
    p = "model.layers.1.moe.experts.0.gate_proj"
    tensors = {
        f"{p}.weight_packed": torch.zeros((4, 4), dtype=torch.uint8),
        f"{p}.weight_scale": torch.ones((4, 1), dtype=torch.float8_e4m3fn),
        f"{p}.weight_global_scale": torch.ones((1,), dtype=torch.float32),
        f"{p}.input_global_scale": torch.zeros((1,), dtype=torch.float32),
        "lm_head.weight": torch.ones((8, 4), dtype=torch.bfloat16),
    }
    save_file(tensors, str(mdir / "model-00001-of-00001.safetensors"))
    rep = ver.verify_target("full", "nvfp4-awq")
    assert rep["ok"] is False
    assert any("global_scale" in e for e in rep["errors"])


def _main() -> int:
    return run_standalone(globals())


if __name__ == "__main__":
    raise SystemExit(_main())
