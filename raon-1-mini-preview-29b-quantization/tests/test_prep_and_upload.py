"""tests/test_prep_and_upload.py — issue-5 (43-prep-upload / 49-upload-hf) 단위 테스트.

pytest가 없는 환경에서도 단독 실행된다:
    .venv/bin/python tests/test_prep_and_upload.py
(regression-tests/verify-issue-5.sh가 이 경로를 쓴다)

네트워크를 전혀 쓰지 않는다 — 합성 압축 체크포인트 + 합성 summary/score json 픽스처로
README.md 모델 카드와 업로드 사전 검증만 확인한다.
"""

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


prep = _load_script("43-prep-upload.py")
up = _load_script("49-upload-hf.py")

NVFP4 = "nvfp4-pack-quantized"
PACK = "pack-quantized"
# 표준 GSM8K 단일 벤치 (37-eval이 표준 GSM8K만 평가)
BENCHES = ("gsm8k",)


# ---------------------------------------------------------------- 픽스처


def _qc(fmt: str) -> dict:
    return {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "float" if fmt == NVFP4 else "int",
                    "symmetric": True,
                    "strategy": "block" if fmt == NVFP4 else "group",
                    "block_size": 16 if fmt == NVFP4 else 64,
                },
            }
        },
        "format": fmt,
        "ignore": ["lm_head"],
    }


def _tensors(fmt: str, prefix: str = "model.layers.1.moe.experts.0.gate_proj"):
    if fmt == NVFP4:
        return {
            f"{prefix}.weight_packed": torch.zeros(4, 8, dtype=torch.uint8),
            f"{prefix}.weight_scale": torch.zeros(4, 8, dtype=torch.float8_e4m3fn),
            f"{prefix}.weight_global_scale": torch.zeros(4, 1, dtype=torch.float32),
        }
    return {
        f"{prefix}.weight_packed": torch.zeros(4, 16, dtype=torch.int32),
        f"{prefix}.weight_scale": torch.zeros(4, 16, dtype=torch.bfloat16),
        f"{prefix}.weight_shape": torch.tensor([8, 128], dtype=torch.int64),
    }


def _make_model_dir(root: Path, name: str, fmt: str = NVFP4, *, custom_code: bool = True) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    save_file(_tensors(fmt), str(d / "model-00001-of-00001.safetensors"))
    (d / "config.json").write_text(
        json.dumps({"model_type": "raon1_mini", "architectures": ["Raon1MiniForCausalLM"],
                    "quantization_config": _qc(fmt)}),
        encoding="utf-8",
    )
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    (d / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    if custom_code:
        (d / "configuration_raon1_mini.py").write_text("# cfg\n", encoding="utf-8")
        (d / "modeling_raon1_mini.py").write_text("# model\n", encoding="utf-8")
        (d / "chat_template.jinja").write_text("{{ messages }}\n", encoding="utf-8")
    return d


@pytest.fixture
def workspace(monkeypatch):
    """$RQ_SHARED/{models,outputs} 를 임시 디렉터리로."""
    root = Path(tempfile.mkdtemp(prefix="rq-prep-"))
    models = root / "models"
    outs = root / "outputs"
    models.mkdir()
    outs.mkdir()
    for mod in (prep, up):
        monkeypatch.setattr(mod, "SHARED", str(root))
        monkeypatch.setattr(mod, "MODELS_DIR", lambda mode, _m=models: str(_m))
    monkeypatch.setattr(prep, "OUT_ROOT", lambda: str(outs))
    (models / "raon-bf16").mkdir()
    for f in ("chat_template.jinja", "configuration_raon1_mini.py", "modeling_raon1_mini.py"):
        (models / "raon-bf16" / f).write_text(f"# bf16 {f}\n", encoding="utf-8")
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _write_metrics(root: Path, mode: str = "smoke") -> None:
    outdir = root / "outputs" / f"ppl-{mode}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(
        json.dumps({"base": {"target": "base", "ppl": 8.11, "verdict": "ref"},
                    "nvfp4-awq": {"target": "nvfp4-awq", "ppl": 8.42, "delta": 0.31,
                                  "verdict": "Conditional"}}),
        encoding="utf-8",
    )
    edir = root / "outputs" / f"eval-{mode}"
    edir.mkdir(parents=True, exist_ok=True)
    (edir / "score.json").write_text(
        json.dumps({
            "base": {"n": 30, "done_rate": 1.0, "benchmarks": {b: {"n": 10, "acc": 0.9} for b in BENCHES}},
            "nvfp4-awq": {"n": 30, "done_rate": 1.0,
                          "benchmarks": {b: {"n": 10, "acc": 0.8} for b in BENCHES}},
        }),
        encoding="utf-8",
    )


# ---------------------------------------------------------------- 43-prep-upload


def test_prep_rejects_dense_fallback(workspace):
    """issue-3 계약과 동일하게 format:"dense" 폴백은 준비 단계에서 막는다."""
    d = _make_model_dir(workspace / "models", "raon-nvfp4-awq-smoke")
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    cfg["quantization_config"]["format"] = "dense"
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    report = prep.prepare_target("smoke", "nvfp4-awq", allow_missing_metrics=True)
    assert report["ok"] is False
    assert "dense" in " ".join(report["errors"])


def test_prep_copies_missing_custom_code_from_bf16(workspace):
    d = _make_model_dir(workspace / "models", "raon-awq-w5-smoke", fmt=PACK, custom_code=False)
    report = prep.prepare_target("smoke", "awq-w5", allow_missing_metrics=True)
    assert report["ok"] is True, report["errors"]
    for f in ("chat_template.jinja", "configuration_raon1_mini.py", "modeling_raon1_mini.py"):
        assert (d / f).is_file(), f"{f} 미보충"
    assert "chat_template.jinja" in report["copied"]


def test_prep_requires_tokenizer_files(workspace):
    d = _make_model_dir(workspace / "models", "raon-awq-w5-smoke", fmt=PACK)
    (d / "tokenizer.json").unlink()
    report = prep.prepare_target("smoke", "awq-w5", allow_missing_metrics=True)
    assert report["ok"] is False
    assert any("tokenizer.json" in e for e in report["errors"])


def test_prep_generates_readme_with_required_sections(workspace):
    d = _make_model_dir(workspace / "models", "raon-nvfp4-awq-smoke")
    _write_metrics(workspace, "smoke")
    report = prep.prepare_target("smoke", "nvfp4-awq", allow_missing_metrics=True)
    assert report["ok"] is True, report["errors"]
    readme = (d / "README.md").read_text(encoding="utf-8")
    for tok in ("license: apache-2.0",
                "base_model: KRAFTON/Raon-1-mini-preview",
                "base_model_relation: quantized",
                "library_name: transformers",
                "pipeline_tag: text-generation",
                "da4cf162c8613993f2189ebb438a5f9a23f3e5ca",
                "trust_remote_code=True",
                "--trust-remote-code",
                "--quantization compressed-tensors",
                "Apache"):
        assert tok in readme, f"README에 '{tok}' 없음"
    assert "extra_gated" not in readme, "원본 KRAFTON gated 필드를 복사하면 안 된다"
    # 검사 2종(PPL/GSM8K) 표가 있어야 한다
    assert "## 검사 1 — PPL" in readme, "PPL 검사 절 없음"
    assert "## 검사 2 — GSM8K" in readme, "GSM8K 검사 절 없음"
    # 기준선(base) 수치와 현재 모델 수치가 표에 함께 있어야 한다
    assert "8.11" in readme and "8.42" in readme
    for b in BENCHES:
        assert b in readme


def test_prep_requires_metrics_unless_allowed(workspace):
    d = _make_model_dir(workspace / "models", "raon-nvfp4-awq-smoke")
    report = prep.prepare_target("smoke", "nvfp4-awq", allow_missing_metrics=False)
    assert report["ok"] is False
    assert any("ppl" in e or "score" in e for e in report["errors"])

    _write_metrics(workspace, "smoke")
    report = prep.prepare_target("smoke", "nvfp4-awq", allow_missing_metrics=False)
    assert report["ok"] is True, report["errors"]
    assert (d / "README.md").is_file()


def test_prep_writes_manifest(workspace):
    _make_model_dir(workspace / "models", "raon-nvfp4-awq-smoke")
    _write_metrics(workspace, "smoke")
    manifest = prep.run("smoke", "nvfp4-awq", allow_missing_metrics=True)
    path = workspace / "outputs" / "prep-upload-smoke.json"
    assert path.is_file()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["mode"] == "smoke"
    assert saved["targets"]["nvfp4-awq"]["ok"] is True
    assert manifest["ok"] is True


def test_prep_manifest_fails_when_target_failed(workspace):
    d = _make_model_dir(workspace / "models", "raon-nvfp4-awq-smoke")
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    cfg["quantization_config"]["format"] = "dense"
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    manifest = prep.run("smoke", "nvfp4-awq", allow_missing_metrics=True)
    assert manifest["ok"] is False
    assert manifest["targets"]["nvfp4-awq"]["errors"]


# ---------------------------------------------------------------- 49-upload-hf


def test_repo_name_keeps_smoke_suffix():
    assert up.repo_name("nvfp4-awq", "full") == "tayaee/Raon-1-mini-preview-nvfp4-awq"
    assert up.repo_name("nvfp4-awq", "smoke") == "tayaee/Raon-1-mini-preview-nvfp4-awq-smoke"


def test_upload_refuses_without_readme(workspace):
    _make_model_dir(workspace / "models", "raon-nvfp4-awq-full")
    with pytest.raises(SystemExit, match="README.md"):
        up.check_ready("nvfp4-awq", "full")


def test_upload_skips_missing_config(workspace, capsys):
    """config.json이 없으면 (기존 관례대로) 조용히 SKIP — 다른 타겟을 계속 올린다."""
    _make_model_dir(workspace / "models", "raon-nvfp4-awq-full")
    (workspace / "models" / "raon-nvfp4-awq-full" / "config.json").unlink()
    assert up.check_ready("nvfp4-awq", "full") is None
    assert "SKIP" in capsys.readouterr().out


def test_upload_refuses_empty_readme(workspace):
    d = _make_model_dir(workspace / "models", "raon-nvfp4-awq-full")
    (d / "README.md").write_text("   \n", encoding="utf-8")
    with pytest.raises(SystemExit, match="README.md"):
        up.check_ready("nvfp4-awq", "full")


def test_upload_dry_run_does_no_network(workspace, capsys):
    d = _make_model_dir(workspace / "models", "raon-nvfp4-awq-full")
    (d / "README.md").write_text("# 카드\n", encoding="utf-8")

    def boom(*a, **k):
        raise AssertionError("dry-run이 네트워크를 호출했다")

    import huggingface_hub

    monkey = {"create_repo": boom, "upload_folder": boom, "HfApi": boom}
    old = {k: getattr(huggingface_hub, k, None) for k in monkey}
    try:
        for k, v in monkey.items():
            setattr(huggingface_hub, k, v)
        up.main("full", "nvfp4-awq", dry_run=True)
    finally:
        for k, v in old.items():
            setattr(huggingface_hub, k, v)
    out = capsys.readouterr().out
    assert "tayaee/Raon-1-mini-preview-nvfp4-awq" in out


def test_upload_refuses_smoke_mode(workspace):
    with pytest.raises(SystemExit, match="smoke"):
        up.main("smoke", "nvfp4-awq", dry_run=True)


def test_upload_skips_missing_model(workspace, capsys):
    up.main("full", "nvfp4-awq", dry_run=True)
    assert "SKIP" in capsys.readouterr().out


def test_prep_then_dry_run_succeeds(workspace, capsys):
    """issue-5 §4: 43-prep 실행 후 49-upload-hf --dry-run이 성공해야 한다.

    READY가 안 나오는 경우가 많아야 하는 흔한 실수는 (a) dry-run이 네트워크를 건드림,
    (b) repo_id 규칙이 깨짐, (c) prep이 README를 다른 경로에 씀 — 셋 다 여기서 잡힌다.
    """
    _make_model_dir(workspace / "models", "raon-nvfp4-awq-full")
    _write_metrics(workspace, "full")

    # dry-run이 네트워크를 건드리면 즉시 실패시킨다
    import huggingface_hub

    def boom(*a, **k):
        raise AssertionError("dry-run이 네트워크를 호출했다")

    old = {k: getattr(huggingface_hub, k, None) for k in ("create_repo", "upload_folder", "HfApi")}
    try:
        for k in ("create_repo", "upload_folder", "HfApi"):
            setattr(huggingface_hub, k, boom)

        with pytest.raises(SystemExit, match="README.md"):
            up.main("full", "nvfp4-awq", dry_run=True)  # prep 전: README 없음
        assert prep.run("full", "nvfp4-awq", allow_missing_metrics=False)["ok"] is True

        readme = workspace / "models" / "raon-nvfp4-awq-full" / "README.md"
        assert readme.is_file(), "prep이 모델 디렉터리에 README를 쓰지 않음"

        up.main("full", "nvfp4-awq", dry_run=True)
    finally:
        for k, v in old.items():
            setattr(huggingface_hub, k, v)

    out = capsys.readouterr().out
    assert "READY" in out
    assert up.repo_name("nvfp4-awq", "full") in out
    assert "uploaded" not in out, "dry-run이 실제 업로드 했다"


def test_dry_run_covers_every_target(workspace, capsys):
    """READY 타겟은 보고하되, 디렉터리가 없는 타겟은 SKIP로 넘기고 rc가 깨지지 않는다."""
    _make_model_dir(workspace / "models", "raon-awq-w5-full", fmt=PACK)
    _write_metrics(workspace, "full")
    assert prep.run("full", "awq-w5", allow_missing_metrics=True)["ok"] is True
    up.main("full", "awq-w5,nvfp4-awq", dry_run=True)
    out = capsys.readouterr().out
    assert "READY" in out
    assert "SKIP nvfp4-awq" in out
    assert "uploaded" not in out


def test_prep_and_upload_accept_space_and_env_targets(workspace, monkeypatch, capsys):
    _make_model_dir(workspace / "models", "raon-awq-w5-full", fmt=PACK)
    _make_model_dir(workspace / "models", "raon-nvfp4-awq-full", fmt=NVFP4)
    _write_metrics(workspace, "full")
    monkeypatch.setenv("TARGETS", "awq-w5 nvfp4-awq")
    m = prep.run("full", None, allow_missing_metrics=True)
    assert m["ok"] is True
    assert set(m["targets"]) == {"awq-w5", "nvfp4-awq"}
    up.main("full", None, dry_run=True)
    out = capsys.readouterr().out
    assert "Raon-1-mini-preview-awq-w5" in out
    assert "Raon-1-mini-preview-nvfp4-awq" in out


# ---------------------------------------------------------------- 단독 실행


def _main() -> int:
    return run_standalone(globals())


if __name__ == "__main__":
    raise SystemExit(_main())