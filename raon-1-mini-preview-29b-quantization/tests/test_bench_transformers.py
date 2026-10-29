"""tests/test_bench_transformers.py — s4-28 transformers 인프로세스 벤치 단위 테스트.

합성 픽스처만 사용한다(모델 로드·GPU·네트워크 없음).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sdlc
from tests._compat import run_standalone


def _load_script(name: str):
    mod_name = name.replace("-", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / name)
    assert spec is not None and spec.loader is not None, f"{name} 로드 실패"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


bt = _load_script("s4-28-measure-speed-with-transformers.py")


def _isolate(d: str) -> None:
    bt.SHARED = Path(d)
    sdlc.SHARED = Path(d)


def test_parse_targets():
    assert bt.parse_targets("all") == ["base", "awq-w3-transformers", "awq-w5-transformers", "nvfp4-awq-transformers", "nvfp4-gptq-transformers"]
    assert bt.parse_targets("base,nvfp4-awq-transformers") == ["base", "nvfp4-awq-transformers"]
    assert bt.parse_targets("nvfp4-awq-transformers") == ["nvfp4-awq-transformers"]
    try:
        bt.parse_targets("nope")
    except SystemExit:
        pass
    else:
        raise AssertionError("unknown target이 SystemExit을 내지 않음")


def test_resolve_target():
    with tempfile.TemporaryDirectory() as d:
        _isolate(d)
        assert bt.resolve_target("base", "full") == Path(d) / "models" / "raon-bf16"
        assert bt.resolve_target("nvfp4-awq-transformers", "full") == Path(d) / "models" / "raon-nvfp4-awq-transformers-full"


def test_summarize_stats():
    per_req = [
        {"wall_s": 2.0, "output_tokens": 100, "tok_per_s": 50.0},
        {"wall_s": 4.0, "output_tokens": 100, "tok_per_s": 25.0},
    ]
    s = bt.summarize(per_req)
    assert s["wall_s_total"] == 6.0
    assert s["output_tokens_total"] == 200
    assert s["tok_per_s_p50"] == 37.5
    assert s["req_latency_s_p50"] == 3.0
    assert s["req_latency_s_max"] == 4.0


def test_merge_preserves_history():
    with tempfile.TemporaryDirectory() as d:
        _isolate(d)
        outdir = Path(d) / "outputs" / "eval-smoke"
        outdir.mkdir(parents=True)
        prior = {"servebench": {"ok": True,
                                "targets": {"base": {"1": {"tok_per_s_p50": 42.0}}},
                                "hardware": {"backend": "transformers"}}}
        (outdir / "checks.json").write_text(json.dumps(prior), encoding="utf-8")
        measured = {"run": {"n": 2, "max_tokens": 16, "tok_per_s_p50": 90.0,
                            "tok_per_s_mean": 91.0, "req_latency_s_p50": 1.1,
                            "req_latency_s_max": 1.2, "wall_s_total": 2.2,
                            "output_tokens_total": 200, "backend": "transformers",
                            "model": "x", "measured_at": "t"},
                    "memory": {"before_load": {"mem_avail_gb": 1.0, "server_rss_gb": 0.0}}}
        payload = bt.merge_target("smoke", "nvfp4-awq-transformers", measured, {"backend": "transformers"})
        assert payload["targets"]["base"] == {"1": {"tok_per_s_p50": 42.0}}
        assert payload["targets"]["nvfp4-awq-transformers"]["1"]["tok_per_s_p50"] == 90.0
        assert payload["targets"]["nvfp4-awq-transformers"]["backend"] == "transformers"
        bench = json.loads((outdir / "bench-transformers-smoke.json").read_text(encoding="utf-8"))
        assert bench["targets"]["nvfp4-awq-transformers"]["1"]["tok_per_s_p50"] == 90.0


def test_main_skips_missing_models_without_gpu():
    """모델 디렉터리가 없으면 GPU 없이 skipped로 기록하고 종료한다."""
    with tempfile.TemporaryDirectory() as d:
        _isolate(d)
        out = bt.main("smoke", ["base", "nvfp4-awq-transformers"], 2, 16)
        assert sorted(out["skipped"]) == ["base", "nvfp4-awq-transformers"]
        assert out["targets"] == {} and out["errors"] == {}
        assert "backend" in out["hardware"]


def test_ensure_base_first_auto_adds_base():
    """양자화만 요청 + base 기록 없음 → base 선행."""
    with tempfile.TemporaryDirectory() as d:
        _isolate(d)
        assert bt.ensure_base_first("smoke", ["nvfp4-awq-transformers"]) == ["base", "nvfp4-awq-transformers"]


def test_ensure_base_first_keeps_explicit_base():
    with tempfile.TemporaryDirectory() as d:
        _isolate(d)
        assert bt.ensure_base_first("smoke", ["base", "nvfp4-awq-transformers"]) == ["base", "nvfp4-awq-transformers"]


def test_ensure_base_first_skips_when_recorded():
    """base 기록이 이미 있으면 재측정하지 않는다."""
    with tempfile.TemporaryDirectory() as d:
        _isolate(d)
        outdir = Path(d) / "outputs" / "eval-smoke"
        outdir.mkdir(parents=True)
        prior = {"servebench": {"targets": {"base": {"1": {"tok_per_s_p50": 42.0}}}}}
        (outdir / "checks.json").write_text(json.dumps(prior), encoding="utf-8")
        assert bt.ensure_base_first("smoke", ["nvfp4-awq-transformers"]) == ["nvfp4-awq-transformers"]


def test_ensure_base_first_base_only_untouched():
    with tempfile.TemporaryDirectory() as d:
        _isolate(d)
        assert bt.ensure_base_first("smoke", ["base"]) == ["base"]


def test_hardware_keys():
    hw = bt.hardware_info()
    for k in ("gpu", "unified_mem_gb", "backend", "engine",
              "tensor_parallel", "max_model_len", "quantization"):
        assert k in hw, k
    assert hw["backend"] == "transformers"


def test_prompts_match_servebench():
    """s4-28과 s4-25는 같은 고정 8프롬프트를 쓴다 (복사본 드리프트 방지)."""
    sb = _load_script("s4-25-servebench.py")
    assert list(bt.PROMPTS) == list(sb.PROMPTS), "벤치 프롬프트 불일치 (입력 데이터 통일 깨짐)"
    assert len(bt.PROMPTS) == 8


if __name__ == "__main__":
    raise SystemExit(run_standalone(dict(globals())))
