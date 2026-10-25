"""tests/test_ppl_and_eval.py — issue-4 (s6-34-measure-ppl / s7-37-eval / s7-40-score) 단위 테스트.

pytest가 없는 환경에서도 단독 실행된다:
    .venv/bin/python tests/test_ppl_and_eval.py
(regression-tests/verify-issue-4.sh가 이 경로를 쓴다)

모델을 실제로 로드하지 않고 seams만 검증한다:
  - ppl: 타겟 해석/저장/summary 집계 (합성 토크나이저·모델 픽스처)
  - eval: 백엔드 선택/fallback/run_transformers 합성 모델 (합성 토크나이저)
  - score: 벤치마크별 정확도 산출
"""

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

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


ppl = _load_script("s6-34-measure-ppl.py")
ev = _load_script("s7-37-eval.py")
sc = _load_script("s7-40-score.py")
BENCHES = ("kmmlu_redux", "gsm8k", "humaneval")


@pytest.fixture
def tmp_out(monkeypatch):
    d = Path(tempfile.mkdtemp(prefix="rq-out-"))
    monkeypatch.setattr(ppl, "SHARED", str(d))
    monkeypatch.setattr(ppl, "OUTDIR", lambda mode: str(d / f"ppl-{mode}"))
    monkeypatch.setattr(ev, "SHARED", str(d))
    monkeypatch.setattr(ev, "OUTDIR", lambda mode: str(d / f"eval-{mode}"))
    monkeypatch.setattr(sc, "SHARED", str(d))
    monkeypatch.setattr(sc, "OUTDIR", lambda mode: str(d / f"eval-{mode}"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ================================================================ 34-measure-ppl


def test_parse_targets_accepts_comma_and_space(tmp_out, monkeypatch):
    monkeypatch.delenv("TARGETS", raising=False)
    assert ppl.parse_targets("nvfp4-awq") == ["nvfp4-awq"]
    assert ppl.parse_targets("awq-w5,nvfp4-gptq") == ["awq-w5", "nvfp4-gptq"]
    assert ppl.parse_targets("awq-w5 nvfp4-gptq") == ["awq-w5", "nvfp4-gptq"]
    assert ppl.parse_targets("all") == ppl.ALL_TARGETS


def test_parse_targets_defaults_from_env(tmp_out, monkeypatch):
    monkeypatch.setenv("TARGETS", "awq-w3,nvfp4-awq")
    assert ppl.parse_targets(None) == ["awq-w3", "nvfp4-awq"]
    monkeypatch.delenv("TARGETS")
    assert ppl.parse_targets(None) == ppl.ALL_TARGETS


def test_parse_targets_normalizes_bf16_alias(tmp_out, monkeypatch):
    """이슈 기본값은 bf16, 기존 산출물 파일명은 base. 둘 다 받아야 한다."""
    monkeypatch.delenv("TARGETS", raising=False)
    assert ppl.parse_targets("bf16") == ["base"]
    assert ppl.parse_targets("base,bf16") == ["base"]
    assert "base" in ppl.parse_targets("bf16,awq-w5")


def test_resolve_paths(tmp_out, monkeypatch):
    monkeypatch.setattr(ppl, "MODELS_DIR", lambda mode: str(tmp_out / "models"))
    assert ppl.resolve("base", "smoke").endswith("models/raon-bf16")
    assert ppl.resolve("nvfp4-awq", "full").endswith("models/raon-nvfp4-awq-full")


def test_measure_writes_per_target_json_and_summary(tmp_out, monkeypatch):
    calls = []

    def fake_ppl(model_name, texts, seq):
        calls.append(model_name)
        return 10.0 + 0.5 * (len(calls) - 1)

    monkeypatch.setattr(ppl, "ppl_of", fake_ppl)
    monkeypatch.setattr(ppl, "load_calib_texts", lambda n, *a, **k: ["a text"] * n)
    monkeypatch.setattr(ppl, "model_exists", lambda name: True)

    outdir = tmp_out / "ppl-smoke"
    ppl.measure_targets("smoke", 2, "base,awq-w5", seq=512, outdir=outdir)

    assert (outdir / "base.json").is_file()
    assert (outdir / "awq-w5.json").is_file()
    summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
    assert summary["base"]["ppl"] == 10.0
    assert summary["base"]["verdict"] == "ref"
    assert summary["awq-w5"]["ppl"] == 10.5
    assert summary["awq-w5"]["delta"] == pytest.approx(0.5)
    assert summary["awq-w5"]["verdict"] == "Conditional"


def test_summarize_uses_cached_base_when_not_measured(tmp_out, monkeypatch):
    """bf16을 측정하지 않아도 캐시된 base.json을 기준으로 판정한다."""
    outdir = tmp_out / "ppl-smoke"
    outdir.mkdir(parents=True)
    (outdir / "base.json").write_text(json.dumps({"target": "base", "ppl": 12.0, "n": 8}), encoding="utf-8")
    seen = []
    monkeypatch.setattr(ppl, "ppl_of", lambda *a, **k: seen.append(a) or 12.5)
    monkeypatch.setattr(ppl, "load_calib_texts", lambda n, *a, **k: ["t"] * n)
    monkeypatch.setattr(ppl, "model_exists", lambda name: True)

    ppl.measure_targets("smoke", 2, "awq-w5", seq=512, outdir=outdir)

    assert not (outdir / "base.json").exists() or json.loads((outdir / "base.json").read_text())["ppl"] == 12.0
    summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
    assert summary["awq-w5"]["delta"] == pytest.approx(0.5)
    assert summary["awq-w5"]["verdict"] == "Conditional"
    assert summary["awq-w5"]["ref_source"] == "cached"


def test_measure_releases_memory_between_targets(tmp_out, monkeypatch):
    monkeypatch.setattr(ppl, "load_calib_texts", lambda n, *a, **k: ["t"] * n)
    monkeypatch.setattr(ppl, "model_exists", lambda name: True)
    monkeypatch.setattr(ppl, "ppl_of", lambda *a, **k: 12.0)
    released = []
    monkeypatch.setattr(ppl, "release_memory", lambda: released.append(1))
    ppl.measure_targets("smoke", 1, "awq-w5,awq-w3", seq=8, outdir=tmp_out / "ppl-smoke")
    assert len(released) == 2


def test_summarize_only_does_not_load_models(tmp_out, monkeypatch):
    outdir = tmp_out / "ppl-full"
    outdir.mkdir(parents=True)
    (outdir / "base.json").write_text(json.dumps({"target": "base", "ppl": 9.0}), encoding="utf-8")
    (outdir / "awq-w5.json").write_text(json.dumps({"target": "awq-w5", "ppl": 9.1}), encoding="utf-8")

    def boom(*a, **k):
        raise AssertionError("summarize_only가 모델을 로드했다")

    monkeypatch.setattr(ppl, "ppl_of", boom)
    ppl.summarize(outdir)
    summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) == {"base", "awq-w5"}
    assert summary["awq-w5"]["verdict"] == "Accept"


# ================================================================ 37-eval


class _FakeTokenizer:
    """AutoTokenizer.from_pretrained 대체 — callable + decode 둘 다 지원.

    배치(padding=True)면 좌측 패딩된 input_ids + attention_mask를 돌려준다.
    """

    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "right"

    def __call__(self, text, **kw):
        texts = [text] if isinstance(text, str) else list(text)
        seqs = [[1] * max(1, len(t.split())) for t in texts]
        width = max(len(sq) for sq in seqs)
        ids = torch.zeros((len(seqs), width), dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i2, sq in enumerate(seqs):
            ids[i2, width - len(sq) :] = torch.tensor(sq, dtype=torch.long)
            mask[i2, width - len(sq) :] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, ids, **kw):
        return " 생성된답변" if len(ids) else ""


class _FakeModel:
    """최소 CausalLM: eval() 필수, generate()는 입력 길이 + k 토큰을 돌려준다."""

    device = "cpu"

    def __init__(self):
        self.eval_called = False
        self.calls: list[dict] = []

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, input_ids, **kw):
        assert self.eval_called, "eval()을 먼저 호출해야 한다 (생성 모드)"
        self.calls.append({"shape": tuple(input_ids.shape), "attention_mask": kw.get("attention_mask")})
        k = int(kw.get("max_new_tokens", 1))
        b = input_ids.size(0)
        return torch.cat([input_ids, torch.full((b, k), 2, dtype=input_ids.dtype)], dim=1)


@pytest.fixture
def fake_transformers(monkeypatch):
    """transformers.AutoModelForCausalLM/AutoTokenizer.from_pretrained을 합성 구현으로."""
    seen: dict = {}

    def fake_model(path, **kw):
        seen["model_kwargs"] = kw
        seen["path"] = path
        m = _FakeModel()
        seen["model_obj"] = m
        seen["model_calls"] = m.calls
        return m

    def fake_tok(path, **kw):
        seen["tok_kwargs"] = kw
        return _FakeTokenizer()

    import transformers

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_model))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(fake_tok))
    ev.AutoModelForCausalLM = None  # pyright: ignore[reportAttributeAccessIssue] — 지연 import 캐시 초기화
    ev.AutoTokenizer = None  # pyright: ignore[reportAttributeAccessIssue] — 지연 import 캐시 초기화
    return seen


def test_run_transformers_generates_only_new_tokens(tmp_out, fake_transformers):
    outs = ev.run_transformers("/models/raon-bf16", ["짧은 프롬프트 두 개"], max_tokens=5)
    assert outs == [" 생성된답변"]
    kw = fake_transformers["model_kwargs"]
    assert kw["trust_remote_code"] is True
    assert kw["device_map"] == "auto"
    assert kw["torch_dtype"] is torch.bfloat16
    assert fake_transformers["tok_kwargs"]["trust_remote_code"] is True


def test_backend_auto_falls_back_when_vllm_unsupported(tmp_out, monkeypatch, fake_transformers):
    monkeypatch.setattr(ev, "run_transformers", lambda *a, **k: ["폴백출력"])
    monkeypatch.setattr(
        ev,
        "run_vllm",
        lambda *a, **k: (_ for _ in ()).throw(ev.unsupported_vllm_error("Raon1MiniForCausalLM")),
    )
    calls = []
    got = ev.generate("base", "/models/x", ["p"], max_tokens=8, backend="auto", on_fallback=lambda w: calls.append(w))
    assert got == ["폴백출력"]
    assert calls, "auto 폴백이 발생했음을 기록해야 한다"


def test_backend_auto_does_not_fall_back_on_real_errors(tmp_out, monkeypatch):
    monkeypatch.setattr(ev, "run_vllm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")))
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        ev.generate("base", "/models/x", ["p"], max_tokens=8, backend="auto")


def test_backend_explicit_vllm_raises_unsupported(tmp_out, monkeypatch):
    monkeypatch.setattr(ev, "run_vllm", lambda *a, **k: (_ for _ in ()).throw(ev.unsupported_vllm_error("X")))
    with pytest.raises(RuntimeError, match="unsupported"):
        ev.generate("base", "/models/x", ["p"], max_tokens=8, backend="vllm")


def test_backend_auto_prefers_vllm_when_supported(tmp_out, monkeypatch):
    monkeypatch.setattr(ev, "run_vllm", lambda *a, **k: ["vllm출력"])
    monkeypatch.setattr(ev, "run_transformers", lambda *a, **k: ["transformers출력"])
    assert ev.generate("base", "/models/x", ["p"], max_tokens=8, backend="auto") == ["vllm출력"]


def test_backend_transformers_skips_vllm(tmp_out, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("transformers 백엔드에서 vllm을 호출했다")

    monkeypatch.setattr(ev, "run_vllm", boom)
    monkeypatch.setattr(ev, "run_transformers", lambda *a, **k: ["hf출력"])
    assert ev.generate("base", "/models/x", ["p"], max_tokens=8, backend="transformers") == ["hf출력"]


def test_unsupported_detection_is_narrow(tmp_out):
    assert ev.is_unsupported_vllm_error(ValueError("Model architecture Raon1MiniForCausalLM not supported"))
    assert ev.is_unsupported_vllm_error(
        ValueError("Transformers modeling backend for MLA models requires transformers>=5.15.0.dev0")
    )
    assert ev.is_unsupported_vllm_error(
        RuntimeError("Engine core initialization failed. See root cause above. Failed core proc(s): {}")
    )
    assert not ev.is_unsupported_vllm_error(RuntimeError("CUDA out of memory"))
    assert not ev.is_unsupported_vllm_error(RuntimeError("NCCL error"))


def test_check_vllm_arch_supported_rejects_unregistered_arch(tmp_out):
    mdir = tmp_out / "models" / "raon-bf16"
    mdir.mkdir(parents=True)
    (mdir / "config.json").write_text(
        json.dumps({"architectures": ["Raon1MiniForCausalLM"]}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="unsupported architecture"):
        ev._check_vllm_arch_supported(str(mdir))


def test_load_prompts_slices_per_benchmark(tmp_out):
    outdir = Path(ev.OUTDIR("smoke"))
    outdir.mkdir(parents=True)
    for b in BENCHES:
        lines = [json.dumps({"prompt": f"{b}-{i}", "gold": "A"}) for i in range(5)]
        (outdir / f"prompts-{b}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = ev.load_prompts("smoke", 2, None, list(BENCHES))
    assert len(rows) == 6
    for b in BENCHES:
        assert sum(1 for r in rows if r["bench"] == b) == 2


def test_load_prompts_falls_back_to_calib_file_only_for_pseudo_bench(tmp_out):
    """표준 벤치마크는 calib 폴백을 쓰지 않는다(레거시 의사 벤치 'smoke'만 허용)."""
    calib = tmp_out / "calib.jsonl"
    lines = [
        json.dumps(
            {
                "text": f"user\nq-{i}\nassistant\n<think></think>\n\nans-{i}\n"
            }
        )
        for i in range(8)
    ]
    calib.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = ev.load_prompts("smoke", 3, str(calib), ["smoke"])
    assert len(rows) == 3
    assert rows[0]["prompt"].endswith("assistant\n<think></think>\n\n")
    assert "ans-" not in rows[0]["prompt"]


def test_load_prompts_fails_when_standard_prompts_missing(tmp_out):
    """표준 GSM8K 프롬프트 파일이 없으면 실패한다(임의 표본 평가 금지)."""
    with pytest.raises(SystemExit, match="36-prep-evalset"):
        ev.load_prompts("smoke", 3, None, ["gsm8k"])


def test_run_transformers_batches_with_left_padding(tmp_out, fake_transformers):
    """배치 생성이 실제로 묶어서 호출되고 attention_mask가 전달된다."""
    outs = ev.run_transformers("/models/raon-bf16", ["짧은", "조금 더 긴 프롬프트", "세 번째"], max_tokens=5, batch_size=3)
    assert len(outs) == 3
    assert fake_transformers["model_calls"], "generate 호출 기록이 남아야 한다"
    call = fake_transformers["model_calls"][-1]
    assert call["shape"][0] == 3, "3개 프롬프트가 한 배치로 호출되어야 한다"
    assert call["attention_mask"] is not None, "attention_mask가 전달되어야 한다"


def test_eval_writes_per_target_jsonl(tmp_out, monkeypatch):
    monkeypatch.setattr(ev, "generate", lambda *a, **k: ["out-1", "out-2"])
    monkeypatch.setattr(ev, "resolve", lambda target, mode: str(tmp_out / "models" / target))
    (tmp_out / "models" / "base").mkdir(parents=True)
    (tmp_out / "models" / "base" / "config.json").write_text("{}", encoding="utf-8")
    prompts = [{"bench": "kmmlu_redux", "prompt": "q1", "gold": "A"},
               {"bench": "gsm8k", "prompt": "q2", "gold": "42"}]
    outdir = Path(ev.OUTDIR("smoke"))
    ev.run_target("smoke", "base", "auto", prompts, max_tokens=16)
    rows = [json.loads(l) for l in (outdir / "base.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 2
    assert rows[0]["bench"] == "kmmlu_redux" and rows[0]["output"] == "out-1"
    assert rows[1]["gold"] == "42"
    assert (outdir / "bench" / "gsm8k" / "base.jsonl").is_file()


# ================================================================ end-to-end 체인 (issue-4 §4)


def test_run_transformers_to_score_json_chain(tmp_out, monkeypatch, fake_transformers):
    """합성 CausalLM → run_transformers → 벤치 JSONL → s7-40-score → score.json.

    issue-4 §4의 요구를 그대로: 벤치마크 JSONL이 생기고, 그것을 40-score가 읽어
    산출물의 정확한 Bench 이름으로 score.json에 정확도를 남기는지까지 확인한다.
    """
    mdir = tmp_out / "models" / "base"
    mdir.mkdir(parents=True)
    (mdir / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ev, "resolve", lambda target, mode: str(mdir))

    prompts = [
        {"bench": "kmmlu_redux", "prompt": "capital of France", "gold": "A"},
        {"bench": "gsm8k", "prompt": "1+1", "gold": "42"},
        {"bench": "humaneval", "prompt": "sum a list",
         "gold": {"test": "def check(candidate):\n    assert candidate([1, 2, 3]) == 6\n",
                  "entry_point": "total"}},
    ]
    # 가짜 모델이 벤치마다 정답을 뱉게 만든다: 셋 다 정답이어야 acc=1.0
    monkeypatch.setattr(ev, "run_transformers", lambda *a, **k: ["정답은 A 입니다", "결과는 #### 42",
                                                                 "```python\ndef total(xs):\n    return sum(xs)\n```"])
    monkeypatch.setattr(ev, "generate", lambda target, path, texts, **k: ev.run_transformers(path, texts, 8))

    ev.run_target("smoke", "base", "auto", prompts, max_tokens=8)

    outdir = Path(sc.OUTDIR("smoke"))
    rows = [json.loads(l) for l in (outdir / "base.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert {r["bench"] for r in rows} == set(BENCHES), "벤치 JSONL 3종이 생성되지 않음"
    for b in BENCHES:
        assert (outdir / "bench" / b / "base.jsonl").is_file(), f"bench/{b}/base.jsonl 누락"

    scored = sc.score("smoke")
    assert scored["base"]["n"] == 3
    assert set(scored["base"]["benchmarks"]) == set(BENCHES)
    for b in BENCHES:
        assert scored["base"]["benchmarks"][b] == {"n": 1, "acc": 1.0}, f"{b} 채점 실패"
    saved = json.loads((outdir / "score.json").read_text(encoding="utf-8"))
    assert saved["base"]["benchmarks"]["humaneval"]["acc"] == 1.0


def test_auto_fallback_to_transformers_chain(tmp_out, monkeypatch, fake_transformers):
    """auto 폴백 경로로도 같은 JSONL이 생성돼야 한다 (vLLM 미지원 → transformers)."""
    mdir = tmp_out / "models" / "base"
    mdir.mkdir(parents=True)
    (mdir / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ev, "resolve", lambda target, mode: str(mdir))
    monkeypatch.setattr(
        ev, "run_vllm",
        lambda *a, **k: (_ for _ in ()).throw(ev.unsupported_vllm_error("Raon1MiniForCausalLM")),
    )
    prompts = [{"bench": "kmmlu_redux", "prompt": "q", "gold": "B"}]
    ev.run_target("smoke", "base", "auto", prompts, max_tokens=4)
    outdir = Path(ev.OUTDIR("smoke"))
    rows = [json.loads(l) for l in (outdir / "base.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1 and rows[0]["bench"] == "kmmlu_redux"
    # 가짜 모델은 빈 문자열을 돌려준다 → 정답 아님, 하지만 JSONL은 만들어져야 한다
    assert sc.score("smoke")["base"]["n"] == 1


# ================================================================ 40-score


def test_score_answer_kmmlu():
    assert sc.score_answer("kmmlu_redux", "A", "정답은 A 입니다") == 1
    assert sc.score_answer("kmmlu_redux", "A", "정답은 B 입니다") == 0
    assert sc.score_answer("kmmlu_redux", "A", "<think>..</think> 최종 답: C") == 0
    assert sc.score_answer("kmmlu_redux", ["B", "C"], "답: C") == 1
    assert sc.score_answer("kmmlu_redux", "A", "A. 두 번째 보기") == 1


def test_score_answer_gsm8k():
    assert sc.score_answer("gsm8k", "42", "계산하면 #### 42 가 나온다") == 1
    assert sc.score_answer("gsm8k", "42", "결과는 41") == 0


def test_score_answer_humaneval_executes_test():
    gold = {"test": "def check(candidate):\n    assert candidate([1, 2, 3]) == 6\n", "entry_point": "total"}
    good = "```python\ndef total(xs):\n    return sum(xs)\n```"
    bad = "```python\ndef total(xs):\n    return 0\n```"
    assert sc.score_answer("humaneval", gold, good) == 1
    assert sc.score_answer("humaneval", gold, bad) == 0
    assert sc.score_answer("humaneval", gold, "```python\ndef total(xs)\n  return sum(xs)\n```") == 0


def test_score_writes_score_json_with_benchmarks(tmp_out):
    outdir = Path(sc.OUTDIR("smoke"))
    outdir.mkdir(parents=True)
    (outdir / "base.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"bench": "kmmlu_redux", "prompt": "q", "gold": "A", "output": "답: A"}),
                json.dumps({"bench": "gsm8k", "prompt": "q", "gold": "42", "output": "#### 42"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = sc.score("smoke")
    assert rows["base"]["n"] == 2
    assert rows["base"]["done_rate"] == 1.0
    assert rows["base"]["benchmarks"]["kmmlu_redux"]["acc"] == 1.0
    assert rows["base"]["benchmarks"]["gsm8k"]["acc"] == 1.0
    score = json.loads((outdir / "score.json").read_text(encoding="utf-8"))
    assert score["base"]["benchmarks"]["kmmlu_redux"]["n"] == 1


def test_score_ignores_bench_subdirectory(tmp_out):
    outdir = Path(sc.OUTDIR("smoke"))
    (outdir / "bench").mkdir(parents=True)
    (outdir / "bench" / "gsm8k").mkdir(parents=True)
    (outdir / "base.jsonl").write_text(json.dumps({"prompt": "q", "output": "x"}) + "\n", encoding="utf-8")
    (outdir / "prompts-gsm8k.jsonl").write_text(json.dumps({"prompt": "q", "gold": "42"}) + "\n", encoding="utf-8")
    (outdir / "bench" / "gsm8k" / "base.jsonl").write_text(json.dumps({"prompt": "q", "output": "y"}) + "\n", encoding="utf-8")
    rows = sc.score("smoke")
    assert set(rows) == {"base"}
    assert rows["base"]["n"] == 1


# ================================================================ 단독 실행


def _main() -> int:
    return run_standalone(globals())


if __name__ == "__main__":
    raise SystemExit(_main())