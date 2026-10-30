"""tests/test_snippets.py — snippets/ 단일 소스 보장 테스트.

모델 카드의 실행 예제와 s6/s7 검증이 같은 파일에서 나와야 한다
(온라인 README를 긁어 다시 테스트할 필요가 없게):

  1. snippets/<target>/* 각 파일은 __REPO_ID__ 치환 후
     render_readme(target, mode) 출력에 verbatim으로 포함된다 (full/smoke).
  2. s7-02-verify-readme-with-<target>-and-vllm-fork.sh 5종은
     자기 타깃의 serve-vllm.sh 스니펫을 exec한다.
  3. s6-02 검증(config/s6-customer.inc)은 Hub 카드의 블록을 스니펫과 바이트 비교한다.

pytest가 없는 환경에서도 단독 실행된다:
    .venv/bin/python tests/test_snippets.py
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests._compat import run_standalone


def _load_script(name: str):
    mod_name = name.replace("-", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / name)
    assert spec is not None and spec.loader is not None, f"{name} 로드 실패"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


prep = _load_script("s4-70-update-readme.py")

SNIPPET_TARGETS = ["awq-w3-transformers", "awq-w5-transformers",
                   "nvfp4-awq-transformers", "nvfp4-gptq-transformers",
                   "nvfp4-awq-vllm"]
UNLOADABLE_TARGETS = ["nvfp4-awq-vllm"]
S7_TARGET_NUM = {
    "awq-w3-transformers": "05",
    "awq-w5-transformers": "10",
    "nvfp4-awq-transformers": "15",
    "nvfp4-awq-vllm": "20",
    "nvfp4-gptq-transformers": "25",
}


def _render(target: str, mode: str) -> str:
    with tempfile.TemporaryDirectory() as d:
        old = prep.SHARED
        prep.SHARED = d  # _checks/_evalset_meta는 파일 없으면 기본값
        try:
            return prep.render_readme(target, mode, {}, {}, 20.0)
        finally:
            prep.SHARED = old


def test_snippets_embedded_verbatim_in_cards():
    for target in SNIPPET_TARGETS:
        # vLLM 전용 타깃(modelopt_fp4)은 transformers 예제를 카드에 싣지 않는다 →
        # 해당 스니펫 파일도 존재하지 않는다.
        names = ["usage-transformers.sh", "app-transformers.py", "serve-vllm.sh"]
        if target in UNLOADABLE_TARGETS:
            names.remove("usage-transformers.sh")
            names.remove("app-transformers.py")
            names.append("build-vllm-gb10.sh")
        for mode in ("full", "smoke"):
            repo = prep.hf_repo_id(target, mode)
            card = _render(target, mode)
            for name in names:
                body = (prep.SNIPPET_DIR / target / name).read_text(encoding="utf-8")
                body = body.removesuffix("\n").replace("__REPO_ID__", repo)
                assert body in card, f"{target}/{mode}: snippets/{name} 내용이 카드에 없음"


def test_unloadable_target_has_no_transformers_snippets():
    """vLLM 전용 타깃에는 transformers 스니펫이 없어야 한다 (있으면 카드 예제와 어긋난다)."""
    for target in UNLOADABLE_TARGETS:
        for name in ("usage-transformers.sh", "app-transformers.py"):
            p = prep.SNIPPET_DIR / target / name
            assert not p.exists(), f"{target}/{name} 이 남아 있다 — 카드는 transformers 예제를 싣지 않는다"


def test_snippet_placeholders_only_for_repo_id():
    for target in SNIPPET_TARGETS:
        for f in (prep.SNIPPET_DIR / target).iterdir():
            text = f.read_text(encoding="utf-8")
            assert "__REPO_ID__" not in text or f.name in ("serve-vllm.sh", "app-transformers.py"), \
                f"{target}/{f.name}: repo id 자리는 serve/app에만 허용"


def test_s7_wrappers_exec_serve_snippet():
    """s7 래퍼는 자기 타깃의 serve-vllm.sh 스니펫을 카드 명령 그대로 실행한다."""
    inc = (ROOT / "config" / "s7-serve.inc").read_text(encoding="utf-8")
    assert "serve-vllm.sh" in inc and "__REPO_ID__" in inc, "s7-serve.inc가 카드 명령을 실행하지 않음"
    for target in SNIPPET_TARGETS:
        w = ROOT / f"s7-{S7_TARGET_NUM[target]}-verify-readme-with-{target}-and-vllm-fork.sh"
        assert w.is_file(), f"{w.name} 없음"
        text = w.read_text(encoding="utf-8")
        assert f's7_serve_frozen "{target}"' in text, f"{w.name}가 자기 타깃 스니펫을 지정하지 않음"
        assert "s7-serve.inc" in text, f"{w.name}가 공용 헬퍼를 source 하지 않음"


def test_s7_wrappers_accept_mode():
    """s7 래퍼는 mode 를 받아 repo id 를 맞춰야 한다.

    mode 를 고정하면 smoke 검증에서 full 모델을 서빙해버린다 (조용히 잘못 검증).
    """
    for target in SNIPPET_TARGETS:
        w = ROOT / f"s7-{S7_TARGET_NUM[target]}-verify-readme-with-{target}-and-vllm-fork.sh"
        text = w.read_text(encoding="utf-8")
        assert 's7_serve_frozen "' in text, f"{w.name}가 공용 헬퍼를 쓰지 않음"
        assert '"$MODE"' in text, f"{w.name}가 mode 를 넘기지 않음"


def test_s7_02_runs_card_blocks_including_build():
    """s7-02 는 카드의 '## Usage - vLLM' 블록을 **순서대로** 실행한다.

    nvfp4-awq-vllm 카드는 Step 1(포크 소스 빌드) + Step 2(서빙) 두 블록이다 —
    빌드를 별도 스텝으로 떼어내지 않고 카드 순서 그대로 돌려야 'as-is' 검증이 된다.
    """
    inc = (ROOT / "config" / "s7-serve.inc").read_text(encoding="utf-8")
    assert "s7_run_card_blocks" in inc, "s7-02가 카드 블록 실행 경로가 없다"
    assert "build-vllm-gb10.sh" in inc, "s7-02가 빌드 스니펫을 카드 대조에 포함하지 않음"
    assert not (ROOT / "s7-04-build-vllm-fork-gb10.sh").exists(), \
        "빌드가 별도 스텝으로 남아 있다 — 카드 블록에 이미 포함되어 있다"


def test_s7_serve_inc_runs_card_blocks():
    """s7-02 는 Hub 카드에서 뽑은 블록을 **순서대로** 실행한다 (as-is 검증의 본체)."""
    text = (ROOT / "config" / "s7-serve.inc").read_text(encoding="utf-8")
    assert "s7_verify_card" in text and "s7_run_card_blocks" in text
    assert "build-vllm-gb10.sh" in text, "s7-02가 빌드 스니펫을 카드 대조에 포함하지 않음"
    assert "readme-blocks.py" in text, "s7-02가 카드 블록을 바이트 대조하지 않음"


def test_s6_readme_verify_uses_snippets():
    """s6-02 검증은 Hub 카드 블록을 snippets/ 와 바이트 비교한다 (as-is 검증)."""
    text = (ROOT / "config" / "s6-customer.inc").read_text(encoding="utf-8")
    assert "usage-transformers.sh" in text, "s6 검증이 usage 스니펫을 참조하지 않음"
    assert "app-transformers.py" in text, "s6 검증이 app 스니펫을 참조하지 않음"
    assert "readme-blocks.py" in text, "s6 검증이 readme 블록 추출기를 쓰지 않음"
    assert (ROOT / "config" / "readme-blocks.py").is_file(), "readme-blocks.py 없음"


def test_transformers_unloadable_target_documents_conversion_path():
    """transformers 적재 불가 타깃(modelopt_fp4)의 카드도 transformers 경로를 안내한다.

    modelopt_fp4는 transformers 적재 불가이므로 실행 예제(코드 블록)는 싣지 않고,
    동봉된 nvfp4_to_bf16.py 로 변환 가능하다는 안내만 담는다.
    """
    card = _render("nvfp4-awq-vllm", "full")
    sec = card.split("## Usage - transformers", 1)[1].split("\n## ", 1)[0]
    assert "```" not in sec, "modelopt_fp4 카드에 transformers 실행 예제가 없어야 한다 (vLLM 전용)"
    assert "nvfp4_to_bf16.py" in sec, "modelopt_fp4 카드가 BF16 변환 도구를 안내하지 않는다"
    assert "modelopt_fp4" in sec, "modelopt_fp4 카드가 왜 변환이 필요한지 설명하지 않는다"
    # 다른 타깃은 transformers 예제가 있어야 한다 (반대 케이스도 함께 고정)
    for t in ("nvfp4-awq-transformers", "awq-w3-transformers", "awq-w5-transformers",
              "nvfp4-gptq-transformers"):
        c = _render(t, "full")
        assert "## Usage - transformers" in c and "```python" in c, f"{t}: transformers 예제 누락"


if __name__ == "__main__":
    raise SystemExit(run_standalone(dict(globals())))
