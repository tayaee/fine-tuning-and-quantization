"""tests/test_snippets.py — snippets/ 단일 소스 보장 테스트.

모델 카드의 실행 예제와 s6/s7 검증이 같은 파일에서 나와야 한다
(온라인 README를 긁어 다시 테스트할 필요가 없게):

  1. snippets/<target>/* 각 파일은 __REPO_ID__ 치환 후
     render_readme(target, mode) 출력에 verbatim으로 포함된다 (full/smoke).
  2. s7-01-serve-<target>-from-hf-with-vllm-fork.sh 5종은
     자기 타깃의 serve-vllm.sh 스니펫을 exec한다.
  3. s6-13-chat.sh는 app-transformers.py 스니펫을 실행한다.

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


prep = _load_script("s4-65-update-readme.py")

SNIPPET_TARGETS = ["awq-w3-transformers", "awq-w5-transformers",
                   "nvfp4-awq-transformers", "nvfp4-gptq-transformers",
                   "nvfp4-awq-vllm"]


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
        names = ["usage-transformers.sh", "app-transformers.py", "serve-vllm.sh"]
        if target == "nvfp4-awq-vllm":
            names.append("build-vllm-gb10.sh")
        for mode in ("full", "smoke"):
            repo = prep.hf_repo_id(target, mode)
            card = _render(target, mode)
            for name in names:
                body = (prep.SNIPPET_DIR / target / name).read_text(encoding="utf-8")
                body = body.removesuffix("\n").replace("__REPO_ID__", repo)
                assert body in card, f"{target}/{mode}: snippets/{name} 내용이 카드에 없음"


def test_snippet_placeholders_only_for_repo_id():
    for target in SNIPPET_TARGETS:
        for f in (prep.SNIPPET_DIR / target).iterdir():
            text = f.read_text(encoding="utf-8")
            assert "__REPO_ID__" not in text or f.name in ("serve-vllm.sh", "app-transformers.py"), \
                f"{target}/{f.name}: repo id 자리는 serve/app에만 허용"


def test_s7_wrappers_exec_serve_snippet():
    for target in SNIPPET_TARGETS:
        w = ROOT / f"s7-01-serve-{target}-from-hf-with-vllm-fork.sh"
        assert w.is_file(), f"{w.name} 없음"
        text = w.read_text(encoding="utf-8")
        assert f"snippets/{target}/serve-vllm.sh" in text, f"{w.name}가 스니펫을 실행하지 않음"
        assert "__REPO_ID__" in text, f"{w.name}에 플레이스홀더 치환 없음"


def test_s6_chat_runs_app_snippet():
    text = (ROOT / "s6-13-chat.sh").read_text(encoding="utf-8")
    assert "snippets/" in text and "app-transformers.py" in text, "s6-13이 app 스니펫을 참조하지 않음"
    assert "__REPO_ID__" in text, "s6-13에 플레이스홀더 치환 없음"


if __name__ == "__main__":
    raise SystemExit(run_standalone(dict(globals())))
