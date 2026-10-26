"""tests/test_serve_command.py — README vLLM 명령 ≡ s8-52-serve-vllm.sh 실행 명령 동등성.

s8-52는 README 모델카드의 vLLM 명령을 as-is로 실행해 검증한다. 양쪽이 어긋나면
"카드대로 했는데 안 돈다"가 된다. 4종 양자화 타깃은 repo id를 제외하고 동일 명령이다.
"""

from __future__ import annotations

import importlib.util
import re
import sys
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


prep = _load_script("s7-56-prep-upload.py")

QUANT_TARGETS = ["awq-w3", "awq-w5", "nvfp4-awq", "nvfp4-gptq"]
S852 = (ROOT / "s8-52-serve-vllm.sh").read_text(encoding="utf-8")


def _vllm_block(readme: str) -> str:
    """README의 vLLM ```bash 블록 1개 (Usage - vLLM 섹션)."""
    sec = readme.split("## Usage - vLLM", 1)[1]
    m = re.search(r"```bash\n(.*?)```", sec, re.DOTALL)
    assert m, "vLLM bash 블록 없음"
    return m.group(1)


def _norm(block: str, model: str) -> str:
    """모델 ref를 플레이스홀더로 치환한 정규형 (README repo id vs s8-52 local 경로 차이 흡수)."""
    return block.replace(model, "<MODEL>")


def test_readme_commands_share_one_template():
    """4종 카드의 vLLM 명령은 repo id만 다르고 플래그는 동일하다."""
    blocks = {}
    for t in QUANT_TARGETS:
        r = prep.render_readme(t, "full", {}, {}, 20.0)
        blocks[t] = _vllm_block(r)
    norms = {_norm(b, prep.hf_repo_id(t, "full")) for t, b in blocks.items()}
    assert len(norms) == 1, f"타깃별 명령 분기 발생: {norms}"
    cmd = norms.pop()
    assert "krafton-ai/vllm.git@${VLLM_REF}" in cmd
    assert "--tensor-parallel-size 1" in cmd
    assert "--quantization compressed-tensors" in cmd
    assert "--trust-remote-code" in cmd
    assert "--served-model-name" in cmd and "--max-model-len 8192" in cmd


def test_s852_matches_readme_spec():
    """s8-52의 case 테이블이 s7-56 VLLM_SERVE 스펙과 일치한다."""    """s8-52의 case 테이블이 s7-56 VLLM_SERVE 스펙과 일치한다."""
    for t in QUANT_TARGETS:
        m = re.search(rf"^\s*{re.escape(t)}\)\s*SUF=\"([^\"]+)\";\s*QUANT=\"([^\"]+)\";\s*VTP_DEF=(\S+)", S852, re.M)
        assert m, f"s8-52 case arm 없음: {t}"
        assert m.group(2) == prep.VLLM_SERVE[t]["quant"], t
        assert m.group(3) == str(prep.VLLM_SERVE[t]["tp"]), t
    assert 'VLLM_REF:-raon-1-mini-preview' in S852
    assert "krafton-ai/vllm.git@${VLLM_REF}" in S852
    assert "--tensor-parallel-size \"$VTP\"" in S852
    assert "--host 127.0.0.1" in S852
    # README 렌더 명령과 s8-52 exec 명령의 플래그 집합이 같다 (모델 ref 제외).
    r = prep.render_readme("nvfp4-awq", "full", {}, {}, 20.0)
    readme_flags = sorted(re.findall(r"--[\w-]+(?:\s+[^\s\\]+)?", _vllm_block(r)))
    s852_flags = sorted(re.findall(r"--[\w-]+(?:\s+\"?[^\s\"\\]+\"?)?", S852))
    for f in ("--tensor-parallel-size", "--quantization", "--trust-remote-code",
              "--served-model-name", "--max-model-len", "--gpu-memory-utilization"):
        assert any(x.startswith(f) for x in readme_flags), f
        assert any(x.startswith(f) for x in s852_flags), f


def test_wrapper_scripts_embed_readme_block():
    """s8-52-vllm-serve-<slug>.sh 본문이 full 카드의 vLLM bash 블록과 바이트 동일하다."""
    for t in QUANT_TARGETS:
        slug = prep.hf_repo_id(t, "full").split("/", 1)[1]
        wrapper = (ROOT / f"s8-52-vllm-serve-{slug}.sh").read_text(encoding="utf-8")
        r = prep.render_readme(t, "full", {}, {}, 20.0)
        assert _vllm_block(r) in wrapper, f"wrapper 본문 불일치: {t}"
        assert wrapper.startswith("#!/usr/bin/env bash"), t


if __name__ == "__main__":
    raise SystemExit(run_standalone(dict(globals())))
