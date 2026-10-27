"""tests/test_serve_command.py — README vLLM 명령 ≡ s7 frozen 래퍼 동등성.

s7-01/04/07/10 래퍼는 README 모델카드의 vLLM 명령을 as-is로 실행한다. 양쪽이
어긋나면 "카드대로 했는데 안 돈다"가 된다. 4종 양자화 타깃은 repo id를 제외하고
동일 명령이다. s7-13-chat.sh이 타깃→래퍼 테이블로 묶는다.
개별 래퍼 번호: awq-w3=01, awq-w5=04, nvfp4-awq=07, nvfp4-gptq=10.
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


prep = _load_script("s4-31-prep-upload.py")

QUANT_TARGETS = ["awq-w3", "awq-w5", "nvfp4-awq", "nvfp4-gptq"]
WRAPPER_NUM = {"awq-w3": "01", "awq-w5": "04", "nvfp4-awq": "07", "nvfp4-gptq": "10"}
S713 = (ROOT / "s7-13-chat.sh").read_text(encoding="utf-8")


def _vllm_block(readme: str) -> str:
    """README의 vLLM ```bash 블록 1개 (Usage - vLLM 섹션)."""
    sec = readme.split("## Usage - vLLM", 1)[1]
    m = re.search(r"```bash\n(.*?)```", sec, re.DOTALL)
    assert m, "vLLM bash 블록 없음"
    return m.group(1)


def _norm(block: str, model: str) -> str:
    """모델 ref를 플레이스홀더로 치환한 정규형 (README repo id vs s7 래퍼 경로 차이 흡수)."""
    return block.replace(model, "<MODEL>")


def test_readme_commands_share_one_template():
    """4종 카드의 vLLM 명령은 repo id만 다르고 플래그는 동일하다 (production 단일 명령)."""
    blocks = {}
    for t in QUANT_TARGETS:
        r = prep.render_readme(t, "full", {}, {}, 20.0)
        assert "## Production serving" not in r, "production 분리가 잔존"
        blocks[t] = _vllm_block(r)
    norms = {_norm(b, prep.hf_repo_id(t, "full")) for t, b in blocks.items()}
    assert len(norms) == 1, f"타깃별 명령 분기 발생: {norms}"
    cmd = norms.pop()
    assert "krafton-ai/vllm.git@${VLLM_REF}" in cmd
    assert "--tensor-parallel-size 1" in cmd
    assert "--quantization compressed-tensors" in cmd
    assert "--trust-remote-code" in cmd
    assert "--served-model-name" in cmd and "--max-model-len 262144" in cmd
    for f in ("--max-num-batched-tokens 16384", "--max-num-seqs 128",
              "--gpu-memory-utilization 0.85", "--kv-cache-dtype fp8_e4m3",
              "--enable-prefix-caching", "--reasoning-parser raon_1",
              "--tool-call-parser raon_1", "--enable-auto-tool-choice"):
        assert f in cmd, f


def test_s713_table_matches_readme_spec():
    """s7-13의 case 테이블이 s4-31 VLLM_SERVE 스펙과 일치한다."""
    for t in QUANT_TARGETS:
        m = re.search(rf"^\s*{re.escape(t)}\)\s*SUF=\"([^\"]+)\";\s*QUANT=\"([^\"]+)\";\s*VTP_DEF=(\S+?);\s*NUM=(\S+)", S713, re.M)
        assert m, f"s7-13 case arm 없음: {t}"
        assert m.group(2) == prep.VLLM_SERVE[t]["quant"], t
        assert m.group(3) == str(prep.VLLM_SERVE[t]["tp"]), t
        assert m.group(4) == str(WRAPPER_NUM[t]), t
    assert 'VLLM_REF:-raon-1-mini-preview' in S713
    assert "krafton-ai/vllm.git@${VLLM_REF}" in S713
    assert "--tensor-parallel-size \"${VTP:-$VTP_DEF}\"" in S713
    assert "--host 127.0.0.1" in S713
    assert "--max-model-len \"${VLLM_MAX_LEN:-262144}\"" in S713
    assert "--gpu-memory-utilization \"${GPU_UTIL:-0.85}\"" in S713
    # README 렌더 명령과 s7 frozen 래퍼 명령의 플래그 집합이 같다 (모델 ref 제외).
    r = prep.render_readme("nvfp4-awq", "full", {}, {}, 20.0)
    readme_flags = sorted(re.findall(r"--[\w-]+(?:\s+[^\s\\]+)?", _vllm_block(r)))
    s713_flags = sorted(re.findall(r"--[\w-]+(?:\s+\"?[^\s\"\\]+\"?)?", S713))
    for f in ("--tensor-parallel-size", "--quantization", "--trust-remote-code",
              "--served-model-name", "--max-model-len", "--gpu-memory-utilization",
              "--max-num-batched-tokens", "--max-num-seqs", "--kv-cache-dtype",
              "--enable-prefix-caching", "--reasoning-parser", "--tool-call-parser"):
        assert any(x.startswith(f) for x in readme_flags), f
        assert any(x.startswith(f) for x in s713_flags), f


def test_wrapper_scripts_embed_readme_block():
    """s7-<num>-vllm-fork-test-<t>.sh 본문이 full 카드의 vLLM bash 블록과 바이트 동일하다."""
    for t in QUANT_TARGETS:
        slug = prep.hf_repo_id(t, "full").split("/", 1)[1]
        wrapper = (ROOT / f"s7-{WRAPPER_NUM[t]}-vllm-fork-test-{t}.sh").read_text(encoding="utf-8")
        r = prep.render_readme(t, "full", {}, {}, 20.0)
        assert _vllm_block(r) in wrapper, f"wrapper 본문 불일치: {t}"
        assert wrapper.startswith("#!/usr/bin/env bash"), t


def test_s713_manages_wrapper_server():
    """s7-13은 frozen 래퍼를 백그라운드로 띄워 slug 확인 후 상주시킨다 (s7-16이 재사용)."""
    assert 'WRAPPER="$RQ_ROOT/s7-$NUM-vllm-fork-test-${TARGET}.sh"' in S713
    assert 'nohup "$WRAPPER" >"$SERVLOG" 2>&1 &' in S713
    assert 'model-slug.sh' in S713
    assert 's7-serve.pid' in S713
    assert '[SERVE-FAIL]' in S713 and '[CHAT-FAIL]' in S713 and '"[OK] $slug"' in S713


def test_no_separate_production_script():
    """production 별도 스크립트 없음 — 01/04/07/10 래퍼 자체가 production이다."""
    assert not list(ROOT.glob("*-serve-vllm-production.sh"))
    for t in QUANT_TARGETS:
        slug = prep.hf_repo_id(t, "full").split("/", 1)[1]
        wrapper = (ROOT / f"s7-{WRAPPER_NUM[t]}-vllm-fork-test-{t}.sh").read_text(encoding="utf-8")
        assert "--max-model-len 262144" in wrapper, t
        assert "--kv-cache-dtype fp8_e4m3" in wrapper, t


if __name__ == "__main__":
    raise SystemExit(run_standalone(dict(globals())))
