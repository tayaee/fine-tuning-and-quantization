"""tests/test_serve_command.py — README vLLM 명령 ≡ snippets ≡ s7 frozen 래퍼 동등성.

실행 예제의 단일 소스는 snippets/<target>/serve-vllm.sh 다:
  - README 모델카드는 __REPO_ID__ 치환 후 verbatim 포함한다 (test_snippets.py).
  - s7 frozen 래퍼는 그 스니펫을 repo id 고정으로 exec한다.
양쪽이 어긋나면 "카드대로 했는데 안 돈다"가 된다. llmcompressor 4종 양자화 타깃은
repo id를 제외하고 동일 명령이다. ModelOpt판(nvfp4-awq-vllm)은
--quantization 없이 자동 감지된다. s7-30-chat.sh이 타깃→래퍼 테이블로 묶는다.
개별 래퍼 번호: awq-w3-transformers=01, awq-w5-transformers=01, nvfp4-awq-vllm=01, nvfp4-awq-transformers=01, nvfp4-gptq-transformers=01.
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


prep = _load_script("s4-70-update-readme.py")

QUANT_TARGETS = ["awq-w3-transformers", "awq-w5-transformers", "nvfp4-awq-vllm", "nvfp4-awq-transformers", "nvfp4-gptq-transformers"]
LLMC_TARGETS = ["awq-w3-transformers", "awq-w5-transformers", "nvfp4-awq-transformers", "nvfp4-gptq-transformers"]
WRAPPER_NUM = {
    "awq-w3-transformers": "05",
    "awq-w5-transformers": "10",
    "nvfp4-awq-transformers": "15",
    "nvfp4-awq-vllm": "20",
    "nvfp4-gptq-transformers": "25",
}
# 래퍼 파일명의 타깃 토큰 (= 타깃 슬러그)
WRAPPER_SLUG = {"awq-w3-transformers": "awq-w3-transformers", "awq-w5-transformers": "awq-w5-transformers", "nvfp4-awq-vllm": "nvfp4-awq-vllm",
                "nvfp4-awq-transformers": "nvfp4-awq-transformers", "nvfp4-gptq-transformers": "nvfp4-gptq-transformers"}
S713 = (ROOT / "s7-30-chat.sh").read_text(encoding="utf-8")


def _vllm_block(readme: str) -> str:
    """README의 vLLM ```bash 블록 1개 (Usage - vLLM 섹션)."""
    sec = readme.split("## Usage - vLLM", 1)[1]
    m = re.search(r"```bash\n(.*?)```", sec, re.DOTALL)
    assert m, "vLLM bash 블록 없음"
    return m.group(1)


def _vllm_serve_block(readme: str) -> str:
    """README Usage - vLLM 섹션의 serve 명령 블록 (`vllm serve` 포함).

    ModelOpt 카드는 소스 빌드 블록이 먼저 나오므로, serve 명령을 포함한
    블록을 골라야 한다 (첫 bash 블록이 빌드 블록이다).
    """
    sec = readme.split("## Usage - vLLM", 1)[1]
    blocks = re.findall(r"```bash\n(.*?)```", sec, re.DOTALL)
    assert blocks, "vLLM bash 블록 없음"
    for b in blocks:
        if "vllm serve" in b:
            return b.removesuffix("\n")
    raise AssertionError("vLLM serve 블록 없음")


def _snippet_serve(target: str, repo: str) -> str:
    """snippets/<target>/serve-vllm.sh 를 repo id 치환한 형태 (카드와 대조용)."""
    text = (prep.SNIPPET_DIR / target / "serve-vllm.sh").read_text(encoding="utf-8")
    return text.removesuffix("\n").replace("__REPO_ID__", repo)


def _norm(block: str, model: str) -> str:
    """모델 ref를 플레이스홀더로 치환한 정규형 (README repo id vs s7 래퍼 경로 차이 흡수)."""
    return block.replace(model, "<MODEL>")


def test_readme_commands_share_one_template():
    """llmcompressor 4종 카드의 vLLM 명령은 repo id만 다르고 플래그는 동일하다 (production 단일 명령)."""
    blocks = {}
    for t in LLMC_TARGETS:
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


def test_readme_modelopt_command_has_no_quantization_flag():
    """ModelOpt판 카드는 --quantization 없이 자동 감지 + 대문자 repo id다."""
    r = prep.render_readme("nvfp4-awq-vllm", "full", {}, {}, 20.0)
    b = _vllm_serve_block(r)
    assert "--quantization" not in b, "ModelOpt 카드에 --quantization 잔존"
    assert "tayaee/Raon-1-mini-preview-NVFP4-AWQ-GB10-vLLM" in b
    for f in ("--tensor-parallel-size 1", "--trust-remote-code",
              "--max-model-len 262144", "--reasoning-parser raon_1"):
        assert f in b, f


def test_s713_table_matches_readme_spec():
    """s7-30의 case 테이블이 s4-31 VLLM_SERVE 스펙과 일치한다."""
    for t in QUANT_TARGETS:
        m = re.search(rf"^\s*{re.escape(t)}\)\s*SUF=\"([^\"]+)\";\s*QUANT=\"([^\"]*)\";\s*VTP_DEF=(\S+?);\s*NUM=(\S+)", S713, re.M)
        assert m, f"s7-30 case arm 없음: {t}"
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
    r = prep.render_readme("nvfp4-awq-transformers", "full", {}, {}, 20.0)
    readme_flags = sorted(re.findall(r"--[\w-]+(?:\s+[^\s\\]+)?", _vllm_block(r)))
    s713_flags = sorted(re.findall(r"--[\w-]+(?:\s+\"?[^\s\"\\]+\"?)?", S713))
    for f in ("--tensor-parallel-size", "--quantization", "--trust-remote-code",
              "--served-model-name", "--max-model-len", "--gpu-memory-utilization",
              "--max-num-batched-tokens", "--max-num-seqs", "--kv-cache-dtype",
              "--enable-prefix-caching", "--reasoning-parser", "--tool-call-parser"):
        assert any(x.startswith(f) for x in readme_flags), f
        assert any(x.startswith(f) for x in s713_flags), f


def test_wrapper_scripts_embed_readme_block():
    """s7 frozen 래퍼 = 스니펫 exec이며, 스니펫은 full 카드의 vLLM serve 블록과 바이트 동일하다."""
    for t in QUANT_TARGETS:
        repo = prep.hf_repo_id(t, "full")
        wrapper = (ROOT / f"s7-{WRAPPER_NUM[t]}-verify-readme-with-{WRAPPER_SLUG[t]}-and-vllm-fork.sh").read_text(encoding="utf-8")
        assert wrapper.startswith("#!/usr/bin/env bash"), t
        assert f's7_serve_frozen "{t}"' in wrapper, f"wrapper가 자기 타깃을 지정하지 않음: {t}"
        r = prep.render_readme(t, "full", {}, {}, 20.0)
        assert _vllm_serve_block(r) == _snippet_serve(t, repo), f"스니펫≠카드 serve 블록: {t}"


def test_s713_manages_wrapper_server():
    """s7-30은 frozen 래퍼를 백그라운드로 띄워 slug 확인 후 상주시킨다 (s7-35가 재사용)."""
    assert 'WRAPPER="$RQ_ROOT/s7-$NUM-verify-readme-with-${TARGET}-and-vllm-fork.sh"' in S713
    assert 'nohup "$WRAPPER" "$MODE" >"$SERVLOG" 2>&1 &' in S713
    assert "model-slug.sh" in S713
    assert 's7-serve.pid' in S713
    assert '[SERVE-FAIL]' in S713 and '[CHAT-FAIL]' in S713 and '"[OK] $slug"' in S713


def test_no_separate_production_script():
    """production 별도 스크립트 없음 — 스니펫 자체가 production 명령이다."""
    assert not list(ROOT.glob("*-serve-vllm-production.sh"))
    for t in QUANT_TARGETS:
        body = _snippet_serve(t, prep.hf_repo_id(t, "full"))
        assert "--max-model-len 262144" in body, t
        assert "--kv-cache-dtype fp8_e4m3" in body, t


if __name__ == "__main__":
    raise SystemExit(run_standalone(dict(globals())))
