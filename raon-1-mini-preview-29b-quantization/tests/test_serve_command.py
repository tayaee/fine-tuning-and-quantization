"""tests/test_serve_command.py — README vLLM 명령 ≡ s8-64-serve-vllm.sh 실행 명령 동등성.

s8-64는 README 모델카드의 vLLM 명령을 as-is로 실행해 검증한다. 양쪽이 어긋나면
"카드대로 했는데 안 돈다"가 된다. 4종 양자화 타깃은 repo id를 제외하고 동일 명령이다.
개별 래퍼 번호: awq-w3=52, awq-w5=55, nvfp4-awq=58, nvfp4-gptq=61.
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
WRAPPER_NUM = {"awq-w3": 52, "awq-w5": 55, "nvfp4-awq": 58, "nvfp4-gptq": 61}
S864 = (ROOT / "s8-64-serve-vllm.sh").read_text(encoding="utf-8")


def _vllm_block(readme: str) -> str:
    """README의 vLLM ```bash 블록 1개 (Usage - vLLM 섹션)."""
    sec = readme.split("## Usage - vLLM", 1)[1]
    m = re.search(r"```bash\n(.*?)```", sec, re.DOTALL)
    assert m, "vLLM bash 블록 없음"
    return m.group(1)


def _norm(block: str, model: str) -> str:
    """모델 ref를 플레이스홀더로 치환한 정규형 (README repo id vs s8-64 local 경로 차이 흡수)."""
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


def test_s864_matches_readme_spec():
    """s8-64의 case 테이블이 s7-56 VLLM_SERVE 스펙과 일치한다."""
    for t in QUANT_TARGETS:
        m = re.search(rf"^\s*{re.escape(t)}\)\s*SUF=\"([^\"]+)\";\s*QUANT=\"([^\"]+)\";\s*VTP_DEF=(\S+?);\s*NUM=(\S+)", S864, re.M)
        assert m, f"s8-64 case arm 없음: {t}"
        assert m.group(2) == prep.VLLM_SERVE[t]["quant"], t
        assert m.group(3) == str(prep.VLLM_SERVE[t]["tp"]), t
        assert m.group(4) == str(WRAPPER_NUM[t]), t
    assert 'VLLM_REF:-raon-1-mini-preview' in S864
    assert "krafton-ai/vllm.git@${VLLM_REF}" in S864
    assert "--tensor-parallel-size \"$VTP\"" in S864
    assert "--host 127.0.0.1" in S864
    # README 렌더 명령과 s8-64 exec 명령의 플래그 집합이 같다 (모델 ref 제외).
    r = prep.render_readme("nvfp4-awq", "full", {}, {}, 20.0)
    readme_flags = sorted(re.findall(r"--[\w-]+(?:\s+[^\s\\]+)?", _vllm_block(r)))
    s864_flags = sorted(re.findall(r"--[\w-]+(?:\s+\"?[^\s\"\\]+\"?)?", S864))
    for f in ("--tensor-parallel-size", "--quantization", "--trust-remote-code",
              "--served-model-name", "--max-model-len", "--gpu-memory-utilization"):
        assert any(x.startswith(f) for x in readme_flags), f
        assert any(x.startswith(f) for x in s864_flags), f


def test_wrapper_scripts_embed_readme_block():
    """s8-<num>-vllm-serve-<slug>.sh 본문이 full 카드의 vLLM bash 블록과 바이트 동일하다."""
    for t in QUANT_TARGETS:
        slug = prep.hf_repo_id(t, "full").split("/", 1)[1]
        wrapper = (ROOT / f"s8-{WRAPPER_NUM[t]}-vllm-serve-{slug}.sh").read_text(encoding="utf-8")
        r = prep.render_readme(t, "full", {}, {}, 20.0)
        assert _vllm_block(r) in wrapper, f"wrapper 본문 불일치: {t}"
        assert wrapper.startswith("#!/usr/bin/env bash"), t


def test_s864_delegates_to_wrapper_for_card_command():
    """full+hf+기본값 호출은 frozen 래퍼에 위임한다 (카드 명령 as-is 실행)."""
    assert 'WRAPPER="$RQ_ROOT/s8-$NUM-vllm-serve-${BASE_MODEL##*/}-${TARGET}.sh"' in S864
    assert 'exec "$WRAPPER"' in S864
    assert '[ "$SOURCE" = hf ] && [ "$MODE" = full ]' in S864


def test_card_production_block_matches_s865():
    """카드 Production 블록 ≡ s8-65 스크립트 (GB10 풀 컨텍스트)."""
    r = prep.render_readme("nvfp4-awq", "full", {}, {}, 20.0)
    assert "## Production serving (DGX Spark, 128 GB unified)" in r
    prod_sec = r.split("## Production serving", 1)[1].split("## License", 1)[0]
    blocks = re.findall(r"```bash\n(.*?)```", prod_sec, re.DOTALL)
    assert blocks, "production bash 블록 없음"
    prod = blocks[0]
    for flag in ("--max-model-len 262144", "--max-num-batched-tokens 16384",
                 "--max-num-seqs 128", "--gpu-memory-utilization 0.85",
                 "--kv-cache-dtype fp8_e4m3", "--enable-prefix-caching",
                 "--reasoning-parser raon_1", "--tool-call-parser raon_1",
                 "--enable-auto-tool-choice"):
        assert flag in prod, flag
    assert "s8-65-serve-vllm-production.sh" in prod_sec
    s865 = (ROOT / "s8-65-serve-vllm-production.sh").read_text(encoding="utf-8")
    for flag in ("--max-model-len 262144", "--kv-cache-dtype fp8_e4m3",
                 "--enable-prefix-caching", "--reasoning-parser raon_1",
                 "--default-chat-template-kwargs"):
        assert flag in s865, flag


if __name__ == "__main__":
    raise SystemExit(run_standalone(dict(globals())))
