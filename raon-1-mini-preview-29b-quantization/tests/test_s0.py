"""tests/test_s0.py — s0 베이스검증 8절차 + 그룹 넘버링 규칙 단위 테스트.

넘버링 규칙: 두 번째 번호는 그룹 내 순서로 00부터 5단위(00, 05, 10, 15, ...),
각 그룹의 첫 스크립트는 항상 s#-00-.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

S0_SCRIPTS = [
    "s0-00-serve-base-from-hf-with-transformers.sh",
    "s0-05-chat-with-transformers.sh",
    "s0-10-pi-tool-call-with-transformers.sh",
    "s0-15-bench-latency-with-transformers.sh",
    "s0-25-serve-base-from-hf-with-vllm-fork.sh",
    "s0-30-chat-with-vllm-fork.sh",
    "s0-35-pi-tool-call-with-vllm-fork.sh",
    "s0-45-bench-latency-with-vllm-fork.sh",
]

GROUP_FIRST = {"s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"}

#: 5종 타깃 (s6/s7 검증 대상).
ALL_TARGETS = ["awq-w3-transformers", "awq-w5-transformers", "nvfp4-awq-vllm",
               "nvfp4-awq-transformers", "nvfp4-gptq-transformers"]
#: transformers 로 적재 불가한 타깃 (modelopt_fp4 = vLLM 전용).
UNLOADABLE_TARGETS = ["nvfp4-awq-vllm"]

S6_TARGET_NUM = {
    "awq-w3-transformers": "05",
    "awq-w5-transformers": "10",
    "nvfp4-awq-transformers": "15",
    "nvfp4-gptq-transformers": "20",
}

S7_TARGET_NUM = {
    "awq-w3-transformers": "05",
    "awq-w5-transformers": "10",
    "nvfp4-awq-transformers": "15",
    "nvfp4-awq-vllm": "20",
    "nvfp4-gptq-transformers": "25",
}

def _steps(group: str) -> list[int]:
    nums = []
    for p in ROOT.glob(f"{group}-*.sh"):
        m = re.match(rf"{group}-(\d{{2}})-", p.name)
        assert m, f"naming rule 위반: {p.name}"
        nums.append(int(m.group(1)))
    return sorted(nums)


def test_s0_eight_procedures_exist_and_executable():
    for name in S0_SCRIPTS:
        p = ROOT / name
        assert p.is_file(), f"{name} 없음"
        assert p.stat().st_mode & 0o111, f"{name} 실행 권한 없음"
    for name in ("s0-20-stop-server-with-transformers.sh", "s0-40-stop-server-with-vllm-fork.sh"):
        p = ROOT / name
        assert p.is_file(), f"{name} 없음"
        assert p.stat().st_mode & 0o111, f"{name} 실행 권한 없음"


def test_s0_scripts_bash_syntax():
    for name in S0_SCRIPTS:
        out = subprocess.run(["bash", "-n", str(ROOT / name)],
                             capture_output=True, text=True, check=False)
        assert out.returncode == 0, f"{name}: {out.stderr}"


def test_s0_pi_helper_defines_fixed_question():
    inc = (ROOT / "config" / "s0-pi.inc").read_text(encoding="utf-8")
    assert "S0_PI_QUESTION=" in inc
    assert "숫자 하나로 답해" in inc
    for fn in ("s0_pi_probe_dir", "s0_pi_register", "s0_pi_ask", "s0_pi_verify"):
        assert fn in inc, f"{fn} 없음"


def test_s0_shared_server_and_bench():
    import ast
    ast.parse((ROOT / "config" / "transformers-server.py").read_text(encoding="utf-8"))
    bench = (ROOT / "config" / "s0-bench.inc").read_text(encoding="utf-8")
    assert "s0_bench_run()" in bench


def _all_steps(group: str) -> list[int]:
    nums = []
    for p in ROOT.glob(f"{group}-*.sh"):
        m = re.match(rf"{group}-(\d{{2}})-", p.name)
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)


def test_group_numbering_follows_convention():
    for g in GROUP_FIRST:
        all_steps = _all_steps(g)
        assert all_steps, f"{g} 그룹에 스크립트 없음"
        nums = _steps(g)
        assert nums, f"{g} 그룹에 단계 스크립트 없음"
        # 전 그룹 00부터 5단위 (n % 5 == 0), 고유 번호(중복 없음), 00부터 N*5까지 연속
        assert all(n % 5 == 0 for n in nums), f"{g} 5단위 위반: {nums}"
        assert nums == list(range(0, len(nums) * 5, 5)), f"{g} 연속 5간격 위반: {nums}"
        assert 0 in all_steps, f"{g} 에 첫 스텝 스크립트(00) 없음: {all_steps}"


def test_s7_serve_set_covers_all_targets():
    """s7(카드의 '## Usage - vLLM' 예제 검증)은 5종 전부 — transformers 적재 불가 타깃도 포함."""
    for target in ALL_TARGETS:
        b = ROOT / f"s7-{S7_TARGET_NUM[target]}-verify-readme-with-{target}-and-vllm-fork.sh"
        assert b.is_file(), f"{b.name} 없음"


def test_s6_transformers_set_covers_targets_with_transformers_example():
    """s6(카드의 '## Usage - transformers' 예제 검증) — 그 예제가 있는 타깃만.

    nvfp4-awq-vllm(modelopt_fp4)은 transformers 가 못 읽는다(2026-10-29 실측) → 카드는
    vLLM 전용이고 transformers 예제가 없다 → 검증할 예제가 없으니 s6 스텝도 없다.
    """
    for target in ALL_TARGETS:
        if target in UNLOADABLE_TARGETS:
            continue
        a = ROOT / f"s6-{S6_TARGET_NUM[target]}-verify-readme-with-{target}-and-transformers.sh"
        assert a.is_file(), f"{a.name} 없음"


STOP_SCRIPTS = ["s0-20-stop-server-with-transformers.sh", "s0-40-stop-server-with-vllm-fork.sh"]


def test_s0_stop_scripts_bash_syntax():
    import subprocess
    for name in STOP_SCRIPTS:
        out = subprocess.run(["bash", "-n", str(ROOT / name)],
                             capture_output=True, text=True, check=False)
        assert out.returncode == 0, f"{name}: {out.stderr}"


if __name__ == "__main__":
    test_s0_eight_procedures_exist_and_executable()
    test_s0_stop_scripts_bash_syntax()
    test_s0_scripts_bash_syntax()
    test_s0_pi_helper_defines_fixed_question()
    test_s0_shared_server_and_bench()
    test_group_numbering_follows_convention()
    test_s7_serve_set_covers_all_targets()
    test_s6_transformers_set_covers_targets_with_transformers_example()
    print("test_s0 OK")
