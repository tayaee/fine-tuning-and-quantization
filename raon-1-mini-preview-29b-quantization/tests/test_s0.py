"""tests/test_s0.py — s0 베이스검증 8절차 + 그룹 넘버링 규칙 단위 테스트.

넘버링 규칙: 두 번째 번호는 그룹 내 순서로 01부터 3단위,
각 그룹의 첫 스크립트는 항상 s#-01-.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

S0_SCRIPTS = [
    "s0-01-serve-base-from-hf-with-transformers.sh",
    "s0-04-chat-with-transformers.sh",
    "s0-07-pi-tool-call-with-transformers.sh",
    "s0-10-bench-latency-with-transformers.sh",
    "s0-13-serve-base-from-hf-with-vllm-fork.sh",
    "s0-16-chat-with-vllm-fork.sh",
    "s0-19-pi-tool-call-with-vllm-fork.sh",
    "s0-22-bench-latency-with-vllm-fork.sh",
]

GROUP_FIRST = {"s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"}

def _steps(group: str) -> list[int]:
    nums = []
    for p in ROOT.glob(f"{group}-*.sh"):
        if "-stop-" in p.name or "-clean-" in p.name:
            continue  # 단독 정리 유틸리티(s0-11/s0-21, s6-00/s7-00)는 3단위 예외
        m = re.match(rf"{group}-(\d{{2}})-", p.name)
        assert m, f"naming rule 위반: {p.name}"
        nums.append(int(m.group(1)))
    return sorted(nums)


def test_s0_eight_procedures_exist_and_executable():
    for name in S0_SCRIPTS:
        p = ROOT / name
        assert p.is_file(), f"{name} 없음"
        assert p.stat().st_mode & 0o111, f"{name} 실행 권한 없음"
    for name in ("s0-11-stop-server-with-transformers.sh", "s0-21-stop-server-with-vllm-fork.sh"):
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


def test_group_numbering_starts_at_01_step_3():
    for g in GROUP_FIRST:
        nums = _steps(g)
        assert nums, f"{g} 그룹에 스크립트 없음"
        assert nums[0] == 1, f"{g} 첫 스텝이 01이 아님: {nums}"
        assert all(n % 3 == 1 for n in nums), f"{g} 3단위 위반: {nums}"


def test_s6_s7_customer_sets_are_parallel():
    """s6 serve 5종과 s7 serve 5종은 같은 스텝·타깃으로 일치한다."""
    targets = ["awq-w3-transformers", "awq-w5-transformers", "nvfp4-awq-vllm", "nvfp4-awq-transformers", "nvfp4-gptq-transformers"]
    steps = ["01", "01", "01", "01", "01"]
    slugs = {"awq-w3-transformers": "awq-w3-transformers", "awq-w5-transformers": "awq-w5-transformers", "nvfp4-awq-vllm": "nvfp4-awq-vllm",
             "nvfp4-awq-transformers": "nvfp4-awq-transformers", "nvfp4-gptq-transformers": "nvfp4-gptq-transformers"}
    for step, target in zip(steps, targets):
        suffix = "-NOOP" if target == "nvfp4-awq-vllm" else ""
        a = ROOT / f"s6-{step}-serve-{slugs[target]}-from-hf-with-transformers{suffix}.sh"
        b = ROOT / f"s7-{step}-serve-{slugs[target]}-from-hf-with-vllm-fork.sh"
        assert a.is_file(), f"{a.name} 없음"
        assert b.is_file(), f"{b.name} 없음"


STOP_SCRIPTS = ["s0-11-stop-server-with-transformers.sh", "s0-21-stop-server-with-vllm-fork.sh"]


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
    test_group_numbering_starts_at_01_step_3()
    test_s6_s7_customer_sets_are_parallel()
    print("test_s0 OK")
