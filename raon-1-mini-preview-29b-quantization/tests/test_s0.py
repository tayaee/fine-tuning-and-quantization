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

#: 5종 타깃 (s6/s7 검증 대상).
ALL_TARGETS = ["awq-w3-transformers", "awq-w5-transformers", "nvfp4-awq-vllm",
               "nvfp4-awq-transformers", "nvfp4-gptq-transformers"]
#: transformers 로 적재 불가한 타깃 (modelopt_fp4 = vLLM 전용).
UNLOADABLE_TARGETS = ["nvfp4-awq-vllm"]

def _steps(group: str) -> list[int]:
    nums = []
    for p in ROOT.glob(f"{group}-*.sh"):
        if "-stop-" in p.name or "-clean-" in p.name:
            continue  # 단독 정리 유틸리티(s0-11/s0-21, s6-00/s7-00)는 3단위 예외
        if re.match(r"s6-02-verify-readme-with-.+-and-transformers\.sh", p.name):
            continue  # README transformers 예제 러너(s6-02)는 3단위 예외
        if re.match(r"s7-02-verify-readme-with-.+-and-vllm-fork\.sh", p.name):
            continue  # README vLLM 예제 러너(s7-02)는 3단위 예외
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


# 그룹별 번호 격자: (나머지, 간격). 기본은 01부터 3단위(01/04/07/...) → n % 3 == 1.
#   s4(측정·평가)만 5단위(00/05/10/...) — 2026-10-28 변경(커밋 4af87cb).
GROUP_GRID = {"s4": (0, 5)}
# 그룹의 첫 스크립트 번호. s6/s7 은 00(정리 유틸) → 02(카드 as-is 러너)로 시작한다 —
# 카드 예제 러너를 01 에 두지 않는 Naming 규칙(2026-10-29 재명명).
GROUP_FIRST_STEP = {"s4": 0, "s6": 0, "s7": 0}
# 3단위 격자 대상이 아닌 그룹. s6 은 정리 유틸(s6-00)과 카드 as-is 러너(s6-02)뿐이다
# — 2026-10-29 s6-13/s6-16 삭제 이후 남은 스크립트가 전부 문서화된 예외다.
EXEMPT_GROUPS = {"s6"}


def _all_steps(group: str) -> list[int]:
    """예외로 제외하지 않은 모든 스크립트의 스텝 번호 (중복 포함)."""
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
        if g in EXEMPT_GROUPS:
            # 격자 검증 대상이 없는 그룹 — 예외 스크립트만 남은 경우다.
            continue
        nums = _steps(g)
        assert nums, f"{g} 그룹에 단계 스크립트 없음"
        rem, step = GROUP_GRID.get(g, (1, 3))
        # 예외 스크립트(정리 유틸·README 러너)를 빼면 최솟값이 격자 첫 칸보다 클 수 있으므로
        # '격자 정렬'만 검증한다.
        assert all(n % step == rem for n in nums), f"{g} {step}단위 위반: {nums}"
        assert GROUP_FIRST_STEP.get(g, 1) in all_steps, \
            f"{g} 에 첫 스텝 스크립트 없음: {all_steps}"


def test_s7_serve_set_covers_all_targets():
    """s7(카드의 '## Usage - vLLM' 예제 검증)은 5종 전부 — transformers 적재 불가 타깃도 포함."""
    for target in ALL_TARGETS:
        b = ROOT / f"s7-02-verify-readme-with-{target}-and-vllm-fork.sh"
        assert b.is_file(), f"{b.name} 없음"


def test_s6_transformers_set_covers_all_targets():
    """s6(카드의 '## Usage - transformers' 예제 검증) — 5종 전부.

    s6/s7 은 'README 의 인스트럭션을 as-is 그대로 타이핑하면 돌아가는가'를 본다.
    타깃별로 transformers 예제(s6-02) 와 vLLM 예제(s7-02) 가 모두 있어야 한다.
    """
    for target in ALL_TARGETS:
        a = ROOT / f"s6-02-verify-readme-with-{target}-and-transformers.sh"
        assert a.is_file(), f"{a.name} 없음"


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
    test_group_numbering_follows_convention()
    test_s7_serve_set_covers_all_targets()
    test_s6_transformers_set_excludes_unloadable_targets()
    print("test_s0 OK")
