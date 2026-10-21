"""io_common.py — 원자적 쓰기 헬퍼 (0바이트/반쯤 쓴 산출물 방지).
규칙: 최종 경로에 직접 쓰지 말고 `<final>.tmp`에 다 쓴 뒤 rename/commit.
중간 kill 시 최종본은 예전 상태 그대로(또는 부재)라 m1b_fresh가 깨진
산출물을 fresh로 오판하지 않는다. 부재/구버전은 rebuild 방향이라 안전.
各 스크립트는 `uv run $M1B_ROOT/*.py` 로 실행되며 스크립트 디렉토리가
sys.path에 들어오므로 plain `import io_common`으로 로드된다.
"""

import os
import shutil


def tmp_path(final: str) -> str:
    return final + ".tmp"


def clean_tmp(path: str) -> str:
    """tmp 경로 초기화 후 반환. save_pretrained/oneshot/gptqmodel save 계열은
    비어 있지 않은 디렉토리에 저장을 거부하므로 디렉토리 출력 전에 호출."""
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.remove(path)
    return path


def commit_file(tmp: str, final: str) -> None:
    """파일 원자 교체 (同 FS 내 os.replace)."""
    os.replace(tmp, final)


def commit_dir(tmp: str, final: str) -> None:
    """디렉토리 교체. 기존 산출물은 통째로 교체되므로 반쯤 쓴
    config.json + 미완성 weight 조합이 남지 않는다."""
    if os.path.isdir(final) and not os.path.islink(final):
        shutil.rmtree(final)
    elif os.path.lexists(final):
        os.remove(final)
    os.rename(tmp, final)
