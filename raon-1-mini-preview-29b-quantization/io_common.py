"""io_common.py — 원자적 쓰기 헬퍼 (0바이트/반쯤 쓴 산출물 방지).
규칙: 최종 경로에 직접 쓰지 말고 `<final>.tmp`에 다 쓴 뒤 rename/commit.
(템플릿 llama-3.2-1b-to-math-model/io_common.py와 동일.)
"""

import os
import shutil


def tmp_path(final: str) -> str:
    return final + ".tmp"


def clean_tmp(path: str) -> str:
    """tmp 경로 초기화 후 반환. save_pretrained/oneshot 계열은
    비어 있지 않은 디렉토리에 저장을 거부하므로 출력 전에 호출."""
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.remove(path)
    return path


def commit_file(tmp: str, final: str) -> None:
    """파일 원자 교체 (同 FS 내 os.replace)."""
    os.replace(tmp, final)


def commit_dir(tmp: str, final: str) -> None:
    """디렉토리 교체. 반쯤 쓴 config.json + 미완성 weight 조합이 남지 않는다."""
    if os.path.isdir(final) and not os.path.islink(final):
        shutil.rmtree(final)
    elif os.path.lexists(final):
        os.remove(final)
    os.rename(tmp, final)
