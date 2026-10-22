"""58-upload-hf.py — 산출물 HF Hub 업로드.
repo 규칙: full → <HF_ID>/<BASE>-<target> (접미사 없음),
smoke → <HF_ID>/<BASE>-<target>-smoke (만약의 오업로드가 full을 덮지 않도록).
smoke는 절대 업로드하지 않는다 (우회 수단 없음). full만 업로드 가능.
로컬 산출물 없으면 SKIP.

  uv run 58-upload-hf.py --mode full --targets all
"""

import argparse
import os

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")
HF_ID = os.environ.get("HF_ID", "tayaee")
BASE_SLUG = os.environ.get("BASE_MODEL", "KRAFTON/Raon-1-mini-preview").split("/")[-1]
ALL_TARGETS = ["awq-w3", "awq-w5", "nvfp4-awq", "nvfp4-gptq"]


def repo_name(target: str, mode: str) -> str:
    name = f"{HF_ID}/{BASE_SLUG}-{target}"
    if mode == "smoke":
        name += "-smoke"
    return name


def main(mode: str, targets: str):
    from huggingface_hub import create_repo, upload_folder

    if mode != "full":
        raise SystemExit("REFUSE: smoke는 절대 업로드하지 않는다")
    ts = ALL_TARGETS if targets == "all" else targets.split(",")
    for t in ts:
        local = f"{SHARED}/models/raon-{t}-{mode}"
        if not os.path.exists(f"{local}/config.json"):
            print(f"SKIP {t}: missing {local}")
            continue
        repo = repo_name(t, mode)
        create_repo(repo, exist_ok=True)
        upload_folder(folder_path=local, repo_id=repo)
        print(f"uploaded {local} -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--targets", default="all")
    a = ap.parse_args()
    main(a.mode, a.targets)
