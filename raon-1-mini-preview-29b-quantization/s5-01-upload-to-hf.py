"""s5-01-upload-to-hf.py — 이미 준비된 산출물을 HuggingFace Hub에 업로드 (전담).
repo 규칙: full → <HF_ID>/<BASE>-<target> (접미사 없음),
smoke → <HF_ID>/<BASE>-<target>-smoke (만약의 오업로드가 full을 덮지 않도록).
smoke는 절대 업로드하지 않는다 (우회 수단 없음). full만 업로드 가능.
로컬 산출물 없으면 SKIP.

메타데이터(README.md) 생성은 s4-65-update-readme.py가 담당한다 — 여기서는
`config.json` + 비어 있지 않은 `README.md`가 이미 있는지만 확인하고 올린다.

  uv run s5-01-upload-to-hf.py --mode full --targets all [--dry-run]
"""

import argparse
import os

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")
HF_ID = os.environ.get("HF_ID", "tayaee")
BASE_SLUG = os.environ.get("BASE_MODEL", "KRAFTON/Raon-1-mini-preview").split("/")[-1]
ALL_TARGETS = ["awq-w3-transformers", "awq-w5-transformers", "nvfp4-awq-transformers", "nvfp4-gptq-transformers", "nvfp4-awq-vllm"]


def MODELS_DIR(mode: str) -> str:
    return f"{SHARED}/models"


def repo_name(target: str, mode: str) -> str:
    name = f"{HF_ID}/{BASE_SLUG}-{target}"
    if mode == "smoke":
        name += "-smoke"
    return name


def check_ready(target: str, mode: str) -> str | None:
    """업로드 전제 조건 확인 → 로컬 모델 디렉터리 경로 반환.

    모델이 없으면 None(조용히 SKIP), README가 없으면 SystemExit — s4-31-prep-upload을
    먼저 돌리라는 안내와 함께 즉시 중단한다 (빈 README는 업로드해도 허브에서 깨진다).
    """
    local = f"{MODELS_DIR(mode)}/raon-{target}-{mode}"
    if not os.path.exists(f"{local}/config.json"):
        print(f"SKIP {target}: missing {local}")
        return None
    readme = f"{local}/README.md"
    blank = True
    if os.path.isfile(readme):
        with open(readme, encoding="utf-8") as f:
            blank = not f.read().strip()
    if blank:
        raise SystemExit(f"README.md missing in {local}; run ./s4-65-update-readme.sh first")
    return local


def main(mode: str, targets: str | None = None, dry_run: bool = False):
    # smoke 업로드 허용 (2026-10-28): smoke 서빙 검증을 빼먹어 1주일 재작업 발생.
    # smoke는 -smoke repo로만 올라가 정식 repo를 덮지 않는다 (repo_name 참조).
    spec = (targets if targets is not None else os.environ.get("TARGETS") or "all").strip()
    tokens = spec.replace(",", " ").split()
    ts = ALL_TARGETS if (not tokens or "all" in tokens) else tokens
    for t in ts:
        local = check_ready(t, mode)
        if local is None:
            continue
        repo = repo_name(t, mode)
        if dry_run:
            print(f"READY {local} -> https://huggingface.co/{repo} (dry-run, 미업로드)")
            continue
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=local, repo_id=repo, commit_message=f"Upload {t} ({mode})")
        print(f"uploaded {local} -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="준비된 산출물 HF Hub 업로드 (전담)")
    ap.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    ap.add_argument("--targets", default="all")
    ap.add_argument("--dry-run", action="store_true", help="준비 상태만 확인 (네트워크 호출 없음)")
    a = ap.parse_args()
    main(a.mode, a.targets, a.dry_run)