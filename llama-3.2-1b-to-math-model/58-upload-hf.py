"""58-upload-hf.py — Stage 7a. 로컬 산출물을 HF Hub(tayaee/*)에 업로드.
HF 산출물은 Stage 8b 서빙의 입력이 된다 (local 경로로도 서빙 가능).

  uv run 58-upload-hf.py --mode mini|full --targets all|fft,qlora,fft-gptq,...,fft-gguf-q4_k_m [--dry-run]
전제: hf auth login (쓰기 권한 토큰).
"""

import argparse
import io
import os

SHARED = os.environ.get("M1B_SHARED", os.environ.get("P5_SHARED", "/rosenas/data/AIML/math-1b-shared"))
HF_ID = os.environ.get("HF_ID", "tayaee")
BASE_MODEL = os.environ.get("BASE_MODEL", "unsloth/Llama-3.2-1B")
# repo명에 베이스 모델명을 포함한다. BASE_MODEL 뒷부분만 사용하므로
# 미러(unsloth/…)와 원본(meta-llama/…)이 같은 repo명으로 수렴한다.
# 예: unsloth/Llama-3.2-1B → tayaee/m1b-Llama-3.2-1B-math-fft-mini
SLUG = os.environ.get("BASE_MODEL", "unsloth/Llama-3.2-1B").split("/")[-1]

# target → (로컬 디렉토리 템플릿, 업로드 방식). {m}=mode, {s}=strat.
# 양자화 총 16종: HF 폴더 6종(gptq/awq/fp8 × fft/qlora) + GGUF 10종(fft/qlora × 5).
# 구이름(gptq/awq/fp8/gguf)은 fft-* 별칭. fft-gguf/qlora-gguf 키는 q8_0 호환용으로 유지.
TARGETS = {
    "fft": ("synthetic-fft-{m}-{s}", "folder"),
    "qlora": ("synthetic-qlora-{m}-{s}-merged", "folder"),
    "fft-gptq": ("synthetic-fft-{m}-{s}-gptq", "folder"),
    "fft-awq": ("synthetic-fft-{m}-{s}-awq", "folder"),
    "fft-fp8": ("synthetic-fft-{m}-{s}-fp8", "folder"),
    "fft-gguf": (
        "synthetic-fft-{m}-{s}-gguf/model-q8_0.gguf",
        "gguf",
    ),
    "qlora-gptq": (
        "synthetic-qlora-{m}-{s}-merged-gptq",
        "folder",
    ),
    "qlora-awq": (
        "synthetic-qlora-{m}-{s}-merged-awq",
        "folder",
    ),
    "qlora-fp8": (
        "synthetic-qlora-{m}-{s}-merged-fp8",
        "folder",
    ),
    "qlora-gguf": (
        "synthetic-qlora-{m}-{s}-merged-gguf/model-q8_0.gguf",
        "gguf",
    ),
}


# HF repo명 규칙 (output-models-tree.txt 설계 결정 3):
# single 동결 ({slug}-math-{target}-{m}, full이면 -{m} 생략),
# ddp/fsdp는 -{s} 삽입, full은 무표기.
def repo_name(target: str, strat: str, mode: str) -> str:
    r = f"{SLUG}-math-{target}"
    if strat != "single":
        r += f"-{strat}"
    if mode != "full":
        r += f"-{mode}"
    return f"{HF_ID}/{r}"


def readme_for(target: str, repo: str, mode: str, strat: str, kind: str) -> str:
    """업로드용 README.md. 없으면 pipeline_tag=None이라 Hub 위젯·검색에 안 걸린다.

    이 모델들은 chat_template이 없어 chat API(hf inference run 같은 Providers
    chat)는 400/미지원이다. completions/transformers/vLLM 경로만 문서화한다.
    """
    base = "fft" if target.startswith("fft") else "qlora" if target.startswith("qlora") else target
    tags = ["llama", "sft", "math", "math-1b-sft-and-quantization", base]
    if "gptq" in target:
        tags += ["gptq", "4-bit"]
    elif "awq" in target:
        tags += ["awq", "compressed-tensors"]
    elif "fp8" in target:
        tags += ["fp8", "compressed-tensors"]
    elif "gguf" in target:
        tags += ["gguf", "endpoints_compatible"]
        for q in ("q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m"):
            if q in target:
                tags += [q]
                break
    tags_yaml = "\n".join(f"- {t}" for t in tags)
    lib_line = "library_name: transformers\n" if kind == "folder" else ""
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
    return f"""---
pipeline_tag: text-generation
{lib_line}base_model: {BASE_MODEL}
tags:
{tags_yaml}
---

# {repo}

math-1b-sft-and-quantization SFT math model (`mode={mode} strat={strat} target={target}`, base `{BASE_MODEL}`).
GSM8K 합성데이터 FFT/QLoRA → 양자화 산출물. 학습 프롬프트는 `### Instruction/### Response` completions 포맷이다.

chat_template이 없으므로 chat API는 미지원이다. `hf inference run`(Providers chat)은
커스텀 파인튜닝에 프로바이더가 없어 동작하지 않는다. 아래 경로로 테스트하라.

## Prompt format

```
{prompt}
```

## Use (transformers, local)

```python
from transformers import pipeline
p = pipeline("text-generation", model="{repo}", max_new_tokens=256)
print(p("### Instruction:\\n2+3은?\\n\\n### Response:")[0]["generated_text"])
```

## Use (vLLM serve + completions)

```bash
SOURCE=hf ./60-serve-vllm.sh {mode} {target}
./62-infer-examples.sh {repo} 3
```
"""


# 구이름 별칭 → 정식 키 (같은 repo에 업로드, 기존 동작 유지).
ALIAS = {"gptq": "fft-gptq", "awq": "fft-awq", "fp8": "fft-fp8", "gguf": "fft-gguf"}

# GGUF 5종 변종 (42/52 산출물): fft/qlora × {q8_0,q6_k,q5_k_m,q4_k_m,q3_k_m}.
# gguf 단일파일 업로드 분기 그대로 사용 (path_in_repo=파일명).
GQUANTS = ["q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m"]
for _base, _local in (
    ("fft", "synthetic-fft-{m}-{s}"),
    ("qlora", "synthetic-qlora-{m}-{s}-merged"),
):
    for _q in GQUANTS:
        TARGETS[f"{_base}-gguf-{_q}"] = (
            f"{_local}-gguf/model-{_q}.gguf",
            "gguf",
        )


def main(mode: str, strat: str, targets: str, dry_run: bool):
    sel = list(TARGETS) if targets == "all" else targets.split(",")
    plan = []
    for t in sel:
        key = ALIAS.get(t, t)
        local_t, kind = TARGETS[key]
        plan.append(
            (
                t,
                f"{SHARED}/models/" + local_t.format(m=mode, s=strat),
                repo_name(key, strat, mode),
                kind,
            )
        )
    for t, local, repo, kind in plan:
        print(f"[{t}] {local} -> {repo} ({kind})")
        key = ALIAS.get(t, t)
        text = readme_for(key, repo, mode, strat, kind)
        if dry_run:
            print(f"  README preview ({len(text)} chars, first 5 lines):")
            for line in text.splitlines()[:5]:
                print(f"    {line}")
            continue
        if not os.path.exists(local):
            print("  SKIP: no local output (run that stage first)")
            continue
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo, exist_ok=True)
        if kind == "folder":
            with open(os.path.join(local, "README.md"), "w", encoding="utf-8") as f:
                f.write(text)
            print("  README: wrote local README.md")
            api.upload_folder(repo_id=repo, folder_path=local)
        else:  # gguf 단일 파일 (tokenizer는 폴더 타깃 repo에 동봉 전제)
            api.upload_file(
                repo_id=repo,
                path_or_fileobj=local,
                path_in_repo=os.path.basename(local),
            )
            api.upload_file(
                repo_id=repo,
                path_or_fileobj=io.BytesIO(text.encode("utf-8")),
                path_in_repo="README.md",
            )
            print("  README: uploaded README.md")
        print(f"  OK: https://huggingface.co/{repo}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument(
        "--strat",
        default=os.environ.get("STRAT", "single"),
        choices=["single", "ddp", "fsdp"],
    )
    ap.add_argument("--targets", default="all")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(a.mode, a.strat, a.targets, a.dry_run)
