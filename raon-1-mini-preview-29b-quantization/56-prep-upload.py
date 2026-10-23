"""56-prep-upload.py — 업로드 전 모델 디렉터리 검증 + 모델 카드(README.md) 생성.
58-upload-hf.py는 이 스크립트가 채운 폴더를 그대로 Hub에 올리기만 한다
(메타데이터는 업로드 지점에서 즉석 생성하면 요약 정보가 빠진다).

각 타겟에 대해:
  1) quant_common.verify_compressed_output으로 패킹 포맷 강제 (dense 폴백 차단, issue-3)
  2) chat_template.jinja / configuration_raon1_mini.py / modeling_raon1_mini.py
     누락 시 models/raon-bf16에서 복사 보충
  3) config.json / tokenizer.json / tokenizer_config.json 존재 확인
  4) outputs/ppl-<mode>/summary.json + outputs/eval-<mode>/score.json에서 지표를 읽어
     HuggingFace 모델 카드 README.md 작성 (YAML frontmatter + 양자화 사양 + 벤치마크 표
     + transformers/vLLM 실행 예제 + 라이선스)
  5) 요약 매니페스트를 outputs/prep-upload-<mode>.json에 기록

  uv run 56-prep-upload.py --mode smoke --targets nvfp4-awq --allow-missing-metrics
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from quant_common import (
    NVFP4_PACK_QUANTIZED,
    PACK_QUANTIZED,
    verify_compressed_output,
)

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")
BASE_MODEL = os.environ.get("BASE_MODEL", "KRAFTON/Raon-1-mini-preview")
# 실측 캐시 스냅샷 (config/common.env BASE_REVISION과 같은 값 — 카드에 박아 둔다)
BASE_REVISION = os.environ.get("BASE_REVISION") or "da4cf162c8613993f2189ebb438a5f9a23f3e5ca"
ALL_TARGETS = ["awq-w3", "awq-w5", "nvfp4-awq", "nvfp4-gptq"]
COPY_FROM_BF16 = ("chat_template.jinja", "configuration_raon1_mini.py", "modeling_raon1_mini.py")
REQUIRED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
# 원본 KRAFTON 레포에는 gated 동의 필드가 있다. 라이선스 요청이 아니므로 절대 복사하지 않는다.
FORBIDDEN_FRONTMATTER = ("extra_gated_",)
BENCH_LABELS = {"kmmlu_redux": "KMMLU-redux", "gsm8k": "GSM8K", "humaneval": "HumanEval"}
BENCH_ORDER = ("kmmlu_redux", "gsm8k", "humaneval")

# 타겟별 on-disk 포맷과 README에 박을 양자화 사양
SPECS: dict[str, dict] = {
    "awq-w5": {
        "format": PACK_QUANTIZED, "tags": ["w5a16", "gptq"],
        "algorithm": "GPTQ (AWQ smoothing 후)", "precision": "W5A16",
        "bits": "5-bit int", "group_size": "group_size=64, symmetric=False",
        "ignored": "`lm_head`, `moe.gate` (router), `self_attn`, `shared_experts`",
    },
    "awq-w3": {
        "format": PACK_QUANTIZED, "tags": ["w3a16", "gptq"],
        "algorithm": "GPTQ (AWQ smoothing 후)", "precision": "W3A16",
        "bits": "3-bit int", "group_size": "group_size=64, symmetric=False",
        "ignored": "`lm_head`, `moe.gate` (router), `self_attn`, `shared_experts`",
    },
    "nvfp4-awq": {
        "format": NVFP4_PACK_QUANTIZED, "tags": ["nvfp4", "w4a4", "awq"],
        "algorithm": "NVFP4 (AWQ smoothing 후)", "precision": "W4A4 (FP8 동적 활성 스케일)",
        "bits": "NVFP4 (E2M1)", "group_size": "block_size=16",
        "ignored": "`lm_head`, `moe.gate` (router), `self_attn`, `shared_experts`",
    },
    "nvfp4-gptq": {
        "format": NVFP4_PACK_QUANTIZED, "tags": ["nvfp4", "w4a4", "gptq"],
        "algorithm": "NVFP4 (GPTQ)", "precision": "W4A4 (FP8 동적 활성 스케일)",
        "bits": "NVFP4 (E2M1)", "group_size": "block_size=16",
        "ignored": "`lm_head`, `moe.gate` (router), `self_attn`, `shared_experts`",
    },
}
# 캘리브레이션 스펙은 config/common.env가 정한 실제 값이다 (full 기본 256/2048).
# 카드에 박히는 값이 파이프라인이 만든 값과 어긋나면 안 되므로 환경변수를 따른다.
CALIB_NOTE = (
    f"{os.environ.get('CALIB_N', '256')} samples, "
    f"seq_len {os.environ.get('SEQ_LEN', '2048')} (in-domain, GSM8K 계열 한국어 혼합)"
)
IGNORE_NOTE = "`lm_head`, `mlp.gate`, `self_attn`, `shared_experts` (BF16 유지)"


def MODELS_DIR(mode: str) -> str:
    return f"{SHARED}/models"


def BF16_DIR() -> str:
    """30-dequant 복원본 경로. 모드 무관(한 번만 만든다)이라 mode를 받지 않는다."""
    return f"{SHARED}/models/raon-bf16"


def OUT_ROOT() -> str:
    return f"{SHARED}/outputs"


def model_dir(mode: str, target: str) -> Path:
    return Path(f"{MODELS_DIR(mode)}/raon-{target}-{mode}")


def _ppl_summary(mode: str) -> dict:
    fp = Path(f"{OUT_ROOT()}/ppl-{mode}/summary.json")
    return json.loads(fp.read_text(encoding="utf-8")) if fp.is_file() else {}


def _eval_scores(mode: str) -> dict:
    fp = Path(f"{OUT_ROOT()}/eval-{mode}/score.json")
    return json.loads(fp.read_text(encoding="utf-8")) if fp.is_file() else {}


def _dir_size_gb(d: Path) -> float:
    return round(sum(f.stat().st_size for f in d.glob("*.safetensors")) / 1024**3, 2)


def _fmt(value, digits: int = 2, dash: str = "—") -> str:
    if value is None:
        return dash
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_readme(target: str, mode: str, ppl: dict, scores: dict, size_gb: float) -> str:
    """HuggingFace 모델 카드 전체를 만든다 (요구사항 2.1-3)."""
    spec = SPECS[target]
    rows = ["base", *ALL_TARGETS]
    header = ("| Model | WikiText-2 PPL | Δ vs bf16 | "
              + " | ".join(f"{BENCH_LABELS[b]} ({b})" for b in BENCH_ORDER) + " |")
    sep = "| --- | ---: | ---: | " + " | ".join("---:" for _ in BENCH_ORDER) + " |"
    lines_table = [header, sep]
    for r in rows:
        p = (ppl.get(r) or {}).get("ppl")
        delta = (ppl.get(r) or {}).get("delta")
        sc = (scores.get(r) or {}).get("benchmarks") or {}
        accs = [_fmt((sc.get(b) or {}).get("acc")) for b in BENCH_ORDER]
        name = "**bf16 (base)**" if r == "base" else f"**{r}**" if r == target else r
        lines_table.append(f"| {name} | {_fmt(p, 3)} | {_fmt(delta)} | " + " | ".join(accs) + " |")

    tags = ", ".join(["raon", "moe", "compressed-tensors", "llm-compressor", *spec["tags"]])
    fm = [
        "---",
        "license: apache-2.0",
        f"base_model: {BASE_MODEL}",
        "base_model_relation: quantized",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "language:",
        "- ko",
        "- en",
        "tags:",
        *(f"- {t}" for t in tags.split(", ")),
        "---",
    ]
    return "\n".join(fm) + f"""

# {BASE_MODEL} — {target} ({mode})

`{BASE_MODEL}`의 **{target}** 양자화 모델입니다. 가중치만 {spec['precision']}로 낮췄고
활성화·정규화 레이어는 BF16 그대로 유지했습니다. `compressed-tensors`
(`{spec['format']}`) 포맷이므로 `transformers`의 `AutoModelForCausalLM`로 바로 로드됩니다.

## 양자화 사양

| 항목 | 값 |
| --- | --- |
| Base model | `{BASE_MODEL}` |
| Base revision | `{BASE_REVISION}` |
| 양자화 알고리즘 | {spec['algorithm']} |
| 가중치 정밀도 | {spec['precision']} |
| 비트폭 | {spec['bits']} |
| 그룹 크기 | {spec['group_size']} |
| 양자화 제외 레이어 | {IGNORE_NOTE} |
| 캘리브레이션 | {CALIB_NOTE} |
| 체크포인트 용량 | {size_gb} GB |
| 저장 포맷 | `{spec['format']}` |

## 성능 비교

WikiText-2 PPL은 in-domain 캘리브레이션 텍스트 기준이며, 나머지는 50-eval/54-score가
산출한 정확도입니다. **굵게** 표시된 행이 이 저장소에 올라간 모델입니다.

{chr(10).join(lines_table)}

## 사용법 — transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{target}"  # 또는 공개된 이 repo
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
messages = [{{"role": "user", "content": "2+2는?"}}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(inputs, max_new_tokens=256)[0], skip_special_tokens=True))
```

## 사용법 — vLLM

Raon-1-mini 계열은 커스텀 아키텍처라 KRAFTON의 vLLM 포크가 필요합니다.

```bash
vllm serve {target} \\
  --trust-remote-code \\
  --quantization compressed-tensors \\
  --tensor-parallel-size 2
```

## 라이선스 및 출처

- Apache License 2.0. 이 저장소의 코드는 Apache-2.0으로 배포됩니다.
- 베이스 모델: [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) (rev `{BASE_REVISION}`), Apache-2.0.
- 원본 KRAFTON 저장소는 접근이 게이트되어 있으나, 이 저장소는 그 요청을 재게이트하지 않습니다.
  베이스 모델 라이선스(`apache-2.0`)의 조건은 그대로 적용됩니다.
"""


def _copy_missing_custom_code(d: Path, copied: list) -> list:
    """누락된 custom_code/*.jinja를 models/raon-bf16에서 복사 보충."""
    src = Path(BF16_DIR())
    for name in COPY_FROM_BF16:
        if (d / name).is_file():
            continue
        s = src / name
        if not s.is_file():
            raise RuntimeError(f"{d}: {name} 누락이고 보충 원본도 없음 ({s})")
        shutil.copy(s, d / name)
        copied.append(name)
    return copied


def prepare_target(mode: str, target: str, allow_missing_metrics: bool = False) -> dict:
    """타겟 1개 준비. 실패해도 예외 대신 ok=False + errors를 담은 보고서를 돌려준다."""
    report: dict = {"target": target, "mode": mode, "ok": False, "errors": [], "copied": [], "size_gb": None}
    d = model_dir(mode, target)
    if not (d / "config.json").is_file():
        report["errors"].append(f"{d}: config.json 없음 — 모델 디렉터리 미생성 (run the quant step first)")
        return report

    # 1) 누락 custom code 보충 (verify_compressed_output도 custom_code를 요구하므로 먼저)
    try:
        report["copied"] = _copy_missing_custom_code(d, report["copied"])
    except Exception as e:  # noqa: BLE001
        report["errors"].append(str(e))

    # 2) 압축 포맷 계약 (issue-3) — dense 폴백 차단
    try:
        verify_compressed_output(d, SPECS[target]["format"])
    except Exception as e:  # noqa: BLE001 — 실패 사유를 보고서로 모아 fatal로 올린다
        report["errors"].append(f"{type(e).__name__}: {e}")
        return report

    # 3) 필수 파일 확인
    for name in REQUIRED_FILES:
        if not (d / name).is_file():
            report["errors"].append(f"{d}: {name} 없음 — 업로드 시 로드 불가")

    # 4) 지표 → 모델 카드
    ppl = _ppl_summary(mode)
    scores = _eval_scores(mode)
    if not allow_missing_metrics:
        if target not in ppl:
            report["errors"].append(f"outputs/ppl-{mode}/summary.json에 '{target}' 측정값 없음 (run 48-measure-ppl first)")
        if target not in scores:
            report["errors"].append(f"outputs/eval-{mode}/score.json에 '{target}' 결과 없음 (run 50-eval/54-score first)")
    report["size_gb"] = _dir_size_gb(d)
    readme = render_readme(target, mode, ppl, scores, report["size_gb"])
    for bad in FORBIDDEN_FRONTMATTER:
        if bad in readme.split("---", 2)[1]:
            report["errors"].append(f"frontmatter에 금지 필드 '{bad}' 포함")
    if report["errors"]:
        return report
    (d / "README.md").write_text(readme, encoding="utf-8")
    report["readme_bytes"] = len(readme.encode("utf-8"))

    report["ok"] = True
    return report


def _parse_targets(targets: str | list | None) -> list[str]:
    if targets is None:
        targets = os.environ.get("TARGETS") or "all"
    if isinstance(targets, str):
        tokens = targets.replace(",", " ").split()
        if not tokens or "all" in tokens:
            return list(ALL_TARGETS)
        return tokens
    return list(targets)


def run(mode: str, targets: str | list | None = None, allow_missing_metrics: bool = False) -> dict:
    """전 타겟 준비 + 매니페스트 기록. 전부 성공해야 ok=True."""
    ts = _parse_targets(targets)
    unknown = [t for t in ts if t not in SPECS]
    if unknown:
        raise SystemExit(f"unknown target(s): {unknown} (허용: {ALL_TARGETS})")
    reports = {t: prepare_target(mode, t, allow_missing_metrics) for t in ts}
    manifest = {
        "mode": mode,
        "allow_missing_metrics": allow_missing_metrics,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "targets": reports,
        "ok": all(r["ok"] for r in reports.values()),
    }
    outdir = Path(f"{OUT_ROOT()}")
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"prep-upload-{mode}.json"
    fp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for t, r in reports.items():
        if r["ok"]:
            print(f"OK {t}: readme={r['readme_bytes']}B size={r['size_gb']}GB copied={r['copied']}")
        else:
            print(f"FAIL {t}: {'; '.join(r['errors'])}")
    print(f"-> {fp} (ok={manifest['ok']})")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="업로드 전 모델 디렉터리 검증 + README.md 생성")
    ap.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    ap.add_argument("--targets", default=None, help="콤마/공백 구분 (기본: $TARGETS 또는 all)")
    ap.add_argument("--allow-missing-metrics", action="store_true",
                    help="PPL/eval 지표가 없어도 카드를 만든다 (정식 업로드에는 쓰지 말 것)")
    a = ap.parse_args()
    _m = run(a.mode, a.targets, a.allow_missing_metrics)
    raise SystemExit(0 if _m["ok"] else 1)