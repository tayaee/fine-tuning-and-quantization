"""s8-43-prep-upload.py — 업로드 전 모델 디렉터리 검증 + 모델 카드(README.md) 생성.
s8-49-upload-hf.py는 이 스크립트가 채운 폴더를 그대로 Hub에 올리기만 한다
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

  uv run s8-43-prep-upload.py --mode smoke --targets nvfp4-awq --allow-missing-metrics
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
# 표준 GSM8K 단일 벤치 (s7-36-prep-evalset.sh가 만든 openai/gsm8k test 샘플)
BENCH_LABELS = {"gsm8k": "GSM8K"}
BENCH_ORDER = ("gsm8k",)

# 타겟별 on-disk 포맷과 README에 박을 양자화 사양
SPECS: dict[str, dict] = {
    "awq-w5": {
        "format": PACK_QUANTIZED, "tags": ["w5a16", "gptq"],
        "algorithm": "GPTQ (after AWQ smoothing)", "precision": "W5A16",
        "bits": "5-bit int", "group_size": "group_size=64, symmetric=False",
        "ignored": "`lm_head`, `moe.gate` (router), `self_attn`, `shared_experts`",
    },
    "awq-w3": {
        "format": PACK_QUANTIZED, "tags": ["w3a16", "gptq"],
        "algorithm": "GPTQ (after AWQ smoothing)", "precision": "W3A16",
        "bits": "3-bit int", "group_size": "group_size=64, symmetric=False",
        "ignored": "`lm_head`, `moe.gate` (router), `self_attn`, `shared_experts`",
    },
    "nvfp4-awq": {
        "format": NVFP4_PACK_QUANTIZED, "tags": ["nvfp4", "w4a4", "awq"],
        "algorithm": "NVFP4 (after AWQ smoothing)", "precision": "W4A4 (FP8 dynamic activation scale)",
        "bits": "NVFP4 (E2M1)", "group_size": "block_size=16",
        "ignored": "`lm_head`, `moe.gate` (router), `self_attn`, `shared_experts`",
    },
    "nvfp4-gptq": {
        "format": NVFP4_PACK_QUANTIZED, "tags": ["nvfp4", "w4a4", "gptq"],
        "algorithm": "NVFP4 (GPTQ)", "precision": "W4A4 (FP8 dynamic activation scale)",
        "bits": "NVFP4 (E2M1)", "group_size": "block_size=16",
        "ignored": "`lm_head`, `moe.gate` (router), `self_attn`, `shared_experts`",
    },
}
# 캘리브레이션 스펙은 config/common.env가 정한 실제 값이다 (full 기본 256/2048).
# 카드에 박히는 값이 파이프라인이 만든 값과 어긋나면 안 되므로 환경변수를 따른다.
CALIB_NOTE = (
    f"{os.environ.get('CALIB_N', '256')} samples, "
    f"seq_len {os.environ.get('SEQ_LEN', '2048')} "
    "(in-domain corpus: Korean instruction data mixed with English chat data)"
)
IGNORE_NOTE = "`lm_head`, `mlp.gate`, `self_attn`, `shared_experts` (kept in BF16)"


def MODELS_DIR(mode: str) -> str:
    return f"{SHARED}/models"


def BF16_DIR() -> str:
    """19-dequant 복원본 경로. 모드 무관(한 번만 만든다)이라 mode를 받지 않는다."""
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


def _gsm8k_acc_n(scores: dict, target: str) -> tuple[float | None, str]:
    """표준 GSM8K (정확도, "정답/전체")."""
    sc = (scores.get(target) or {}).get("benchmarks") or {}
    row = sc.get("gsm8k") or {}
    acc = row.get("acc")
    n = row.get("n")
    if acc is None or not n:
        return None, "-"
    return acc, f"{round(float(acc) * int(n))}/{int(n)}"


def _gsm8k_acc(scores: dict, target: str) -> float | None:
    acc, _ = _gsm8k_acc_n(scores, target)
    return acc


def _gsm8k_verdict(acc: float | None, base_acc: float | None, drop_max: float) -> str:
    """base 대비 정확도 하락이 허용폭 이내면 PASS."""
    if acc is None:
        return "-"
    if base_acc is None:
        return "n/a"
    return "PASS" if (float(base_acc) - float(acc)) <= float(drop_max) else "FAIL"


def _acceptance_target_spec(target: str) -> dict:
    try:
        from sdlc import load_acceptance, target_spec

        return target_spec(load_acceptance(), target)
    except Exception:  # noqa: BLE001 — 계약 파일이 없으면 기본값
        return {"ppl_delta_warn": 0.3, "ppl_delta_max": 1.0, "gsm8k_acc_drop_max": 0.02}


def _evalset_meta(mode: str) -> str:
    """평가 세트 매니페스트 한 줄 요약 (데이터셋·리비전·시드·표본수)."""
    path = Path(SHARED) / "outputs" / f"eval-{mode}" / f"evalset-{mode}.json"
    if not path.is_file():
        return "evalset manifest not found"
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "evalset manifest could not be read"
    rev = str(m.get("revision", ""))[:12]
    return f"{m.get('dataset')}@{rev}, seed={m.get('seed')}, n={m.get('written_n')}"


def _acceptance_rows(target: str, mode: str, ppl: dict, size_gb: float) -> list[str]:
    """Stage 1 수용 기준(acceptance.toml) vs 실측 대조표(모델 카드용)."""
    try:
        from sdlc import load_acceptance, target_spec, verdict_for_delta

        data = load_acceptance()
    except Exception:  # noqa: BLE001 — 계약 파일이 없으면 기준 표를 생략
        return ["_Acceptance contract file (config/acceptance.toml) not found - comparison table omitted._"]
    spec = target_spec(data, target)
    row = (ppl or {}).get(target) or {}
    delta = row.get("delta")
    verdict = row.get("verdict")
    if delta is not None and not verdict:
        try:
            verdict = verdict_for_delta(float(delta), spec)
        except Exception:  # noqa: BLE001
            verdict = None
    lo, hi = (spec.get("size_gb") or [None, None])[:2]
    size_ok = "n/a"
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        size_ok = "PASS" if float(lo) <= float(size_gb) <= float(hi) else "FAIL"
    rows = [
        "| Criterion | Threshold | Measured | Verdict |",
        "| --- | --- | --- | --- |",
        (
            f"| PPL delta vs base | < {spec.get('ppl_delta_max')} (Conditional limit) | "
            f"{_fmt(delta)} | {verdict or 'n/a'} |"
        ),
        (
            f"| PPL delta (warning line) | < {spec.get('ppl_delta_warn')} (Accept limit) | "
            f"{_fmt(delta)} | {'Accept' if verdict == 'Accept' else 'n/a'} |"
        ),
        f"| Checkpoint size (GB) | {lo} ~ {hi} | {size_gb} | {size_ok} |",
        (
            f"| Storage format | `{spec.get('quant_format')}` | `{SPECS[target]['format']}` | "
            f"{'PASS' if spec.get('quant_format') == SPECS[target]['format'] else 'FAIL'} |"
        ),
    ]
    return rows


def render_readme(target: str, mode: str, ppl: dict, scores: dict, size_gb: float) -> str:
    """HuggingFace 모델 카드 전체를 만든다 (요구사항 2.1-3)."""
    spec = SPECS[target]
    _spec = _acceptance_target_spec(target)
    ppl_delta_warn = _spec.get("ppl_delta_warn", 0.3)
    ppl_delta_max = _spec.get("ppl_delta_max", 1.0)
    gsm8k_drop = _spec.get("gsm8k_acc_drop_max", 0.02)
    gsm8k_meta = _evalset_meta(mode)
    acceptance_rows = _acceptance_rows(target, mode, ppl, size_gb)
    rows = ["base", *ALL_TARGETS]
    # 검사 1 — PPL (in-domain perplexity, base 대비 Δ + 판정)
    ppl_table = [
        "| Model | PPL | Delta vs bf16 | Verdict |",
        "| --- | ---: | ---: | --- |",
    ]
    # 검사 2 — GSM8K (표준 테스트셋 정확도)
    gsm_table = [
        "| Model | GSM8K accuracy | Correct / Total | Verdict |",
        "| --- | ---: | ---: | --- |",
    ]
    base_acc = _gsm8k_acc(scores, "base")
    for r in rows:
        name = "**bf16 (base)**" if r == "base" else f"**{r}**" if r == target else r
        pr = ppl.get(r) or {}
        ppl_table.append(
            f"| {name} | {_fmt(pr.get('ppl'), 3)} | {_fmt(pr.get('delta'))} | {pr.get('verdict', '-')} |"
        )
        acc, cnt = _gsm8k_acc_n(scores, r)
        gsm_table.append(
            f"| {name} | {_fmt(acc, 3)} | {cnt} | {_gsm8k_verdict(acc, base_acc, gsm8k_drop)} |"
        )


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

This is a **{target}** quantized version of `{BASE_MODEL}`. Only the weights are
compressed to {spec['precision']}; activations and normalization layers are kept in BF16.
The checkpoint is stored in the `compressed-tensors` format (`{spec['format']}`), so it
loads directly with `transformers`' `AutoModelForCausalLM`.

## Quantization spec

| Item | Value |
| --- | --- |
| Base model | `{BASE_MODEL}` |
| Base revision | `{BASE_REVISION}` |
| Quantization algorithm | {spec['algorithm']} |
| Weight precision | {spec['precision']} |
| Bit width | {spec['bits']} |
| Group size | {spec['group_size']} |
| Layers left unquantized | {IGNORE_NOTE} |
| Calibration | {CALIB_NOTE} |
| Checkpoint size | {size_gb} GB |
| Storage format | `{spec['format']}` |

## Acceptance criteria (Stage 1 contract)

The acceptance criteria are the pipeline contract defined in `config/acceptance.toml`.
The table below contrasts that contract with the measured values.

{chr(10).join(acceptance_rows)}

## Check 1 - PPL (in-domain perplexity)

In-domain perplexity measured on the calibration corpus. The verdict is based on the
delta against the bf16 base model (`delta < {ppl_delta_warn}` Accept,
`< {ppl_delta_max}` Conditional, otherwise Discard; thresholds from
`config/acceptance.toml`). The **bold** row is the model published in this repository.

{chr(10).join(ppl_table)}

## Check 2 - GSM8K (standard test set accuracy)

Accuracy on a deterministic sample of the standard GSM8K test set
(`openai/gsm8k`, main/test). The prompts are generated deterministically by
`s7-36-prep-evalset.sh` ({gsm8k_meta}) and scored by s7-37-eval / s7-40-score.
The verdict is PASS when the accuracy drop against the base model stays within
`{gsm8k_drop}`.

{chr(10).join(gsm_table)}

## Usage - transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{target}"  # or the published repo id below
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
messages = [{{"role": "user", "content": "What is 2+2?"}}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(inputs, max_new_tokens=256)[0], skip_special_tokens=True))
```

## Usage - vLLM

The Raon-1-mini family uses a custom architecture, so the KRAFTON vLLM fork is required.

```bash
vllm serve {target} \\
  --trust-remote-code \\
  --quantization compressed-tensors \\
  --tensor-parallel-size 2
```

## License and source

- Apache License 2.0. The code in this repository is distributed under Apache-2.0.
- Base model: [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) (rev `{BASE_REVISION}`), Apache-2.0.
- The original KRAFTON repository is access-gated; this repository does not re-gate it.
  The base model license terms (`apache-2.0`) still apply.
"""


def _copy_missing_custom_code(d: Path, copied: list) -> list:
    """누락된 custom_code/*.jinja를 models/raon-bf16에서 복사 보충."""
    src = Path(BF16_DIR())
    for name in COPY_FROM_BF16:
        if (d / name).is_file():
            continue
        s = src / name
        if not s.is_file():
            raise RuntimeError(f"{d}: {name} is missing and no source is available to copy it from ({s})")
        shutil.copy(s, d / name)
        copied.append(name)
    return copied


def prepare_target(mode: str, target: str, allow_missing_metrics: bool = False) -> dict:
    """타겟 1개 준비. 실패해도 예외 대신 ok=False + errors를 담은 보고서를 돌려준다."""
    report: dict = {"target": target, "mode": mode, "ok": False, "errors": [], "copied": [], "size_gb": None}
    d = model_dir(mode, target)
    if not (d / "config.json").is_file():
        report["errors"].append(f"{d}: config.json missing - model directory not created (run the quantization step first)")
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
            report["errors"].append(f"{d}: {name} missing - the model cannot be loaded after upload")

    # 4) 지표 → 모델 카드
    ppl = _ppl_summary(mode)
    scores = _eval_scores(mode)
    if not allow_missing_metrics:
        if target not in ppl:
            report["errors"].append(f"no measured value for '{target}' in outputs/ppl-{mode}/summary.json (run s6-34-measure-ppl first)")
        if target not in scores:
            report["errors"].append(f"no result for '{target}' in outputs/eval-{mode}/score.json (run s7-37-eval/s7-40-score first)")
    report["size_gb"] = _dir_size_gb(d)
    readme = render_readme(target, mode, ppl, scores, report["size_gb"])
    for bad in FORBIDDEN_FRONTMATTER:
        if bad in readme.split("---", 2)[1]:
            report["errors"].append(f"frontmatter contains a forbidden field '{bad}'")
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
        raise SystemExit(f"unknown target(s): {unknown} (allowed: {ALL_TARGETS})")
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