"""s4-65-update-readme.py — 업로드 전 모델 디렉터리 검증 + 모델 카드(README.md) 생성.
s8-01-upload-hf.py는 이 스크립트가 채운 폴더를 그대로 Hub에 올리기만 한다
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

  uv run s4-65-update-readme.py --mode smoke --targets nvfp4-awq-transformers --allow-missing-metrics
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from quant_common import (
    MOPT_NVFP4,
    NVFP4_PACK_QUANTIZED,
    PACK_QUANTIZED,
    verify_compressed_output,
)

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")
BASE_MODEL = os.environ.get("BASE_MODEL", "KRAFTON/Raon-1-mini-preview")
# 실측 캐시 스냅샷 (config/common.env BASE_REVISION과 같은 값 — 카드에 기록한다)
BASE_REVISION = os.environ.get("BASE_REVISION") or "da4cf162c8613993f2189ebb438a5f9a23f3e5ca"
ALL_TARGETS = ["awq-w3-transformers", "awq-w5-transformers", "nvfp4-awq-transformers", "nvfp4-gptq-transformers", "nvfp4-awq-vllm"]
COPY_FROM_BF16 = ("chat_template.jinja", "configuration_raon1_mini.py", "modeling_raon1_mini.py")
REQUIRED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
# 원본 KRAFTON 레포에는 gated 동의 필드가 있다. 라이선스 요청이 아니므로 절대 복사하지 않는다.
FORBIDDEN_FRONTMATTER = ("extra_gated_",)
# 표준 GSM8K 단일 벤치 (s4-05-prep-gsm8k.sh가 만든 openai/gsm8k test 샘플)
BENCH_LABELS = {"gsm8k": "GSM8K"}
BENCH_ORDER = ("gsm8k",)

# 타겟별 on-disk 포맷과 README에 기재할 양자화 사양
SPECS: dict[str, dict] = {
    "awq-w5-transformers": {
        "format": PACK_QUANTIZED, "tags": ["w5a16", "gptq"],
        "algorithm": "GPTQ (after AWQ smoothing)", "precision": "W5A16",
        "bits": "5-bit int", "group_size": "group_size=64, symmetric=False",
        "ignored": "`lm_head`, `moe.gate` (router), `self_attn`, `shared_experts`",
    },
    "awq-w3-transformers": {
        "format": PACK_QUANTIZED, "tags": ["w3a16", "gptq"],
        "algorithm": "GPTQ (after AWQ smoothing)", "precision": "W3A16",
        "bits": "3-bit int", "group_size": "group_size=64, symmetric=False",
        "ignored": "`lm_head`, `moe.gate` (router), `self_attn`, `shared_experts`",
    },
    "nvfp4-awq-transformers": {
        "format": NVFP4_PACK_QUANTIZED, "tags": ["nvfp4", "w4a4", "awq"],
        "algorithm": "NVFP4 (after AWQ smoothing)", "precision": "W4A4 (FP8 dynamic activation scale)",
        "bits": "NVFP4 (E2M1)", "group_size": "block_size=16",
        "ignored": "`lm_head`, `moe.gate` (router), `self_attn`, `shared_experts`",
    },
    "nvfp4-gptq-transformers": {
        "format": NVFP4_PACK_QUANTIZED, "tags": ["nvfp4", "w4a4", "gptq"],
        "algorithm": "NVFP4 (GPTQ)", "precision": "W4A4 (FP8 dynamic activation scale)",
        "bits": "NVFP4 (E2M1)", "group_size": "block_size=16",
        "ignored": "`lm_head`, `moe.gate` (router), `self_attn`, `shared_experts`",
    },
    "nvfp4-awq-vllm": {
        "format": MOPT_NVFP4, "tags": ["nvfp4", "w4a4", "awq", "modelopt", "gb10"],
        "algorithm": "NVFP4 (AWQ Lite, ModelOpt)", "precision": "W4A4 (FP8 dynamic activation scale)",
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


#: Hub repo id 규칙 — s8-01-upload-hf.py의 repo_name과 같아야 한다
#: (full → <HF_ID>/<BASE>-<target>, smoke → ...-<target>-smoke)
HF_ID = os.environ.get("HF_ID", "tayaee")
BASE_SLUG = os.environ.get("BASE_MODEL", "KRAFTON/Raon-1-mini-preview").split("/")[-1]

#: vLLM 서빙 명령 스펙 — s9-13-vllm-serve-one.sh의 case 테이블과 값을 공유한다.
#: 4종 양자화 타깃은 repo id를 제외하고 동일 명령이다. TP=1 (KRAFTON NVFP4 단일 GPU
#: 프로파일; 20GB급 체크포인트는 단일 GPU에 올라간다). base는 Hub 카드가 없어
#: 여기서 다루지 않는다 (s9-13에서만 로컬 서빙, TP는 infra 기본값).
VLLM_SERVE = {
    "awq-w3-transformers": {"tp": 1, "quant": "compressed-tensors"},
    "awq-w5-transformers": {"tp": 1, "quant": "compressed-tensors"},
    "nvfp4-awq-transformers": {"tp": 1, "quant": "compressed-tensors"},
    "nvfp4-gptq-transformers": {"tp": 1, "quant": "compressed-tensors"},
    "nvfp4-awq-vllm": {"tp": 1, "quant": ""},  # ModelOpt NVFP4: 자동 감지, --quantization 지정 금지
}
VLLM_REF = "raon-1-mini-preview"
VLLM_REPO = "https://github.com/krafton-ai/vllm.git"

#: 코드 샘플 단일 소스 — 모델 카드에 박히는 실행 예제와 s6/s7 검증이 같은 파일에서 나온다.
#: 카드 렌더·검증 스크립트가 이 파일들을 직접 읽어 쓰므로, 온라인 README를 긁어
#: 다시 테스트할 필요가 없다 (s6/s7 검증 = README instruction 검증).
#: __REPO_ID__ 자리는 렌더/실행 시점에 full/smoke repo id(또는 로컬 모델 경로)로 바뀐다.
SNIPPET_DIR = Path(__file__).resolve().parent / "snippets"


def load_snippet(target: str, name: str, repo_id: str) -> str:
    """snippets/<target>/<name> 내용을 repo id 치환 후 돌려준다 (fence 제외 본문)."""
    text = (SNIPPET_DIR / target / name).read_text(encoding="utf-8")
    return text.removesuffix("\n").replace("__REPO_ID__", repo_id)


def hf_repo_id(target: str, mode: str) -> str:
    """정식 HF 모델명(Hub repo id). 표의 Model 컬럼에 쓴다."""
    if target == "nvfp4-awq-vllm":
        slug = "Raon-1-mini-preview-NVFP4-AWQ-GB10-vLLM"
    else:
        slug = f"{BASE_SLUG}-{target}"
    name = f"{HF_ID}/{slug}"
    if mode == "smoke":
        name += "-smoke"
    return name


def model_label(target: str, mode: str) -> str:
    """표의 Model 컬럼 값 — base는 원본 repo, 양자화본은 업로드 repo id."""
    if target == "base":
        return BASE_MODEL
    return hf_repo_id(target, mode)


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


def _vllm_base_bench():
    """s0-13 컨테이너 경로로 잰 vLLM 베이스(NVFP4 릴리스) 고정-8문항 결과. 없으면 {}."""
    try:
        with open(os.path.join(SHARED, "outputs", "s0-bench-vllm-s4",
                               "results.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}




def _checks(mode: str) -> dict:
    """s7 검사 결과(checks.json): gsm8k 통계 · kmmluko · format · servebench."""
    path = Path(SHARED) / "outputs" / f"eval-{mode}" / "checks.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _kmmluko_acc(checks: dict, target: str) -> float | None:
    row = (((checks.get("kmmluko") or {}).get("targets") or {}).get(target)) or {}
    return row.get("acc")


def _format_rate(checks: dict, target: str) -> float | None:
    row = (((checks.get("format") or {}).get("targets") or {}).get(target)) or {}
    return row.get("pass_rate")


def _gsm8k_drop(checks: dict, target: str) -> float | None:
    d = ((checks.get("gsm8k") or {}).get("delta_vs_base") or {}).get(target)
    return None if d is None else float(d)


def _kmmluko_drop(checks: dict, target: str) -> float | None:
    d = ((checks.get("kmmluko") or {}).get("delta_vs_base") or {}).get(target)
    return None if d is None else float(d)


def _gsm8k_contract_verdict(checks: dict, target: str) -> str:
    return str(((checks.get("gsm8k") or {}).get("verdict") or {}).get(target, "-"))


def _kmmluko_contract_verdict(checks: dict, target: str) -> str:
    return str(((checks.get("kmmluko") or {}).get("verdict") or {}).get(target, "-"))


def _format_contract_verdict(checks: dict, target: str) -> str:
    """형식/지시 준수 판정 — 채점 결과(s4-22)의 verdict를 따른다.

    verdict가 없으면(구 산출물) 계약 하한으로 직접 계산한다. 통과율이 있다는
    이유만으로 PASS를 돌려주면 **가짜 통과**가 카드에 찍힌다.
    """
    v = ((checks.get("format") or {}).get("verdict") or {}).get(target)
    if v:
        return str(v)
    row = (((checks.get("format") or {}).get("targets") or {}).get(target)) or {}
    rate = row.get("pass_rate")
    if rate is None:
        return "-"
    min_rate = float(row.get("pass_rate_min", 0.9))
    return "PASS" if float(rate) >= min_rate else "FAIL"



def _existing_targets(mode: str, ppl: dict, scores: dict) -> list[str]:
    """이 모드에서 **존재하는** 양자화 버전(모델 디렉터리 또는 실측 지표가 있는 것)."""
    out: list[str] = []
    for t in ALL_TARGETS:
        if model_dir(mode, t).is_dir() or t in ppl or t in scores:
            out.append(t)
    return out


def _numbered_rows(items: list[tuple[bool, str]]) -> list[str]:
    """(표시 여부, 본문) 목록에서 표시분만 꺼내 `| n | 본문 |` 형태로 번호를 다시 매긴다."""
    return [f"| {i} | {body} |" for i, (_, body) in
            enumerate([it for it in items if it[0]], 1)]


def _comparison_table(target: str, mode: str, ppl: dict, scores: dict, size_gb: float,
                      checks: dict) -> str:
    """Base 대비 바뀐 점 1표 (모델 카드용).

    Acceptance criteria ~ Serving performance 섹션을 이 1표로 대체한다.
    기준값과 Verdict는 컬럼 4에 함께 적는다. 카드는 영어 유지.
    """
    spec = _acceptance_target_spec(target)
    base_p = (ppl.get("base") or {})
    cur_p = (ppl.get(target) or {})
    delta = cur_p.get("delta")
    warn = spec.get("ppl_delta_warn", 0.3)
    maxd = spec.get("ppl_delta_max", 1.0)
    ppl_note = f"\u0394 {_fmt(delta, 3)} \u2014 {cur_p.get('verdict', '-')} (Accept < {warn}, Conditional < {maxd})"

    base_size = _dir_size_gb(Path(BF16_DIR()))
    lo, hi = (spec.get("size_gb") or [None, None])[:2]
    size_ok = "PASS" if (isinstance(lo, (int, float)) and isinstance(hi, (int, float))
                         and float(lo) <= float(size_gb) <= float(hi)) else "-"
    size_note = f"{size_ok} (contract {lo} ~ {hi} GB)"

    fmt_ok = "PASS" if spec.get("quant_format") == SPECS[target]["format"] else "FAIL"
    fmt_note = f"{fmt_ok} (contract `{spec.get('quant_format')}`)"

    b_acc, b_cnt = _gsm8k_acc_n(scores, "base")
    c_acc, c_cnt = _gsm8k_acc_n(scores, target)
    gdrop_max = spec.get("gsm8k_acc_drop_max", 0.02)
    gsm_v = _gsm8k_verdict(c_acc, b_acc, gdrop_max)
    gsm_drop = "-" if b_acc is None or c_acc is None else f"{float(b_acc) - float(c_acc):.3f}"
    gsm_note = f"{gsm_v} (drop {gsm_drop} <= {gdrop_max})"

    k_acc = _kmmluko_acc(checks, target)
    kb_acc = _kmmluko_acc(checks, "base")
    k_row = ((checks.get("kmmluko") or {}).get("targets") or {}).get(target) or {}
    kb_row = ((checks.get("kmmluko") or {}).get("targets") or {}).get("base") or {}
    kdrop_max = spec.get("kmmluko_acc_drop_max", 0.03)
    k_v = _kmmluko_contract_verdict(checks, target)
    k_cnt = "-" if not k_row.get("n") else f"{round(float(k_acc or 0) * float(k_row['n']))}/{k_row['n']}"
    kb_cnt = "-" if not kb_row.get("n") else f"{round(float(kb_acc or 0) * float(kb_row['n']))}/{kb_row['n']}"
    k_note = f"{k_v} (drop {_fmt(_kmmluko_drop(checks, target), 3)} <= {kdrop_max})"

    f_row = ((checks.get("format") or {}).get("targets") or {}).get(target) or {}
    fb_row = ((checks.get("format") or {}).get("targets") or {}).get("base") or {}
    f_rate = _format_rate(checks, target)
    fb_rate = _format_rate(checks, "base")
    f_cnt = "-" if not f_row.get("n") else f"{round((f_rate or 0) * f_row['n'])}/{f_row['n']}"
    fb_cnt = "-" if not fb_row.get("n") else f"{round((fb_rate or 0) * fb_row['n'])}/{fb_row['n']}"
    f_min = float(f_row.get("pass_rate_min", spec.get("format_pass_rate_min", 0.9)))
    f_note = f"{_format_contract_verdict(checks, target)} (>= {f_min})"

    sb = checks.get("servebench") or {}
    sbt = sb.get("targets") or {}
    run = (sbt.get(target) or {}).get("1") or {}
    brun = (sbt.get("base") or {}).get("1") or {}
    bkend = sb.get("backend") or (sb.get("hardware") or {}).get("backend") or "transformers"
    tps = f"{run.get('tok_per_s_p50', '—')} tok/s"
    btps = f"{brun.get('tok_per_s_p50', '—')} tok/s"
    lat = f"{run.get('req_latency_s_p50', '—')} s"
    blat = f"{brun.get('req_latency_s_p50', '—')} s"
    serve_note = f"in-process {bkend} (no fused kernels; not a serving claim)"

    vb = _vllm_base_bench()
    v_rows = vb.get("rows") or []
    v_ok = sum(1 for r in v_rows if r.get("ok"))
    v_lat = (round(sum(r.get("latency_s", 0) for r in v_rows) / len(v_rows), 3)
             if v_rows else None)

    def _pct(new, old):
        try:
            return f"{(float(new) / float(old) - 1) * 100:+.0f}%"
        except (TypeError, ValueError, ZeroDivisionError):
            return "—"

    b_tps = brun.get('tok_per_s_p50')
    c_tps = run.get('tok_per_s_p50')
    tf_change = ("—" if b_tps is None or c_tps is None
                 else f"{b_tps} → {c_tps} ({_pct(c_tps, b_tps)})")
    b_lat = brun.get('req_latency_s_p50')
    c_lat = run.get('req_latency_s_p50')
    lat_change = ("—" if b_lat is None or c_lat is None
                  else f"{b_lat} → {c_lat} ({_pct(c_lat, b_lat)})")
    size_change = ("—" if base_size is None else f"{base_size} → {size_gb} ({_pct(size_gb, base_size)})")
    # 측정 스택 주석: base 는 transformers in-process, 이 타깃은 vLLM 포크 서빙일 수 있다.
    # 스택이 다르면 Δ 에 양자화 손실과 스택 차이가 섞인다 — 카드에 명시한다.
    _stack_note = ""
    _bmark = Path(OUT_ROOT()) / f"eval-{mode}" / f"backend-{target}.json"
    if _bmark.is_file():
        try:
            _bm = json.loads(_bmark.read_text(encoding="utf-8"))
        except ValueError:
            _bm = {}
        if _bm.get("backend") == "server":
            _stack_note = (
                "[^3]: This model's accuracy rows were measured through the vLLM fork "
                "(the checkpoint cannot be loaded in-process by `transformers`), while "
                "the Base column is transformers in-process. The drop therefore mixes "
                "quantization loss with stack/kernel differences — read it as an upper "
                "bound on the quantization loss, not an exact delta."
            )
    rows = [
        "| # | Configuration | Base Model | This Model | Change | Notes |",
        "| --- | --- | ---: | ---: | --- | --- |",
        # 측정 불가 항목은 빼고 번호를 다시 매긴다 (예: ModelOpt는 in-process PPL 불가).
        *_numbered_rows([
            (SPECS[target]["format"] != MOPT_NVFP4,
             f"PPL (in-domain) | {_fmt(base_p.get('ppl'), 3)} | {_fmt(cur_p.get('ppl'), 3)} | Δ {_fmt(delta, 3)} | {cur_p.get('verdict', '-')} (Accept < {warn}, Conditional < {maxd})"),
            (True,
             f"Checkpoint size | {base_size} GB | {size_gb} GB | {size_change} | {size_note}"),
            (True,
             f"Storage format | BF16 (dequantized) | `{SPECS[target]['format']}` | — | {fmt_note}"),
            (True,
             f"GSM8K accuracy | {_fmt(b_acc, 3)} ({b_cnt}) | {_fmt(c_acc, 3)} ({c_cnt}) | drop {gsm_drop} | {gsm_note}"),
            (True,
             f"KMMLU accuracy | {_fmt(kb_acc, 3)} ({kb_cnt}) | {_fmt(k_acc, 3)} ({k_cnt}) | drop {_fmt(_kmmluko_drop(checks, target), 3)} | {k_note}"),
            (True,
             f"Format compliance | {_fmt(fb_rate, 3)} ({fb_cnt}) | {_fmt(f_rate, 3)} ({f_cnt}) | — | {f_note}"),
            (True,
             f"Throughput C=1, transformers | {btps} | {tps} | {tf_change} | reference only[^2] — both slow eager (stack, not a quantization regression)"),
            (True,
             f"Latency p50, transformers | {blat} | {lat} | {lat_change} | same fixed workload[^2]"),
            (True,
             f"vLLM fork latency, s (C=1; n={len(v_rows)}, ok={v_ok}) | {v_lat if v_lat is not None else '—'} | n/a[^1] | unmeasurable — see note"),
        ]),
        "",
        "[^2]: Rows 7–8 measured on a single NVIDIA GB10 (DGX Spark), in-process "
        "transformers, same fixed 8 prompts. Numbers move with hardware — compare "
        "Base vs This Model relatively, not absolutely.",
        "[^1]: Quant·vLLM unmeasurable: the pinned fork deletes llmcompressor "
        "`quantization_config` at engine init (serves only MXFP8-release/ModelOpt "
        "profiles → `TypeError`), and upstream vLLM lacks RDA `linear_attention`. "
        "Serving these PTQ checkpoints needs a ModelOpt-NVFP4 recipe.",
    ]
    if _stack_note:
        rows.append(_stack_note)
    return "\n".join(rows) + "\n"


def _check_tables(
    rows: list[str],
    bold: set[str],
    mode: str,
    ppl: dict,
    scores: dict,
    checks: dict,
    gsm8k_drop: float,
) -> tuple[list[str], list[str], list[str], list[str], str]:
    """검사 표 4종(PPL·GSM8K·KMMLU·형식 준수) + 서빙 성능 줄.

    `rows`에 들어있는 모델만 표에 실린다 — 모델 카드는 base + 대상 1개,
    요약 리포트는 base + 그 모드의 전체 양자화 버전.
    Model 컬럼에는 **정식 HF 모델명**(Hub repo id)을 쓴다.
    """
    def _name(r: str) -> str:
        label = model_label(r, mode)
        if r == "base":
            return f"**{label}** (base)"
        return f"**{label}**" if r in bold else label

    ppl_table = [
        "| Model | PPL | Delta vs bf16 | Verdict |",
        "| --- | ---: | ---: | --- |",
    ]
    gsm_table = [
        "| Model | GSM8K accuracy | Correct / Total | Verdict |",
        "| --- | ---: | ---: | --- |",
    ]
    ko_table = [
        "| Model | KMMLU accuracy | Correct / Total | Verdict |",
        "| --- | ---: | ---: | --- |",
    ]
    fmt_table = [
        "| Model | Pass rate | Passed / Total | Failed cases | Verdict |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for r in rows:
        acc = _kmmluko_acc(checks, r)
        cnt = ((checks.get("kmmluko") or {}).get("targets") or {}).get(r) or {}
        n = cnt.get("n")
        correct = round(float(acc) * float(n)) if (acc is not None and n) else None
        ko_table.append(
            f"| {_name(r)} | {_fmt(acc, 3)} | "
            f"{'-' if correct is None else f'{correct}/{n}'} | "
            f"{'ref' if r == 'base' else _kmmluko_contract_verdict(checks, r)} |"
        )
    for r in rows:
        row = (((checks.get("format") or {}).get("targets") or {}).get(r)) or {}
        rate = row.get("pass_rate")
        passed = round((rate or 0) * (row.get("n") or 0))
        fmt_table.append(
            f"| {_name(r)} | {_fmt(rate, 3)} | {passed}/{row.get('n', '-')} | "
            f"{', '.join(row.get('failed') or []) or '-'} | "
            f"{'ref' if r == 'base' else _format_contract_verdict(checks, r)} |"
        )
    base_acc = _gsm8k_acc(scores, "base")
    for r in rows:
        pr = ppl.get(r) or {}
        ppl_table.append(
            f"| {_name(r)} | {_fmt(pr.get('ppl'), 3)} | {_fmt(pr.get('delta'), 3)} | {pr.get('verdict', '-')} |"
        )
        acc, cnt = _gsm8k_acc_n(scores, r)
        gsm_table.append(
            f"| {_name(r)} | {_fmt(acc, 3)} | {cnt} | "
            f"{'ref' if r == 'base' else _gsm8k_verdict(acc, base_acc, gsm8k_drop)} |"
        )
    sb = checks.get("servebench") or {}
    serve_lines = _serve_section(sb, rows, mode)
    return ppl_table, gsm_table, ko_table, fmt_table, serve_lines


def _serve_section(sb: dict, rows: list[str], mode: str) -> str:
    """서빙 성능 비교표 + 측정 하드웨어 조건.

    `s4-60-measure-speed__transformers-only.py`가 타깃별로 누적한 checks.json servebench.targets를
    타깃 × concurrency 표로 렌더링한다. 구 형태(단일 측정 payload)도 fallback 지원.
    backend가 transformers면 인프로세스 측정 조건을, vLLM이면 서버 조건을 표기한다
    (하드웨어 키가 달라 `vLLM ?` 같은 빈 값을 찍지 않기 위함).
    """
    hw = sb.get("hardware") or {}
    backend = sb.get("backend") or hw.get("backend") or ""
    if hw:
        if backend == "transformers":
            hw_line = (
                f"Hardware: {hw.get('gpu', '?')} (unified {hw.get('unified_mem_gb', '?')} GB) · "
                f"{hw.get('engine', 'transformers')} · in-process (no server) · "
                f"TP={hw.get('tensor_parallel', '?')} · "
                f"max-model-len={hw.get('max_model_len', '?')} · "
                f"temperature=0.0, max_tokens={sb.get('max_tokens', '?')}, "
                f"warmup=1, n={sb.get('n', '?')} sequential (C=1 greedy)."
            )
        else:
            hw_line = (
                f"Hardware: {hw.get('gpu', '?')} (unified {hw.get('unified_mem_gb', '?')} GB) · "
                f"vLLM {hw.get('vllm_version', '?')} · "
                f"TP={hw.get('tensor_parallel', '?')} · "
                f"gpu-mem-util={hw.get('gpu_memory_utilization', '?')} · "
                f"max-model-len={hw.get('max_model_len', '?')} · "
                f"temperature=0.0, max_tokens=256, warmup=2, n=32 per concurrency."
            )
    else:
        hw_line = "Hardware: not recorded."
    targets = sb.get("targets") or {}
    lines = [hw_line, ""]
    if isinstance(targets, dict) and targets:
        lines += [
            "| Model | C=1 tok/s (p50) | C=8 tok/s (p50) | C=32 tok/s (p50) | C=1 latency p50 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for r in [t for t in rows if t in targets] + [t for t in targets if t not in rows]:
            runs = targets.get(r) or {}
            label = model_label(r, mode) if r in rows else r
            cells = []
            for c in ("1", "8", "32"):
                v = (runs.get(c) or {}).get("tok_per_s_p50", "—")
                cells.append(str(v))
            lat = (runs.get("1") or {}).get("req_latency_s_p50", "—")
            lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {lat} s |")
        mem_lines = []
        for r in [t for t in rows if t in targets] + [t for t in targets if t not in rows]:
            mem = (targets.get(r) or {}).get("memory") or {}
            if not mem:
                continue
            avail = {p: (mem.get(p) or {}).get("mem_avail_gb", "—") for p in
                     ("before_load", "after_warmup", "before_teardown", "after_teardown")}
            rss = max([float((mem.get(p) or {}).get("server_rss_gb") or 0)
                       for p in mem] + [0.0])
            label = model_label(r, mode) if r in rows else r
            mem_lines.append(
                f"- {label} unified-mem avail (GB): "
                f"before_load {avail['before_load']} → after_warmup {avail['after_warmup']} → "
                f"before_teardown {avail['before_teardown']} → after_teardown {avail['after_teardown']} "
                f"(server RSS peak {rss:.1f} GB)")
        if mem_lines:
            lines += ["", *mem_lines]
    elif sb.get("ok"):
        lines += [
            "| Model | Concurrency | Output tok/s (p50) | Request latency (p50) | Output tokens |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| {sb.get('model', '-')} | {sb.get('concurrency', '-')} | {sb.get('tok_per_s_p50', '-')} | "
            f"{sb.get('req_latency_s_p50', '-')} s | {sb.get('output_tokens_total', '-')} |",
        ]
    else:
        lines.append("_Not measured (no running server during the last run)._")
    return "\n".join(lines) + "\n"



def render_readme(target: str, mode: str, ppl: dict, scores: dict, size_gb: float) -> str:
    """HuggingFace 모델 카드 (요구사항 2.1-3).

    업로드되는 산출물이므로 **base + 이 모델 1개**만 비교한다.
    (전체 양자화 버전 비교는 models/README-<mode>.md 요약 리포트에 있다)
    """
    spec = SPECS[target]
    checks = _checks(mode)
    gsm8k_meta = _evalset_meta(mode)
    repo_id = hf_repo_id(target, mode)
    comparison = _comparison_table(target, mode, ppl, scores, size_gb, checks)
    is_mopt = target == "nvfp4-awq-vllm"
    storage_format = "modelopt_fp4" if is_mopt else "compressed-tensors"
    # 실행 예제는 snippets/<target>/ 단일 소스에서 읽는다 (s6/s7 검증과 동일 파일).
    gb10_build_block = ("```bash\n" + load_snippet(target, "build-vllm-gb10.sh", repo_id) + "\n```"
                        if is_mopt else "")
    serve_body = load_snippet(target, "serve-vllm.sh", repo_id)
    # transformers 예제는 vLLM 전용 타깃(modelopt_fp4)에선 카드에 없으므로 읽지 않는다.
    usage_body = "" if is_mopt else load_snippet(target, "usage-transformers.sh", repo_id)
    app_body = "" if is_mopt else load_snippet(target, "app-transformers.py", repo_id)
    if is_mopt:
        vllm_step1 = ("Step 1 — build vLLM from source (GB10). Prerequisites: `nvcc`\n"
                      "from CUDA 13.0 on `PATH`, ~50 GB free disk. Run verbatim in order;\n"
                      "the last line verifies the install.\n\n" + gb10_build_block + "\n\n"
                      "Step 2 — serve with the built binary.")
    else:
        vllm_step1 = ""
    prebuilt_note_lines = [
        "> On B300 / RTX 5090 — the GPUs KRAFTON tested — the precompiled fork works",
        "> without a source build: replace `$VENV/bin/vllm` above with",
        '> `VLLM_USE_PRECOMPILED=1 uvx --from "vllm @ git+https://github.com/krafton-ai/vllm.git@raon-1-mini-preview" vllm`.',
    ]
    vllm_prebuilt_note = ("\n".join(prebuilt_note_lines) + "\n") if is_mopt else ""
    if is_mopt:
        vllm_intro = ("The Raon-1-mini family uses a custom architecture, so stock vLLM cannot serve\n"
                      "it. On GB10 (sm_121) the KRAFTON precompiled fork fails at engine init, so\n"
                      "build the GB10 fork from source first (one-time), then serve with it.")
    else:
        vllm_intro = (f"The Raon-1-mini family uses a custom architecture, so stock vLLM cannot serve\n"
                      f"it. Use the KRAFTON vLLM fork (`{VLLM_REF}` branch). `uvx --from`\n"
                      "pins that exact fork without touching your environment\n"
                      "(`VLLM_USE_PRECOMPILED=1` selects the precompiled wheels; the first start\n"
                      "compiles the model's kernels and takes about 10 minutes).")

    # transformers 로 읽을 수 없는 타깃(modelopt_fp4)의 카드에는 transformers 실행 예제를
    # 싣지 않는다 — 싣으면 고객이 그대로 복사해 실행하고 반드시 실패한다 (2026-10-29 실측:
    # transformers 5.14.1 에 modelopt_fp4 양자화기가 없고, NVFP4 패킹된 가중치가 raw uint8 로
    # 적재되어 shape mismatch 로 죽는다).
    # BF16 변환 경로(동봉된 nvfp4_to_bf16.py)는 예제가 아니라 '참고'로만 언급한다 —
    # 변환본은 56GB 로 BF16 베이스(53.6GB)보다 커서, 양자화 모델의 transformers 경로로는
    # 이득이 없다. 이 모델의 고객 경로는 vLLM 포크 하나뿐이다.
    if is_mopt:
        transformers_section = """## Usage - transformers

**`transformers` cannot load this checkpoint.** It is stored in the `modelopt_fp4`
format (NVFP4-packed weights plus ModelOpt scales), which `transformers` does not
implement: `AutoModelForCausalLM.from_pretrained` logs
`Unknown quantization type, got modelopt_fp4` and then aborts with a weight shape
mismatch, because the packed tensors are never unpacked.

Serve it with the vLLM fork instead (see **Usage - vLLM** below) — the fork
implements the `modelopt_fp4` loader path natively, so the checkpoint stays at its
19 GB size.

If you specifically need a `transformers`-loadable copy, this repository ships
`nvfp4_to_bf16.py`, which converts the packed weights to BF16. Note that the result
is ~56 GB — larger than the BF16 base model — so it is a convenience, not the
intended path.
"""
        load_claim = (", which `transformers` cannot read —\n"
                      "serve it with the vLLM fork (see **Usage - vLLM** below)")
    else:
        transformers_section = f"""## Usage - transformers

```bash
{usage_body}
```

```python
{app_body}
```
```bash
python app.py  # runs in .venv above
```
"""
        load_claim = (", so it\n"
                      "loads directly with `transformers`' `AutoModelForCausalLM`")

    tags = ", ".join(["raon", "moe", storage_format, *spec["tags"]])
    fm = [
        "---",
        "license: apache-2.0",
        f"base_model: {BASE_MODEL}",
        "base_model_relation: quantized",
        "library_name: vllm" if is_mopt else "library_name: transformers",
        "pipeline_tag: text-generation",
        "language:",
        "- ko",
        "- en",
        "tags:",
        *(f"- {t}" for t in tags.split(", ")),
        "---",
    ]
    if is_mopt:
        intro_banner = f"""
> [!IMPORTANT]
> **Target Hardware: NVIDIA GB10 (DGX Spark, sm_121)**
> 
> * **Model Lineage**: [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}) (Base) ➔ **`{repo_id}`** (4-bit AWQ NVFP4)
> * **Required Serving Engine**: This model requires our customized vLLM engine targeting GB10 Blackwell hardware:  
>   👉 **[tayaee/raon-vllm-spark (branch: gb10-support)](https://github.com/tayaee/raon-vllm-spark/tree/gb10-support)**
> * **Notice for Non-GB10 Users**: This checkpoint is stored in `modelopt_fp4` format and is optimized strictly for GB10. If you are using other hardware (e.g., B300, H100, RTX 4090), standard `transformers` or stock `vLLM` will **not** load this checkpoint. Please use the official base model [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}) instead.
"""
    else:
        intro_banner = ""

    return "\n".join(fm) + f"""

# {repo_id}
{intro_banner}
This is a **{repo_id}** quantized version of `{BASE_MODEL}`{', targeting DGX Spark GB10' if is_mopt else ''}. Only the weights are
compressed to {spec['precision']}; activations and normalization layers are kept in BF16.
The checkpoint is stored in the `{storage_format}` format (`{spec['format']}`){load_claim}.

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

## Base vs this model

What changed against the bf16 base model (`{BASE_MODEL}`). Eval sets: GSM8K
({gsm8k_meta}). Verdict thresholds are listed in the table notes.

{comparison}

{transformers_section}
## Usage - vLLM

{vllm_intro}

{vllm_step1}

Full-context serving on a single GB10: flags follow the vendor B300 profile,
scaled to 128 GB. The 18 RDA layers hold fixed-size state and only the 7 MLA
layers keep KV cache (~8 KiB/token), so 262K context needs ~2 GB of KV cache.

```bash
{serve_body}
```

> `--kv-cache-dtype fp8_e4m3` and the `raon_1` parsers are fork features —
> verified on KRAFTON checkpoints{'' if is_mopt else ', untested on `compressed-tensors` builds'}.
> On a busy node lower `--gpu-memory-utilization` (e.g. `0.3`).
> Compatibility note: the fork is verified against KRAFTON's own checkpoints
> (`modelopt_fp4` NVFP4).{(' This checkpoint uses the same `modelopt_fp4` loader path.' if is_mopt else ' This repository’s other checkpoints use the `compressed-tensors` format, which is a different loader path — serving them on the fork is untested.')}
{vllm_prebuilt_note}## License and source

- Apache License 2.0. The code in this repository is distributed under Apache-2.0.
- Base model: [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) (rev `{BASE_REVISION}`), Apache-2.0.
- The original KRAFTON repository is access-gated; this repository does not re-gate it.
  The base model license terms (`apache-2.0`) still apply.
"""


def summary_path(mode: str) -> Path:
    """모드별 전체 비교 리포트 경로 (models/README-<mode>.md)."""
    return Path(MODELS_DIR(mode)) / f"README-{mode}.md"


def render_summary(mode: str, ppl: dict, scores: dict) -> str:
    """모드의 **전체 양자화 버전 비교 리포트** → models/README-<mode>.md.

    모델 카드(업로드 대상)는 base + 그 모델 1개만 비교한다. 여기서는
    같은 모드의 모든 양자화 버전을 한 표에 모아 서로 비교한다.
    """
    checks = _checks(mode)
    targets = _existing_targets(mode, ppl, scores)
    rows = ["base", *targets]
    _spec = _acceptance_target_spec(targets[0]) if targets else {}
    gsm8k_drop = _spec.get("gsm8k_acc_drop_max", 0.02)
    ppl_table, gsm_table, ko_table, fmt_table, serve_lines = _check_tables(
        rows, set(targets), mode, ppl, scores, checks, gsm8k_drop
    )

    spec_rows = [
        "| Model | Algorithm | Precision | Bits | Group size | Storage format | Checkpoint (GB) | Model card |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for t in targets:
        sp = SPECS[t]
        size = _dir_size_gb(model_dir(mode, t))
        spec_rows.append(
            f"| **{hf_repo_id(t, mode)}** | {sp['algorithm']} | {sp['precision']} | {sp['bits']} | "
            f"{sp['group_size']} | `{sp['format']}` | {size} | `raon-{t}-{mode}/README.md` |"
        )

    return f"""# Quantization comparison - {BASE_MODEL} ({mode})

Every quantized build of `{BASE_MODEL}` produced for the **{mode}** profile, compared
against the bf16 base model. Base model: `{BASE_MODEL}` (rev `{BASE_REVISION}`),
Apache-2.0.

Each build is published as its own repository with its own model card that compares
**the base model and that build only**. This file is the cross-version comparison and
is not a model card for any single repository.

## Quantized variants

{chr(10).join(spec_rows)}

## Check 1 - PPL (in-domain perplexity)

{chr(10).join(ppl_table)}

## Check 2 - GSM8K (standard test set accuracy)

{chr(10).join(gsm_table)}

## Check 3 - KMMLU (standard Korean benchmark)

{chr(10).join(ko_table)}

## Check 4 - Instruction and format compliance

{chr(10).join(fmt_table)}

## Serving performance

{serve_lines}

## License and source

- Apache License 2.0.
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
            report["errors"].append(f"no measured value for '{target}' in outputs/ppl-{mode}/summary.json (run s4-00-measure-ppl first)")
        if target not in scores:
            report["errors"].append(f"no result for '{target}' in outputs/eval-{mode}/score.json (run s4-15-answer-benchmarks-quantized__transformers-only.sh/s4-20-aggregate-gsm8k__transformers_vllm_both.sh first)")
        else:
            # score.json에 타깃 행이 있어도 GSM8K 정확도가 없으면 카드의 Check 2가 빈다
            # (일부 벤치만 다시 생성한 경우). base도 있어야 하락폭을 판정할 수 있다.
            for r in ("base", target):
                rerun = (f"./s4-10-answer-benchmarks-base__transformers-only.sh {mode}" if r == "base"
                         else f"./s4-15-answer-benchmarks-quantized__transformers-only.sh {mode} {r}")
                if _gsm8k_acc(scores, r) is None:
                    report["errors"].append(
                        f"no GSM8K accuracy for '{r}' in outputs/eval-{mode}/score.json "
                        f"(rerun: EVAL_BENCHMARKS=gsm8k {rerun} && ./s4-20-aggregate-gsm8k__transformers_vllm_both.sh {mode})"
                    )
    report["size_gb"] = _dir_size_gb(d)
    readme = render_readme(target, mode, ppl, scores, report["size_gb"])
    for bad in FORBIDDEN_FRONTMATTER:
        if bad in readme.split("---", 2)[1]:
            report["errors"].append(f"frontmatter contains a forbidden field '{bad}'")
    if report["errors"]:
        return report
    (d / "README.md").write_text(readme, encoding="utf-8")
    report["readme_bytes"] = len(readme.encode("utf-8"))

    # 5) 모드 전체 비교 리포트 → models/README-<mode>.md (업로드 대상 아님)
    summary = render_summary(mode, ppl, scores)
    sp = summary_path(mode)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(summary, encoding="utf-8")
    report["summary_path"] = str(sp)
    report["summary_bytes"] = len(summary.encode("utf-8"))

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
            print(f"-> {model_dir(mode, t)}/README.md (모델 카드, 업로드됨)")
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