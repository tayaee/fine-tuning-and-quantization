"""s8-61-release-notes.py — SDLC 릴리스 노트 생성 (Stage 8 산출물).

단계 원장(outputs/stage-ledger-<mode>.json) + 수용 기준(Stage 1) + 실측 지표
(PPL summary / eval score)를 합쳐 "이 릴리스가 무엇이고 어떤 계약을 만족하는지"를
한 파일로 남긴다. SDLC의 "증빙"이 이 문서다.

  uv run s8-61-release-notes.py --mode smoke
  uv run s8-61-release-notes.py --mode full --print
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sdlc import QUANT_TARGETS, STAGES, read_ledger  # sys.path 조작 후 import

SHARED = Path(os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared"))


def outputs_dir(mode: str) -> Path:
    return SHARED / "outputs"


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def collect(mode: str) -> dict:
    out = outputs_dir(mode)
    ledger = read_ledger(mode)
    acceptance = read_json(out / f"acceptance-{mode}.json")
    ppl = read_json(out / f"ppl-{mode}" / "summary.json")
    score = read_json(out / f"eval-{mode}" / "score.json")
    prep = read_json(out / f"prep-upload-{mode}.json")
    verify = read_json(out / f"verify-upload-{mode}.json")
    return {
        "mode": mode,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ledger": ledger,
        "acceptance": acceptance,
        "ppl": ppl,
        "score": score,
        "prep": prep,
        "verify": verify,
    }


def _stage_rows(data: dict) -> list[str]:
    stages = (data.get("ledger") or {}).get("stages") or {}
    rows = ["| # | 단계 | 상태 | 판정 | 산출물 |", "| ---: | --- | --- | --- | --- |"]
    for spec in STAGES:
        e = stages.get(str(spec["id"])) or {}
        status = e.get("status", "-")
        arts = ", ".join(f"`{a}`" for a in (e.get("artifacts") or [])) or "-"
        targets = e.get("targets") or {}
        tgt_note = ""
        if targets:
            tgt_note = " (" + ", ".join(f"{k}:{v.get('status')}" for k, v in sorted(targets.items())) + ")"
        rows.append(
            f"| {spec['id']} | {spec['title']} (`{spec['name']}`) | {status}{tgt_note} | "
            f"{e.get('verdict', '-')} | {arts} |"
        )
    return rows


def _metric_rows(data: dict) -> list[str]:
    acceptance = data.get("acceptance") or {}
    specs = acceptance.get("targets") or {}
    ppl = data.get("ppl") or {}
    score = data.get("score") or {}
    rows = [
        (
            "| 타겟 | PPL | Δ vs bf16 | 판정 | 체크포인트(GB) | 수용 용량(GB) | "
            "GSM8K (표준) |"
        ),
        "| --- | ---: | ---: | --- | ---: | --- | ---: |",
    ]
    base_ppl = (ppl.get("base") or {}).get("ppl")
    for t in QUANT_TARGETS:
        p = ppl.get(t) or {}
        sc = (score.get(t) or {}).get("benchmarks") or {}
        spec = specs.get(t) or {}
        size = spec.get("size_gb") or []
        size_txt = f"{size[0]}~{size[1]}" if len(size) == 2 else "-"
        def _acc(b: str, _sc: dict = sc) -> str:
            v = (_sc.get(b) or {}).get("acc")
            return "-" if v is None else str(v)

        rows.append(
            f"| {t} | {_num(p.get('ppl'))} | {_num(p.get('delta'))} | {p.get('verdict', '-')} | "
            f"{_num(p.get('size_gb'), '-')} | {size_txt} | {_acc('gsm8k')} |"
        )
    if base_ppl is not None:
        rows.append(f"| base (bf16) | {_num(base_ppl)} | - | ref | - | - | - |")
    return rows


def _num(value, dash: str = "-") -> str:
    if value is None:
        return dash
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _check_rows(data: dict) -> list[str]:
    """s7 검사 결과(GSM8K 통계·한국어·형식 준수·서빙) 표."""
    checks = {}
    for f in ("checks.json",):
        p = outputs_dir(data["mode"]) / f
        if p.is_file():
            try:
                loaded = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    checks = loaded
            except (OSError, json.JSONDecodeError):
                checks = {}
    rows = ["| 검사 | 대상 | 결과 | 판정 |", "| --- | --- | ---: | --- |"]
    gsm = (checks.get("gsm8k") or {}).get("targets") or {}
    for t, v in sorted(gsm.items()):
        verdict = str(((checks.get("gsm8k") or {}).get("verdict") or {}).get(t, "-"))
        rows.append(
            f"| GSM8K (표준) | {t} | acc {v.get('acc')} (answer_rate {v.get('answer_rate')}) | {verdict} |"
        )
    ko = (checks.get("kmmluko") or {}).get("targets") or {}
    for t, v in sorted(ko.items()):
        verdict = str(((checks.get("kmmluko") or {}).get("verdict") or {}).get(t, "-"))
        rows.append(f"| KMMLU (한국어) | {t} | acc {v.get('acc')} (n {v.get('n')}) | {verdict} |")
    fmt = (checks.get("format") or {}).get("targets") or {}
    for t, v in sorted(fmt.items()):
        verdict = str(((checks.get("format") or {}).get("verdict") or {}).get(t, "-"))
        if verdict in ("", "-"):
            verdict = ",".join(v.get("failed") or []) or "PASS"
        rows.append(
            f"| 형식/지시 준수 | {t} | pass_rate {v.get('pass_rate')} (n {v.get('n')}) | {verdict} |"
        )
    sb = checks.get("servebench") or {}
    if sb.get("ok"):
        rows.append(
            f"| 서빙 처리량 | {sb.get('model')} | tok/s(p50) {sb.get('tok_per_s_p50')} @c{sb.get('concurrency')} | measured |"
        )
    return rows


def render(data: dict) -> str:
    mode = data["mode"]
    acceptance = data.get("acceptance") or {}
    prep = data.get("prep") or {}
    verify = data.get("verify") or {}
    gate_lines = [
        "| 게이트 | 상태 | 근거 |",
        "| --- | --- | --- |",
    ]
    acc_ok = bool(acceptance.get("targets"))
    gate_lines.append(f"| Stage 1 수용 기준 | {'PASS' if acc_ok else 'MISSING'} | `acceptance-{mode}.json` |")
    prep_ok = bool(prep.get("ok"))
    gate_lines.append(f"| Stage 7 모델카드/디렉터리 | {'PASS' if prep_ok else 'FAIL/MISSING'} | `prep-upload-{mode}.json` |")
    ver_ok = bool(verify.get("ok"))
    gate_lines.append(f"| Stage 7 업로드 전 감사 | {'PASS' if ver_ok else 'FAIL/MISSING'} | `verify-upload-{mode}.json` |")

    lines = [
        f"# Release notes — {mode}",
        "",
        f"- 생성: `{data['generated_at']}`",
        "- 파이프라인: 9단계 SDLC (docs/SDLC.md)",
        f"- 원장: `outputs/stage-ledger-{mode}.json`",
        f"- 수용 기준: `config/acceptance.toml` (version {acceptance.get('version', '-')})",
        "",
        "## 1. Stage별 상태",
        "",
    ]
    lines += _stage_rows(data)
    lines += ["", "## 2. 수용 기준 대비 실측", ""]
    lines += _metric_rows(data)
    lines += ["", "## 3. 검사 결과 (s7)", ""]
    lines += _check_rows(data)
    lines += ["", "## 4. 릴리스 게이트", ""]
    lines += gate_lines
    lines += [
        "",
        "## 4. 판정 규칙",
        "",
        "- ΔPPL < `ppl_delta_warn` → **Accept**, < `ppl_delta_max` → **Conditional**, 그 이상 → **Discard**",
        "- 게이트 실패 시 다음 단계로 진행하지 않는다(9단계 SDLC의 단계 간 계약)",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="SDLC 릴리스 노트 생성")
    ap.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    ap.add_argument("--print", dest="show", action="store_true")
    a = ap.parse_args()

    print("[s8-61-release-notes] Stage 8/9 - build the release notes from the stage ledger and the measured metrics")
    data = collect(a.mode)
    text = render(data)
    if a.show:
        print(text)
    out = outputs_dir(a.mode) / f"release-notes-{a.mode}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"release notes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
