"""sdlc.py — SDLC 공용 라이브러리 (수용 기준 · 단계 원장 · 판정).

이 파이프라인의 SDLC는 8개 그룹으로 정의된다(문서: docs/SDLC.md).

  0 베이스검증 → 1 환경체인 → 2 베이스복원 → 3 양자화 → 4 측정·평가
  → 5 업로드 → 6 고객인수(transformers) → 7 고객인수(vLLM 서빙)

이 모듈은 그중 "계약(acceptance)"과 "증빙(ledger)" 두 가지를 담당한다.

  - config/acceptance.toml = Stage 1 산출물(수용 기준의 단일 소스).
    s4-00-measure-ppl__transformers-only.py(Δ 판정), s4-70-verify-upload.py(용량 범위),
    s4-65-update-readme.py(모델 카드) 모두 여기 정의된 값만 사용한다.
    파일이 없으면 DEFAULT_ACCEPTANCE로 폴백(하위 호환).
  - outputs/stage-ledger-<mode>.json = 단계 원장. 각 단계가
    status/artifact/verdict를 남기고, s4-75-release-notes.py가 릴리스 노트로 굽는다.

표준 라이브러리만 사용한다(uv sync/lock을 건드리지 않기 위함).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SHARED = Path(os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared"))
ACCEPTANCE_PATH = ROOT / "config" / "acceptance.toml"

#: 양자화 타깃 4종 (base는 복원본이라 양자화 대상이 아님).
QUANT_TARGETS: tuple[str, ...] = ("awq-w3-transformers", "awq-w5-transformers", "nvfp4-awq-transformers", "nvfp4-gptq-transformers")
#: 평가/측정에 참여하는 전체 대상 (base가 기준선).
ALL_TARGETS: tuple[str, ...] = ("base",) + QUANT_TARGETS

#: 수용 기준 파일이 없을 때 쓰는 기본값 — 2026-10 기준 기존 하드코딩 값과 동일.
DEFAULT_ACCEPTANCE: dict[str, Any] = {
    "version": 1,
    "targets": {
        "awq-w3-transformers": {
            "quant_format": "pack-quantized",
            "size_gb": [14.0, 26.0],
            "ppl_delta_warn": 0.3,
            "ppl_delta_max": 1.0,
        },
        "awq-w5-transformers": {
            "quant_format": "pack-quantized",
            "size_gb": [20.0, 35.0],
            "ppl_delta_warn": 0.3,
            "ppl_delta_max": 1.0,
        },
        "nvfp4-awq-transformers": {
            "quant_format": "nvfp4-pack-quantized",
            "size_gb": [15.0, 28.0],
            "ppl_delta_warn": 0.3,
            "ppl_delta_max": 1.0,
        },
        "nvfp4-gptq-transformers": {
            "quant_format": "nvfp4-pack-quantized",
            "size_gb": [15.0, 28.0],
            "ppl_delta_warn": 0.3,
            "ppl_delta_max": 1.0,
        },
    },
    "gates": {
        "release": {
            "require_model_card": True,
            "require_metrics": True,
            "require_finite_ppl": True,
        }
    },
}

#: SDLC 그룹 정의 (id → 이름/산출물). 문서·원장·릴리스 노트가 공유한다.
#: s8(업로드)는 s5로, s9(CAT)는 s6(고객-transformers)+s7(고객-vLLM)로 재편됐다.
STAGES: tuple[dict[str, Any], ...] = (
    {"id": 0, "name": "base", "title": "베이스모델 검증", "artifact": "outputs/s0-<mode>.json"},
    {"id": 1, "name": "envchain", "title": "환경·수용기준·데이터·호환성", "artifact": "outputs/acceptance-<mode>.json"},
    {"id": 2, "name": "dequant", "title": "베이스 복원(MXFP8→BF16)", "artifact": "models/raon-bf16"},
    {"id": 3, "name": "quant", "title": "양자화 실행", "artifact": "models/raon-<target>-<mode>"},
    {"id": 4, "name": "measure-eval", "title": "PPL·벤치마크 평가·카드·감사", "artifact": "outputs/eval-<mode>/<target>.jsonl"},
    {"id": 5, "name": "upload", "title": "Hub 업로드", "artifact": "Hub (tayaee/*)"},
    {"id": 6, "name": "cat-tf", "title": "고객인수(transformers)", "artifact": "outputs/s6-<mode>/"},
    {"id": 7, "name": "cat-vllm", "title": "고객인수(vLLM 서빙)", "artifact": "outputs/serve-<mode>.json"},
)

VERDICT_ACCEPT = "Accept"
VERDICT_CONDITIONAL = "Conditional"
VERDICT_DISCARD = "Discard"


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- 수용 기준 (Stage 1)


def load_acceptance(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """config/acceptance.toml을 읽어 기본값과 병합한다. 파일이 없으면 기본값만.

    병합은 타깃 단위로 수행한다(파일에서 일부 타깃만 정의해도 나머지는 기본값 유지).
    """
    p = Path(path) if path is not None else ACCEPTANCE_PATH
    data: dict[str, Any] = {
        "version": DEFAULT_ACCEPTANCE["version"],
        "targets": {k: dict(v) for k, v in DEFAULT_ACCEPTANCE["targets"].items()},
        "gates": {k: dict(v) for k, v in DEFAULT_ACCEPTANCE["gates"].items()},
        "source": str(p),
    }
    if not p.is_file():
        data["source"] = "default"
        return data
    with p.open("rb") as f:
        raw = tomllib.load(f)
    if isinstance(raw.get("version"), int):
        data["version"] = raw["version"]
    targets = raw.get("targets")
    if isinstance(targets, dict):
        for name, spec in targets.items():
            if not isinstance(spec, dict):
                continue
            merged = dict(data["targets"].get(name, {}))
            merged.update(spec)
            data["targets"][name] = merged
    gates = raw.get("gates")
    if isinstance(gates, dict):
        for name, spec in gates.items():
            if not isinstance(spec, dict):
                continue
            merged = dict(data["gates"].get(name, {}))
            merged.update(spec)
            data["gates"][name] = merged
    data["source"] = str(p)
    return data


def acceptance_errors(data: dict[str, Any]) -> list[str]:
    """수용 기준 스키마 오류 목록. 비어 있으면 정상."""
    errors: list[str] = []
    if not isinstance(data.get("version"), int):
        errors.append("version 정수 아님")
    targets = data.get("targets")
    if not isinstance(targets, dict) or not targets:
        errors.append("targets 없음")
        return errors
    for name in QUANT_TARGETS:
        spec = targets.get(name)
        if not isinstance(spec, dict):
            errors.append(f"targets.{name} 없음")
            continue
        if not isinstance(spec.get("quant_format"), str):
            errors.append(f"targets.{name}.quant_format 문자열 아님")
        size = spec.get("size_gb")
        if not (isinstance(size, list) and len(size) == 2 and all(isinstance(x, (int, float)) for x in size)):
            errors.append(f"targets.{name}.size_gb = [min,max] 숫자 2개 아님")
        elif float(size[0]) >= float(size[1]):
            errors.append(f"targets.{name}.size_gb 범위 역전 (min >= max)")
        for key in ("ppl_delta_warn", "ppl_delta_max"):
            if not isinstance(spec.get(key), (int, float)):
                errors.append(f"targets.{name}.{key} 숫자 아님")
        warn = spec.get("ppl_delta_warn")
        mx = spec.get("ppl_delta_max")
        if isinstance(warn, (int, float)) and isinstance(mx, (int, float)) and float(warn) >= float(mx):
            errors.append(f"targets.{name}.ppl_delta_warn >= ppl_delta_max")
    return errors


def target_spec(data: dict[str, Any], target: str) -> dict[str, Any]:
    """타깃별 수용 기준. 미지정 타깃은 base용 기본 판정만 준다."""
    targets = data.get("targets", {})
    spec = targets.get(target)
    if isinstance(spec, dict):
        return spec
    return {"quant_format": "", "size_gb": [0.0, 0.0], "ppl_delta_warn": 0.3, "ppl_delta_max": 1.0}


def verdict_for_delta(delta: float, spec: dict[str, Any]) -> str:
    """base 대비 ΔPPL 판정: <warn Accept / <max Conditional / 그 이상 Discard."""
    warn = float(spec.get("ppl_delta_warn", 0.3))
    mx = float(spec.get("ppl_delta_max", 1.0))
    if delta < warn:
        return VERDICT_ACCEPT
    if delta < mx:
        return VERDICT_CONDITIONAL
    return VERDICT_DISCARD


# ---------------------------------------------------------------- 단계 원장 (ledger)


def ledger_path(mode: str, shared: str | os.PathLike[str] | None = None) -> Path:
    base = Path(shared) if shared is not None else SHARED
    return base / "outputs" / f"stage-ledger-{mode}.json"


def read_ledger(mode: str, shared: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """원장 읽기. 없으면 빈 원장."""
    p = ledger_path(mode, shared)
    if not p.is_file():
        return {"mode": mode, "updated_at": None, "stages": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"mode": mode, "updated_at": None, "stages": {}}
    if not isinstance(data, dict):
        return {"mode": mode, "updated_at": None, "stages": {}}
    data.setdefault("mode", mode)
    stages = data.get("stages")
    if not isinstance(stages, dict):
        data["stages"] = {}
    return data


def update_ledger(
    mode: str,
    stage: int,
    name: str,
    status: str,
    *,
    artifacts: list[str] | None = None,
    verdict: str | None = None,
    duration_s: float | None = None,
    detail: str | None = None,
    target: str | None = None,
    shared: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """원장에 단계 상태를 기록한다(병렬 실행 대비 파일 락 사용).

    target을 주면 단계 entry 안의 targets 맵에 타깃별 상태로 쌓이고,
    단계 status는 "타깃 하나라도 실패면 fail"로 집계된다.
    같은 stage id를 다시 기록하면 갱신된다(중복 실행 이력은 artifacts에 누적하지 않는다).
    """
    p = ledger_path(mode, shared)
    p.parent.mkdir(parents=True, exist_ok=True)
    lock_path = p.with_suffix(p.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        data = read_ledger(mode, shared)
        entry: dict[str, Any] = dict(data["stages"].get(str(int(stage))) or {})
        entry.update(
            {
                "id": int(stage),
                "name": name,
                "status": status,
                "updated_at": _now(),
            }
        )
        if artifacts:
            entry["artifacts"] = list(dict.fromkeys(artifacts))
        if verdict:
            entry["verdict"] = verdict
        if duration_s is not None:
            entry["duration_s"] = round(float(duration_s), 3)
        if detail:
            entry["detail"] = detail
        if target:
            targets = dict(entry.get("targets") or {})
            targets[target] = {
                "status": status,
                "updated_at": _now(),
                "artifacts": list(artifacts or []),
            }
            entry["targets"] = targets
            entry["status"] = (
                "fail" if any(t.get("status") == "fail" for t in targets.values()) else "pass"
            )
        data["stages"][str(int(stage))] = entry
        data["mode"] = mode
        data["updated_at"] = _now()
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(p)
    return data


def checks_path(mode: str, shared: str | os.PathLike[str] | None = None) -> Path:
    """검사 결과 원장 (GSM8K 통계·한국어·형식 준수 등)."""
    base = Path(shared) if shared is not None else SHARED
    return base / "outputs" / f"eval-{mode}" / "checks.json"


def merge_check(mode: str, name: str, payload: dict, shared: str | os.PathLike[str] | None = None) -> dict:
    """검사 결과를 checks.json에 병합한다(병렬 실행 대비 파일 락)."""
    p = checks_path(mode, shared)
    p.parent.mkdir(parents=True, exist_ok=True)
    lock_path = p.with_suffix(p.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        data: dict[str, Any] = {}
        if p.is_file():
            try:
                loaded = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except (OSError, json.JSONDecodeError):
                data = {}
        data[name] = payload
        data["updated_at"] = _now()
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(p)
    return data


def write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    """JSON 산출물 atomic write."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)
    return p


# ---------------------------------------------------------------- CLI


def _cmd_ledger(a: argparse.Namespace) -> int:
    update_ledger(
        a.mode,
        a.stage,
        a.name,
        a.status,
        artifacts=a.artifact or [],
        verdict=a.verdict,
        duration_s=a.duration_s,
        detail=a.detail,
        target=a.target,
    )
    return 0


def _cmd_show(a: argparse.Namespace) -> int:
    data = read_ledger(a.mode)
    stages = data.get("stages", {})
    print(f"ledger mode={a.mode} updated_at={data.get('updated_at')}")
    for key in sorted(stages, key=lambda k: int(k)):
        e = stages[key]
        arts = ",".join(e.get("artifacts", []))
        print(f"  [{e.get('id')}] {e.get('name'):<12} {e.get('status'):<6} {e.get('verdict', ''):<12} {arts}")
    return 0


def _cmd_acceptance(a: argparse.Namespace) -> int:
    data = load_acceptance(a.path)
    errors = acceptance_errors(data)
    if errors:
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SDLC 공용 도구 (수용 기준 · 원장)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_led = sub.add_parser("ledger", help="단계 상태를 원장에 기록")
    p_led.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    p_led.add_argument("--stage", type=int, required=True)
    p_led.add_argument("--name", required=True)
    p_led.add_argument("--status", required=True)
    p_led.add_argument("--artifact", action="append")
    p_led.add_argument("--verdict")
    p_led.add_argument("--duration-s", type=float, dest="duration_s")
    p_led.add_argument("--detail")
    p_led.add_argument("--target")
    p_led.set_defaults(func=_cmd_ledger)

    p_show = sub.add_parser("show", help="원장 출력")
    p_show.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    p_show.set_defaults(func=_cmd_show)

    p_acc = sub.add_parser("acceptance", help="수용 기준 검증/출력")
    p_acc.add_argument("--path", default=None)
    p_acc.set_defaults(func=_cmd_acceptance)

    a = ap.parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":
    raise SystemExit(main())
