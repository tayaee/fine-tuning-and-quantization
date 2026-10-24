"""46-verify-upload.py — HF Hub 업로드 전 모델·지표·README·vLLM 호환성 종합 감사(Audit).

43-prep-upload.py가 모델 디렉터리 보충과 README.md 생성을 마친 뒤, 49-upload-hf.py로
실제 업로드하기 직전에 지금까지 발견된 모든 결함 유형을 전수 점검한다:

  1) 상류 BF16 복원본(models/raon-bf16) 무결성 (issue-2):
     - config.json 내 quantization_config 잔존 금지
     - safetensors 내 F8_E4M3 / U8(weight_scale_inv) 잔존 금지
     - modeling_raon1_mini.py 내 _routed_projection -> nn.Linear 트레이싱 심 확인
  2) 양자화 포맷 및 safetensors 헤더/인덱스 계약 (issue-3):
     - quantization_config.quant_method == "compressed-tensors"
     - format == pack-quantized(awq-w3/w5) 또는 nvfp4-pack-quantized(nvfp4-*) ("dense" 폴백 차단)
     - weight_packed / weight_scale / weight_global_scale / input_global_scale 개수·dtype 일치
     - 29B 본 모델(num_hidden_layers>=25)에서 양자화된 expert 프로젝션 수(9,216개) 확인
     - 비양자화 보호 레이어(lm_head, embed_tokens 등)가 BF16/F32 .weight로 유지되는지 확인
     - model.safetensors.index.json의 weight_map <-> 실제 샤드 파일·텐서 키 1:1 일치 확인
  3) 체크포인트 용량(모델 사이즈) 정상 범위 검증:
     - 29B 본 모델 기준 타겟별 예상 압축 용량 범위(nvfp4: 15~28GB, awq-w3: 14~26GB, awq-w5: 20~35GB)
     - 미압축 dense 폴백(~55GB)이나 샤드 누락(<10GB) 즉시 검출
  4) vLLM 및 transformers 실행 준비 상태 (issue-1, issue-4):
     - 필수 파일(config.json, tokenizer.json, tokenizer_config.json, chat_template.jinja,
       configuration_raon1_mini.py, modeling_raon1_mini.py, README.md) 존재 및 비어 있지 않음
     - custom code(.py) AST 문법 및 Raon1MiniConfig / Raon1MiniForCausalLM 클래스 정의 확인
     - config.json의 architectures, model_type, auto_map, eos_token_id, num_experts,
       num_experts_per_token 필드 확인
     - chat_template.jinja의 ChatML(<|im_start|>, <|im_end|>) 및 <think> 마커 확인
  5) PPL·벤치마크 평가 지표 및 README.md 정합성 (issue-4, issue-5):
     - outputs/ppl-<mode>/summary.json 내 base 및 타겟 PPL 유한성·Δ·verdict 확인
     - outputs/eval-<mode>/score.json 내 done_rate > 0, prompts-* 오염 부재, 3대 벤치 존재 확인
     - README.md YAML frontmatter 필수 키, extra_gated_* 금지, 실제 체크포인트 용량(GB) 및
       성능 비교표 내 타겟 행의 실제 수치 기입 여부 확인

  uv run 46-verify-upload.py --mode full --targets nvfp4-awq
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
from pathlib import Path

from quant_common import (
    NVFP4_PACK_QUANTIZED,
    PACK_QUANTIZED,
    verify_compressed_output,
)

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")
BASE_MODEL = os.environ.get("BASE_MODEL", "KRAFTON/Raon-1-mini-preview")
BASE_REVISION = os.environ.get("BASE_REVISION") or "da4cf162c8613993f2189ebb438a5f9a23f3e5ca"
ALL_TARGETS = ["awq-w3", "awq-w5", "nvfp4-awq", "nvfp4-gptq"]
BENCH_ORDER = ("kmmlu_redux", "gsm8k", "humaneval")
ROUTED_LINEAR_MARKER = "return nn.Linear(in_features, out_features, bias=False)"
MOE_CALIB_MARKER = "get_calibrate_all_experts_flag"

REQUIRED_MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "configuration_raon1_mini.py",
    "modeling_raon1_mini.py",
    "README.md",
)

# 29B 본 모델(24개 MoE 레이어 × 128 experts × 3개 프로젝션 = 9,216개) 기준 타겟 사양
TARGET_RULES: dict[str, dict] = {
    "awq-w3": {
        "format": PACK_QUANTIZED,
        "num_bits": 3,
        "w_type": "int",
        "size_gb_range": (14.0, 26.0),
        "needs_input_global_scale": False,
    },
    "awq-w5": {
        "format": PACK_QUANTIZED,
        "num_bits": 5,
        "w_type": "int",
        "size_gb_range": (20.0, 35.0),
        "needs_input_global_scale": False,
    },
    "nvfp4-awq": {
        "format": NVFP4_PACK_QUANTIZED,
        "num_bits": 4,
        "w_type": "float",
        "size_gb_range": (15.0, 28.0),
        "needs_input_global_scale": True,
    },
    "nvfp4-gptq": {
        "format": NVFP4_PACK_QUANTIZED,
        "num_bits": 4,
        "w_type": "float",
        "size_gb_range": (15.0, 28.0),
        "needs_input_global_scale": True,
    },
}

EXPECTED_FULL_EXPERT_LINEAR_COUNT = 24 * 128 * 3  # 9,216


def MODELS_DIR(mode: str) -> str:
    return f"{SHARED}/models"


def BF16_DIR() -> str:
    return f"{SHARED}/models/raon-bf16"


def OUT_ROOT() -> str:
    return f"{SHARED}/outputs"


def model_dir(mode: str, target: str) -> Path:
    return Path(f"{MODELS_DIR(mode)}/raon-{target}-{mode}")


def _scan_safetensors_headers(d: Path) -> tuple[dict[str, set[str]], dict[str, str], dict[str, int]]:
    """샤드별 텐서 키 집합, 전체 텐서->dtype 매핑, 접미사별 개수를 헤더만으로 스캔한다."""
    from safetensors import safe_open

    shard_keys: dict[str, set[str]] = {}
    tensor_dtypes: dict[str, str] = {}
    suffix_counts: dict[str, int] = {}
    for shard in sorted(d.glob("*.safetensors")):
        keys: set[str] = set()
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                keys.add(name)
                dt = str(f.get_slice(name).get_dtype())
                tensor_dtypes[name] = dt
                suffix = name.rsplit(".", 1)[-1]
                suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        shard_keys[shard.name] = keys
    return shard_keys, tensor_dtypes, suffix_counts


def check_bf16_dir(bf16_path: Path | None = None) -> list[str]:
    """상류 models/raon-bf16 디렉터리의 복원 무결성 점검 (issue-2)."""
    d = bf16_path or Path(BF16_DIR())
    errors: list[str] = []
    if not d.is_dir():
        return [f"bf16 디렉터리 없음: {d}"]
    cfg_file = d / "config.json"
    if not cfg_file.is_file():
        return [f"{d}: config.json 없음"]
    try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return [f"{cfg_file}: JSON 파싱 실패 ({e})"]
    if "quantization_config" in cfg:
        errors.append(f"{cfg_file}: quantization_config가 제거되지 않음")

    modeling = d / "modeling_raon1_mini.py"
    if not modeling.is_file():
        errors.append(f"{d}: modeling_raon1_mini.py 없음")
    else:
        mod_src = modeling.read_text(encoding="utf-8")
        if ROUTED_LINEAR_MARKER not in mod_src:
            errors.append(f"{modeling}: '{ROUTED_LINEAR_MARKER}' 트레이싱 심 누락")
        if "class Raon1MiniMoE" in mod_src and MOE_CALIB_MARKER not in mod_src:
            errors.append(f"{modeling}: '{MOE_CALIB_MARKER}' MoE 전-expert 캘리브레이션 심 누락")

    shards = sorted(d.glob("*.safetensors"))
    if not shards:
        errors.append(f"{d}: .safetensors 파일 없음")
    else:
        _, tensor_dtypes, _ = _scan_safetensors_headers(d)
        bad_dtypes = {dt for dt in tensor_dtypes.values() if dt in ("F8_E4M3", "U8")}
        if bad_dtypes:
            errors.append(f"{d}: 미복원 FP8/스케일 dtype 잔존 ({sorted(bad_dtypes)})")
        scale_keys = [k for k in tensor_dtypes if k.endswith(".weight_scale_inv")]
        if scale_keys:
            errors.append(f"{d}: weight_scale_inv 텐서 {len(scale_keys)}개 잔존")
    return errors


def _check_custom_code_ast(d: Path) -> list[str]:
    """configuration_raon1_mini.py / modeling_raon1_mini.py AST 문법 및 핵심 클래스 확인."""
    errors: list[str] = []
    expected_classes = {
        "configuration_raon1_mini.py": {"Raon1MiniConfig"},
        "modeling_raon1_mini.py": {"Raon1MiniModel", "Raon1MiniForCausalLM"},
    }
    for fname, req_classes in expected_classes.items():
        fp = d / fname
        if not fp.is_file():
            continue
        try:
            tree = ast.parse(fp.read_text(encoding="utf-8"), filename=str(fp))
        except SyntaxError as e:
            errors.append(f"{fp}: 파이썬 문법 오류 ({e})")
            continue
        found = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        missing = req_classes - found
        if missing:
            errors.append(f"{fp}: 필수 클래스 누락 {sorted(missing)}")

    jinja_fp = d / "chat_template.jinja"
    if jinja_fp.is_file():
        jtxt = jinja_fp.read_text(encoding="utf-8")
        for token in ("<|im_start|>", "<|im_end|>", "<think>", "</think>"):
            if token not in jtxt:
                errors.append(f"{jinja_fp}: 필수 ChatML/thinking 토큰 '{token}' 누락")
    return errors


def _check_vllm_and_config(d: Path, cfg: dict, target: str) -> list[str]:
    """vLLM 및 transformers 로드에 필요한 config.json 필드 점검."""
    errors: list[str] = []
    rule = TARGET_RULES[target]

    archs = cfg.get("architectures") or []
    if "Raon1MiniForCausalLM" not in archs:
        errors.append(f"{d}/config.json: architectures={archs} ('Raon1MiniForCausalLM' 필요)")
    if cfg.get("model_type") != "raon1_mini":
        errors.append(f"{d}/config.json: model_type={cfg.get('model_type')!r} ('raon1_mini' 필요)")

    auto_map = cfg.get("auto_map") or {}
    for key, expected_ref in (
        ("AutoConfig", "configuration_raon1_mini.Raon1MiniConfig"),
        ("AutoModelForCausalLM", "modeling_raon1_mini.Raon1MiniForCausalLM"),
    ):
        if auto_map.get(key) != expected_ref:
            errors.append(f"{d}/config.json: auto_map[{key!r}]={auto_map.get(key)!r} ({expected_ref!r} 필요)")

    for key in ("eos_token_id", "pad_token_id", "num_experts", "num_experts_per_token"):
        if cfg.get(key) is None:
            errors.append(f"{d}/config.json: vLLM/생성 필수 필드 '{key}' 누락")

    qc = cfg.get("quantization_config") or {}
    if qc.get("quant_method") != "compressed-tensors":
        errors.append(
            f"{d}/config.json: quantization_config.quant_method={qc.get('quant_method')!r} "
            "('compressed-tensors' 필요)"
        )
    if qc.get("format") != rule["format"]:
        errors.append(
            f"{d}/config.json: quantization_config.format={qc.get('format')!r} "
            f"({rule['format']!r} 필요)"
        )

    ignore = qc.get("ignore") or []
    if not any("lm_head" in str(x) for x in ignore):
        errors.append(f"{d}/config.json: quantization_config.ignore에 'lm_head' 보호 규칙 없음")

    groups = qc.get("config_groups") or {}
    if not groups:
        errors.append(f"{d}/config.json: quantization_config.config_groups 비어 있음")
    else:
        for gname, gspec in groups.items():
            if not isinstance(gspec, dict):
                continue
            w = gspec.get("weights") or {}
            if w.get("num_bits") != rule["num_bits"]:
                errors.append(
                    f"{d}/config.json: {gname}.weights.num_bits={w.get('num_bits')} "
                    f"(expected {rule['num_bits']})"
                )
            if w.get("type") and w.get("type") != rule["w_type"]:
                errors.append(
                    f"{d}/config.json: {gname}.weights.type={w.get('type')!r} "
                    f"(expected {rule['w_type']!r})"
                )
    return errors


def _check_safetensors_and_index(
    d: Path,
    cfg: dict,
    target: str,
    *,
    allow_tiny: bool = False,
) -> tuple[list[str], int, int]:
    """safetensors 헤더·인덱스 정합성 및 패킹 텐서 수 점검."""
    errors: list[str] = []
    rule = TARGET_RULES[target]
    shard_keys, tensor_dtypes, suffix_counts = _scan_safetensors_headers(d)
    num_shards = len(shard_keys)
    packed_count = suffix_counts.get("weight_packed", 0)
    scale_count = suffix_counts.get("weight_scale", 0)

    if packed_count == 0:
        errors.append(f"{d}: weight_packed 텐서 0개 (미압축 체크포인트)")
    if scale_count != packed_count:
        errors.append(f"{d}: weight_scale({scale_count}) != weight_packed({packed_count}) 개수 불일치")

    if rule["needs_input_global_scale"]:
        wgs = suffix_counts.get("weight_global_scale", 0)
        igs = suffix_counts.get("input_global_scale", 0)
        if wgs != packed_count:
            errors.append(f"{d}: weight_global_scale({wgs}) != weight_packed({packed_count})")
        if igs != packed_count:
            errors.append(f"{d}: input_global_scale({igs}) != weight_packed({packed_count}) (vLLM W4A4 필수)")

        import torch
        from safetensors import safe_open

        bad_scales: list[str] = []
        for shard in sorted(d.glob("*.safetensors")):
            with safe_open(shard, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if not name.endswith(("weight_global_scale", "input_global_scale")):
                        continue
                    t = f.get_tensor(name).float()
                    if not bool(torch.isfinite(t).all()) or bool((t <= 0).any() or (t >= 1e10).any()):
                        bad_scales.append(f"{name}={float(t.flatten()[0]):.3e}")
        if bad_scales:
            errors.append(f"{d}: 미보정/비정상 global_scale {len(bad_scales)}개 (예: {bad_scales[:3]})")

    # 보호 레이어가 실수로 패킹되지 않았는지 점검
    for protected_prefix in ("lm_head.", "model.embed_tokens."):
        bad_packed = [k for k in tensor_dtypes if k.startswith(protected_prefix) and k.endswith(".weight_packed")]
        if bad_packed:
            errors.append(f"{d}: 보호 레이어가 양자화됨 ({bad_packed[:3]})")

    is_full_arch = int(cfg.get("num_hidden_layers") or 0) >= 25 and int(cfg.get("num_experts") or 0) >= 128
    if is_full_arch and not allow_tiny and packed_count < EXPECTED_FULL_EXPERT_LINEAR_COUNT:
        errors.append(
            f"{d}: 29B 모델의 weight_packed={packed_count} < 예상 expert 프로젝션 수({EXPECTED_FULL_EXPERT_LINEAR_COUNT})"
        )

    # model.safetensors.index.json 1:1 대조
    idx_fp = d / "model.safetensors.index.json"
    if idx_fp.is_file():
        try:
            idx_data = json.loads(idx_fp.read_text(encoding="utf-8"))
            weight_map: dict[str, str] = idx_data.get("weight_map") or {}
        except Exception as e:  # noqa: BLE001
            errors.append(f"{idx_fp}: JSON 파싱 실패 ({e})")
            weight_map = {}
        if not weight_map:
            errors.append(f"{idx_fp}: weight_map 비어 있음")
        else:
            referenced_shards = set(weight_map.values())
            actual_shards = set(shard_keys.keys())
            missing_shards = referenced_shards - actual_shards
            if missing_shards:
                errors.append(f"{idx_fp}: 디스크에 없는 샤드 참조 {sorted(missing_shards)}")
            unindexed_shards = actual_shards - referenced_shards
            if unindexed_shards:
                errors.append(f"{idx_fp}: 인덱스에 누락된 샤드 존재 {sorted(unindexed_shards)}")
            all_header_keys = set().union(*shard_keys.values()) if shard_keys else set()
            missing_keys = set(weight_map.keys()) - all_header_keys
            extra_keys = all_header_keys - set(weight_map.keys())
            if missing_keys:
                errors.append(f"{idx_fp}: 샤드 헤더에 없는 텐서 키 {len(missing_keys)}개 (예: { sorted(missing_keys)[:2] })")
            if extra_keys:
                errors.append(f"{idx_fp}: weight_map에 누락된 텐서 키 {len(extra_keys)}개 (예: { sorted(extra_keys)[:2] })")
    elif num_shards > 1:
        errors.append(f"{d}: 멀티 샤드({num_shards}개)인데 model.safetensors.index.json 없음")

    return errors, num_shards, packed_count


def _check_size_gb(d: Path, cfg: dict, target: str, *, allow_tiny: bool = False) -> tuple[list[str], float]:
    """체크포인트 총 safetensors 용량(GB) 검증."""
    errors: list[str] = []
    total_bytes = sum(f.stat().st_size for f in d.glob("*.safetensors"))
    size_gb = round(total_bytes / 1024**3, 2)
    if total_bytes <= 0:
        errors.append(f"{d}: .safetensors 총 용량이 0 바이트")
        return errors, size_gb

    is_full_arch = int(cfg.get("num_hidden_layers") or 0) >= 25 and int(cfg.get("num_experts") or 0) >= 128
    if is_full_arch and not allow_tiny:
        lo, hi = TARGET_RULES[target]["size_gb_range"]
        if not (lo <= size_gb <= hi):
            errors.append(
                f"{d}: 체크포인트 용량 {size_gb} GB가 {target} 정상 범위({lo}~{hi} GB)를 벗어남 "
                "(dense 미압축 폴백 또는 샤드 누락 의심)"
            )
    return errors, size_gb


def _check_metrics(
    mode: str,
    target: str,
    *,
    allow_missing_metrics: bool = False,
    allow_discard: bool = False,
) -> tuple[list[str], list[str], dict]:
    """PPL(summary.json) 및 벤치마크(score.json / <target>.jsonl) 지표 검증."""
    errors: list[str] = []
    warnings: list[str] = []
    meta: dict = {"ppl": None, "ppl_delta": None, "ppl_verdict": None, "benchmarks": {}}

    ppl_fp = Path(f"{OUT_ROOT()}/ppl-{mode}/summary.json")
    score_fp = Path(f"{OUT_ROOT()}/eval-{mode}/score.json")
    eval_jsonl = Path(f"{OUT_ROOT()}/eval-{mode}/{target}.jsonl")

    if allow_missing_metrics and not ppl_fp.is_file() and not score_fp.is_file():
        warnings.append("allow_missing_metrics=True: PPL/eval 지표 파일 생략")
        return errors, warnings, meta

    if not ppl_fp.is_file():
        if not allow_missing_metrics:
            errors.append(f"{ppl_fp}: 파일 없음 (34-measure-ppl.sh 먼저 실행)")
    else:
        ppl_data = json.loads(ppl_fp.read_text(encoding="utf-8"))
        for req_key in ("base", target):
            if req_key not in ppl_data:
                if not allow_missing_metrics:
                    errors.append(f"{ppl_fp}: '{req_key}' 항목 없음")
                continue
            val = ppl_data[req_key].get("ppl")
            if not isinstance(val, (int, float)) or not math.isfinite(val) or not (1.0 < val < 100.0):
                errors.append(f"{ppl_fp}: '{req_key}' PPL 비정상 ({val!r})")
        if target in ppl_data:
            t_row = ppl_data[target]
            meta["ppl"] = t_row.get("ppl")
            meta["ppl_delta"] = t_row.get("delta")
            meta["ppl_verdict"] = t_row.get("verdict")
            if not allow_missing_metrics and (
                meta["ppl_delta"] is None or not math.isfinite(float(meta["ppl_delta"]))
            ):
                errors.append(f"{ppl_fp}: '{target}' delta 누락 또는 비정상 ({meta['ppl_delta']!r})")
            if meta["ppl_verdict"] == "Discard":
                msg = f"{ppl_fp}: '{target}' PPL 판정이 Discard (delta={meta['ppl_delta']})"
                if allow_discard:
                    warnings.append(msg)
                else:
                    errors.append(msg)
            elif meta["ppl_verdict"] == "Conditional":
                warnings.append(f"{ppl_fp}: '{target}' PPL 판정이 Conditional (delta={meta['ppl_delta']})")

    if not score_fp.is_file():
        if not allow_missing_metrics:
            errors.append(f"{score_fp}: 파일 없음 (37-eval.sh / 40-score.sh 먼저 실행)")
    else:
        score_data = json.loads(score_fp.read_text(encoding="utf-8"))
        bad_keys = [k for k in score_data if k.startswith("prompts-")]
        if bad_keys:
            errors.append(f"{score_fp}: 프롬프트 입력 파일이 타겟으로 오집계됨 ({bad_keys})")
        for req_key in ("base", target):
            if req_key not in score_data:
                if not allow_missing_metrics:
                    errors.append(f"{score_fp}: '{req_key}' 항목 없음")
                continue
            s_row = score_data[req_key]
            if float(s_row.get("done_rate") or 0.0) <= 0.0:
                errors.append(f"{score_fp}: '{req_key}' done_rate={s_row.get('done_rate')} (빈 응답만 생성됨)")
            benches = s_row.get("benchmarks") or {}
            for b in BENCH_ORDER:
                if b not in benches:
                    errors.append(f"{score_fp}: '{req_key}'에 벤치마크 '{b}' 결과 누락")
        if target in score_data:
            meta["benchmarks"] = (score_data[target].get("benchmarks") or {}).copy()

    if not eval_jsonl.is_file():
        if not allow_missing_metrics:
            errors.append(f"{eval_jsonl}: 생성 응답 JSONL 없음")
    else:
        lines = [json.loads(line) for line in eval_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            errors.append(f"{eval_jsonl}: 생성 응답이 0줄임")
        elif all(not str(r.get("output") or "").strip() for r in lines):
            errors.append(f"{eval_jsonl}: 모든 생성 응답(output)이 비어 있음")

    return errors, warnings, meta


def _check_readme(
    d: Path,
    target: str,
    size_gb: float,
    *,
    allow_missing_metrics: bool = False,
) -> list[str]:
    """README.md YAML frontmatter, 필수 섹션, 사양·성능표 정합성 검증."""
    errors: list[str] = []
    readme_fp = d / "README.md"
    if not readme_fp.is_file():
        return [f"{readme_fp}: 파일 없음 (43-prep-upload.sh 먼저 실행)"]
    text = readme_fp.read_text(encoding="utf-8")
    if len(text.strip()) < 500:
        errors.append(f"{readme_fp}: 내용이 너무 짧음 ({len(text)} chars)")
    if not text.startswith("---\n") or text.count("---") < 2:
        errors.append(f"{readme_fp}: YAML frontmatter(---) 블록 누락")
        return errors

    fm = text.split("---", 2)[1]
    for req_fm in (
        "license: apache-2.0",
        f"base_model: {BASE_MODEL}",
        "base_model_relation: quantized",
        "library_name: transformers",
        "pipeline_tag: text-generation",
    ):
        if req_fm not in fm:
            errors.append(f"{readme_fp}: frontmatter에 '{req_fm}' 누락")
    if "extra_gated_" in fm:
        errors.append(f"{readme_fp}: frontmatter에 금지 필드 'extra_gated_*' 포함")

    rule = TARGET_RULES[target]
    for req_body in (
        BASE_REVISION,
        rule["format"],
        f"{size_gb} GB",
        "## 양자화 사양",
        "## 성능 비교",
        "## 사용법 — transformers",
        "## 사용법 — vLLM",
        "## 라이선스 및 출처",
        "--trust-remote-code",
        "--quantization compressed-tensors",
    ):
        if req_body not in text:
            errors.append(f"{readme_fp}: 본문에 필수 항목 '{req_body}' 누락")

    # 성능 비교표에서 현재 타겟 강조 행 확인
    row_re = re.compile(rf"^\|\s*\*\*{re.escape(target)}\*\*\s*\|(.+)\|$", re.MULTILINE)
    m = row_re.search(text)
    if not m:
        errors.append(f"{readme_fp}: 성능 비교표에 현재 타겟 강조 행('| **{target}** |') 없음")
    elif not allow_missing_metrics:
        cells = [c.strip() for c in m.group(1).split("|")]
        if any(c in ("—", "", "None", "nan") for c in cells):
            errors.append(f"{readme_fp}: 성능 비교표의 '**{target}**' 행에 미기입 결측치('—') 존재: {cells}")

    return errors


def verify_target(
    mode: str,
    target: str,
    *,
    allow_missing_metrics: bool = False,
    allow_discard: bool = False,
    allow_tiny: bool = False,
) -> dict:
    """타겟 1개에 대해 8대 범주 전수 검사를 수행하고 결과 사전을 반환한다."""
    d = model_dir(mode, target)
    report: dict = {
        "target": target,
        "mode": mode,
        "model_dir": str(d),
        "ok": False,
        "errors": [],
        "warnings": [],
        "checks": {
            "files": False,
            "custom_code": False,
            "quant_format": False,
            "safetensors_index": False,
            "model_size": False,
            "vllm_ready": False,
            "metrics": False,
            "readme": False,
        },
        "size_gb": None,
        "num_shards": 0,
        "packed_tensors": 0,
        "ppl": None,
        "ppl_delta": None,
        "ppl_verdict": None,
        "benchmarks": {},
    }

    if not d.is_dir():
        report["errors"].append(f"{d}: 모델 디렉터리 없음")
        return report

    # 1) 필수 파일 존재 및 비어 있지 않음
    file_errs: list[str] = []
    for fname in REQUIRED_MODEL_FILES:
        fp = d / fname
        if not fp.is_file():
            file_errs.append(f"{d}: 필수 파일 '{fname}' 없음")
        elif fp.stat().st_size == 0:
            file_errs.append(f"{d}: 필수 파일 '{fname}'이 비어 있음 (0 bytes)")
    if not file_errs:
        report["checks"]["files"] = True
    report["errors"].extend(file_errs)

    if not (d / "config.json").is_file():
        return report
    try:
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        report["errors"].append(f"{d}/config.json: JSON 파싱 실패 ({e})")
        return report

    # 2) custom code & chat_template 검증
    cc_errs = _check_custom_code_ast(d)
    if not cc_errs:
        report["checks"]["custom_code"] = True
    report["errors"].extend(cc_errs)

    # 3) 압축 포맷 계약 (verify_compressed_output)
    fmt_errs: list[str] = []
    try:
        verify_compressed_output(d, TARGET_RULES[target]["format"])
    except Exception as e:  # noqa: BLE001
        fmt_errs.append(f"{type(e).__name__}: {e}")
    if not fmt_errs:
        report["checks"]["quant_format"] = True
    report["errors"].extend(fmt_errs)

    # 4) safetensors 헤더/인덱스 정합성
    if list(d.glob("*.safetensors")):
        idx_errs, n_shards, n_packed = _check_safetensors_and_index(d, cfg, target, allow_tiny=allow_tiny)
        report["num_shards"] = n_shards
        report["packed_tensors"] = n_packed
        if not idx_errs:
            report["checks"]["safetensors_index"] = True
        report["errors"].extend(idx_errs)
    else:
        report["errors"].append(f"{d}: .safetensors 파일 없음")

    # 5) 모델 사이즈 검증
    sz_errs, size_gb = _check_size_gb(d, cfg, target, allow_tiny=allow_tiny)
    report["size_gb"] = size_gb
    if not sz_errs:
        report["checks"]["model_size"] = True
    report["errors"].extend(sz_errs)

    # 6) vLLM 및 transformers config 호환성
    vllm_errs = _check_vllm_and_config(d, cfg, target)
    if not vllm_errs:
        report["checks"]["vllm_ready"] = True
    report["errors"].extend(vllm_errs)

    # 7) PPL 및 벤치마크 지표 검증
    m_errs, m_warns, meta = _check_metrics(
        mode,
        target,
        allow_missing_metrics=allow_missing_metrics,
        allow_discard=allow_discard,
    )
    report["ppl"] = meta["ppl"]
    report["ppl_delta"] = meta["ppl_delta"]
    report["ppl_verdict"] = meta["ppl_verdict"]
    report["benchmarks"] = meta["benchmarks"]
    report["warnings"].extend(m_warns)
    if not m_errs:
        report["checks"]["metrics"] = True
    report["errors"].extend(m_errs)

    # 8) README.md 모델 카드 검증
    rm_errs = _check_readme(d, target, size_gb, allow_missing_metrics=allow_missing_metrics)
    if not rm_errs:
        report["checks"]["readme"] = True
    report["errors"].extend(rm_errs)

    report["ok"] = len(report["errors"]) == 0 and all(report["checks"].values())
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


def run(
    mode: str,
    targets: str | list | None = None,
    *,
    allow_missing_metrics: bool = False,
    allow_discard: bool = False,
    allow_tiny: bool = False,
    skip_bf16: bool = False,
) -> dict:
    """전 타겟 + (선택) 상류 raon-bf16 감사 후 outputs/verify-upload-<mode>.json 기록."""
    ts = _parse_targets(targets)
    unknown = [t for t in ts if t not in TARGET_RULES]
    if unknown:
        raise SystemExit(f"unknown target(s): {unknown} (허용: {ALL_TARGETS})")

    bf16_errors: list[str] = []
    if not skip_bf16 and Path(BF16_DIR()).is_dir():
        bf16_errors = check_bf16_dir(Path(BF16_DIR()))

    reports = {
        t: verify_target(
            mode,
            t,
            allow_missing_metrics=allow_missing_metrics,
            allow_discard=allow_discard,
            allow_tiny=allow_tiny,
        )
        for t in ts
    }
    all_ok = len(bf16_errors) == 0 and all(r["ok"] for r in reports.values())
    summary = {
        "mode": mode,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "bf16_ok": len(bf16_errors) == 0,
        "bf16_errors": bf16_errors,
        "ok": all_ok,
        "targets": reports,
    }
    out_dir = Path(OUT_ROOT())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"verify-upload-{mode}.json"
    out_fp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not all_ok:
        failed = [t for t, r in reports.items() if not r["ok"]]
        raise SystemExit(f"46-verify-upload FAILED: bf16_errors={len(bf16_errors)}, failed_targets={failed}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="업로드 전 모델·지표·README·vLLM 호환성 종합 감사")
    ap.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    ap.add_argument("--targets", default=None, help="콤마 또는 공백 구분 (기본: $TARGETS 또는 all)")
    ap.add_argument("--allow-missing-metrics", action="store_true", help="PPL/eval 지표 누락 허용")
    ap.add_argument("--allow-discard", action="store_true", help="PPL Discard 판정도 통과 허용")
    ap.add_argument("--allow-tiny", action="store_true", help="소형 테스트 모델 용량/텐서 수 예외 허용")
    ap.add_argument("--skip-bf16", action="store_true", help="models/raon-bf16 상류 검사 생략")
    a = ap.parse_args()
    run(
        a.mode,
        a.targets,
        allow_missing_metrics=a.allow_missing_metrics,
        allow_discard=a.allow_discard,
        allow_tiny=a.allow_tiny,
        skip_bf16=a.skip_bf16,
    )
