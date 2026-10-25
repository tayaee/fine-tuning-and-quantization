"""quant_common.py — 양자화 스크립트(32/33/34/35/36) 공용 헬퍼.
캘리브레이션 텍스트 로딩 단일 정의. s2-13-prep-calibration이 chat template 적용済み
`{"text": ...}` jsonl을 만들므로 로더는 text를 그대로 읽는다 (구 GSM8K source/target
형식도 호환 유지).
각 스크립트는 `uv run $RQ_ROOT/<nn>-quant-*.py` 로 실행되며 스크립트 디렉토리가
sys.path에 들어오므로 plain `import quant_common`으로 로드된다.
"""

import json
import os
from typing import NamedTuple

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")
DEFAULT_CALIB_FILE = f"{SHARED}/datasets/raon-calibration-8.jsonl"
BASE_MODEL = os.environ.get("BASE_MODEL", "KRAFTON/Raon-1-mini-preview")
BASE_REVISION = os.environ.get("BASE_REVISION") or None

# 벤더(MXFP8) recipe와 동일하게 gate/router + lm_head 제외.
# moe router 실체는 moe.gate라 "gate"에 포함, gate_proj 제외도 벤더 관례 (K-EXAONE 동일).
DEFAULT_IGNORE = ["lm_head", "gate"]

# 양자화 타깃: 기본 experts-only (벤더 MXFP8 ignored_layers와 동일 — attn/mlp/router/shared/lm_head는 BF16 유지).
# RQ_TARGETS=all 이면 전 Linear (9496 모듈, AWQ 매우 느림).
EXPERT_TARGETS = ["re:.*\\.moe\\.experts\\.[0-9]+\\.(gate_proj|up_proj|down_proj)"]


# AWQ smoothing 매핑 (Raon1-mini 전용): 양자화 타깃인 MoE expert만.
# default auto-mappings는 q/k/v_proj 기준이라 이 아키텍처에서 매칭 실패한다.
#  - post_attention_layernorm → experts gate/up. 단 0층은 dense(mlp)라 smooth에서
#    제외해야 한다: 0층 norm이 balance 없이 먼저 매칭되면 parent_context가
#    "model.layers"로 굳어져 24개 MoE층이 한 그룹으로 합쳐지고(single smoothlayer
#    위반) 실패한다. MoE층은 1~24 (probe: match_modules_set 1그룹 [25,3072,3072] 확인).
#  - experts up → down (SiTU-GLU 출력 스케일, expert 단위 그룹이라 무관)
RAON_AWQ_MAPPINGS = [
    (
        "re:.*\\.layers\\.([1-9]|1[0-9]|2[0-4])\\.post_attention_layernorm$",
        ["re:.*moe\\.experts\\.[0-9]+\\.gate_proj$", "re:.*moe\\.experts\\.[0-9]+\\.up_proj$"],
    ),
    (
        "re:.*moe\\.experts\\.[0-9]+\\.up_proj$",
        ["re:.*moe\\.experts\\.[0-9]+\\.down_proj$"],
    ),
]


def build_awq_smoothing():
    """Raon 매핑 고정 AWQ smoothing (mappings 자동 추론 미사용)."""
    from llmcompressor.modifiers.transform.awq import AWQMapping
    from llmcompressor.modifiers.transform.awq import AWQModifier as AWQSmoothing

    return AWQSmoothing(mappings=[AWQMapping(s, b) for s, b in RAON_AWQ_MAPPINGS])


def resolve_targets() -> list[str]:
    which = os.environ.get("RQ_TARGETS", "experts")
    if which == "all":
        return ["Linear"]
    if which == "experts":
        return EXPERT_TARGETS
    return [which]  # custom 패턴 직접 지정

TARGETS = ["awq-w3", "awq-w5", "nvfp4-awq", "nvfp4-gptq"]


class _PackedContract(NamedTuple):
    """압축 포맷별 on-disk 텐서 계약 (compressed_tensors 컴프레서 기준)."""

    packed_dtype: str  # weight_packed safetensors dtype 문자열
    scale_dtype: str | None  # weight_scale dtype (None = 제약 없음)
    needs_global_scale: bool  # weight_global_scale(F32) 필수 여부
    forbid_zero_point: bool  # weight_zero_point 잔존 금지 (대칭 전용 포맷)


PACK_QUANTIZED = "pack-quantized"
NVFP4_PACK_QUANTIZED = "nvfp4-pack-quantized"
CUSTOM_CODE_FILES = ("configuration_raon1_mini.py", "modeling_raon1_mini.py")

# nvfp4: U8 패킹 + F8_E4M3 블록 스케일 + F32 전역 스케일, 대칭 전용.
# pack-quantized(int): num_bits개를 I32에 몰아넣음, 대칭 여부는 레시피가 정한다.
PACKED_CONTRACTS: dict[str, _PackedContract] = {
    PACK_QUANTIZED: _PackedContract("I32", None, False, False),
    NVFP4_PACK_QUANTIZED: _PackedContract("U8", "F8_E4M3", True, True),
}


def _scan_shard_dtypes(model_dir) -> tuple[dict[str, set[str]], dict[str, int], int]:
    """모든 .safetensors 헤더를 스캔해 텐서명 접미사 → dtype 집합/개수, shard 수 반환.
    텐서 데이터는 읽지 않고 헤더만 본다(수십 GB 체크포인트를 훑지 않기 위함)."""
    from pathlib import Path

    from safetensors import safe_open

    shards = sorted(Path(model_dir).glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"{model_dir}: .safetensors 없음 — 압축 체크포인트 아님")
    dtypes: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            # safetensors safe_open은 순회 불가 → keys()로 텐서명 나열
            for name in f.keys():  # noqa: SIM118
                suffix = name.rsplit(".", 1)[-1]  # layer.weight_packed → weight_packed
                dtypes.setdefault(suffix, set()).add(str(f.get_slice(name).get_dtype()))
                counts[suffix] = counts.get(suffix, 0) + 1
    return dtypes, counts, len(shards)


def _verify_global_scale_values(model_dir, *, has_input_scale: bool) -> None:
    """전역 스케일(F32 스칼라)이 미보정(0/음수/NaN/Inf/쓰레기값) 상태로 저장됐는지 검사한다."""
    from pathlib import Path

    import torch
    from safetensors import safe_open

    bad: list[str] = []
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                if not name.endswith(("weight_global_scale", "input_global_scale")):
                    continue
                t = f.get_tensor(name).float()
                if not bool(torch.isfinite(t).all()):
                    bad.append(f"{name}=non-finite")
                elif (has_input_scale or name.endswith("input_global_scale")) and bool(
                    (t <= 0).any() or (t >= 1e10).any()
                ):
                    bad.append(f"{name}={float(t.flatten()[0]):.3e}")
    if bad:
        raise RuntimeError(
            f"{model_dir}: 비정상/미보정 global_scale {len(bad)}개 (예: {bad[:3]})"
            " — MoE 캘리브레이션 누락 가능성"
        )


def _weights_symmetric(qc: dict) -> bool:
    """config_groups 전체가 대칭 가중치 양자화인지 (QuantizationArgs 기본값 True).
    하나라도 weights.symmetric=False면 비대칭(zero_point 정상 잔존)."""
    groups = qc.get("config_groups") or {}
    for scheme in groups.values():
        for one in scheme if isinstance(scheme, list) else [scheme]:
            if not isinstance(one, dict):
                continue
            weights = one.get("weights")
            if isinstance(weights, dict) and not weights.get("symmetric", True):
                return False
    return True


def verify_compressed_output(model_dir, expected_format: str) -> dict:
    """저장 직후(commit_dir 전) 압축 포맷 계약을 검증한다. 실패 시 RuntimeError.

    llmcompressor oneshot은 커스텀 모듈에서 can_compress=False면 에러 없이
    DenseCompressor("format": "dense")로 폴백한다. 그 결과 미패킹 59GB BF16
    체크포인트가 정상 산출물로 커밋되고, 나중에 vLLM 로드/정확도 단계에서야 드러난다.
    그래서 config.json의 format과 safetensors 헤더의 텐서 계약(패킹 dtype,
    scale dtype, global scale, zero_point 잔존)을 여기서 강제한다.

    반환값: {"format", "packed", "shards"}
    """
    from pathlib import Path

    model_dir = Path(model_dir)
    contract = PACKED_CONTRACTS.get(expected_format)
    if contract is None:
        raise RuntimeError(f"unknown expected_format={expected_format!r}")

    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        raise RuntimeError(f"{model_dir}: config.json 없음 — 압축 산출물 아님")
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{cfg_path}: config.json 파싱 실패: {e}") from e
    qc = cfg.get("quantization_config")
    if not isinstance(qc, dict):
        # 압축 미적용(폴백) 실패 계약이라 TypeError가 아닌 RuntimeError를 던진다
        raise RuntimeError(f"{cfg_path}: quantization_config 없음 — 양자화가 적용되지 않음")  # noqa: TRY004
    fmt = qc.get("format")
    if fmt != expected_format:
        raise RuntimeError(
            f"quantization_config.format={fmt!r} != 기대값 {expected_format!r}"
            " — compressor가 can_compress=False로 dense 폴백했을 가능성"
        )

    dtypes, counts, n_shards = _scan_shard_dtypes(model_dir)
    packed = dtypes.get("weight_packed")
    if not packed:
        raise RuntimeError(f"{model_dir}: weight_packed 텐서 없음 — 패킹되지 않은 dense 체크포인트")
    if packed != {contract.packed_dtype}:
        raise RuntimeError(
            f"weight_packed dtype={sorted(packed)} != [{contract.packed_dtype}] — 포맷 {fmt!r} 불일치"
        )
    scale = dtypes.get("weight_scale")
    if not scale:
        raise RuntimeError(f"{model_dir}: weight_scale 텐서 없음 — 역양자화 불가")
    if contract.scale_dtype and scale != {contract.scale_dtype}:
        raise RuntimeError(f"weight_scale dtype={sorted(scale)} != [{contract.scale_dtype}]")
    if contract.needs_global_scale:
        gscale = dtypes.get("weight_global_scale")
        if not gscale:
            raise RuntimeError(f"{model_dir}: weight_global_scale 텐서 없음 — {fmt}는 F32 전역 스케일 필수")
        if gscale != {"F32"}:
            raise RuntimeError(f"weight_global_scale dtype={sorted(gscale)} != [F32]")
        iscale = dtypes.get("input_global_scale")
        if iscale and iscale != {"F32"}:
            raise RuntimeError(f"input_global_scale dtype={sorted(iscale)} != [F32]")
        _verify_global_scale_values(model_dir, has_input_scale=bool(iscale))
    if contract.forbid_zero_point and "weight_zero_point" in dtypes:
        raise RuntimeError(f"{model_dir}: weight_zero_point 잔존 — {fmt}는 대칭 전용 (vLLM 로드 불가)")
    if _weights_symmetric(qc) and "weight_zero_point" in dtypes:
        raise RuntimeError(f"{model_dir}: 대칭 양자화인데 weight_zero_point 잔존 (vLLM 로드 불가)")

    missing = [name for name in CUSTOM_CODE_FILES if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"{model_dir}: custom code 누락 {missing} — trust_remote_code 재로드 불가")

    info = {"format": fmt, "packed": counts.get("weight_packed", 0), "shards": n_shards}
    print(f"verify_compressed_output OK {model_dir} {info}")
    return info


def save_custom_code(out_dir: str, src_dir: str | None = None) -> None:
    """custom_code *.py를 출력 디렉토리에 복사.
    src_dir(로컬 복원본 등)이 주어지면 거기서, 없으면 Hub 스냅샷에서 가져온다.
    save_pretrained는 modeling/configuration .py를 저장하지 않으므로,
    로컬 디렉토리를 trust_remote_code로 다시 로드하려면 이 복사가 필수다.
    """
    import glob
    import shutil

    if src_dir is None:
        from huggingface_hub import snapshot_download

        src_dir = snapshot_download(
            repo_id=BASE_MODEL, revision=BASE_REVISION, allow_patterns=["*.py"]
        )
    copied = 0
    for src in glob.glob(f"{src_dir}/*.py"):
        shutil.copy(src, f"{out_dir}/{src.split('/')[-1]}")
        copied += 1
    print(f"custom_code copied={copied} from {src_dir}")
    if copied == 0:
        print("WARN: no *.py in source — custom_arch reload may fail")


def patch_unified_memory_info() -> None:
    """GB10(128GB 통합 메모리, is_integrated=1)에서 OS 페이지 캐시(buff/cache) 때문에
    cudaMemGetInfo의 free가 수 GB로 낮게 보고되는 문제를 보정한다.
    실제 CUDA 할당 시 커널이 페이지 캐시를 즉시 회수하므로 MemAvailable을 반영한다.
    """
    try:
        import psutil
        import torch

        if not torch.cuda.is_available():
            return
        if not getattr(torch.cuda.get_device_properties(0), "is_integrated", 0):
            return
        _orig = getattr(torch.cuda, "_orig_mem_get_info", None) or torch.cuda.mem_get_info
        torch.cuda._orig_mem_get_info = _orig  # type: ignore[attr-defined]

        def _unified_mem_get_info(device=None):
            free, total = _orig(device)
            avail = int(psutil.virtual_memory().available)
            return (max(free, avail), total)

        torch.cuda.mem_get_info = _unified_mem_get_info
        if hasattr(torch, "accelerator"):
            if hasattr(torch.accelerator, "memory"):
                torch.accelerator.memory.get_memory_info = _unified_mem_get_info
            if hasattr(torch.accelerator, "get_memory_info"):
                torch.accelerator.get_memory_info = _unified_mem_get_info
    except (ImportError, RuntimeError):  # pragma: no cover - CPU 전용 환경
        return


def allow_trusted_remote_code() -> None:
    """llmcompressor 내부 AutoConfig 호출(trust 미전파) 우회.
    pinned revision 코드를 직접 검토했으므로 trust=None만 True로 해결
    (명시적 False는 유지). transformers configuration_auto 경유 전제.
    """
    patch_unified_memory_info()
    try:
        import transformers.models.auto.configuration_auto as caa
    except ImportError:
        return
    if not hasattr(caa, "resolve_trust_remote_code"):
        raise SystemExit(
            "transformers API changed: no resolve_trust_remote_code in configuration_auto"
        )
    _orig = caa.resolve_trust_remote_code  # pyright: ignore[reportPrivateImportUsage]

    def _resolve(trust_remote_code, *args, **kwargs):
        if trust_remote_code is None:
            return True
        return _orig(trust_remote_code, *args, **kwargs)

    caa.resolve_trust_remote_code = _resolve  # pyright: ignore[reportPrivateImportUsage]
    print("trust_remote_code: None->True (pinned revision, reviewed)")


def load_calib_texts(calib: int, calib_file: str | None = None) -> list[str]:
    """캘리브레이션 jsonl 앞 `calib`줄을 텍스트로 반환.
    `{"text"}` 우선, 없으면 구 형식 source/target 결합.
    파일 부재·행 부족 시 명시 종료 (s2-13-prep-calibration.sh로 생성).
    """
    calib_file = calib_file or os.environ.get("CALIB_FILE", DEFAULT_CALIB_FILE)
    if not os.path.exists(calib_file):
        raise SystemExit(
            f"missing calibration file: {calib_file} (run s2-13-prep-calibration.sh first)"
        )
    texts = []
    with open(calib_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "text" in r:
                texts.append(r["text"])
            else:
                src = r.get("source", "")
                tgt = r.get("target", "")
                src = src[0] if isinstance(src, list) else src
                tgt = tgt[0] if isinstance(tgt, list) else tgt
                texts.append(f"{src}\n{tgt}")
            if len(texts) >= calib:
                break
    if len(texts) < calib:
        raise SystemExit(
            f"calibration file has {len(texts)} rows < requested calib={calib}"
        )
    print(f"calib_src={calib_file} n={len(texts)}")
    return texts
