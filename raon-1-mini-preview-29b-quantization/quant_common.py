"""quant_common.py — 양자화 스크립트(32/33/34/35/36) 공용 헬퍼.
캘리브레이션 텍스트 로딩 단일 정의. 12-prep-calibration이 chat template 적용済み
`{"text": ...}` jsonl을 만들므로 로더는 text를 그대로 읽는다 (구 GSM8K source/target
형식도 호환 유지).
각 스크립트는 `uv run $RQ_ROOT/<nn>-quant-*.py` 로 실행되며 스크립트 디렉토리가
sys.path에 들어오므로 plain `import quant_common`으로 로드된다.
"""

import json
import os

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


def allow_trusted_remote_code() -> None:
    """llmcompressor 내부 AutoConfig 호출(trust 미전파) 우회.
    pinned revision 코드를 직접 검토했으므로 trust=None만 True로 해결
    (명시적 False는 유지). transformers configuration_auto 경유 전제.
    """
    try:
        import transformers.models.auto.configuration_auto as caa
    except ImportError:
        return
    if not hasattr(caa, "resolve_trust_remote_code"):
        raise SystemExit(
            "transformers API changed: no resolve_trust_remote_code in configuration_auto"
        )
    _orig = caa.resolve_trust_remote_code

    def _resolve(trust_remote_code, *args, **kwargs):
        if trust_remote_code is None:
            return True
        return _orig(trust_remote_code, *args, **kwargs)

    caa.resolve_trust_remote_code = _resolve
    print("trust_remote_code: None->True (pinned revision, reviewed)")


def load_calib_texts(calib: int, calib_file: str | None = None) -> list[str]:
    """캘리브레이션 jsonl 앞 `calib`줄을 텍스트로 반환.
    `{"text"}` 우선, 없으면 구 형식 source/target 결합.
    파일 부재·행 부족 시 명시 종료 (12-prep-calibration.sh로 생성).
    """
    calib_file = calib_file or os.environ.get("CALIB_FILE", DEFAULT_CALIB_FILE)
    if not os.path.exists(calib_file):
        raise SystemExit(
            f"missing calibration file: {calib_file} (run 12-prep-calibration.sh first)"
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
