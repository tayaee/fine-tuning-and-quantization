"""s3-01-quant-nvfp4-awq-vllm.py — BF16 원본 → AWQ Lite + NVFP4 (ModelOpt).

transformers판(s3-01)과 같은 NVFP4-AWQ지만 표지가 modelopt_fp4라 vLLM 포크
nvfp4칸으로 적재된다. 레시피 뼈대 = NVFP4_EXPERTS_ONLY (KRAFTON NVFP4와 동일
범위: MoE expert만, router/conv/lm_head 제외, KV 무양자화) + awq_lite 탐색.

  uv run s3-01-quant-nvfp4-awq-vllm.py --mode smoke|full [--calib N] [--calib-file PATH]
"""

import argparse
import copy
import os

from io_common import clean_tmp, commit_dir, tmp_path
from quant_common import (
    MOPT_NVFP4,
    allow_trusted_remote_code,
    load_calib_texts,
    save_custom_code,
    verify_mopt_output,
)

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")
BASE_MODEL = os.environ.get("BASE_MODEL", "KRAFTON/Raon-1-mini-preview")
BASE_REVISION = os.environ.get("BASE_REVISION") or "main"
SRC = os.environ.get("RQ_SRC", f"{SHARED}/models/raon-bf16")
TARGET = "nvfp4-awq-vllm"


def build_config(with_awq: bool = True) -> dict:
    import modelopt.torch.quantization as mtq

    cfg = copy.deepcopy(dict(mtq.NVFP4_EXPERTS_ONLY_CFG))
    if with_awq:
        cfg["algorithm"] = {"method": "awq_lite"}
    return cfg


def _strip_mopt_extras(out_dir: str) -> None:
    """포크 Marlin MoE가 등록하지 않는 여분 텐서를 export 산출물에서 제거한다.
    - *pre_quant_scale: awq_lite 탐색 메타데이터 (수치에 반영済み, KRAFTON 표에 없음)
    weight_scale_2(전역 스케일)는 Marlin 커널이 w13/w2_weight_scale_2로 등록해
    사용하므로 반드시 유지한다 (삭제 시 미초기화 메모리 → 음절 샐러드, 2026-10-28 확인).
    input_scale/weight_scale도 유지한다."""
    import json

    import torch
    from safetensors.torch import load_file, save_file

    out = out_dir
    idx_fp = f"{out}/model.safetensors.index.json"
    idx = json.load(open(idx_fp, encoding="utf-8"))
    wmap = idx["weight_map"]
    drop = [k for k in wmap if k.endswith("pre_quant_scale")]
    if not drop:
        print("strip: no extras")
        return
    by_file: dict[str, list[str]] = {}
    for k in drop:
        by_file.setdefault(wmap[k], []).append(k)
    import glob as _glob
    import os as _os

    for fn, keys in by_file.items():
        fp = f"{out}/{fn}"
        if not _os.path.isfile(fp):
            # index shard명과 실제 파일명 불일치 시 glob으로 탐색
            cands = _glob.glob(f"{out}/*.safetensors")
            fp = next((c for c in cands if c.endswith(fn) or fn in c), fp)
        tensors = load_file(fp)
        for k in keys:
            del tensors[k]
            del wmap[k]
        save_file(tensors, fp)
    json.dump(idx, open(idx_fp, "w"), ensure_ascii=False, indent=2)
    print(f"strip: removed {len(drop)} extras (pre_quant_scale only)")


def _stamp_modelopt_fp4(out_dir: str) -> None:
    """export_hf_checkpoint은 quant_method='modelopt', quant_algo='NVFP4_AWQ'로 찍는다.
    vLLM 포크 nvfp4칸(KRAFTON NVFP4와 동일)은 quant_method='modelopt_fp4' +
    quant_algo in ('NVFP4', 'MIXED_PRECISION')를 보므로, 가중치 실물은 그대로
    두고 표지만 교정한다. 실물은 NVFP4(E2M1, block 16)라 표기가 맞다."""
    import json

    fp = f"{out_dir}/config.json"
    with open(fp, encoding="utf-8") as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config") or {}
    changed = []
    if qc.get("quant_method") != "modelopt_fp4":
        qc["quant_method"] = "modelopt_fp4"
        changed.append("quant_method=modelopt_fp4")
    if qc.get("quant_algo") not in ("NVFP4", "MIXED_PRECISION"):
        qc["quant_algo"] = "NVFP4"
        changed.append("quant_algo=NVFP4")
    if changed:
        # AWQ 판별 정보는 별도 필드에 보존 (포크 게이트 무관, 카드·감사 근거용)
        qc["producer"] = {"recipe": "nvfp4-awq-vllm", "calibration": "NVFP4_AWQ"}
        cfg["quantization_config"] = qc
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print(f"stamped {', '.join(changed)} ({fp})")


def main(mode: str, calib: int, calib_file: str | None, seq_len: int):
    import torch
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = f"{SHARED}/models/raon-{TARGET}-{mode}"
    allow_trusted_remote_code()
    texts = load_calib_texts(calib, calib_file)
    print(f"[{mode}] {TARGET} calib={calib} seq={seq_len} src={SRC}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, revision=BASE_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        SRC, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    def forward_loop(m):
        for t in texts[:calib]:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=seq_len)["input_ids"].to(model.device)
            with torch.no_grad():
                m(ids)

    last_err = None
    for with_awq in ([True, False] if os.environ.get("RQ_AWQ_FIRST", "1") == "1" else [False]):
        cfg = build_config(with_awq)
        tmp = clean_tmp(tmp_path(out))
        try:
            qmodel = mtq.quantize(model, cfg, forward_loop)
            export_hf_checkpoint(qmodel, export_dir=tmp)
            _strip_mopt_extras(tmp)
            _stamp_modelopt_fp4(tmp)
            tok.save_pretrained(tmp)
            save_custom_code(tmp, SRC)
            verify_mopt_output(tmp)
            commit_dir(tmp, out)
            print(f"[{mode}] {TARGET} awq={with_awq} calib={calib} seq={seq_len} -> {out}")
            return
        except Exception as e:  # noqa: BLE001 — 폴백 체인 (awq → pure)
            import gc
            import traceback

            traceback.print_exc()
            last_err = f"{type(e).__name__}: {e}"
            print(f"awq={with_awq} failed: {last_err}")
            del cfg
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise SystemExit(f"all recipes failed, last: {last_err}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--calib", type=int, default=None)
    ap.add_argument("--calib-file", default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    a = ap.parse_args()
    d_calib = int(os.environ.get("CALIB_N", "8" if a.mode == "smoke" else "256"))
    d_seq = int(os.environ.get("SEQ_LEN", "512" if a.mode == "smoke" else "2048"))
    main(a.mode, a.calib or d_calib, a.calib_file, a.seq_len or d_seq)
