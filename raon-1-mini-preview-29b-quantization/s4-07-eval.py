"""s4-07-eval.py — 벤치마크 프롬프트 생성 (vLLM 우선, transformers 폴백).
입력: outputs/eval-<mode>/prompts-<bench>.jsonl (있으면) — 없으면 calib 파일 뒤 EVAL_N개.
출력: outputs/eval-<mode>/<target>.jsonl ({bench, prompt, gold, output}).
        outputs/eval-<mode>/bench/<bench>/<target>.jsonl (벤치별 상세, s4-13-gsm8k는 최상위만 집계)

백엔드(요구사항 2):
  KRAFTON Raon-1-mini-preview는 자체 vllm 포크(krafton-ai/vllm@raon-1-mini-preview)를
  요구하는데 현재 venv의 업스트림 vllm에는 Raon1MiniForCausalLM이 등록돼 있지 않다.
  업스트림의 TransformersMoEForCausalLM 폴백도 num_experts_per_token/top_k 필드명과
  transformers>=5.15 버전 체크 때문에 실패한다. 반면 transformers의
  from_pretrained(trust_remote_code=True)는 compressed-tensors 후크로 양자화 모델을
  정상 로드하므로, backend=auto는 vLLM을 먼저 시도하고 커스텀 아키텍처 미지원일 때만
  transformers로 폴백한다 (CUDA OOM 같은 진짜 오류는 삼키지 않고 그대로 올린다).

  uv run s4-07-eval.py --mode smoke --target nvfp4-gptq-transformers --backend auto --tp 2
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Any

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")
ALL_TARGETS = ["base", "awq-w3-transformers", "awq-w5-transformers", "nvfp4-awq-transformers", "nvfp4-gptq-transformers"]
# 평가 벤치마크 = 표준 GSM8K (openai/gsm8k main/test).
# 프롬프트는 s4-04-prep-gsm8k.sh가 결정론적으로 만든다(수작업 표본 금지).
BENCHMARKS = ["gsm8k"]
QUANT_OF = {"nvfp4-awq-transformers": "compressed-tensors",
            "nvfp4-gptq-transformers": "compressed-tensors", "awq-w5-transformers": "compressed-tensors", "awq-w3-transformers": "compressed-tensors"}

# transformers 심볼은 런타임에 한 번만 채운다 (테스트가 모듈 속성을 주입할 수 있게).
AutoModelForCausalLM: Any = None
AutoTokenizer: Any = None


def MODELS_DIR(mode: str) -> str:
    return f"{SHARED}/models"


def OUTDIR(mode: str) -> str:
    return f"{SHARED}/outputs/eval-{mode}"


def resolve(target: str, mode: str) -> str:
    if target == "base":
        return f"{MODELS_DIR(mode)}/raon-bf16"  # 19-dequant 복원본
    return f"{MODELS_DIR(mode)}/raon-{target}-{mode}"


def _ensure_transformers() -> None:
    global AutoModelForCausalLM, AutoTokenizer
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        from transformers import AutoModelForCausalLM as _M
        from transformers import AutoTokenizer as _T

        AutoModelForCausalLM, AutoTokenizer = _M, _T


def _default_max_tokens() -> int:
    return int(os.environ.get("EVAL_MAX_TOKENS", "256"))


def release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except (ImportError, RuntimeError):  # pragma: no cover - CPU 전용 환경
        pass


# ---------------------------------------------------------------- vLLM 미지원 판정

_UNSUPPORTED_HINTS = (
    "raon1miniforcausallm",
    "raon1minicausallm",
    "raon1mini",
    "unsupported architecture",
    "not supported",
    "is not supported",
    "unrecognized model",
    "no module named",
    "transformers modeling backend",
    "engine core initialization failed",
)


def unsupported_vllm_error(detail: str) -> RuntimeError:
    """커스텀 아키텍처 미지원 — auto 폴백이 정당한 유일한 예외 타입."""
    return RuntimeError(
        f"vllm: unsupported architecture ({detail}) — 업스트림 vllm는 이 커스텀 아키텍처를 미지원"
        " (KRAFTON 포크 필요), transformers 백엔드로 폴백한다"
    )


def is_unsupported_vllm_error(exc: BaseException) -> bool:
    """vLLM이 이 아키텍처를 모른다고 말했는지. OOM/결속 오류는 폴백 대상이 아니다."""
    if isinstance(exc, RuntimeError) and str(exc) is not None and "unsupported architecture" in str(exc):
        return True
    blob = f"{type(exc).__name__}: {exc}".lower()
    return any(h in blob for h in _UNSUPPORTED_HINTS)


def _check_vllm_arch_supported(model_path: str) -> None:
    """vLLM EngineCore 서브프로세스를 띄우기 전에 ModelRegistry 등록 여부를 먼저 확인.

    업스트림 vllm(0.31.0)에는 Raon1MiniForCausalLM이 없어 자식 프로세스가
    TransformersMoEForCausalLM으로 폴백하다 죽는다. 미리 검사해 불필요한 GPU 프로세스
    기동 없이 즉시 unsupported_vllm_error를 던진다.
    """
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        return
    try:
        with open(cfg_path, encoding="utf-8") as f:
            archs = json.load(f).get("architectures") or []
    except (OSError, json.JSONDecodeError):
        return
    if not archs:
        return
    try:
        from vllm.model_executor.models.registry import ModelRegistry

        supported = set(ModelRegistry.get_supported_archs())
    except Exception:  # noqa: BLE001 — vllm 미설치나 내부 API 변경 시 LLM() 호출에 맡긴다
        return
    if not any(a in supported for a in archs):
        raise unsupported_vllm_error(f"{archs} not in vllm ModelRegistry")


# ---------------------------------------------------------------- 백엔드


def run_vllm(model_path: str, prompts: list[str], max_tokens: int, target: str = "", tp: int = 1) -> list[str]:
    """vLLM offline greedy 생성. 아키텍처 미지원이면 unsupported_vllm_error로 표면화."""
    _check_vllm_arch_supported(model_path)
    from vllm import LLM, SamplingParams

    kw: dict = {
        "model": model_path,
        "tensor_parallel_size": tp,
        "trust_remote_code": True,
        "gpu_memory_utilization": float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.9")),
        "max_model_len": int(os.environ.get("VLLM_MAX_LEN", "8192")),
    }
    if target in QUANT_OF:
        kw["quantization"] = QUANT_OF[target]
    try:
        llm = LLM(**kw)
    except Exception as e:  # 판정 후 재던짐/폴백
        if is_unsupported_vllm_error(e):
            raise unsupported_vllm_error(str(e)[:200]) from e
        raise
    try:
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens))
        return [o.outputs[0].text for o in outs]
    finally:
        del llm
        release_memory()


def run_transformers(model_path: str, prompts: list[str], max_tokens: int, batch_size: int = 1) -> list[str]:
    """transformers 백엔드 — trust_remote_code + greedy decoding.

    입력 프롬프트를 토큰화해 길이를 기억하고, 생성 결과에서 그 이후 토큰만 디코드한다
    (프롬프트가 출력에 중복되지 않게).

    batch_size > 1 이면 **왼쪽 패딩**으로 묶어 한 번에 생성한다.
    왼쪽 패딩을 쓰는 이유: 생성 토큰이 항상 뒤에 붙으므로 새 토큰 구간이
    `ids.size(1)`부터 시작해 행별로 입력 길이가 달라도 동일하게 잘라낼 수 있다.
    """
    import torch

    from quant_common import patch_unified_memory_info

    patch_unified_memory_info()
    _ensure_transformers()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token", None):
        tok.pad_token = tok.eos_token
    try:
        tok.padding_side = "left"
    except Exception:  # noqa: BLE001 — padding_side를 못 바꾸는 토크나이저는 단건 처리로 폴백
        batch_size = 1
    model.eval()
    bs = max(1, int(batch_size))
    max_len = int(os.environ.get("EVAL_MAX_LEN", "4096"))
    texts: list[str] = []
    try:
        for start in range(0, len(prompts), bs):
            chunk = prompts[start : start + bs]
            enc = tok(chunk, return_tensors="pt", padding=bs > 1, truncation=True, max_length=max_len)
            ids = enc["input_ids"].to(model.device)
            mask = enc.get("attention_mask")
            if mask is not None:
                mask = mask.to(model.device)
            out = model.generate(
                ids, attention_mask=mask, do_sample=False, max_new_tokens=max_tokens, use_cache=True
            )
            gen = out[:, ids.size(1) :]
            for row in gen:
                texts.append(tok.decode(row, skip_special_tokens=True))
    finally:
        del model
        release_memory()
    return texts


BACKEND = os.environ.get("RQ_EVAL_BACKEND", "auto")


def generate(target: str, model_path: str, prompts: list[str], *, max_tokens: int | None = None,
             tp: int | None = None, backend: str | None = None, on_fallback=None,
             batch_size: int | None = None) -> list[str]:
    """backend로 생성. auto는 vLLM → (아키텍처 미지원 시) transformers 순."""
    backend = backend or BACKEND
    max_tokens = max_tokens or _default_max_tokens()
    tp = tp or int(os.environ.get("TP", "1"))
    if backend in ("auto", "vllm"):
        try:
            return run_vllm(model_path, prompts, max_tokens, target, tp)
        except Exception as e:  # 명시 backend면 그대로, auto면 조건부 폴백
            if backend == "vllm" or not is_unsupported_vllm_error(e):
                raise
            print(f"[s4-07-eval] vllm 미지원 → transformers 폴백: {e}")
            if on_fallback is not None:
                on_fallback(str(e))
    bs = batch_size if batch_size is not None else int(os.environ.get("EVAL_BATCH", "1"))
    return run_transformers(model_path, prompts, max_tokens, bs)


# ---------------------------------------------------------------- 프롬프트/산출물


#: 캘리브레이션 텍스트에서 "assistant 응답 시작" 마커.
#: 데이터 형식에 따라 ChatML 마커와 평문 마커가 함께 쓰인다 — 둘 다 처리해야 한다.
#: (평문 마커만 보면 실제 캘리브 데이터의 응답 꼬리가 잘려나가지 않아
#:  평가 프롬프트에 정답이 새어 들어갈 수 있다.)
_STRIP_MARKERS = ("<|" + "im_start|>assistant\n", "assistant\n")


def _strip_assistant_completion(text: str) -> str:
    """캘리브레이션 텍스트에 포함된 assistant 응답 꼬리를 잘라내 생성용 프롬프트만 남긴다."""
    for marker in _STRIP_MARKERS:
        idx = text.find(marker)
        if idx == -1:
            continue
        after = text[idx + len(marker) :]
        if after.startswith("<think></think>\n\n"):
            return text[: idx + len(marker)] + "<think></think>\n\n"
        return text[: idx + len(marker)]
    return text


def _read_all_texts(path: str) -> list[str]:
    """JSONL 파일에서 가능한 모든 텍스트 행을 읽는다 (행 수 상한 검사 없이 전체 반환)."""
    if not os.path.exists(path):
        raise SystemExit(f"missing calibration/eval file: {path} (run s1-07-prep-calibration.sh first)")
    texts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "text" in r:
                texts.append(_strip_assistant_completion(r["text"]))
            elif "prompt" in r:
                texts.append(r["prompt"])
            else:
                src = r.get("source", "")
                src = src[0] if isinstance(src, list) else src
                texts.append(str(src))
    if not texts:
        raise SystemExit(f"empty calibration/eval file: {path}")
    return texts


def load_prompts(mode: str, n: int, calib_file: str | None, benchmarks: list[str]) -> list[dict]:
    """벤치별 프롬프트 {bench, prompt, gold} 목록.

    1) outputs/eval-<mode>/prompts-<bench>.jsonl (정식 벤치 데이터 — 표준 GSM8K는
       s4-04-prep-gsm8k.sh가 만든 파일이다)
    2) **정식 파일이 없으면 실패한다** — calib-tail 폴백은 표준 벤치마크가 아니라
       임의 텍스트 평가라(수작업 표본) 표준 GSM8K 평가로는 쓰지 않는다.
       단, 의사 벤치 "smoke"만 레거시 폴백을 허용한다.
    """
    outdir = OUTDIR(mode)
    rows: list[dict] = []
    missing: list[str] = []
    for b in benchmarks:
        fp = os.path.join(outdir, f"prompts-{b}.jsonl")
        if not os.path.exists(fp):
            missing.append(b)
            continue
        bench_rows: list[dict] = []
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r.setdefault("bench", b)
                    bench_rows.append(r)
        rows.extend(bench_rows[:n] if n else bench_rows)
    if missing and [b for b in missing if b != "smoke"]:
        raise SystemExit(
            f"bench 프롬프트 파일 없음: {missing} — ./s4-04-prep-gsm8k.sh {mode} 로 먼저 준비하라"
        )
    if rows:
        return rows

    eval_file = os.environ.get("EVAL_FILE", f"{SHARED}/datasets/raon-eval.jsonl")
    if os.path.exists(eval_file):
        texts = _read_all_texts(eval_file)
        print(f"eval_src={eval_file}")
    else:
        default_calib = f"{SHARED}/datasets/raon-calibration-{8 if mode == 'smoke' else 256}.jsonl"
        cfile = calib_file or os.environ.get("CALIB_FILE") or default_calib
        all_texts = _read_all_texts(cfile)
        texts = all_texts[-n:] if n else all_texts
        print("eval_src=calib-tail (전용 eval 파일 없음, 12 뒤쪽 슬라이스)")
    picked = texts[:n] if n else texts
    return [{"bench": benchmarks[i % len(benchmarks)], "prompt": t, "gold": None} for i, t in enumerate(picked)]


def build_row(p: dict, output: str) -> dict:
    """프롬프트 행 + 생성 결과를 응답 행으로 만든다.

    프롬프트 행의 메타데이터(case/check/subject 등)를 그대로 보존한다 —
    형식 검사(s4-22)는 응답 행에서 검사 키를 읽어야 하는데, 여기서 버리면
    "검사 키 없음"이 되어 모든 케이스가 통과로 집계된다(가짜 통과).
    """
    row = {k: v for k, v in p.items() if k != "prompt"}
    row.setdefault("bench", "smoke")
    row["prompt"] = p["prompt"]
    row["output"] = output
    return row


def preserved_rows(main_path: str, regenerated: set[str]) -> list[dict]:
    """기존 <target>.jsonl에서 이번에 다시 생성하지 않는 벤치의 행만 돌려준다."""
    if not os.path.exists(main_path):
        return []
    kept: list[dict] = []
    with open(main_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("bench", "smoke") not in regenerated:
                kept.append(r)
    return kept


def run_target(mode: str, target: str, backend: str | None, prompts: list[dict], *,
               max_tokens: int | None = None, tp: int | None = None,
               batch_size: int | None = None) -> int:
    """타겟 1개를 생성해 <target>.jsonl + bench/<bench>/<target>.jsonl 저장."""
    name = resolve(target, mode)
    if not os.path.exists(f"{name}/config.json"):
        hint = " (run s2-01-dequant-mxfp8.sh first)" if target == "base" else ""
        raise SystemExit(f"missing model: {name}{hint}")
    texts = [p["prompt"] for p in prompts]
    outputs = generate(target, name, texts, max_tokens=max_tokens, tp=tp, backend=backend,
                       batch_size=batch_size)

    outdir = OUTDIR(mode)
    os.makedirs(outdir, exist_ok=True)
    for b in {p.get("bench", "smoke") for p in prompts}:
        os.makedirs(os.path.join(outdir, "bench", b), exist_ok=True)
    main_path = os.path.join(outdir, f"{target}.jsonl")
    # 이번 실행에 없는 벤치의 기존 응답은 보존한다 — 일부 벤치만 다시 돌렸을 때
    # (예: EVAL_BENCHMARKS=kmmluko,format) 다른 벤치 응답이 사라지면 채점·카드가 비게 된다.
    kept = preserved_rows(main_path, {p.get("bench", "smoke") for p in prompts})
    detail: dict[str, list] = {}
    with open(main_path, "w", encoding="utf-8") as f:
        for p, o in zip(prompts, outputs):
            b = p.get("bench", "smoke")
            row = build_row(p, o)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            detail.setdefault(b, []).append(row)
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for b, rows in detail.items():
        with open(os.path.join(outdir, "bench", b, f"{target}.jsonl"), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    kept_benches = sorted({r.get("bench", "smoke") for r in kept})
    print(f"[{mode}/{target}] n={len(prompts)} backend={backend or BACKEND} -> {main_path}"
          + (f" (kept {len(kept)} rows of {kept_benches})" if kept else ""))
    return len(prompts)


def main(mode: str, target: str, n: int, tp: int, calib_file: str | None, backend: str | None,
         benchmarks: list[str], batch_size: int | None = None):
    targets = ALL_TARGETS if target == "all" else [target]
    for t in targets:
        prompts = load_prompts(mode, n, calib_file, benchmarks)
        run_target(mode, t, backend, prompts, max_tokens=_default_max_tokens(), tp=tp,
                   batch_size=batch_size)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="벤치마크 프롬프트 생성 (vLLM/transformers)")
    ap.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    ap.add_argument("--target", default="nvfp4-gptq-transformers", choices=["all"] + ALL_TARGETS)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--calib-file", default=None)
    ap.add_argument("--backend", default=None, choices=["auto", "vllm", "transformers"],
                    help="기본: $RQ_EVAL_BACKEND 또는 auto (vllm 미지원 시 transformers 폴백)")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS),
                    help="콤마 구분 (기본: 표준 GSM8K). s4-04-prep-gsm8k.sh가 만든 prompts-<bench>.jsonl 사용")
    ap.add_argument("--batch-size", type=int, dest="batch_size", default=None,
                    help="transformers 생성 배치 크기 (기본 EVAL_BATCH, 1=단건)")
    a = ap.parse_args()
    d_n = int(os.environ.get("EVAL_N", "8" if a.mode == "smoke" else "200"))
    d_tp = int(os.environ.get("TP", "2"))
    _bench = [b for b in a.benchmarks.replace(" ", "").split(",") if b] or ["smoke"]
    main(a.mode, a.target, a.n or d_n, a.tp or d_tp, a.calib_file, a.backend, _bench, a.batch_size)