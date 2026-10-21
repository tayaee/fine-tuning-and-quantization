"""48-measure-ppl.py — Stage 6. 전 타깃 PPL 측정 + Accept/Conditional/Discard 판정.
대상 19종: base, fft, qlora(merged) + HF 양자화 6종 + GGUF 10종(fft/qlora × 5).
GGUF는 transformers 대신 llama-perplexity로 측정 (미빌드 시 해당 타깃만 SKIP).

판정 기준 (부모 FP 대비 ΔPPL):
  Δ < 0.3            → Accept (그대로 배포)
  0.3 ≤ Δ < 1.0      → Conditional (50-eval GSM8K 2차 심사 필요)
  Δ ≥ 1.0            → Discard (추론 깨짐 구간)
부모: fft-* vs fft, qlora-* vs qlora. fft/qlora 자체는 vs base 참고용으로 표시.

텍스트: $M1B_SHARED/datasets/gsm8k-test.jsonl 앞 N개를 학습 프롬프트 템플릿으로
감싸서 in-domain PPL 측정 (WikiText가 아니라 수학 특화 유지 여부용).

  uv run 48-measure-ppl.py --mode mini|full [--n 32] [--targets all|base,fft,...]
  출력: stdout 표 + $M1B_SHARED/outputs/ppl-<mode>-<strat>/ (summary.json + 타깃별 json)
  summary.json은 중첩 트리 {meta, tree}. 각 노드는 method/parent/delta/delta_vs/
  verdict를 가지며, delta는 같은 method끼리만 (gguf는 q8_0 형제 대비).
"""

import argparse
import json
import math
import os

from io_common import commit_file, tmp_path

SHARED = os.environ.get("M1B_SHARED", os.environ.get("P5_SHARED", "/rosenas/data/AIML/math-1b-shared"))
BASE = os.environ.get("BASE_MODEL", "unsloth/Llama-3.2-1B")
LLAMACPP = os.environ.get("LLAMACPP", os.path.expanduser("~/git/llama.cpp"))
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context.\n"
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\nGive a response as the assistant with the input conversation history\n\n"
    "### Response:\n{response}"
)

# target → (kind, 경로템플릿, 부모타깃).
# GGUF 5종 변종 (42/52 산출물): kind="gguf"는 llama-perplexity로 측정.
# fft-gguf/qlora-gguf 키는 F16 원본 측정 (분기 기준점). q8_0 포함 5종 변종은
# GQUANTS 루프에서 생성되어 F16 대비로 판정한다.
GQUANTS = ["q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m"]
TARGETS = {
    "base": ("hf", BASE, None),
    "fft": ("hf", f"{SHARED}/models/synthetic-fft-{{m}}-{{s}}", "base"),
    "qlora": ("hf", f"{SHARED}/models/synthetic-qlora-{{m}}-{{s}}-merged", "base"),
    "fft-gguf": (
        "gguf",
        f"{SHARED}/models/synthetic-fft-{{m}}-{{s}}-gguf/model-f16.gguf",
        "fft",
    ),
    "fft-gptq": ("hf", f"{SHARED}/models/synthetic-fft-{{m}}-{{s}}-gptq", "fft"),
    "fft-awq": ("hf", f"{SHARED}/models/synthetic-fft-{{m}}-{{s}}-awq", "fft"),
    "fft-fp8": ("hf", f"{SHARED}/models/synthetic-fft-{{m}}-{{s}}-fp8", "fft"),
    "qlora-gguf": (
        "gguf",
        f"{SHARED}/models/synthetic-qlora-{{m}}-{{s}}-merged-gguf/model-f16.gguf",
        "qlora",
    ),
    "qlora-gptq": (
        "hf",
        f"{SHARED}/models/synthetic-qlora-{{m}}-{{s}}-merged-gptq",
        "qlora",
    ),
    "qlora-awq": (
        "hf",
        f"{SHARED}/models/synthetic-qlora-{{m}}-{{s}}-merged-awq",
        "qlora",
    ),
    "qlora-fp8": (
        "hf",
        f"{SHARED}/models/synthetic-qlora-{{m}}-{{s}}-merged-fp8",
        "qlora",
    ),
}
for _base, _local, _parent in (
    ("fft-gguf", "synthetic-fft-{m}-{s}-gguf", "fft-gguf"),
    ("qlora-gguf", "synthetic-qlora-{m}-{s}-merged-gguf", "qlora-gguf"),
):
    for _q in GQUANTS:
        TARGETS.setdefault(
            f"{_base}-{_q}",
            ("gguf", f"{SHARED}/models/{_local}/model-{_q}.gguf", _parent),
        )
ORDER = [
    "base",
    "fft",
    "qlora",
    "fft-gguf",
    "fft-gptq",
    "fft-awq",
    "fft-fp8",
    "qlora-gguf",
    "qlora-gptq",
    "qlora-awq",
    "qlora-fp8",
] + [f"{b}-{q}" for b in ("fft-gguf", "qlora-gguf") for q in GQUANTS]


def verdict(delta):
    if delta is None:
        return "SKIP"
    if delta < 0.3:
        return "Accept"
    if delta < 1.0:
        return "Conditional"
    return "Discard"


GGUF_REF = {"fft-gguf", "qlora-gguf"}


def score_results(results, sel):
    """Δ 및 판정. 같은 method끼리만 비교한다 (transformers vs llama.cpp 혼합 금지).
    gguf 양자화의 직접 부모는 F16 노드 (변환 체인: HF → F16 → quant) —
    parent가 곧 비교 대상이라 ref_sibling 같은 우회가 필요 없다."""
    for t in sel:
        r = results[t]
        kind = TARGETS[t][0]
        r["method"] = "llama.cpp" if kind == "gguf" else "transformers"
        r["parent"] = TARGETS[t][2]
        if r.get("ppl") is None:
            r.update(delta=None, delta_vs=None, verdict="SKIP")
            continue
        if t == "base":
            r.update(delta=None, delta_vs=None, verdict="ref")
        elif t in ("fft", "qlora"):
            # SFT 자체의 도메인 이동이라 판정 대상 아님 (ref), delta는 참고용.
            b = results.get("base", {})
            r["delta_vs"] = "base"
            r["delta"] = (
                round(r["ppl"] - b["ppl"], 3) if b.get("ppl") is not None else None
            )
            r["verdict"] = "ref"
        elif kind == "gguf":
            if t in GGUF_REF:
                r.update(delta=None, delta_vs=None, verdict="ref")
            else:
                p = TARGETS[t][2]  # F16 노드 (직접 부모이자 비교 대상)
                s = results.get(p, {})
                if s.get("ppl") is not None:
                    d = r["ppl"] - s["ppl"]
                    r.update(delta=round(d, 3), delta_vs=p, verdict=verdict(d))
                else:
                    r.update(delta=None, delta_vs=p, verdict="SKIP")
        else:
            p = TARGETS[t][2]
            pp = results.get(p, {})
            if pp.get("ppl") is not None:
                d = r["ppl"] - pp["ppl"]
                r.update(delta=round(d, 3), delta_vs=p, verdict=verdict(d))
            else:
                r.update(delta=None, delta_vs=p, verdict="SKIP")


def print_table(results, sel):
    print(f"\n{'target':12} {'ppl':>8} {'delta':>8} {'vs':12} {'method':12} {'verdict':>11}")
    for t in sel:
        r = results[t]
        p = f"{r['ppl']:.3f}" if r.get("ppl") is not None else "-"
        d = f"{r['delta']:+.3f}" if r.get("delta") is not None else "-"
        print(
            f"{t:12} {p:>8} {d:>8} {(r.get('delta_vs') or '-'):12} "
            f"{r.get('method', '-'):12} {r.get('verdict', '-')}"
        )


def shell(name, parent, method, note):
    """null 조상 노드 (ppl 없음). llamacpp forest의 base/fft/qlora,
    또는 양쪽 forest의 미측정 중간 노드용."""
    return {"name": name, "parent": parent, "method": method,
            "ppl": None, "delta": None, "delta_vs": None, "verdict": None,
            "note": note, "children": {}}


def build_tree(results, sel):
    """method별 2 forest.
    transformers: 실측 그대로 (미측정 중간은 null shell).
    llamacpp: base/fft/qlora 조상은 항상 null shell (ppl 비교 제외),
    F16/quant는 실측 (미측정 F16은 null shell). 실측 없는 forest는 생략."""
    sel = set(sel)
    meas = {t: dict(results[t], name=t, children={}) for t in sel}

    def parent_of(t):
        return TARGETS.get(t, (None, None, None))[2]

    def tf_node(name):
        # 실측+transformers면 실노드, 아니면 null shell (미측정 중간 자리 보장).
        # memo 먼저 등록 후 부모로 올라가서 순환에도 안전.
        if name in tf_nodes:
            return tf_nodes[name]
        if name in meas and meas[name].get("method") == "transformers":
            n = meas[name]
        else:
            n = shell(name, parent_of(name), "transformers", "미측정")
        tf_nodes[name] = n
        p = n.get("parent") or parent_of(name)
        if p:
            pn = tf_node(p)
            pn["children"][name] = n
        return n

    def ll_node(name, stack=()):
        if name in ("base", "fft", "qlora"):
            n = shell(name, parent_of(name), "transformers",
                      "ppl 비교 제외 (transformers 측정)")
        elif name in meas and meas[name].get("method") == "llama.cpp":
            n = meas[name]
        else:
            n = shell(name, parent_of(name), "llama.cpp", "미측정")
        kids = {}
        for t in sorted(sel):
            r = results[t]
            if r.get("method") == "llama.cpp" and r.get("parent") == name \
                    and t not in stack:
                kids[t] = ll_node(t, stack + (name,))
        n["children"] = kids
        return n

    def has_meas(n):
        return n.get("ppl") is not None or \
            any(has_meas(c) for c in n["children"].values())

    tf_nodes = {}
    tree = {}
    for t in sel:
        if results[t].get("method") == "transformers":
            tf_node(t)
    # parent 링크가 전부 base에서 끝나므로 루트는 base 단일 (미측정이면 shell).
    if "base" in tf_nodes and has_meas(tf_nodes["base"]):
        tree["transformers"] = tf_nodes["base"]
    # llamacpp forest: base/fft/qlora shell 뼈대 + 실측 gguf.
    # F16 미측정이어도 shell로 유지 (밑 quant의 자리 보장).
    ll_base = shell("base", None, "transformers",
                    "ppl 비교 제외 (transformers 측정)")
    for branch, f16 in (("fft", "fft-gguf"), ("qlora", "qlora-gguf")):
        bnode = shell(branch, "base", "transformers",
                      "ppl 비교 제외 (transformers 측정)")
        fnode = ll_node(f16)
        if has_meas(fnode):
            bnode["children"][f16] = fnode
            ll_base["children"][branch] = bnode
    if has_meas(ll_base):
        tree["llamacpp"] = ll_base

    # 형제 키 alpha-numeric 정렬 (삽입 순서가 뒤죽박죽으로 보임 방지).
    def sort_tree(n):
        n["children"] = {k: sort_tree(v) for k, v in sorted(n["children"].items())}
        return n

    return {k: sort_tree(v) for k, v in tree.items()}


def first(v):
    return v[0] if isinstance(v, list) else v


def load_texts(mode: str, n: int):
    texts = []
    with open(f"{SHARED}/datasets/gsm8k-test.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                texts.append(
                    PROMPT_TEMPLATE.format(
                        instruction=first(r["source"]), response=first(r["target"])
                    )
                )
                if len(texts) >= n:
                    break
    return texts


def ppl_of(model_path: str, texts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    nll_sum, tok_count = 0.0, 0
    with torch.no_grad():
        for t in texts:
            ids = (
                tok(
                    t, return_tensors="pt", truncation=True, max_length=1024
                ).input_ids.cuda()
                if torch.cuda.is_available()
                else tok(
                    t, return_tensors="pt", truncation=True, max_length=1024
                ).input_ids
            )
            if torch.cuda.is_available():
                ids = ids.cuda()
                model = (
                    model.cuda()
                    if next(model.parameters()).device.type == "cpu"
                    else model
                )
            labels = ids.clone()
            out = model(input_ids=ids, labels=labels)
            # loss는 평균 NLL → 토큰 수 가중 합산
            n = ids.numel()
            nll_sum += out.loss.item() * n
            tok_count += n
    return math.exp(nll_sum / max(tok_count, 1))


def ppl_of_gguf(model_path: str, texts) -> float:
    """llama-perplexity로 GGUF PPL 측정. 바이너리 미빌드 시 FileNotFoundError.
    텍스트는 load_texts()와 동일물이라 HF 타깃과 입력 정의가 같다.
    단 청킹 방식이 달라 부모(fft/qlora, transformers per-text)와 엔진이 상이 —
    결과에 note를 남기고 Δ 판정은 동일 임계로 적용한다.
    """
    import re
    import subprocess
    import tempfile

    clip = os.environ.get("LLAMACPP_BIN", f"{LLAMACPP}/build/bin/llama-perplexity")
    if not (os.path.isfile(clip) and os.access(clip, os.X_OK)):
        raise FileNotFoundError(
            f"llama-perplexity 없음: {clip} (llama.cpp에서 cmake --build로 빌드)"
        )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n\n".join(texts) + "\n")
        prompt_file = f.name
    try:
        ctx = os.environ.get("LLAMACPP_CTX", "2048")
        cmd = [clip, "-m", model_path, "-f", prompt_file, "-c", ctx]
        ngl = os.environ.get("LLAMACPP_NGL", "")
        if ngl:
            cmd += ["-ngl", ngl]
        print(f"+ {subprocess.list2cmdline(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    finally:
        os.unlink(prompt_file)
    out = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", out)
    if not m:
        raise RuntimeError(
            f"llama-perplexity PPL 파싱 실패 (rc={proc.returncode}): {out[-500:]}"
        )
    return float(m.group(1))


def cached_ppl(outdir: str, t: str, path: str):
    """신선한 per-target json이 있으면 ppl 반환, 없으면 None.
    base(HF 원격)는 파일 존재만으로 스킵. 구 스키마 파일도 ppl 숫자만 있으면
    재사용한다 (verdict 등은 score pass에서 다시 붙는다)."""
    if os.environ.get("FORCE", "0") == "1":
        return None
    cp = f"{outdir}/{t}.json"
    try:
        with open(cp, encoding="utf-8") as f:
            old = json.load(f)
        p = old.get("ppl")
        if not isinstance(p, (int, float)):
            return None
        if t == "base" or (
            os.path.exists(path) and os.path.getmtime(cp) > os.path.getmtime(path)
        ):
            return p
    except (OSError, ValueError):
        pass
    return None


def write_one(outdir: str, t: str, entry: dict):
    tmp = tmp_path(f"{outdir}/{t}.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)
    commit_file(tmp, f"{outdir}/{t}.json")


def main(mode: str, strat: str, n: int, targets: str):
    sel = ORDER if targets == "all" else targets.split(",")
    texts = load_texts(mode, n)
    print(f"[ppl][{mode}][{strat}] texts={len(texts)}")
    outdir = f"{SHARED}/outputs/ppl-{mode}-{strat}"
    os.makedirs(outdir, exist_ok=True)
    results = {}
    for t in sel:
        kind, tmpl, parent = TARGETS[t]
        path = tmpl.format(m=mode, s=strat) if "{m}" in tmpl else tmpl
        cp = cached_ppl(outdir, t, path)
        if cp is not None:
            results[t] = {"ppl": cp, "path": path}
            print(f"{t:12} SKIP (fresh json)")
            continue
        if kind == "gguf":
            if not os.path.exists(path):
                results[t] = {
                    "ppl": None,
                    "delta": None,
                    "verdict": "SKIP",
                    "note": f"산출물 없음: {path}",
                }
                print(f"{t:12} SKIP (no output)")
                continue
            try:
                p = ppl_of_gguf(path, texts)
                results[t] = {
                    "ppl": round(p, 3),
                    "path": path,
                }
                write_one(outdir, t, results[t])  # 즉시 기록 (중단 시 완료분 보존)
                print(f"{t:12} ppl={p:.3f} (llamacpp)")
            except Exception as e:  # noqa: BLE001 — 미빌드/파싱 실패도 표에 남김
                results[t] = {
                    "ppl": None,
                    "delta": None,
                    "verdict": "SKIP",
                    "note": f"{type(e).__name__}: {e}",
                }
                print(f"{t:12} SKIP ({e})")
            continue
        if not os.path.exists(path) and t != "base":
            results[t] = {
                "ppl": None,
                "delta": None,
                "verdict": "SKIP",
                "note": f"산출물 없음: {path}",
            }
            print(f"{t:12} SKIP (no output)")
            continue
        try:
            p = ppl_of(path, texts)
            results[t] = {"ppl": round(p, 3), "path": path}
            write_one(outdir, t, results[t])  # 즉시 기록 (중단 시 완료분 보존)
            print(f"{t:12} ppl={p:.3f}")
        except Exception as e:  # noqa: BLE001 — 전 타깃 표를 끝까지 뽑기 위함
            results[t] = {
                "ppl": None,
                "delta": None,
                "verdict": "SKIP",
                "note": f"{type(e).__name__}: {e}",
            }
            print(f"{t:12} SKIP ({e})")

    score_results(results, sel)
    print_table(results, sel)

    summary = {
        "meta": {
            "mode": mode,
            "strat": strat,
            "n": n,
            "rule": "같은 method끼리만 delta/verdict (transformers vs llama.cpp 혼합 금지)",
            "thresholds": {"accept_lt": 0.3, "conditional_lt": 1.0},
        },
        "tree": build_tree(results, sel),
    }

    outdir = f"{SHARED}/outputs/ppl-{mode}-{strat}"  # 상단에서 이미 생성됨
    for t in sel:
        tmp = tmp_path(f"{outdir}/{t}.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(results[t], f, ensure_ascii=False, indent=2)
        commit_file(tmp, f"{outdir}/{t}.json")
    outp = f"{outdir}/summary.json"
    tmp = tmp_path(outp)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    commit_file(tmp, outp)
    print(f"-> {outdir}/ (summary.json + {len(sel)} targets)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument(
        "--strat",
        default=os.environ.get("STRAT", "single"),
        choices=["single", "ddp", "fsdp"],
    )
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--targets", default="all")
    a = ap.parse_args()
    main(a.mode, a.strat, a.n, a.targets)
