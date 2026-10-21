"""56-report.py — Stage 7d. PPL + score 종합 리포트 (베이스 대비 향상 모델 식별).
입력: $M1B_SHARED/outputs/ppl-<mode>-<strat>/summary.json (48-measure-ppl.sh 산출)
      $M1B_SHARED/outputs/eval-<mode>-<strat>/score.json (54-score.sh 산출)
출력: stdout 4컬럼 표 + $M1B_SHARED/outputs/report-<mode>-<strat>/report.json

컬럼: (1) 모델 [hf|gguf] / (2) ppl + 개선여부 (비교불가=n/a) /
      (3) score(acc) + 랭킹 (gguf끼리 / hf 부모·자식 가문별) / (4) HF 업로드 추천 Yes/No

랭킹: acc 내림차순, 동률은 invalid 오름차순 (competition 순위 1,2,2,4).
  - gguf 가문: gguf 전종(분기 F16 포함) 한 그룹.
  - hf 가문: base{base,fft,qlora} / fft{fft+자식3} / qlora{qlora+자식3}.
    fft·qlora는 부모 가문과 자기 가문 순위를 둘 다 표시.
추천(Yes): base 대비 acc 엄격 우위 AND ppl 게이트 통과
  (verdict Accept/Conditional/ref, ppl 측정됨). base 본인은 No(기준점).
  base acc 부재 시 부모 acc를 기준으로 삼고 사유에 명시, 기준도 없으면 No.

  uv run 56-report.py --mode mini|full [--strat single|ddp|fsdp]
    [--format table|json]
  --format table: stdout에 고정폭 4컬럼 표 (기본값).
  --format json: stdout에 report.json과 동일 내용을 JSON으로만 출력 (파이프용).
  --from-report는 내부용 (fresh-skip 시 캐시 재출력).
"""

import argparse
import importlib.util
import json
import os

from io_common import commit_file, tmp_path

SHARED = os.environ.get("M1B_SHARED", os.environ.get("P5_SHARED", "/rosenas/data/AIML/math-1b-shared"))

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_ppl_spec():
    """48-measure-ppl.py의 ORDER/TARGETS 재사용 (번호 바뀌어도 단일 정의 유지)."""
    path = os.path.join(_HERE, "48-measure-ppl.py")
    spec = importlib.util.spec_from_file_location("m1b_ppl_spec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ORDER, mod.TARGETS


ORDER, TARGETS = _load_ppl_spec()
KIND = {t: TARGETS[t][0] for t in TARGETS}
PARENT = {t: TARGETS[t][2] for t in TARGETS}

FAM_BASE = ["base", "fft", "qlora"]
FAM_FFT = ["fft", "fft-awq", "fft-fp8", "fft-gptq"]
FAM_QLORA = ["qlora", "qlora-awq", "qlora-fp8", "qlora-gptq"]
FAM_GGUF = [t for t in ORDER if KIND.get(t) == "gguf"]


def families_of(t: str) -> list[str]:
    if KIND.get(t) == "gguf":
        return ["gguf"]
    if t == "base":
        return ["base"]
    if t == "fft":
        return ["base", "fft"]
    if t == "qlora":
        return ["base", "qlora"]
    if t in FAM_FFT:
        return ["fft"]
    if t in FAM_QLORA:
        return ["qlora"]
    return []


FAM_MEMBERS = {"base": FAM_BASE, "fft": FAM_FFT, "qlora": FAM_QLORA, "gguf": FAM_GGUF}

PASS_VERDICTS = {"Accept", "Conditional", "ref"}


def flatten(tree: dict) -> dict:
    """summary.json tree → {target: node}. ppl·verdict 둘 다 없는 shell은 제외."""
    out = {}

    def walk(n):
        if n.get("ppl") is not None or n.get("verdict") is not None:
            out[n["name"]] = n
        for c in (n.get("children") or {}).values():
            walk(c)

    for root in tree.values():
        walk(root)
    return out


def load_inputs(mode: str, strat: str):
    sp = f"{SHARED}/outputs/ppl-{mode}-{strat}/summary.json"
    sc = f"{SHARED}/outputs/eval-{mode}-{strat}/score.json"
    if not os.path.exists(sp):
        raise SystemExit(f"missing ppl summary: {sp} (run 48-measure-ppl.sh first)")
    if not os.path.exists(sc):
        raise SystemExit(f"missing eval score: {sc} (run 54-score.sh first)")
    with open(sp, encoding="utf-8") as f:
        summary = json.load(f)
    with open(sc, encoding="utf-8") as f:
        scores = json.load(f)
    nodes = flatten(summary.get("tree", {}))
    # tree에 없는 타깃은 per-target json으로 보충 (구 스키마는 ppl만 있을 수 있음).
    for t in ORDER:
        if t not in nodes:
            try:
                with open(
                    f"{SHARED}/outputs/ppl-{mode}-{strat}/{t}.json", encoding="utf-8"
                ) as f:
                    r = json.load(f)
                if isinstance(r.get("ppl"), (int, float)):
                    nodes[t] = {"name": t, "ppl": r["ppl"], "method": None,
                                "parent": PARENT.get(t), "delta": None,
                                "delta_vs": None, "verdict": None}
            except OSError:
                pass
    return nodes, scores, sp, sc


def col_ppl(n: dict) -> tuple[str, bool | None]:
    """(표시문자열, improved). 비교불가·미측정은 n/a, improved=None."""
    ppl = n.get("ppl")
    if not isinstance(ppl, (int, float)):
        return "n/a", None
    d = n.get("delta")
    if not isinstance(d, (int, float)):
        tag = "기준점" if n.get("verdict") == "ref" else "비교불가"
        return f"{ppl:.3f} (n/a·{tag})", None
    vs = n.get("delta_vs") or "?"
    arrow = "개선" if d < 0 else ("유지" if d == 0 else "악화")
    v = n.get("verdict")
    tail = f"·{v}" if v not in (None, "ref") else ""
    return f"{ppl:.3f} (Δ{d:+.3f} vs {vs}·{arrow}{tail})", d < 0


def rank_family(members: list[str], scores: dict, nodes: dict) -> dict:
    """{target: (rank, size)}. acc 있는 멤버만 순위 pool (competition 순위)."""
    pool = [t for t in members if isinstance(scores.get(t, {}).get("acc"), (int, float))]
    if not pool:
        return {}
    n = lambda t: nodes.get(t, {}).get("ppl")  # noqa: E731
    pool.sort(key=lambda t: (-scores[t]["acc"], scores[t].get("invalid", 9),
                             n(t) if isinstance(n(t), (int, float)) else 1e18, t))
    ranks, prev, rank = {}, None, 0
    for i, t in enumerate(pool):
        key = (scores[t]["acc"], scores[t].get("invalid", 9))
        if key != prev:
            rank = i + 1
            prev = key
        ranks[t] = (rank, len(pool))
    return ranks


def decide(t: str, n: dict, scores: dict) -> tuple[str, str]:
    """(Yes/No, 사유). base 본인은 No. gguf 변종도 HF 업로드 대상이라 동일 규칙."""
    if t == "base":
        return "No", "기준점(업스트림)"
    ppl = n.get("ppl")
    v = n.get("verdict")
    s = scores.get(t, {})
    acc = s.get("acc")
    if not isinstance(ppl, (int, float)) or v not in PASS_VERDICTS:
        return "No", f"ppl 게이트 탈락({v or '미판정'})"
    if not isinstance(acc, (int, float)):
        return "No", "score 없음"
    basis, basis_name = None, ""
    b = scores.get("base", {}).get("acc")
    if isinstance(b, (int, float)):
        basis, basis_name = b, "base"
    else:
        p = PARENT.get(t)
        pb = scores.get(p, {}).get("acc") if p else None
        if isinstance(pb, (int, float)):
            basis, basis_name = pb, f"부모 {p}"
    if basis is None:
        return "No", "비교 기준 없음"
    if acc > basis:
        return "Yes", f"acc {acc:.3f} > {basis_name} {basis:.3f}·ppl {v}"
    return "No", f"acc {acc:.3f} ≤ {basis_name} {basis:.3f}"


def tags_of(t: str, rank: dict) -> list[str]:
    return [f"{f} {rk[0]}/{rk[1]}" for f in families_of(t) if (rk := rank.get(f))]


def score_text(acc, n_, inv, tags: list[str]) -> str:
    if not isinstance(acc, (int, float)):
        return "n/a"
    s = (f"{acc:.3f} (n={n_ if n_ is not None else '?'}, "
         f"inv={inv if inv is not None else '?'})")
    return s + (f" [{', '.join(tags)}]" if tags else "")


SLIM_KEYS = ("target", "kind", "ppl", "delta", "delta_vs", "verdict",
             "improved", "acc", "n", "invalid", "rank", "upload", "reason")


def emit(rows: list[dict], rec: list[str], fmt: str):
    if fmt == "json":
        print(json.dumps(
            {"recommend": rec,
             "rows": [{k: r.get(k) for k in SLIM_KEYS} for r in rows]},
            ensure_ascii=False, indent=2))
        return
    w1 = max(len(r["c1"]) for r in rows)
    w2 = max(len(r["c2"]) for r in rows)
    w3 = max(len(r["c3"]) for r in rows)
    print(f"\n{'모델':{w1}}  {'ppl(개선여부)':{w2}}  "
          f"{'score(랭킹)':{w3}}  업로드추천")
    for r in rows:
        print(f"{r['c1']:{w1}}  {r['c2']:{w2}}  {r['c3']:{w3}}  {r['c4']}")
    print(f"\n추천 {len(rec)}종: {', '.join(rec) if rec else '-'}")


def render_cached(path: str, fmt: str):
    """report.json 캐시에서 재계산 없이 다시 출력 (fresh-skip용)."""
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    rows = []
    for r in report["rows"]:
        ppl_txt, _ = col_ppl({k: r.get(k)
                              for k in ("ppl", "delta", "delta_vs", "verdict")})
        rows.append({**r,
                     "c1": f"{r['target']} [{r.get('kind', '?')}]",
                     "c2": ppl_txt,
                     "c3": score_text(r.get("acc"), r.get("n"), r.get("invalid"),
                                      tags_of(r["target"], r.get("rank") or {})),
                     "c4": r.get("upload")})
    rec = report.get("recommend",
                     [r["target"] for r in rows if r.get("upload") == "Yes"])
    emit(rows, rec, fmt)
    if fmt == "json":
        import sys
        print(f"-> {path}", file=sys.stderr)
    else:
        print(f"-> {path}")


def main(mode: str, strat: str, fmt: str):
    nodes, scores, sp, sc = load_inputs(mode, strat)
    fam_ranks = {f: rank_family(m, scores, nodes) for f, m in FAM_MEMBERS.items()}
    universe = [t for t in ORDER if t in nodes or t in scores]
    universe += sorted((set(nodes) | set(scores)) - set(ORDER))

    rows = []
    for t in universe:
        n = nodes.get(t, {"name": t})
        ppl_txt, improved = col_ppl(n)
        s = scores.get(t, {})
        acc = s.get("acc")
        tags = []
        for f in families_of(t):
            r = fam_ranks.get(f, {}).get(t)
            if r:
                tags.append(f"{f} {r[0]}/{r[1]}")
        score_txt = score_text(acc, s.get("n"), s.get("invalid"), tags)
        up, reason = decide(t, n, scores)
        rows.append({"target": t, "kind": KIND.get(t, "?"),
                     "ppl": n.get("ppl"), "delta": n.get("delta"),
                     "delta_vs": n.get("delta_vs"), "verdict": n.get("verdict"),
                     "improved": improved, "acc": acc,
                     "n": s.get("n"), "invalid": s.get("invalid"),
                     "rank": {f: fam_ranks[f][t]
                              for f in families_of(t) if t in fam_ranks.get(f, {})},
                     "upload": up, "reason": reason,
                     "c1": f"{t} [{KIND.get(t, '?')}]",
                     "c2": ppl_txt, "c3": score_txt, "c4": up})

    rec = [r["target"] for r in rows if r["upload"] == "Yes"]
    slim = [{k: r[k] for k in SLIM_KEYS} for r in rows]
    emit(rows, rec, fmt)

    outdir = f"{SHARED}/outputs/report-{mode}-{strat}"
    os.makedirs(outdir, exist_ok=True)
    report = {"meta": {"mode": mode, "strat": strat,
                       "ppl_summary": sp, "eval_score": sc,
                       "rule": ("acc가 base(부재 시 부모)보다 엄격히 크고 "
                                "ppl verdict가 Accept/Conditional/ref인 타깃 추천. "
                                "랭킹은 acc 내림차순·invalid 오름차순 competition 순위. "
                                "gguf는 gguf끼리, hf는 부모·자식 가문별.")},
              "rows": slim,
              "recommend": rec}
    outp = f"{outdir}/report.json"
    tmp = tmp_path(outp)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    commit_file(tmp, outp)
    if fmt == "json":
        import sys
        print(f"-> {outp}", file=sys.stderr)
    else:
        print(f"-> {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument(
        "--strat",
        default=os.environ.get("STRAT", "single"),
        choices=["single", "ddp", "fsdp"],
    )
    ap.add_argument("--format", default="table", choices=["table", "json"])
    ap.add_argument("--from-report", default=None,
                    help="report.json 캐시에서 재계산 없이 다시 출력 (fresh-skip용)")
    a = ap.parse_args()
    if a.from_report:
        render_cached(a.from_report, a.format)
    else:
        main(a.mode, a.strat, a.format)
