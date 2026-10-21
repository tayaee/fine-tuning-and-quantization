#!/usr/bin/env python3
"""check_report_to.py — train job에 report_to를 주기 전 미리 체크.

실제 검증 로직은 report_check.py(train-entry.py 기동 게이트와 공유)에 있다.
Trainer와 동일한 해석+인스턴스화 경로로 검증하므로, 여기서 OK면 잡도 통과한다.

사용 (math-1b-sft-and-quantization 디렉토리에서):
    uv run --no-sync check_report_to.py --report_to wandb,trackio
    uv run --no-sync check_report_to.py --report_to all --out /path/to/model-out

종료 코드: 0=학습 시 로그 남음(요청 전체 활성), 1=로그 안 남거나 문제 있음.
"""

import argparse
import importlib.metadata
import sys

from report_check import STATIC_KEYS, summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report_to", required=True, help='예: "all", "none", "trackio,wandb"')
    p.add_argument("--out", default=None, help="모델 출력 디렉토리 (경로 표시에 사용)")
    a = p.parse_args()

    try:
        tf_ver = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        tf_ver = "(미설치)"
    res = summarize(a.report_to, out=a.out)
    print(f"input: {a.report_to!r}  ->  Trainer report_to: {res['effective']!r} (transformers {tf_ver})")

    if res["effective"] == "none":
        print("-> 로깅 없음. Trainer 콜백 추가 안 됨.")
        return 0

    if a.report_to.strip() == "all":
        got = sorted(x["name"] for x in res["rows"] if x["name"] != "all")
        print(f"-> all 해석: {got if got else '(설치된 연동 없음)'}")
        missing = sorted(set(STATIC_KEYS) - set(got))
        if missing:
            from report_check import KNOWN_META
            print("not installed (all에 포함 안 됨):")
            for n in missing:
                print(f"  [----] {n}: pip install {KNOWN_META[n]['pip']}")

    for x in res["rows"]:
        if x["status"] == "OK":
            print(f"  [OK] {x['name']}: {x['detail']} | 출력 -> {x['output']}{x['env']}")
            if x["note"]:
                print(f"       {x['note']}")
        elif x["status"] == "MISSING":
            print(f"  [MISSING] {x['name']}: {x['detail']}")
            print(f"           처방: pip install {x['pip']} 후 재확인, 또는 report_to에서 제외")
        elif x["status"] in ("BROKEN", "ENV-BROKEN"):
            print(f"  [BROKEN] {x['name']}: {x['detail']}")
            print(f"           처방: pip install -U {x['pip']} 후 재확인, 또는 report_to에서 제외")
        elif x["status"] == "UNKNOWN":
            print(f"  [UNKNOWN] {x['name']}: {x['detail']}")
        else:
            print(f"  [{x['status']}] {x['name']}: {x['detail']}")

    if res["problems"]:
        print(f"RESULT: FAIL — {res['problems']}건 문제 (위 MISSING/BROKEN/UNKNOWN 해결 후 재실행).")
        return 1
    print("RESULT: OK — 요청 전체 활성. 학습 시 위 위치에 보고서 생성.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
