"""quant_common.py — 양자화 스크립트(40/41/43/50/51/53) 공용 헬퍼.
캘리브레이션 텍스트 로딩(PROMPT_TEMPLATE + gsm8k-calibration 파일) 단일 정의.
각 스크립트는 `uv run $M1B_ROOT/<nn>-quant-*.py` 로 실행되며 스크립트 디렉토리가
sys.path에 들어오므로 plain `import quant_common`으로 로드된다.
"""

import json
import os

SHARED = os.environ.get("M1B_SHARED", os.environ.get("P5_SHARED", "/rosenas/data/AIML/math-1b-shared"))
DEFAULT_CALIB_FILE = f"{SHARED}/datasets/gsm8k-calibration-256.jsonl"

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context.\n"
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\nGive a response as the assistant with the input conversation history\n\n"
    "### Response:\n{response}"
)


def first(v):
    return v[0] if isinstance(v, list) else v


def load_calib_texts(
    calib: int, calib_file: str | None = None, template: str = PROMPT_TEMPLATE
) -> list[str]:
    """캘리브레이션 jsonl 앞 `calib`줄을 템플릿 적용 텍스트로 반환.
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
            if line:
                r = json.loads(line)
                texts.append(
                    template.format(
                        instruction=first(r["source"]), response=first(r["target"])
                    )
                )
            if len(texts) >= calib:
                break
    if len(texts) < calib:
        raise SystemExit(
            f"calibration file has {len(texts)} rows < requested calib={calib}"
        )
    print(f"calib_src={calib_file} n={len(texts)}")
    return texts
