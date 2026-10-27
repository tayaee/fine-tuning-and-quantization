"""s9-16-inference-test.py — 서빙 중인 모델에 한국어 스모크 3문항 요청 (completions API).
서버는 별도 터미널에서 s9-13-vllm-serve-one.sh로 실행. chat template이 아닌 completions 기준
스모크라 thinking/한국어 출력 여부만 확인한다.

  uv run s9-16-inference-test.py [--n 3] [--model tayaee/Raon-1-mini-preview-nvfp4-gptq]
"""

import argparse
import os

PROMPTS = [
    "다음을 한국어로 한 문장으로 요약하세요: 세종대왕은 훈민정음을 창제하여 백성이 쉽게 글을 읽고 쓸 수 있게 하였다.",
    "사과 3개가 있고 배는 사과보다 2개 많습니다. 배는 몇 개입니까? 단계별로 생각하세요.",
    "1부터 10까지 합을 구하는 파이썬 함수를 작성하세요.",
]


def main(n: int, model: str | None, base_url: str, port: int):
    from openai import OpenAI

    url = base_url or f"http://localhost:{port}/v1"
    client = OpenAI(base_url=url, api_key="dummy")
    model_id = model or client.models.list().data[0].id
    for i, p in enumerate(PROMPTS[:n]):
        r = client.completions.create(model=model_id, prompt=p, temperature=0.6,
                                      top_p=0.95, max_tokens=512)
        text = r.choices[0].text
        print(f"===== Q{i+1}: {p}")
        print(text)
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", ""))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    a = ap.parse_args()
    main(a.n, a.model, a.base_url, a.port)
