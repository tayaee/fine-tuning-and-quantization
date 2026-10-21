# REPRODUCIBILITY — 1년 후 동일 재현용 핀 목록

기준일: 2026-10-19 (spark1 실측). 복원 기준: `uv.lock` + `.python-version` +
`requirements-frozen.txt` + 아래 핀.

## 1. 툴체인

| 항목 | 핀 | 확인 |
|---|---|---|
| python | `3.12.13` (`.python-version`) | `uv python pin` 기준, `.venv/bin/python --version` |
| uv | `>=0.11.8` (`UV_MIN`, 실측 `0.11.8 aarch64`) | `10-setup.sh`가 미만이면 REFUSE |
| torch backend | `cu130` (`UV_TORCH_BACKEND`, infra env) | `pyproject.toml: torch==2.13.0` + lock의 cu130 휠 |
| OS/GPU | DGX Spark, driver `580.178.04`, `aarch64` | `nvidia-smi -L`, `pyproject [tool.uv] required-environments` |

## 2. Python 패키지

- 직접 의존: `pyproject.toml` 전 항목 `==` 고정.
- 전이 의존: `uv.lock` (hash 포함, `required-markers: aarch64+linux`).
- 설치는 반드시 `uv sync --frozen` (10-setup.sh). `--frozen`은 lock/pyproject
  불일치 시 실패시켜 드리프트를 즉시 감지한다.
- lock 밖 1종: `llmcompressor==0.14.0` (`--no-deps`, resolver 충돌 회피).
  `compressed-tensors`는 lock 기준 `==0.17.0` 사용 (별도 0.19.0 오버라이드 삭제됨 —
  `uv sync`가 되돌려 실측 0.17.0으로 전 단계 동작 확인).
- 감사 스냅샷: `requirements-frozen.txt` (232개, `uv pip freeze` 실측).
  설치 기준이 아니라 감사/비교용. 복원 후 `diff <(uv pip freeze|sort) requirements-frozen.txt`
  로 드리프트 확인.

## 3. HF 모델 리비전

| 변수 | 값 (commit SHA) |
|---|---|
| `BASE_REVISION` (unsloth/Llama-3.2-1B) | `9535bd9b1d1dea6acafbdc4813b728796aeb28da` |
| `TEACHER_REVISION` (meta-llama/Llama-3.1-8B-Instruct) | `0e9e39f249a16976918f6564b8830bc894c89659` |

`config/common.env` 기본값. `--revision` 미지정 시 env 기본 사용, 빈 값이면
floating(비재현). 전달 경로:

- `26/28-*-train.sh` → `train-entry.py --revision $BASE_REVISION`
- `30-merge-lora.sh` → `30-merge-lora.py --revision $BASE_REVISION`
- `22-*.sh` → `22-teacher-*.py --revision $TEACHER_REVISION` (tokenizer + vLLM `LLM(revision=)`)
- 로컬 경로 로드(quant/eval/merge adapter·GGUF)는 Hub를 안 타서 핀 불필요.

## 4. 외부 git

| repo | 핀 HEAD | 용도 |
|---|---|---|
| `~/git/maskedthought` | `9a467afe60b8cd3d89b7b7728557d54539998f48` | 참조 전용 (데이터는 `data/` 벤더링 우선) |
| `~/git/llama.cpp` | `98d5e8ba8a2642710c9871d05ac1033a3328b884` | GGUF 변환/양자화 바이너리 |

`10-setup.sh` + `36/44-*-gguf.sh`가 HEAD 불일치 시 WARN (참조/바이너리라 REFUSE 아님 —
GGUF 바이트 동일성은 보장 안 됨). 1년 후 바이트 재현이 필요하면 해당 SHA로
`git checkout` 후 `build/bin/llama-quantize` 재빌드.

## 5. 1년 후 복원 절차

```bash
uv --version            # >=0.11.8
uv sync --frozen
uv pip install --no-deps "llmcompressor==0.14.0"
./10-setup.sh           # 핀 검증 + 공유 datasets/models skeleton
diff <(uv pip freeze | sort) <(grep -v '^#' requirements-frozen.txt)
./run-mini-single.sh    # 또는 run-full-single.sh
```

## 6. 의도적 비고정 (재현에 영향 없음)

- `GPU_UTIL` (60-serve 기본 `0.5`): 환경 VRAM 여유에 따른 실행 튜닝. 가중치 불변.
- `PORT/SOURCE/QTYPE/STRAT/PROFILE`: 실행 방법 선택. 산출물 경로 규칙은 `models-tree.txt`.
- `REPORT_TO/wandb/trackio`: 로깅 sink. `train-time-log.md`에만 영향.
