# issue-20: 카드 표의 Model 컬럼 = 정식 HF 모델명

agent-tier: local-ok

## 1. 배경 및 문제점

모델 카드의 검사 표는 Model 컬럼에 **내부 식별자**를 쓰고 있었다.

```
| **bf16 (base)** | 6.055 | — | ref |
| **nvfp4-awq**   | 6.083 | 0.03 | Accept |
```

이 카드는 HuggingFace Hub에 그대로 올라가는 **공개 산출물**이다. 읽는 사람은
`bf16`/`nvfp4-awq`가 무엇인지 모른다 — Hub에서 그 모델을 찾으려면 정식 repo id가
필요하다. 즉 카드가 "어떤 모델을 받아야 하는지"를 알려주지 않는다.

동일한 이유로 카드 제목·사용 예제도 내부 이름(`nvfp4-awq`)을 쓰고 있었다.

## 2. 요구사항

1. 모든 표의 Model 컬럼은 **정식 HF 모델명(Hub repo id)**을 쓴다.
   - base → `KRAFTON/Raon-1-mini-preview` (원본 repo)
   - 양자화본 → `tayaee/Raon-1-mini-preview-<target>` (full) / `...-<target>-smoke` (smoke)
   - repo id 규칙은 `s8-49-upload-hf.py`와 **동일해야 한다**(어긋나면 카드가 엉뚱한 모델을 가리킨다).
2. 카드 제목과 사용 예제(transformers `model_id`, `vllm serve`)도 같은 repo id를 쓴다.
3. 요약 리포트(`models/README-<mode>.md`)의 Model 컬럼도 동일 표기.
4. 업로드 전 감사(`s8-46`)가 repo id 기준으로 강조 행·타깃 혼입을 검사한다.
5. 카드는 영어 유지(issue-16) — 한국어 주석 금지.

## 3. 승인 기준

1. 카드에 `| **KRAFTON/Raon-1-mini-preview** (base) |` 행이 있다.
2. 카드에 `| **tayaee/Raon-1-mini-preview-<target>[-smoke]** |` 행이 있다.
3. 카드 제목이 repo id다(`# tayaee/Raon-1-mini-preview-nvfp4-awq-smoke`).
4. `s8-46-verify-upload`가 repo id 기준으로 검사하고 통과한다.
5. verify-issue-1~20 통과, 단위 테스트 전부 통과, ruff/pyright 신규 오류 0건.

## 4. 검증

- 실제 공유 디렉터리 재생성(smoke/full) → 카드·요약의 Model 컬럼 확인
- 업로드 전 감사 smoke/full 통과
- 단위 테스트 + 회귀 전수

## 5. 의존성 / 관계

- **선행**: issue-19(카드/요약 분리)
- **후행**: 없음

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-25
- **변경 파일 목록**:
  - `s8-43-prep-upload.py` (수정):
    - `hf_repo_id()` / `model_label()` 신설 — `s8-49-upload-hf.py`의 `repo_name`과 같은 규칙
      (full → `<HF_ID>/<BASE>-<target>`, smoke → `...-<target>-smoke`)
    - `_check_tables()` — Model 컬럼을 정식 HF 모델명으로
    - 카드 제목 → `# <repo id>`, transformers `model_id`·`vllm serve` → repo id
    - 요약 리포트의 "Build" 컬럼 → "Model"(repo id)
  - `s8-46-verify-upload.py` (수정): `hf_repo_id()` 추가 · 강조 행 검사·타깃 혼입 검사를
    repo id 기준으로 · `_check_readme()`에 `mode` 전달
  - `tests/test_prep_and_upload.py` (수정): repo id 표기 단언으로 갱신
  - `tests/test_verify_upload.py` (수정): 결측치 주입 대상 행을 repo id 표기로 갱신
- **계획 대비 편차**:
  - base 행은 `**KRAFTON/Raon-1-mini-preview** (base)` 로 "원본 repo"임을 함께 표시했다
    (HF 모델명만으로는 base인지 양자화본인지 식별되지 않으므로).
  - repo id 규칙(HF_ID·BASE_SLUG)은 `s8-49-upload-hf.py`와 **중복 정의**한다
    (스크립트가 서로 독립 실행되므로 공용 모듈로 빼는 편이 좋지만, 업로드 스크립트의
    규칙이 단일 진실이라 여기서는 동일 규칙을 명시적으로 유지하고 주석으로 연결한다).
- **검증 결과**:
  - 실제 재생성: `models/raon-nvfp4-awq-{smoke,full}/README.md` 제목·표 모두 repo id 표기
  - `models/README-{smoke,full}.md` Model 컬럼 repo id ✓
  - `s8-46-verify-upload.sh` smoke/full → exit 0 (ok=true)
  - `pytest tests/` 134 passed · `ruff check .` All checks passed · `bash -n` 전 스크립트 통과
  - `verify-issue-1~13`·`17`·`18`·`19` 전부 통과
  - pyright: 변경분 신규 오류 0건 (s5-* 기존 7건 잔존)
