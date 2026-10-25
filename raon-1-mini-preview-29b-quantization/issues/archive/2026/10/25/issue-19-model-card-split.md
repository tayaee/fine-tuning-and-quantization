# issue-19: 모델 카드 = base+대상 1개 / 전체 버전 비교는 요약 리포트로 분리

agent-tier: local-ok

## 1. 배경 및 문제점

`s8-43-prep-upload`가 만드는 모델 카드는 **그 모드의 모든 양자화 버전을 한 표에** 실었다.

```
| **bf16 (base)** | 6.055 | — | ref |
| awq-w3          | —     | — | -   |     ← 실측 없음(noise)
| awq-w5          | —     | — | -   |
| **nvfp4-awq**   | 6.083 | 0.03 | Accept |
| nvfp4-gptq      | —     | — | -   |
```

문제는 두 가지다.

1. **업로드 산출물로서 부적절** — Hub의 모델 repo 1개 = 모델 1개인데, 카드에는 다른
   버전의 행(대부분 "—")까지 올라가 "이 repo에 뭐가 들어있는지"가 흐려진다.
2. **비교의 의미를 잃음** — "이 모델이 base보다 나은가/나쁜가"라는 질문에 다른 버전의
   결측 행이 끼어든다. 비교는 base ↔ 해당 모델 두 개여야 한다.

전체 버전 비교는 필요하되 **업로드 카드와 분리**되어야 한다.

## 2. 요구사항

1. 모델 디렉터리의 `README.md`(업로드 대상)는 **base + 해당 양자화 버전 1개**만 비교한다.
2. 같은 모드의 **전체 양자화 버전 비교**는 `models/README-<mode>.md`로 분리한다.
   - 존재하는(모델 디렉터리 또는 실측 지표가 있는) 버전만 표에 실린다.
   - 각 버전의 사양(알고리즘·정밀도·비트·group size·포맷·체크포인트 크기) + 모델 카드 링크.
3. 업로드 전 감사(`s8-46-verify-upload`)가 다음을 검사한다.
   - 모델 카드에 **다른 타깃의 행이 있으면 실패**
   - `models/README-<mode>.md` 존재 + 내용 크기
4. 회귀 가드 + 단위 테스트.

## 3. 승인 기준

1. `models/raon-<target>-<mode>/README.md`에 base 행과 대상 행만 있다.
2. `models/README-<mode>.md`에 그 모드의 전체 양자화 버전이 표로 실린다.
3. `s8-46-verify-upload`가 위 두 불변식을 검사한다(카드에 다른 버전 행 → 실패).
4. `verify-issue-1~19` 통과, 단위 테스트 전부 통과, ruff/pyright 신규 오류 0건.

## 4. 검증

- 실제 공유 디렉터리 재생성: smoke/full 카드 + `models/README-{smoke,full}.md`
- 업로드 전 감사(`s8-46`) smoke 통과(8대 검사 전부 True)
- 단위 테스트 3건 신규 + 회귀 전수

## 5. 의존성 / 관계

- **선행**: issue-17(s7 평가 확장), issue-18(MODE 환경변수)
- **후행**: 없음

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-25
- **변경 파일 목록**:
  - `s8-43-prep-upload.py` (수정):
    - `_check_tables()` 공통화 — 비교 대상 행(`rows`)·강조 대상(`bold`)을 인자로 받음
    - `render_readme()` — `rows = ["base", target]` (base + 이 모델 1개만)
    - `render_summary()`(신규) — 모드 전체 버전 비교 리포트(양자화 버전 사양 표 + 검사 표 4종 + 서빙 + 라이선스)
    - `summary_path()`(신규) — `models/README-<mode>.md`
    - `_existing_targets()`(신규) — 존재하는(모델 디렉터리 또는 실측 지표) 버전만
    - `prepare_target()` — 요약 리포트 생성 + 매니페스트에 경로/크기 기록
  - `s8-46-verify-upload.py` (수정): "다른 양자화 버전 행 금지" 검사 + 요약 리포트 존재 검사
  - `tests/test_prep_and_upload.py` (수정): 카드 1개 비교·요약 생성·존재 타깃만 3건 신규
  - `regression-tests/verify-issue-19.sh` (신규): 카드/요약 분리 4항목
  - `README.md` · `PROCEDURE.md` · `cheatsheet.md` · `docs/SDLC.md` (수정): 산출물 설명 정리
- **계획 대비 편차**:
  - GSM8K 표의 base 행 판정을 `PASS` → `ref`로 통일(KMMLU·형식 표와 동일), KMMLU
    "Correct / Total"을 GSM8K와 같은 `정답/전체` 형식으로 통일했다(표 일관성).
  - 요약 리포트는 HF frontmatter가 없는 **비-모델카드** 문서다(어떤 repo에도 업로드되지 않는다).
- **검증 결과**:
  - 실제 재생성: `models/raon-nvfp4-awq-smoke/README.md`(base+nvfp4-awq만),
    `models/raon-nvfp4-awq-full/README.md`, `models/README-smoke.md`, `models/README-full.md` 생성
  - `s8-46-verify-upload.sh smoke nvfp4-awq` → ok=true (8대 검사 전부 통과)
  - `pytest tests/` 134 passed · `ruff check .` All checks passed · `bash -n` 전 스크립트 통과
  - `verify-issue-1~13`·`17`·`18`·`19` 전부 통과
