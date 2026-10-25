# issue-16: 모델 카드(README) 전량 영문화

agent-tier: local-ok

## 1. 배경 및 문제점

`s8-43-prep-upload.sh`가 생성하는 모델 카드 README.md는 **한국어 섹션/표/서술**로
작성돼 있었다. 카드가 HuggingFace Hub에 그대로 올라가는 **공개 산출물**이므로
영문이어야 한다.

영문화해야 했던 부분:
- 섹션 제목: `## 양자화 사양`, `## 수용 기준 (Stage 1 계약) 대비 판정`,
  `## 검사 1 — PPL`, `## 검사 2 — GSM8K`, `## 사용법 — transformers`,
  `## 사용법 — vLLM`, `## 라이선스 및 출처`
- 사양 표 헤더/라벨, 수용 기준 대조표 라벨, PPL/GSM8K 검사 표 헤더
- SPECS 값(`GPTQ (AWQ smoothing 후)` → `GPTQ (after AWQ smoothing)` 등),
  캘리브레이션/제외 레이어 설명
- 본문 서술, 사용 예시 프롬프트(`2+2는?` → `What is 2+2?`), 라이선스 문구
- `evalset manifest not found` 등 카드에 노출되는 폴백 문자열

## 2. 요구사항

1. 생성되는 README.md에 **한글이 한 줄도 없어야 한다**.
2. 카드에 노출되는 모든 문자열(사양 값·표 헤더·폴백 메시지) → 영어.
3. 업로드 전 감사(`s8-46-verify-upload.py`)의 README 필수 섹션 목록을 영어 제목으로 갱신.
4. 회귀 가드: 단위 테스트에서 "README에 한글이 없음"을 단언.
5. 스크립트 내부의 개발자용 주석/도그문은 한국어 유지(산출물 아님).

## 3. 승인 기준

1. 생성된 카드(실제 smoke/full nvfp4-awq)에서 한글 줄 0건.
2. `s8-46-verify-upload.sh` smoke/full `ok=true`.
3. `rq_gate_release` smoke/full 통과.
4. verify-issue-1~13, 단위 테스트 7파일 통과, ruff/pyright 0건.

## 4. 검증

- 실제 카드 재생성 후 한글 줄 카운트(0)
- 46-verify 재감사 + 릴리스 게이트
- 회귀/단위/정적분석 전수

## 5. 의존성 / 관계

- **선행**: issue-11(카드 2종 검사), issue-14(파일 개명)
- **후행**: 없음

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-24
- **변경 파일 목록**:
  - `s8-43-prep-upload.py` (수정): 카드 전체 영어화(사양·표·본문·폴백) + 매니페스트 오류 메시지 영어화
  - `s8-46-verify-upload.py` (수정): README 필수 섹션을 영어 제목으로 갱신
  - `tests/test_prep_and_upload.py` (수정): 영어 섹션 단언 + "한글 0줄" 회귀 가드
  - `regression-tests/verify-issue-11.sh` (수정): 카드 섹션 토큰 영어화
