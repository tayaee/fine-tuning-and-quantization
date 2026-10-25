# issue-13: 스크립트 시작 배너 — 1줄 영어 설명을 모든 진입 스크립트에

agent-tier: local-ok

## 1. 배경 및 문제점

`07-acceptance.sh`이 무엇을 하는지 스크립트 이름만으로는 알 수 없었다(09번이 무엇인지,
36번이 무엇인지도 마찬가지). 사용자가 "이거 뭐 하는 거지?"를 매번 물어야 했다.

기존 관례는 헤더 **주석** 설명 + `rq_log` 단계 메시지 + 마지막 `---- result ----`였고,
**실행 시작 시 그 스크립트가 무엇을 하는지 1줄로 말해주는 출력**은 없었다.

## 2. 요구사항

1. `config/common.env`에 공용 헬퍼 `rq_banner "<script-id>" "<description>"` 추가.
   출력 형식: `[<script-id>] <description>`
2. 모든 실행 진입 스크립트가 **시작 시 1줄 영어 설명**을 출력한다.
   - `07/10/13/16/19/22/25/28/31/34/36/37/40/43/46/49/52/55/58-*.sh` (19종)
   - 빌드 wrapper 8종(`10-16,19,X-40-build-*.sh`) — 각자 자기 target 배너
   - `run-{smoke,full}.sh`
   - 파이썬 진입점 `61-release-notes.py`, `64-verify-serving.py`
3. 설명은 **영어** 1줄(Stage 번호 포함): `[07-acceptance] Stage 1/9 - validate the ...`
4. 공용 엔진 `build-ppl-eval.inc`에는 배너를 두지 않는다(각 wrapper가 출력 → 중복 방지).

## 3. 승인 기준

1. 번호가 붙은 실행 스크립트 전부가 `rq_banner` 호출을 포함한다.
2. 배너 설명에 한글이 섞이지 않고 영어 문장으로 읽힌다.
3. `rq_banner` 헬퍼가 `config/common.env`에 있다.
4. `./07-acceptance.sh smoke` 실행 시 배너가 출력된다.
5. 빌드 wrapper 8종이 각자 배너를 출력한다.

## 4. 검증

- `regression-tests/verify-issue-13.sh` — 대상 스크립트 전수 검사 + 실제 출력 확인

## 5. 의존성 / 관계

- **선행**: 없음 (문서/ UX 개선)
- **후행**: 없음

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-24
- **변경 파일 목록**:
  - `config/common.env` (수정): `rq_banner()` 헬퍼
  - `07/10/13/16/19/22/25/28/31/34/36/37/40/43/46/49/52/55/58-*.sh` (수정): 시작 배너
  - `10-16,19,X-40-build-*.sh` 8종 (수정): 타깃별 배너
  - `run-smoke.sh` / `run-full.sh` (수정): 파이프라인 배너
  - `61-release-notes.py` / `64-verify-serving.py` (수정): 배너 출력
  - `build-ppl-eval.inc` (수정): 중복 배너 제거
  - `regression-tests/verify-issue-13.sh` (신규)
- **계획 대비 변경 사항**:
  1. 설명에 Stage 번호(`Stage N/9`)를 넣었다 — 9단계 SDLC 구조가 한 줄로 읽힌다.
  2. 배너는 `rq_log`과 별도(접두사 없음)라 스크립트 목록 출력처럼 정돈된다.
