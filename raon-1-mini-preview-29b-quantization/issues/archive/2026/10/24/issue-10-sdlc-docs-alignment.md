# issue-10: 문서 정렬 — `docs/SDLC.md` 신설 + `PROCEDURE.md`/`README.md`/`cheatsheet.md` 갱신

agent-tier: local-ok

## 1. 배경 및 문제점

1. **SDLC 정의 문서가 없다**: 9단계 정의가 "`PROCEDURE.md`의 stage 표"에만 흩어져 있고,
   단계 간 계약(게이트)·증빙(원장/릴리스 노트)·멱등성 원칙이 문서화되지 않았다.
2. **`PROCEDURE.md`의 stage 표가 9단계와 어긋난다**: 현재 표는 0~8 + `7b`(10행)로,
   "요구사항·수용 기준" 단계가 없고 업로드감사가 별도 단계로 들어 있다.
3. **새 스크립트가 문서에 없다**: `07-acceptance`, `61-release-notes`, `64-verify-serving`,
   `gates.inc`, `sdlc.py`가 사용법(cheatsheet)에 없다.

## 2. 요구사항

1. `docs/SDLC.md` (신규): 9단계 정의표, Stage 1 계약 설명, 게이트 표, 원장/릴리스 노트
   설명, **멱등성(재실행 즉시 종료) 판정 표**, stage↔issue 관계, 11단계 확장안.
2. `PROCEDURE.md`: stage 표를 9단계 기준으로 재작성 + 게이트 위치 명시 + 새 스크립트.
3. `README.md`: 실행 순서에 `07-acceptance` 추가, SDLC 요약 링크.
4. `cheatsheet.md`: 새 스크립트·`STEPS=acceptance/verify`·릴리스 노트 생성 명령.

## 3. 승인 기준

1. `docs/SDLC.md`가 존재하고 9단계 표·게이트 표·멱등성 표를 포함한다.
2. `PROCEDURE.md`에 9단계 표와 게이트가 반영되어 있다.
3. `README.md`/`cheatsheet.md`에 `07-acceptance`/`61-release-notes`/`64-verify-serving`이
   등장한다.

## 4. 검증

- `regression-tests/verify-issue-10.sh` — 승인 기준 grep

## 5. 의존성 / 관계

- **선행**: issue-6 ~ issue-9 (문서화할 대상 구현)

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-24
- **변경 파일 목록**:
  - `docs/SDLC.md` (신규)
  - `PROCEDURE.md`, `README.md`, `cheatsheet.md` (수정)
  - `regression-tests/verify-issue-10.sh` (신규)
- **검증 결과**:
  - `regression-tests/verify-issue-10.sh`: 통과(rc=0)
  - `docs/sdlc-verification-2026-10-24.md`에 실제 검증 결과(재실행 시간·결함 3건·측정값)를 기록
