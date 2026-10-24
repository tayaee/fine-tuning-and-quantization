# issue-9: 배포 후 검증(Stage 8) — `64-verify-serving` + `gate-serving`

agent-tier: local-ok

## 1. 배경 및 문제점

1. **Stage 8에 exit criteria가 없다**: 9단계 정의에서 8단계(릴리스·배포)는
   "업로드·서빙까지"인데, 파이프라인에는 업로드 전 감사(`46-verify-upload`)까지만
   있고 **배포 후**에 서빙이 실제로 성공했는지 확인하는 장치가 없다.
   `58-serve-infer-all.sh`는 각 타깃의 서빙 로그(`logs/55.<slug>.log`)를 남기지만
   이를 판정하는 코드가 없다(사람이 눈으로 봐야 한다).
2. **게이트가 없으면 배포 실패가 조용히 지나간다**: 업로드는 성공했는데 서빙이
   죽으면(커스텀 아키텍처 미지원 등) 다음 소비자가 첫 발견자가 된다.

## 2. 요구사항

### 2.1. `64-verify-serving.py` (신규)
1. `logs/55.<slug>.log`를 타깃별로 읽어 상태 판정:
   - `SERVE-FAIL` / `INFER-FAIL` / `[OK]` 마커 없음 → `fail`
   - `[OK]` 있으면 `ok`
   - 로그 파일 없음 → `missing`(서빙 미수행)
2. `outputs/serve-<mode>.json`에 결과 기록(`ok`/`status`/`errors`/`targets`).
3. exit code: 실패 타깃이 있으면 1, 없으면 0(missing만 있으면 skip 메시지와 함께 0).

### 2.2. `gates.inc`의 `rq_gate_serving`
1. `outputs/serve-<mode>.json`이 없으면 **skip(0)** — 배포는 수동 단계이므로
   파이프라인을 막지 않는다.
2. 파일이 있고 `ok=false`이면 실패(1).

## 3. 승인 기준

1. `64-verify-serving.py`가 로그 상태를 판정해 JSON 리포트를 만든다.
2. `SERVE-FAIL`이 있는 로그 → rc=1.
3. 로그가 전혀 없으면 rc=0 + skip 메시지.
4. `gates.inc`에 `rq_gate_serving`이 있다.
5. `tests/test_sdlc.py` 통과.

## 4. 검증

- `tests/test_sdlc.py` — 게이트 skip/성공/실패 3경로
- `regression-tests/verify-issue-9.sh` — 승인 기준 grep + 단위 테스트

## 5. 의존성 / 관계

- **선행**: issue-7 (게이트 공통 모듈), issue-8 (결과를 원장에 기록)

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-24
- **변경 파일 목록**:
  - `64-verify-serving.py` (신규)
  - `gates.inc` (수정): `rq_gate_serving` 추가
  - `tests/test_sdlc.py` (수정), `regression-tests/verify-issue-9.sh` (신규)
- **계획 대비 변경 사항**:
  1. 서빙 로그를 "판정"하는 것은 사람이 읽던 grep을 코드로 옮긴 것이다.
     `58-serve-infer-all.sh`의 요약(`OK=.. FAIL=..`)과 같은 기준을 쓴다.
  2. 서빙을 실제로 돌리지는 않았다(GPU·업로드 토큰 필요). 로그 fixture로만 검증했다.
- **검증 결과**:
  - `tests/test_sdlc.py`: gate-serving skip/성공 경로 통과
  - `regression-tests/verify-issue-9.sh`: 통과(rc=0)
  - 실환경 확인: 서빙 로그가 없는 상태에서 `64-verify-serving.py --mode smoke` →
    `outputs/serve-smoke.json`(status=skipped) 생성, rc=0(파이프라인을 막지 않음)
  - **미실행**: 실제 vLLM 서빙은 업로드 토큰·GPU가 필요해 수행하지 않음
  - **추가 수정(릴리스 게이트 실측에서 발견)**: `46-verify-upload.py`가
    `eos_token_id`/`pad_token_id`를 무조건 요구했으나 **KRAFTON 원본 config.json에도
    두 필드가 없다**(토크나이저 설정만 보유) → compliant 모델이 오탐으로 막혔다.
    `_base_config()`를 기준 삼아 "베이스에 있을 때만" 요구하도록 수정했다.
    수정 후 smoke/full 모두 `ok=true`, `rq_gate_release` 통과.
