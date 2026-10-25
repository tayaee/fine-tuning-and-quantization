# issue-12: `run-{smoke,full}.sh`의 TARGETS 필수화 — 자동 전체 실행 금지

agent-tier: local-ok

## 1. 배경 및 문제점

`run-{smoke,full}.sh`는 `TARGETS` 미지정 시 **4종 전부**(awq-w3/awq-w5/nvfp4-awq/
nvfp4-gptq)를 자동으로 돌았다.

```bash
_targets="${TARGETS:-all}"     # ← 미지정 = 전부
```

1. **인수 깜빡 → 장시간 작업**: 타깃 하나만 고려하고 실행하면 4종이 각각 20~40분
   (양자화) + 30분(측정)을 소비한다. 전체 4종이면 수 시간이다.
2. **의도와 실행이 어긋남**: "nvfp4-awq만 확인"하려는 사람이 smoke/full 전체가 돌고
   있다는 사실을 실행 후에야 알 수 있다.
3. **실측 사례**: issue-11 작업 중 `./run-smoke.sh`를 인자 없이 실행해 4종 빌드가
   시작될 뻔했다(게이트 위치 덕에 양자화 전에 멈췄지만, 설계 의도와 어긋난다).

## 2. 요구사항

1. `TARGETS` 미지정 시:
   - **어떤 스텝도 실행하지 않고** 가능한 값(`awq-w3`/`awq-w5`/`nvfp4-awq`/
     `nvfp4-gptq`/`all`)과 사용 예시를 출력한 뒤 종료한다(exit 2).
   - 종료 위치는 **파이프라인 시작 직후**(setup/acceptance 등 어떤 스텝도 돌지 않음).
2. `TARGETS=all` → 4종 전부(기존 동작 유지).
3. `TARGETS=<부분집합>` → 해당 타깃만(콤마 구분, 2노드 분담 방식 유지).
4. 알 수 없는 타깃 이름 → 즉시 거부(exit 2, 사용 가능한 값 안내).
5. 문서(README/PROCEDURE/cheatsheet)에 `TARGETS=all` 사용법 반영.

## 3. 승인 기준

1. `./run-smoke.sh`(TARGETS 미지정) → 가능 값 출력 + exit 2, **스텝 실행 0건**.
2. `TARGETS=all STEPS=validate ./run-smoke.sh` → 정상 실행.
3. `TARGETS=nvfp4-awq ./run-smoke.sh` → nvfp4-awq만 대상.
4. `TARGETS=bogus ./run-smoke.sh` → exit 2 + 안내.
5. `run-full.sh`도 동일.
6. 빌드 wrapper(8종)는 타깃이 파일명에 고정되어 있어 영향 없음.

## 4. 검증

- `regression-tests/verify-issue-12.sh` — 승인 기준 기계 검증(그리퍼 + 실제 실행)
- 문서 grep

## 5. 의존성 / 관계

- **선행**: 없음
- **후행**: 없음

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-24
- **변경 파일 목록**:
  - `run-smoke.sh` / `run-full.sh` (수정): TARGETS 필수 검사(파이프라인 시작 직후),
    `all` 확장, 알 수 없는 타깃 거부, 헤더 주석 갱신
  - `README.md` / `PROCEDURE.md` / `cheatsheet.md` (수정): `TARGETS=all` 사용법
  - `regression-tests/verify-issue-12.sh` (신규)
- **계획 대비 변경 사항**:
  1. 종료 코드는 사용법 오류이므로 **2**(0이 아니다) — 셸 스크립트 관례대로 "2 = usage".
  2. "가능한 값" 출력은 실제 타깃 목록을 한 변수(`RQ_TARGETS_ALL`)에서 뽑아 만든다 —
     타깃이 늘면 안내도 자동으로 따라간다.
