# issue-18: MODE 환경변수 무시 결함 (항상 smoke 실행)

agent-tier: local-ok

## 1. 배경 및 문제점

`export MODE=full` 후 `./s8-43-prep-upload.sh`를 인자 없이 실행하면 **무조건 smoke**가 돌았다.

```bash
export TARGETS=nvfp4-awq
export MODE=full
./s8-43-prep-upload.sh
# → + FORCE=1 PROFILE=1x INFRA=dgx-spark-1x MODE=smoke ./s8-43-prep-upload.sh
# → models/raon-nvfp4-awq-smoke/README.md 만 갱신 (full 카드는 그대로)
```

원인은 모드 해석 책임이 두 곳에 나뉘어 있었다는 것이다.

- `config/common.env`: `MODE="${1:-${MODE:-smoke}}"` — 인자 > 환경변수 > 기본값. **올바름.**
- 그런데 stage wrapper 20종이 전부 `source config/common.env "${1:-smoke}"`로 호출했다.
  인자가 없을 때 **"smoke"를 만들어서 넘기므로** common.env는 항상 `$1`이 있다고 판단하고,
  export 된 `MODE`는 **영영 반영되지 않았다.**

결과: "full로 하려고 했는데 smoke 산출물만 갱신" — 공개 업로드 대상 카드가 옛 값으로 남는
실수(무의미한 산출물 덮어쓰기 + 필요한 full 카드 미생성)로 이어진다.

## 2. 요구사항

1. wrapper가 `common.env`에 모드를 넘길 때 **환경변수 `MODE`를 죽이지 않는다.**
   - 우선순위 유지: 명시적 인자 > 환경변수 `MODE` > 기본값 `smoke`
2. 모드를 인자로 받지 않는 스크립트(`s8-55`, `s8-58`, `s0-10-setup`)도 동일한 규칙을 따른다.
3. `s0-s7-build-*.sh`는 파일명에 모드가 박혀 있어 `$BPE_MODE`를 명시적으로 넘긴다(그대로 유지).
4. 회귀 가드로 "MODE 하드코딩 패턴 0건"을 고정한다.
5. 문서에 모드 지정 규칙을 명시한다(인자 / 환경변수).

## 3. 승인 기준

1. `common.env` source 라인에 `"${1:-smoke}"` 또는 리터럴 `smoke` 하드코딩 0건.
2. stage wrapper 20종이 `"${1:-${MODE:-smoke}}"` 형태.
3. 런타임: `MODE=full` 반영 / 명시적 인자 우선 / 미지정 시 smoke.
4. `MODE=full ./s8-43-prep-upload.sh`가 full 모드로 실행(repro echo에 `MODE=full`).
5. verify-issue-1~18 통과, 기존 회귀 전수 통과.

## 4. 검증

- 런타임 모드 해석 3케이스(환경변수 / 인자 우선 / 기본값)
- `MODE=full ./s8-43-prep-upload.sh full nvfp4-awq` → repro echo `MODE=full`
- `bash -n` 전 스크립트 · 회귀 스크립트 전수

## 5. 의존성 / 관계

- **선행**: issue-17(s7 평가 확장)
- **후행**: 없음

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-25
- **변경 파일 목록**:
  - stage wrapper 20종 (수정): `source ... "${1:-smoke}"` → `"${1:-${MODE:-smoke}}"`
    (s1-07 · s2-13 · s3-16 · s4-19 · s5-22 · s5-25 · s5-28 · s5-31 · s6-34 ·
    s7-36 · s7-37 · s7-40 · s7-43 · s7-46 · s7-49 · s7-52 · s8-43 · s8-46 · s8-49 · s8-52)
  - `s8-55-infer-examples.sh` (수정): `common.env smoke` → `"${MODE:-smoke}"`
  - `s8-58-serve-infer-all.sh` (수정): `MODE_ARG="${1:-smoke}"` → `"${1:-${MODE:-smoke}}"`
  - `regression-tests/verify-issue-18.sh` (신규): 하드코딩 0건 + 런타임 모드 해석 3케이스
  - `README.md` · `cheatsheet.md` (수정): 모드 지정 규칙(인자 > `MODE` 환경변수 > smoke) 명시
  - `PROCEDURE.md` (수정): 잔존 구참조 `40-score` → `40-gsm8k`
- **계획 대비 편차**:
  - `s8-52-serve-vllm.sh`의 "인자 없이 실행 → 명령 목록 출력" 분기도 `common.env smoke`을 쓰고 있어
    함께 `"${MODE:-smoke}"`로 정리했다(해당 분기에서 MODE 값은 무관하지만 규칙은 통일).
  - `s0-10-setup.sh`(`source common.env` 인자 없음)와 `s0-s7-build-*.sh`(`$BPE_MODE` 명시)는
    이미 환경변수를 죽이지 않으므로 **변경하지 않았다.**
- **검증 결과**:
  - `verify-issue-18.sh` 통과(3항목) · `verify-issue-1~13`·`17` 전부 통과
  - `pytest tests/` 131 passed · `ruff check .` All checks passed · `bash -n` 전 스크립트 통과
  - 실사용 재현: `MODE=full ./s8-43-prep-upload.sh full nvfp4-awq` → repro echo `MODE=full`
