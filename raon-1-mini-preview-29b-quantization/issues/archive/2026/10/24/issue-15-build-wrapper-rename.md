# issue-15: 빌드 wrapper 개명 — `10-16,19,X-40-build-*` → `s0-s7-build-<mode>-<target>`

agent-tier: local-ok

## 1. 배경 및 문제점

issue-14에서 Stage 스크립트(`##-<slug>`)를 `s<stage>-##-<slug>`로 바꿨지만,
**빌드 wrapper 8종**(`10-16,19,X-40-build-<mode>-<target>.sh`)만 예외로 남아 있었다.
파일명이 스텝 번호 나열(`10-16,19,28-40`)이라 다른 규칙처럼 보였고, Stage 0~7을
한 번에 돌리는 진입점이라는 사실이 이름에 드러나지 않았다.

## 2. 요구사항

1. 빌드 wrapper를 `s0-s7-build-<mode>-<target>.sh`로 개명한다.
   - `s0-s7` = 이 진입점이 **Stage 0~7**(setup→acceptance→calib→compat→dequant→
     quant→ppl→eval)을 한 번에 돌린다는 뜻. Stage 8(카드·감사·업로드·서빙)은
     `STEPS=prep,verify`로 선택한다.
2. 참조 갱신: wrapper 자체의 헤더/사용 예, `build-ppl-eval.inc`, `README.md`,
   `PROCEDURE.md`, `cheatsheet.md`, `docs/SDLC.md`, `docs/sdlc-verification-*.md`,
   `regression-tests/verify-issue-13.sh`(glob).
3. 배치 배너 id도 파일 basename으로(`[s0-s7-build-smoke-nvfp4-awq]`), 설명은
   "Stage 0-7 pipeline for a single target (...)"으로.
4. **파이프라인은 실행하지 않는다**(이름·참조만 변경).

## 3. 승인 기준

1. `10-*.sh` 잔존 0건, `s0-s7-build-*.sh` 8종 존재.
2. 깨진 파일 참조 0건(참조 대상 파일이 모두 존재).
3. 이름 규칙 위반 0건.
4. `bash -n` 전부 통과, ruff/pyright 0건, 단위 테스트 7파일 전부 통과.

## 4. 검증

- 정적 검증만 수행(실행 금지): bash -n / 참조 존재성 / 이름 규칙 / ruff / pyright / 단위 테스트
- 단위 테스트는 합성 픽스처만 사용하며 파이프라인(GPU·모델)을 실행하지 않는다.

## 5. 의존성 / 관계

- **선행**: issue-14 (Stage 스크립트 개명)
- **후행**: 없음

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-24
- **변경 파일 목록**:
  - `git mv` 8종: `10-16,19,{22,25,28,31}-40-build-*.sh` → `s0-s7-build-*.sh`
  - 참조 갱신 11개 파일 + 서술·메시지 22개 파일
  - `PROCEDURE.md` / `docs/SDLC.md`: wrapper 명명 규칙(`s0-s7-`) 설명 추가
  - `issues/archive/2026/10/24/issue-15-build-wrapper-rename.md` (신규)
- **계획 대비 변경 사항**:
  1. 파일명에서 스텝 번호 나열(`10-16,19,28-40`)을 뺐다. Stage 범위(`s0-s7`)가
     같은 정보를 더 정확히 담고, 스텝 번호는 wrapper 헤더와 `build-ppl-eval.inc`에 남긴다.
  2. issue-14에서 남겨둔 **스텝 라벨**(`rq_step "10-setup"`, `rq_stage_of` 패턴,
     타이밍 로그 키)은 이름 변경 대상이 아니므로 그대로 두었다(로그 포맷 유지).
