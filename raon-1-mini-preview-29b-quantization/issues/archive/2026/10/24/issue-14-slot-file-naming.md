# issue-14: 파일 번호 체계 변경 — `##-<slug>` → `s<stage>-##-<slug>`

agent-tier: local-ok

## 1. 배경 및 문제점

스크립트 파일명이 `07-acceptance.sh`, `34-measure-ppl.sh`처럼 **Stage 정보 없이** 스텝 번호만
가지고 있었다. 그런데 이 파이프라인은 9단계 SDLC로 정의되어 있으므로, 파일명만 보면
"이게 몇 단계짜리 작업인지" 알 수 없었다(07이 무엇인지, 36이 무엇인지 매번 물어야 했다).

9단계 구조가 파일명에서 바로 읽히게 하기로 했다.

## 2. 요구사항

1. 파일명을 `s<stage>-##-<slug>.{sh,py}`로 변경한다.
   - `s#` = 9단계 중 몇 번째 단계(0~8, **한 자리 유지**)
   - `##` = 그 Stage 안에서의 스텝 번호(기존 호환성·실행 순서 그대로)
   - `<slug>` = 작업 이름
2. Stage 매핑:

   | stage | 단계 | 파일 |
   |---:|---|---|
   | 0 | 환경 | `s0-10-setup.sh` |
   | 1 | 수용 기준 | `s1-07-acceptance.{sh,py}` |
   | 2 | 데이터 | `s2-13-prep-calibration.{sh,py}` |
   | 3 | 호환성 | `s3-16-compat-check.{sh,py}` |
   | 4 | 베이스 복원 | `s4-19-dequant-mxfp8.{sh,py}` |
   | 5 | 양자화 | `s5-22-quant-awq-w3.{sh,py}` … `s5-31-quant-nvfp4-gptq.{sh,py}` |
   | 6 | PPL | `s6-34-measure-ppl.{sh,py}` |
   | 7 | 평가 | `s7-36-prep-evalset.{sh,py}`, `s7-37-eval.{sh,py}`, `s7-40-score.{sh,py}` |
   | 8 | 릴리스·배포 | `s8-43-prep-upload.{sh,py}` … `s8-64-verify-serving.py` |

3. **참조 전부 갱신**: 스크립트 간 호출, `run-{smoke,full}.sh` validate 목록,
   `build-ppl-eval.inc`, 단위 테스트의 스크립트 로더, 회귀 스크립트의 grep,
   문서(PROCEDURE/README/cheatsheet/docs/SDLC.md), 게이트 메시지.
4. **빌드 wrapper 8종(`10-16,19,X-40-build-*.sh`)은 Stage를 넘나드는 진입점**이라 접두사를
   붙이지 않는다(파일명 자체가 스텝 나열이다).
5. 스텝 라벨(`rq_step "34-ppl"`, 타이밍 로그, 원장 단계명)은 **그대로** 둔다 — 파일명과
   로그의 스텝 번호는 별개다.

## 3. 승인 기준

1. `s?-*.{sh,py}` 37개가 존재하고, 각 파일명의 `s#`가 9단계 정의와 일치한다.
2. 옛 파일명 참조(확장자 포함)가 코드·문서에 0건.
3. `run-{smoke,full}.sh`의 validate 목록이 새 이름을 쓴다(구 이름이면 스크립트가 죽는다).
4. 단위 테스트 7파일 + 회귀 스크립트 verify-issue-1~13 전부 통과.
5. 파이프라인 재실행이 여전히 즉시 종료(2~3초)하고 게이트가 통과한다.

## 4. 검증

- `verify-issue-1~13.sh` 전수 + 단위 테스트 전수 + 빌드 재실행 2회
- 이름 규칙 점검: `ls s?-*.sh` + 단계 매핑 대조

## 5. 의존성 / 관계

- **선행**: issue-6~13 (9단계 SDLC·게이트·표준 GSM8K·TARGETS 필수·배너)
- **후행**: 없음

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-24
- **변경 파일 목록**:
  - 파일 37개 개명(`git mv`)
  - 참조 갱신 79개 파일(스크립트·인클루드·테스트·회귀 스크립트·문서)
  - `PROCEDURE.md` / `docs/SDLC.md`: 명명 규칙 표 추가
  - `regression-tests/verify-issue-13.sh`: 대상 탐색 glob을 `s?-*.sh` + wrapper로 변경
- **계획 대비 변경 사항**:
  1. **1회 재계산이 발생했다**(순수 개명인데도). 개명 후 스크립트 *내용*을 다시 써서
     mtime이 올라갔고, freshness가 "산출물 > 스크립트 mtime"으로判定하기 때문이었다.
     → dequant 999s + quant 1190/2227s + PPL/eval 재실행(총 3.5시간 GPU).
     **결과는 동일**하게 재현됐다(PPL 7.7011, GSM8K 0.935/0.945, 게이트 통과) — 재현성
     측면에서는 오히려 좋은 확인이었다.
     앞으로는 **순수 개명·참조 치환은 mtime을 보존**(`touch -r`)해 불필요한 재계산을 막는다.
  2. 배너 id도 새 파일 basename(`[s7-37-eval]`)으로 바꿔 "복사해 쓸 수 있는 이름"으로 했다.
