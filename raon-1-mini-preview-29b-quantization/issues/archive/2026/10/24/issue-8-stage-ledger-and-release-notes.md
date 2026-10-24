# issue-8: 단계 원장(ledger) + 릴리스 노트 — SDLC 증빙 체계

agent-tier: local-ok

## 1. 배경 및 문제점

1. **단계 상태를 한눈에 볼 수 없다**: `outputs/`에 `ppl-*/summary.json`,
   `eval-*/score.json`, `prep-upload-*.json`이 산발적으로 흩어져 있고,
   "어느 단계가 통과했고 그 단계의 산출물이 무엇인가"를 보여주는 파일이 없다.
2. **릴리스 노트가 없다**: "이 산출물이 왜 배포 가능한가"를 증명하는 문서
   (수용 기준 vs 실측 대조)가 없다. `43-prep-upload`가 만드는 모델 카드는
   *사용자용* 설명서라 파이프라인 내부의 계약 대조증빙과는 목적이 다르다.
3. **실패 이력이 남지 않는다**: 스텝이 실패해도 타이밍 로그에만 rc가 남는다.

## 2. 요구사항

### 2.1. 단계 원장 (`outputs/stage-ledger-<mode>.json`)
1. `sdlc.py`에 `update_ledger()/read_ledger()` — 파일 락(`fcntl`)으로 병렬 실행 안전.
2. 단계 항목: `id`(0~8), `name`, `status`(pass/fail), `artifacts`, `verdict`,
   `duration_s`, `detail`, `targets`(타깃별 상태 맵).
3. 타깃이 있으면 단계 status는 "타깃 하나라도 실패면 fail"로 집계.
4. CLI: `sdlc.py ledger --mode ... --stage ... --name ... --status ... [--target ...]`,
   `sdlc.py show --mode ...`.

### 2.2. 파이프라인 연동
1. `config/common.env`: `rq_stage_record`(단계 단위), `rq_stage_target`(타깃 단위),
   `rq_stage_of`(스텝 이름 → 9단계 ID 매핑).
2. `run-smoke.sh`/`run-full.sh`의 `rq_step`과 `build-ppl-eval.inc`의 `rq_bpe_step`이
   스텝 완료 시 자동으로 원장을 갱신(실패도 기록).

### 2.3. `61-release-notes.py` (신규)
1. 원장 + `acceptance-<mode>.json` + `ppl summary` + `eval score` + prep/verify를 합쳐
   `outputs/release-notes-<mode>.md` 생성.
2. 구성: ① Stage별 상태 표 ② 수용 기준 대비 실측 표 ③ 릴리스 게이트 표 ④ 판정 규칙.

## 3. 승인 기준

1. 원장 파일이 파이프라인 실행 후 갱신되고, 9개 단계 ID가 채워진다.
2. 타깃별 실패가 단계 status를 fail로 집계한다.
3. `61-release-notes.py --mode smoke`가 마크다운 릴리스 노트를 생성한다.
4. `tests/test_sdlc.py` 통과.

## 4. 검증

- `tests/test_sdlc.py` — 원장 roundtrip/타깃 집계/파일 부재 시 빈 원장
- `regression-tests/verify-issue-8.sh` — 승인 기준 grep + 단위 테스트

## 5. 의존성 / 관계

- **선행**: issue-6, issue-7 (단계 정의·게이트 결과가 원장에 쌓임)
- **후행**: issue-9 (배포 후 검증 결과를 원장에 반영)

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-24
- **변경 파일 목록**:
  - `sdlc.py` (수정): `update_ledger(target=...)`, `ledger`/`show` CLI
  - `config/common.env` (수정): `rq_stage_record`/`rq_stage_target`/`rq_stage_of`
  - `run-smoke.sh` / `run-full.sh` (수정): `rq_step`에서 원장 기록
  - `build-ppl-eval.inc` (수정): `rq_bpe_step`에서 원장 기록
  - `61-release-notes.py` (신규)
  - `tests/test_sdlc.py` (수정), `regression-tests/verify-issue-8.sh` (신규)
- **계획 대비 변경 사항**:
  1. 원장 기록 실패는 파이프라인을 막지 않는다(증빙 부재로 죽이면 측정 작업이 날간다).
  2. 원장은 `$RQ_OUTPUTS`(공유 디스크)에 쓴다 — 2노드에서 각자 쓰되 파일 락으로 직렬화.
- **검증 결과**:
  - `tests/test_sdlc.py`: 원장 roundtrip·타깃 집계·빈 원장 통과
  - `regression-tests/verify-issue-8.sh`: 통과(rc=0)
  - 실환경 확인: nvfp4-awq smoke 빌드 실행 후 `outputs/stage-ledger-smoke.json`에
    0~7단계가 status/artifact와 함께 기록됨(5단계는 타깃별 맵 포함),
    `61-release-notes.py --mode smoke` → `outputs/release-notes-smoke.md` 생성
