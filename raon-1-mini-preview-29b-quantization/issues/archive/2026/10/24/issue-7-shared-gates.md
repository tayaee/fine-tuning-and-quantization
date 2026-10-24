# issue-7: 게이트 공통화 — `gates.inc` 신설 + `run-{smoke,full}.sh` 경로에도 게이트 적용

agent-tier: local-ok

## 1. 배경 및 문제점

1. **게이트가 단일 타깃 빌드 경로에만 있다**:
   ```
   $ grep -n "gate" run-smoke.sh run-full.sh   →  0 hits
   ```
   `rq_bpe_gate_quant/ppl/eval`은 `build-ppl-eval.inc` 안에만 정의돼 있었다.
   즉 **주 파이프라인(`run-smoke.sh`/`run-full.sh`)에는 게이트가 하나도 없다** —
   패킹 계약 위반(dense 폴백)·NaN PPL·빈 응답이 그대로 eval/배포까지 흘러갈 수 있다.
   (README가 경고한 "NaN 모델이 0점 벤치만 남는" 상황의 재현 경로가 남아 있었다.)
2. **게이트 정의가 파이프라인 경로에 묶여 있다**: `build-ppl-eval.inc`가 `uv run python -c '...'`
   인라인 스크립트로 게이트를 구현해 있어, 다른 경로에서 재사용할 수 없다.
3. **기준 대조 게이트가 없다**: PPL 게이트는 "유한값"만 보고, 그 값이 **수용 기준
   (issue-6의 acceptance.toml)** 과 맞는 verdict인지 검사하지 않았다.

## 2. 요구사항

### 2.1. `gates.inc` (신규) — 공용 게이트
1. `rq_gate_acceptance <mode>`: Stage 1 계약 산출물 존재·구조 확인.
2. `rq_gate_quant <model_dir> <format>`: `quant_common.verify_compressed_output`
   (패킹 포맷·`global_scale` 값·dense 폴백 차단).
3. `rq_gate_ppl <ppl.json> [target]`: PPL 유한값 + (target이 주어지면) verdict가
   `acceptance.toml` 기준과 일치하는지.
4. `rq_gate_eval <eval.jsonl>`: 응답 파일이 존재하고 1줄 이상.
5. `rq_gate_release <mode>`: `prep-upload-<mode>.json` + `verify-upload-<mode>.json`의 `ok`.
6. `rq_gate_serving <mode>`: `outputs/serve-<mode>.json` 대조(로그 미수행은 skip).
7. 모든 게이트는 통과=0/실패=1, 실패 사유는 stderr.

### 2.2. 경로 연결
1. `run-smoke.sh` / `run-full.sh`: quant 직후 `gate-quant`, PPL 직후 `gate-ppl`,
   eval 직후 `gate-eval` 호출. 게이트 실패 시 파이프라인 중단(다음 단계로 안 넘어감).
2. `build-ppl-eval.inc`: 인라인 게이트 구현을 삭제하고 `gates.inc` 공용 함수로 대체.
   `rq_bpe_gate_*` 이름은 외부 참조가 없어 제거.
3. `config/common.env`가 `gates.inc`를 source → 모든 스크립트가 같은 게이트를 쓴다.

## 3. 승인 기준

1. `run-smoke.sh`/`run-full.sh`에 `gate-quant`/`gate-ppl`/`gate-eval` 호출이 존재.
2. `build-ppl-eval.inc`에 인라인 게이트 구현이 남아있지 않고 `rq_gate_*`를 호출.
3. `gates.inc`가 `rq_gate_acceptance/quant/ppl/eval/release/serving` 6종을 정의.
4. 게이트 실패 시 rc=1로 파이프라인이 중단된다(테스트로 확인).
5. `tests/test_sdlc.py` 통과.

## 4. 검증

- `tests/test_sdlc.py` — 게이트 통과/차단(NaN·빈 파일·verdict 불일치·acceptance 누락)
- `regression-tests/verify-issue-7.sh` — 승인 기준 grep + 단위 테스트

## 5. 의존성 / 관계

- **선행**: issue-6 (PPL 게이트가 acceptance 기준을 참조)
- **후행**: issue-8(원장이 게이트 결과를 기록), issue-9(배포 후 게이트)

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-24
- **변경 파일 목록**:
  - `gates.inc` (신규): 6종 게이트 + `rq_py` 헬퍼
  - `config/common.env` (수정): `gates.inc` source, `RQ_PY`, `rq_stage_record`,
    `rq_stage_of`, `rq_quant_format` 추가
  - `run-smoke.sh` / `run-full.sh` (수정): `07-acceptance` 스텝 + 게이트 3종 연결
  - `build-ppl-eval.inc` (수정): 인라인 게이트 삭제 → `rq_gate_*` 사용, acceptance 단계 추가
- **계획 대비 변경 사항**:
  1. Stage 8 릴리스 게이트는 **기본 경로에 넣지 않았다**. `46-verify-upload`가 아직
     실행되지 않은 상태에서 `gate-release`를 기본에 넣으면 파이프라인이 항상 실패한다.
     `STEPS=verify`로 **명시 요청 시에만** 실행되도록 `want_step_explicit`를 추가했다.
  2. `rq_gate_serving`은 서빙 로그가 없으면 skip(0)한다 — 배포는 수동 단계라
     파이프라인을 막으면 안 된다.
- **검증 결과**:
  - `tests/test_sdlc.py`: 게이트 통과/차단 6경로 통과(NaN·빈 파일·verdict 불일치·acceptance 누락·skip)
  - `regression-tests/verify-issue-7.sh`: 통과(rc=0)
  - 실환경 확인: `run-smoke.sh`에 gate-quant/ppl/eval 연결 후 nvfp4-awq smoke 빌드가
    게이트 3종을 통과했고(각 0~1s), 재실행 2회차 2초로 종료됨
  - **기존 결함 검출**: 재수집 전 산출물에 대해 gate-quant가
    "비정상/미보정 global_scale 267개"로 실패 → 결함 확정 후 재양자화(재수집 후 통과)
