# issue-6: Stage 1 신설 — 수용 기준(acceptance) 계약화 (`config/acceptance.toml` + `07-acceptance`)

agent-tier: local-ok

## 1. 배경 및 문제점

9단계 SDLC(문서: `docs/SDLC.md`)에서 **Stage 1(요구사항·수용 기준)** 만 코드와 대응물이
없다. 나머지 8단계는 스크립트가 있지만 "무엇이 계약인가"는 코드에 흩어져 있다.

1. **수용 기준이 3곳에 산재**:
   - `34-measure-ppl.py`의 `verdict()` → Δ 임계값 `0.3` / `1.0` 하드코딩
   - `46-verify-upload.py`의 `TARGET_RULES[...]["size_gb_range"]` → 용량 범위 하드코딩
   - `README.md` 본문 문장("Δ<0.3 Accept / 0.3~1.0 Conditional / ≥1.0 Discard")
   → 기준을 한 곳에서 바꾸려면 3곳을 같이 고쳐야 하고, 하나만 고치면 서로 모순된다.
2. **Stage 1 산출물이 없다**: 파이프라인이 "판정 기준 없이" 돌아간다. 기준을
   나중에 만들어도 이미 나온 산출물에 대한 계약 검증이 불가능하다.
3. **다음 단계(게이트)가 기준을 못 참조한다**: PPL 게이트가 "verdict가 맞는가"를
   검사하려면 기준 원본이 필요했다.

## 2. 요구사항

### 2.1. `config/acceptance.toml` (신규) — 수용 기준의 단일 소스
1. TOML 형식(python 3.12 표준 `tomllib`로 읽힘 → **새 의존성 0**, `uv.lock` 무변경).
2. 타깃 4종(`awq-w3`/`awq-w5`/`nvfp4-awq`/`nvfp4-gptq`)별:
   - `quant_format` (기대 저장 포맷)
   - `size_gb = [min, max]` (체크포인트 용량 정상 범위)
   - `ppl_delta_warn` (Accept 상한), `ppl_delta_max` (Conditional 상한)
3. `[gates.release]` — Stage 8 게이트 요구사항(모델카드/지표/유한 PPL).

### 2.2. `07-acceptance.py` / `07-acceptance.sh` (신규) — Stage 1 실행기
1. `07-acceptance.py --mode {smoke,full}`: 계약 스키마 검증 → `outputs/acceptance-<mode>.json`
   (계약 스냅샷 + 9단계 정의 + 판정 규칙 문자열) 생성. `--check`(검증만), `--print` 지원.
2. 스키마 오류 시 exit 1 (fail fast).
3. `07-acceptance.sh [smoke|full]`: `config/common.env` 로드 → `uv run` 호출 →
   `$RQ_OUTPUTS/acceptance-$MODE.json` 존재 확인. 2노드 동시 실행은 `rq_lock`으로 직렬화.

### 2.3. 기존 판정 코드의 기준 이관 (하위 호환 유지)
1. `34-measure-ppl.py`: `verdict(delta, spec=None)` — 임계값을 `acceptance.toml`에서 읽고,
   파일이 없으면 기존 값(0.3/1.0) 사용. `summarize()`는 타깃별 spec으로 판정.
2. `46-verify-upload.py`: `TARGET_RULES[*]["size_gb_range"]`를 `acceptance.toml` 값으로 덮어씀.
   계약 파일이 없으면 기존 하드코딩 범위를 그대로 사용.
3. `43-prep-upload.py`: 모델 카드에 **"수용 기준 대비 판정"** 절 추가(기준 vs 실측 대조표).

## 3. 승인 기준

1. `config/acceptance.toml`이 존재하고 `sdlc.acceptance_errors()`가 빈 리스트를 반환.
2. `./07-acceptance.sh smoke` 실행 시 `outputs/acceptance-smoke.json` 생성, rc=0.
3. `34-measure-ppl.py`의 verdict가 `acceptance.toml`의 `ppl_delta_warn/max`를 따른다.
4. `46-verify-upload.py`의 용량 범위가 `acceptance.toml`의 `size_gb`를 따른다.
5. 계약 파일 부재 시 기존 동작(하드코딩 기본값)으로 폴백한다.
6. `tests/test_sdlc.py` 통과.

## 4. 검증

- `tests/test_sdlc.py` — 계약 스키마/기본값 폴백/verdict 경계/07-acceptance 산출물/게이트
- `regression-tests/verify-issue-6.sh` — 위 승인 기준을 grep·실행으로 기계 검증

## 5. 의존성 / 관계

- **선행**: 없음 (SDLC 1단계이므로 파이프라인 첫머리에 위치)
- **후행**: issue-7(게이트가 `rq_gate_ppl`에서 기준을 참조), issue-8(릴리스 노트)

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-24
- **변경 파일 목록**:
  - `config/acceptance.toml` (신규): 타깃별 기준 + `[gates.release]`
  - `sdlc.py` (신규): `load_acceptance`/`acceptance_errors`/`target_spec`/`verdict_for_delta`
    + 단계 원장(ledger) + CLI. 표준 라이브러리만 사용.
  - `07-acceptance.py` / `07-acceptance.sh` (신규): Stage 1 실행기
  - `34-measure-ppl.py` (수정): `acceptance_spec()` + `verdict(delta, spec)`
  - `46-verify-upload.py` (수정): `_apply_acceptance_size_ranges()`
  - `43-prep-upload.py` (수정): `_acceptance_rows()` — 카드에 수용 기준 대조표 추가
  - `tests/test_sdlc.py` (신규)
  - `regression-tests/verify-issue-6.sh` (신규)
- **계획 대비 변경 사항**:
  1. 제안했던 `config/acceptance.yaml` 대신 **TOML**을 사용했다. 이유: `tomllib`가 표준
     라이브러리라 `pyyaml` 추가 → `uv.lock` 변경 → `10-setup` 스탬프(uv.lock 해시) 무효화
     → setup 재실행이 필요해진다. "2회 실행 즉시 종료" 요구와 충돌하므로 의존성 0이 우선이다.
  2. `sdlc.py`는 원장·판정까지 겸하는 공용 모듈로 두었다(게이트·릴리스 노트가 공유).
- **검증 결과**:
  - `tests/test_sdlc.py`: 15 passed (계약 스키마/기본값 폴백/verdict 경계/07-acceptance 산출물)
  - `regression-tests/verify-issue-6.sh`: 통과(rc=0). verify-issue-1~5, 7~10도 통과
  - `ruff check` (변경 파일): All checks passed · `pyright --pythonpath .venv/bin/python`: 0 errors
  - 실환경 확인: `./07-acceptance.sh smoke` → `outputs/acceptance-smoke.json` 생성, rc=0
  - 판정 기준이 실제로 적용됨을 확인: `34-measure-ppl.py` verdict=Accept(Δ=0.0275),
    `46-verify-upload.py` 용량 범위가 `acceptance.toml` 값에서 로드됨
