# issue-21: 일부 벤치만 재생성하면 GSM8K 응답이 사라져 카드 Check 2가 빔 + PPL delta 3자리

agent-tier: local-ok

## 1. 배경 및 문제점

1. `models/raon-nvfp4-awq-full/README.md`의 PPL delta가 `-0.00`으로 찍혔다
   (실측 `-0.0010078…`). 2자리 반올림이라 Accept 선 근처의 미세한 차이가 보이지 않는다.
2. KMMLU·형식 준수(issue-17)를 채우려고 Stage 7을 다시 돌린 뒤 카드를 재생성하자
   **Check 2(GSM8K) 표가 통째로 `—`**가 됐다.
   - `outputs/eval-full/{base,nvfp4-awq}.jsonl` = kmmluko 200행 + format 10행, **gsm8k 0행**
   - `score.json`에 gsm8k 정확도 없음, `checks.json`의 gsm8k 섹션 `targets: {}`
   - 원인: `s7-37-eval.py`의 `run_target()`이 `<target>.jsonl`을 **이번 실행의 벤치로만
     덮어쓴다**. 벤치 목록에서 gsm8k가 빠진 채 실행되면(예: `EVAL_BENCHMARKS=kmmluko,format`)
     기존 GSM8K 응답(약 50분/모델)이 조용히 사라진다. `bench/gsm8k/*.jsonl` 상세 사본은
     남아 있지만 채점기(s7-40/46/49)는 최상위 파일만 읽는다.
   - 2차 결함: `s7-37-eval.sh`의 freshness는 mtime만 본다 → 그 뒤 `EVAL_BENCHMARKS=gsm8k`로
     다시 돌려도 "fresh"로 판정돼 skip된다(FORCE=1이 필요).
   - 3차 결함: `s8-43-prep-upload.py`는 `score.json`에 타깃 **행**만 있으면 통과시켜
     GSM8K가 빈 카드를 만들었다.

## 2. 요구사항

1. 카드의 PPL delta(수용 기준 표 2행 + Check 1 표)는 소수 셋째 자리까지 표시한다.
2. `s7-37-eval.py`: 이번에 재생성하지 않는 벤치의 기존 응답은 `<target>.jsonl`에 보존한다.
   같은 벤치를 다시 돌리면 교체한다(중복 누적 금지).
3. `s7-37-eval.sh`: 요청한 벤치 중 하나라도 `<target>.jsonl`에 응답이 없으면 stale로 본다.
4. `s8-43-prep-upload.py`: base/타깃 중 GSM8K 정확도가 없으면 카드 생성을 거부한다
   (`--allow-missing-metrics`일 때만 허용).

## 3. 승인 기준

1. 실측 카드: `| PPL delta vs base | < 1.0 (Conditional limit) | -0.001 | Accept |`
2. 실측 카드: Check 2 표에 base·nvfp4-awq GSM8K 정확도와 `n/200`이 있다.
3. 실측 카드: 수용 기준 표의 GSM8K·KMMLU·형식 준수 3행에 값과 판정이 있다.
4. GSM8K만 재생성한 뒤에도 `<target>.jsonl`에 kmmluko/format 응답이 남아 있다.
5. verify-issue-1~21 통과, 단위 테스트 전부 통과, ruff/pyright 신규 오류 0건.

## 4. 검증

- 단위: `test_eval_partial_bench_run_keeps_other_bench_rows`,
  `test_prep_requires_gsm8k_accuracy_unless_allowed`, `test_prep_readme_ppl_delta_has_three_decimals`
- 실측: `EVAL_BENCHMARKS=gsm8k ./s7-37-eval.sh full base,nvfp4-awq` → s7-40/46/49 → s8-43

## 5. 의존성 / 관계

- **선행**: issue-17(s7 평가 확장), issue-20(카드 Model 컬럼)
- **후행**: 없음

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-25
- **변경 파일 목록**:
  - `s8-43-prep-upload.py` (수정): PPL delta `_fmt(..., 3)` (수용 기준 표 2행 + Check 1 표) ·
    `prepare_target()`이 base/타깃 GSM8K 정확도 누락을 오류로 보고(재실행 명령 안내)
  - `s7-37-eval.py` (수정): `preserved_rows()` 신설 — `run_target()`이 재생성하지 않는 벤치의
    기존 행을 `<target>.jsonl` 뒤에 보존, 보존 건수를 로그로 출력
  - `s7-37-eval.sh` (수정): freshness 판정에 "요청 벤치 응답 존재" 조건 추가, 누락 시 stale 로그
  - `tests/test_ppl_and_eval.py` · `tests/test_prep_and_upload.py` (테스트 추가),
    `tests/test_verify_upload.py` (delta 3자리 표기에 맞춰 치환 문자열 갱신)
  - `regression-tests/verify-issue-21.sh` 신규
- **계획 대비 편차**: 소실된 GSM8K 응답은 `bench/gsm8k/` 사본으로 복원하지 않고 실제로
  재생성했다(새 freshness 로직이 stale을 감지해 FORCE 없이 재실행됨을 실측으로 함께 확인).
- **검증 결과**:
  - `pytest tests/` 136 passed · `ruff check .` All checks passed · pyright 변경 파일 0 errors
  - verify-issue-1~21 전부 통과
  - 실측: 수정된 `s7-37-eval.sh`가 `base.jsonl has no rows for bench(es) gsm8k`로 stale을
    감지해 FORCE 없이 GSM8K 재생성을 시작함을 확인. 재생성 후 카드는 `s8-43-prep-upload.sh full`로 갱신
