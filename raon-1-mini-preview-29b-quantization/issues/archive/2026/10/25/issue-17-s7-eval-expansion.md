# issue-17: s7 평가 확장 (한국어 KMMLU · 지시/형식 준수 · 서빙 벤치) + 명시적 개명

agent-tier: local-ok

## 1. 배경 및 문제점

Stage 7 평가가 **표준 GSM8K 하나**였다. 그 결과:

- 이 모델은 한국어 우선 모델인데, GSM8K는 영어 수학이라 **한국어 능력을 아무것도 재지 않는다**.
- PPL은 teacher-forced, GSM8K는 "정답이 맞는가"만 본다. **"요청한 형식을 지켰는가"**를
  검사하는 계층이 없다(언어 지정 무시 · JSON 미출력 · Rambling은 배포에서 가장 흔한 실패).
- 양자화의 이유(속도·메모리)는 품질 검사로 증명되지 않는다. 공개 카드(KRAFTON)의
  처리량 표와 **같은 축**의 자체 측정이 없다.

또한 파일명이 내용을 말하지 않았다 — `s7-36-prep-evalset`는 GSM8K만 만들고,
`s7-40-score`는 GSM8K만 채점한다. 확장되면 이름이 거짓말이 된다.

## 2. 요구사항

1. **표준 한국어 KMMLU**(`HAERAE-HUB/KMMLU`) 평가 세트를 결정론적으로 생성·채점한다.
   - 프롬프트: `s7-43-prep-kmmluko.sh` → `outputs/eval-<mode>/prompts-kmmluko.jsonl` + 매니페스트
   - 채점: `s7-46-kmmluko.sh` → `checks.json`의 `kmmluko` 섹션 (base 대비 하락 허용폭 대조)
2. **지시/형식 준수 스위트**(고정 10케이스)를 생성·검사한다.
   - 프롬프트: `s7-49-format.sh --prepare` → `prompts-format.jsonl`
   - 검사: `s7-49-format.sh` → `checks.json`의 `format` 섹션 (통과율 + 계약 하한 대조)
   - 케이스: 언어지시(한글/영문) · JSON · `####` 마커 · 길이 · bullet · 되묻기 · 문장수 · 줄바꿈
3. **서빙 처리량**(`s7-52-servebench.sh`) — 이미 떠 있는 OpenAI-호환 서버에 요청을 보내
   TTFT·출력 tok/s를 잰다. 서버가 없으면 skip(사유 기록), 파이프라인을 막지 않는다.
4. **명시적 개명**: `s7-36-prep-evalset` → `s7-36-prep-gsm8k`, `s7-40-score` → `s7-40-gsm8k`
   (참조 전수 갱신).
5. 계약(`config/acceptance.toml`)에 `kmmluko_acc_drop_max`·`format_pass_rate_min`을 추가하고,
   게이트(`rq_gate_kmmluko`·`rq_gate_format`)·카드(Check 3/4)·릴리스 노트에 반영한다.
6. **가짜 통과 방지**: 검사 키를 못 찾거나 계약 하한 미달이면 PASS가 아니라 FAIL이다.

## 3. 승인 기준

1. `s7-40-gsm8k` / `s7-43-prep-kmmluko` / `s7-46-kmmluko` / `s7-49-format` / `s7-52-servebench` 존재.
2. 옛 이름(`s7-36-prep-evalset`, `s7-40-score`) 잔존 0건.
3. `config/acceptance.toml`에 `kmmluko_acc_drop_max`(4타깃)·`format_pass_rate_min` 존재.
4. `gates.inc`에 `rq_gate_kmmluko`·`rq_gate_format` 존재.
5. 카드에 `## Check 3 - KMMLU` · `## Check 4 - Instruction and format compliance` ·
   `## Serving performance` 절이 있고, `s8-46-verify-upload`가 Check 3를 필수로 요구한다.
6. 파이프라인 배선(`run-smoke.sh` · `run-full.sh` · `build-ppl-eval.inc`) + `bash -n` 통과.
7. `verify-issue-1~13` + `verify-issue-17` 통과, 단위 테스트 전부 통과, ruff/pyright 신규 오류 0건.

## 4. 검증

- 합성 응답으로 end-to-end: 형식 prepare → 채점(40/46/49) → 게이트(eval/kmmluko/format)
  → 카드 렌더링(KMMLU 하락 0.25 → FAIL, 형식 0.9 → PASS, 카드 한글 0줄)
- 단위 테스트(`tests/test_s7_checks.py`) + 전체 테스트 스위트
- ruff(0.17.2) `check .` / pyright(1.1.414) `--pythonpath .venv/bin/python`
- 회귀 스크립트 전수(verify-issue-1~13, 17)

## 5. 의존성 / 관계

- **선행**: issue-11(표준 GSM8K 도입), issue-14(파일 번호 체계), issue-15(빌드 wrapper 개명),
  issue-16(카드 영문화)
- **후행**: 없음

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-25
- **변경 파일 목록**:
  - `s7-36-prep-gsm8k.{sh,py}` (개명, `s7-36-prep-evalset`에서)
  - `s7-40-gsm8k.{sh,py}` (개명, `s7-40-score`에서) + GSM8K 통계(답 마커율·절단률)·`drop_max` 기록
  - `s7-43-prep-kmmluko.{sh,py}` (신규): 표준 한국어 KMMLU 프롬프트 결정론 생성
  - `s7-46-kmmluko.{sh,py}` (신규): 한국어 채점 + 계약 대조
  - `s7-49-format.{sh,py}` (신규): 고정 10케이스 형식/지시 준수 검사 + 계약 하한 판정
  - `s7-52-servebench.{sh,py}` (신규): 서빙 처리량(서버 없으면 skip)
  - `s7-37-eval.{sh,py}` (수정): 프롬프트 행 메타데이터(case/check) 보존 — 형식 검사가
    검사 키를 잃지 않게(잃으면 전부 통과로 집계된다)
  - `s8-43-prep-upload.py` (수정): 카드에 Check 3(KMMLU)·Check 4(형식)·서빙 성능 절 +
    수용 기준 대조표에 GSM8K/KMMLU/형식 3행 추가. **형식 verdict는 계약 하한 기준**(수정 전에는
    통과율이 있으면 무조건 PASS → 가짜 통과)
  - `s8-61-release-notes.py` (수정): 검사 표에 KMMLU·형식·서빙 행 추가
  - `s8-46-verify-upload.py` (수정): 필수 섹션에 Check 3 추가
  - `sdlc.py` (수정): `checks_path()` · `merge_check()` (파일 락 포함 병합)
  - `gates.inc` (수정): `rq_gate_kmmluko` · `rq_gate_format`
  - `config/acceptance.toml` (수정): `kmmluko_acc_drop_max=0.03` · `format_pass_rate_min=0.9`
  - `config/common.env` (수정): `EVAL_BENCHMARKS` 기본값 3종, `KMMLU_N`·`KMMLU_SUBJECTS`,
    `rq_stage_of`에 7단계 스텝 추가
  - `run-smoke.sh` · `run-full.sh` · `build-ppl-eval.inc` (수정): 신규 스텝 배선
    (서빙 벤치는 `STEPS=serve`로 명시 요청 시에만)
  - `tests/test_s7_checks.py` (신규), `tests/test_evalset.py` · `tests/test_ppl_and_eval.py` (수정)
  - `regression-tests/verify-issue-17.sh` (신규), `verify-issue-4.sh` · `verify-issue-11.sh` (수정)
  - `README.md` · `PROCEDURE.md` · `cheatsheet.md` · `docs/SDLC.md` (수정): 검사 5종 구조 반영
- **계획 대비 편차**:
  - `s2-13-prep-calibration.py` · `s3-16-compat-check.py` — 이슈 범위 밖이지만 `ruff check .`
    전체 통과를 위해 기존 린트 부채(미사용 import·f-string·import 정렬)를 정리했다.
  - `s5-*` 양자화 스크립트의 pyright 오류 7건(`oneshot` private import · `config_groups` ·
    `actorder="static"`)은 issue-3에서 이미 "기존 오류"로 인지한 항목이라 이번엔 손대지 않았다.
- **검증 결과**:
  - `verify-issue-17.sh` 통과(7개 항목) · `verify-issue-1~13` 전부 통과
  - 단위 테스트: `pytest tests/` **131 passed** (신규 파일 `tests/test_s7_checks.py` 15건 포함)
  - `ruff check .` → All checks passed (0건)
  - `pyright --pythonpath .venv/bin/python` → 7 errors (전부 `s5-*` 기존 오류, 이번 변경분 0건)
  - `bash -n` 전체 스크립트 통과 · `compileall` 통과
  - end-to-end(합성 응답): gate-kmmluko FAIL(하락 0.250 > 0.030) · gate-format PASS(0.900 ≥ 0.900)
    · 카드에 FAIL/PASS 정확 반영 · 카드 한글 0줄
