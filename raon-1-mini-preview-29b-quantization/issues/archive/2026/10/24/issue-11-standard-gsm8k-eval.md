# issue-11: 표준 GSM8K 평가 도입 + 모델 카드 2종 검사(PPL·GSM8K) + 배치 생성

agent-tier: local-ok

## 1. 배경 및 문제점

1. **평가가 표준 벤치마크가 아니었다**.
   `37-eval.py`는 `outputs/eval-<mode>/prompts-<bench>.jsonl`을 읽는데, **이 파일을 만드는
   스크립트가 파이프라인에 아예 없었다.** 실존 파일은 이전 세션에서 수작업으로 만든
   표본(smoke 3행/bench, full 10행/bench)이었다. 즉:
   - 표본이 극히 작아 정확도 신뢰도가 없다(10문제 → ±15%p)
   - 어떤 데이터에서 뽑았는지 재현 불가(데이터셋·리비전·시드 기록 없음)
   - `kmmlu_redux`/`humaneval`도 같은 상태(표준 집합 아님)
2. **모델 카드가 "성능 비교 3벤치" 표**라, PPL 검사와 벤치마크 검사가 한 표에 섞여 있고
   "무엇을 통과했는지"가 불명확했다.
3. **평가가 느렸다**: transformers 백엔드가 프롬프트 1개씩 생성 → 20문제에 560초.
   표준 GSM8K 테스트셋(1319문제) 실측은 배치 생성 없이는 불가능했다.

## 2. 요구사항

### 2.1. `36-prep-evalset.py` / `36-prep-evalset.sh` (신규) — 표준 GSM8K 준비
1. `openai/gsm8k` (config `main`, revision 고정) **test** split에서 결정론적 샘플링
   (seed 고정, `GSM8K_N`으로 표본 크기, 0=전체 1319).
2. 프롬프트: Raon chat template(`user\n…\nassistant\n<think></think>\n\n`) +
   `#### <number>` 지시. gold: 데이터셋 정답의 `#### N` 숫자.
3. 출력: `outputs/eval-<mode>/prompts-gsm8k.jsonl` +
   `outputs/eval-<mode>/evalset-<mode>.json`(데이터셋/리비전/seed/n/sha256 매니페스트).
4. **멱등**: (dataset, revision, seed, n) + 파일 sha256가 같으면 재생성 없이 skip.

### 2.2. `37-eval.py` — 표준 GSM8K + 배치 생성
1. 기본 벤치를 `gsm8k` 단일로 변경. 정식 프롬프트 파일이 없으면 **실패**(calib-tail 폴백 금지).
2. `batch_size` 지원(좌측 패딩). `EVAL_BATCH` 환경변수(기본 8), `--batch-size` CLI.
3. `37-eval.sh`: freshness 입력에 프롬프트 파일 추가(평가 세트가 바뀌면 측정값도 무효).

### 2.3. 모델 카드 — 검사 2종
1. "검사 1 — PPL (in-domain Perplexity)" 표(모델/ PPL / Δ / 판정)
2. "검사 2 — GSM8K (표준 테스트셋 정확도)" 표(모델/ 정확도 / 정답·전체 / 판정)
3. 판정 기준은 `config/acceptance.toml`의 `gsm8k_acc_drop_max`(base 대비 하락 허용폭).

### 2.4. 게이트
1. `gates.inc`의 `rq_gate_eval`가 **표준 GSM8K 정확도**를 base와 대조(허용폭 초과 시 실패).
2. 파이프라인: `run-{smoke,full}.sh`/`build-ppl-eval.inc`에 `36-prep-evalset` 단계 추가.

## 3. 승인 기준

1. `./36-prep-evalset.sh full` → `prompts-gsm8k.jsonl`(1319행) + 매니페스트 생성, 재실행 시 skip.
2. `37-eval.py`가 표준 GSM8K만 평가하고, 프롬프트 파일이 없으면 실패.
3. 배치 vs 단건의 채점 결과를 실모델 20문제로 대조(bs=1 19/20 vs bs=8 20/20 —
   1문제 차이). 결과 동일성은 보장되지 않으므로 **프로토콜을 고정**(동일 배치 크기)하고
   표본 노이즈를 판정 근거에 반영한다.
4. 모델 카드에 PPL/GSM8K 두 검사 표가 있고, GSM8K 표의 판정이 계약 기준으로 나온다.
5. `rq_gate_eval`이 GSM8K 정확도 하락 초과 시 실패.
6. `tests/test_evalset.py` + 기존 테스트 전부 통과.

## 4. 검증

- `tests/test_evalset.py` (신규): gold 추출·프롬프트 포맷·결정론 샘플링·멱등성
- `tests/test_ppl_and_eval.py` (수정): 표준 프롬프트 필수·배치 생성 테스트 추가
- `regression-tests/verify-issue-11.sh` (신규)

## 5. 의존성 / 관계

- **선행**: issue-6(수용 기준 — GSM8K 판정 기준이 여기에 속함), issue-7(게이트 공통 모듈)
- **후속**: 없음 (Stage 7이 표준 벤치마크로 정합)

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-24
- **변경 파일 목록**:
  - `36-prep-evalset.py` / `36-prep-evalset.sh` (신규)
  - `37-eval.py` (수정): BENCHMARKS=["gsm8k"], 정식 프롬프트 필수, 배치 생성(`batch_size`)
  - `37-eval.sh` (수정): `--benchmarks`/`--batch-size` 전달, freshness에 프롬프트 파일 포함
  - `43-prep-upload.py` (수정): 카드 = 검사1(PPL) + 검사2(GSM8K), `_gsm8k_verdict`/`_evalset_meta` 추가
  - `config/acceptance.toml` (수정): `gsm8k_acc_drop_max` 추가
  - `config/common.env` (수정): `GSM8K_N`/`EVAL_BENCHMARKS`/`EVAL_BATCH`, smoke `EVAL_N` 8→20
  - `gates.inc` (수정): `rq_gate_eval`이 GSM8K 정확도 계약 대조
  - `run-smoke.sh` / `run-full.sh` / `build-ppl-eval.inc` (수정): 36 단계 배선
  - `tests/test_evalset.py` (신규), `tests/test_ppl_and_eval.py` (수정)
  - `regression-tests/verify-issue-11.sh` (신규)
- **계획 대비 변경 사항**:
  1. `kmmlu_redux`/`humaneval`은 **표준 집합을 준비하는 수단이 같은 경로로 없으므로** 이번
     단계에서는 eval 대상에서 제외했다(기본 벤치 = 표준 GSM8K 단일). 표준 KMMLU/HumanEval을
     추가하려면 36-prep-evalset에 같은 방식으로 벤치 분기를 늘리면 된다.
  2. GSM8K_N(준비할 표본)과 EVAL_N(실제로 푸는 문제 수)을 분리했다. 전체 1319를 준비해
     두고도 smoke는 20문제만 풀어 파이프라인 점검 비용을 유지한다.
  3. 배치 생성이 어휘를 바꾸는 점(패딩/배치 모양에 따른 수치 차이)은 **실측으로 확인**했다.
     동일 20문제를 bs=1 / bs=8로 돌린 결과: bs=1 **19/20(0.95)**, bs=8 **20/20(1.00)** —
     1문제가 달라졌다(bs=8이 정답). 즉 배치는 "결과 동일"을 보장하지 않는다.
     → **프로토콜 고정**: base와 양자화본 모두 같은 배치 크기·같은 max_new_tokens로 재고
     (동일 프로토콜이면 비교가 공정하다). 허용폭 판정에는 표본 노이즈가 있으므로
     n=200 기준 2%p는 약 1σ 수준이라는 점을 문서에 명시한다.
  4. `37-eval.py`의 `_strip_assistant_completion`이 ChatML 마커만 보고 있어 실제 캘리브
     데이터(평문 `assistant\n` 형식)의 응답 꼬리를 못 잘라내고 있었다(정답 누출 위험).
     두 마커를 모두 처리하도록 고쳤다.
