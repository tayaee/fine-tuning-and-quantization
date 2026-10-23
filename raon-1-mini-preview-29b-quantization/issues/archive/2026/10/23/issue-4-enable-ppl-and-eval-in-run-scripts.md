# issue-4: `./run-{smoke,full}.sh`가 `48-measure-ppl` 및 `50-eval`(`54-score`)까지 완주하도록 메모리 격리 및 평가 백엔드 수정

agent-tier: local-ok

## 1. 배경 및 문제점

모델 빌드(smoke 및 full) 완료 후 `./run-smoke.sh`와 `./run-full.sh`가 `48-measure-ppl`과 `50-eval` (`54-score` 포함) 단계까지 안정적으로 완주하려면 다음 3가지 구조적 문제를 해결해야 한다.

### 문제 1: `48-measure-ppl.py` 단일 프로세스 순차 로드로 인한 통합 메모리(128GB) OOM 위험 및 타겟 필터링 부재
- 현재 `48-measure-ppl.py`는 단일 Python 프로세스 안에서 `bf16`과 4종의 양자화 모델(총 5개의 29B 모델)을 연속으로 `AutoModelForCausalLM.from_pretrained`로 로드한다.
- GB10(128GB 통합 메모리) 환경에서 이전 모델의 PyTorch/CUDA 캐시와 파편화된 메모리가 완전히 반환되지 않으면 두 번째 또는 세 번째 모델 로드 중 OOM이 발생한다.
- 또한 `--targets` 인자를 지원하지 않아 일부 타겟만 측정하거나 타겟별로 프로세스를 분리 실행할 수 없다.

### 문제 2: 업스트림 `vllm==0.31.0`의 `Raon1MiniForCausalLM` 미지원으로 인한 `50-eval.py` 실패
- KRAFTON의 `Raon-1-mini-preview`는 자체 커스텀 포크(`krafton-ai/vllm@raon-1-mini-preview`)를 사용하며, 현재 `.venv`에 설치된 업스트림 `vllm==0.31.0`에는 `Raon1MiniForCausalLM` 아키텍처가 등록되어 있지 않다.
- 또한 업스트림 vLLM의 범용 `TransformersMoEForCausalLM` 폴백도 `Raon1MiniConfig`가 `top_k` 대신 `num_experts_per_token` 필드명을 사용하는 점과 MLA 관련 `transformers>=5.15.0.dev0` 버전 체크 때문에 실패한다.
- 반면 `transformers`의 `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`는 `compressed-tensors` 후크를 통해 양자화 모델을 정상 로드하고 `model.generate()`를 수행할 수 있다. 따라서 `50-eval.py`가 `transformers` 생성 백엔드(또는 `auto` 폴백)를 지원해야 `./run-{smoke,full}.sh`에서 `50-eval` 및 `54-score`가 중단 없이 완료된다.

### 문제 3: `./run-{smoke,full}.sh` 실행 시 `TARGETS` 전파 및 GPU 가시성 보장
- 양자화 단계(`32`~`36`)를 CPU 전용(`CUDA_VISIBLE_DEVICES=""`)으로 실행한 경우, 같은 쉘 환경에서 이어지는 `48-ppl`, `50-eval`까지 GPU가 가려지면 평가 속도가 극단적으로 저하되거나 실패한다.
- 또한 `run-smoke.sh`와 `run-full.sh`에서 `48-measure-ppl.sh` 호출 시에도 `TARGETS` 환경변수가 일관되게 반영되어야 한다.

---

## 2. 요구사항

1. **`48-measure-ppl.py` 및 `48-measure-ppl.sh` 개선**:
   - `48-measure-ppl.py`에 `--targets` 인자(콤마 또는 공백 구분, 기본값은 환경변수 `TARGETS` 또는 `bf16,awq-w5,awq-w3,nvfp4-awq,nvfp4-gptq`)를 추가한다 (`bf16`이 누락된 경우에도 기존 캐시된 `bf16.json`이 있으면 읽고, 없으면 `bf16`을 포함하거나 지정된 타겟만 측정).
   - 각 타겟 측정 결과를 `outputs/ppl-<mode>/<target>.json`에 개별 저장하고, 각 모델 측정 직후 `del model; gc.collect(); torch.cuda.empty_cache()`를 호출한다.
   - 디렉터리 내 존재하는 `<target>.json` 파일들을 취합하여 `outputs/ppl-<mode>/summary.json`을 생성/갱신하는 집계 로직(`--summarize-only` 또는 매 실행 끝 자동 갱신)을 구현한다.
   - `48-measure-ppl.sh`는 각 타겟을 별도의 `uv run 48-measure-ppl.py --mode "$MODE" --targets "$t"` 서브프로세스로 실행하여 모델 간 OS 메모리를 100% 회수한 뒤 최종 `summary.json`을 생성한다.
2. **`50-eval.py`에 `transformers` 백엔드 및 `auto` 폴백 추가**:
   - `--backend {auto,vllm,transformers}` 인자(기본값: 환경변수 `RQ_EVAL_BACKEND` 또는 `auto`)를 추가한다.
   - `run_transformers(model_path: Path, prompts: list[str], max_tokens: int) -> list[str]` 함수를 구현한다:
     - `AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)` 및 `AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)`로 모델을 로드한다.
     - 각 프롬프트에 대해 greedy decoding (`do_sample=False`, `max_new_tokens=max_tokens`, `use_cache=True`)으로 생성하고 입력 프롬프트 이후의 신규 토큰만 디코딩하여 반환한다.
     - 완료 후 `del model; gc.collect(); torch.cuda.empty_cache()`를 수행한다.
   - `backend == "auto"`일 때: `vllm` 백엔드를 시도하되 `Raon1MiniForCausalLM` 미지원 예외 등이 발생하면 자동으로 `run_transformers`로 폴백한다.
3. **`./run-smoke.sh` 및 `./run-full.sh` 파이프라인 연동**:
   - `48-ppl` 및 `50-eval` 단계가 `TARGETS` 환경변수를 전달받아 실행되도록 정리한다.
   - 양자화 단계에서만 CPU를 강제하고 `48-ppl`/`50-eval`에서는 GPU를 사용할 수 있도록 `QUANT_CPU_ONLY=1` 옵션을 지원하거나, `48-measure-ppl.sh` / `50-eval.sh`가 실행될 때 정상적으로 완료되어 `outputs/ppl-<mode>/summary.json` 및 `outputs/eval-<mode>/score.json`을 생성하도록 보장한다.

---

## 3. 승인 기준

- [ ] `48-measure-ppl.py --mode smoke --targets nvfp4-awq`와 같이 단일 타겟 지정 실행이 동작하며 `outputs/ppl-smoke/nvfp4-awq.json` 및 `outputs/ppl-smoke/summary.json`을 정상 기록함
- [ ] `48-measure-ppl.sh`가 타겟별 서브프로세스로 `48-measure-ppl.py`를 호출하여 프로세스 간 메모리를 격리함
- [ ] `50-eval.py`가 `--backend transformers` 및 `--backend auto` 모드에서 업스트림 vLLM이 커스텀 아키텍처를 지원하지 않더라도 `transformers`로 폴백하여 벤치마크 응답 JSONL을 정상 생성함
- [ ] `./run-smoke.sh` 및 `./run-full.sh`가 `48-ppl` → `50-eval` (`54-score.sh` 포함) 단계까지 호출 체인이 끊기지 않고 연결됨

---

## 4. 검증

- 단위 테스트 (`tests/test_ppl_and_eval.py`):
  - 초소형 합성 토크나이저/모델 픽스처에서 `48-measure-ppl.py`의 타겟별 JSON 저장 및 `summary.json` 집계 검증
  - 초소형 합성 CausalLM 픽스처에서 `50-eval.py`의 `run_transformers` 및 `auto` 폴백이 벤치마크 JSONL(`kmmlu_redux.jsonl`, `gsm8k.jsonl`, `humaneval.jsonl`)을 생성하고 `54-score.py`가 `score.json`을 산출하는지 검증
- 회귀 스크립트 (`regression-tests/verify-issue-4.sh`) 작성 및 실행

---

## 5. 의존성 / 관계

- 선행 이슈: `issue-1` (`50-eval.sh`, `50-eval.py` 기본 수정), `issue-2`, `issue-3`
- 후속 이슈: `issue-5` (`48-measure-ppl`의 `summary.json`과 `54-score`의 `score.json`을 읽어 모델 카드 `README.md`에 기입함)

---

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-23T09:16:35-04:00
- **변경 파일 목록**:
  - `48-measure-ppl.py` (수정): `parse_targets`(콤마/공백/`$TARGETS`/`all`/`bf16`→`base` 별칭) + `MODEL_DIR`/`OUTDIR` + `model_exists` + `release_memory`(gc + `cuda.empty_cache` + `ipc_collect`) + `summarize`(캐시 집계, 모델 미로드) + `measure_targets` + `--summarize-only` + `--targets`
  - `48-measure-ppl.sh` (수정): 타겟마다 `48-measure-ppl.py`를 별도 프로세스로 실행 → 마지막에 `--summarize-only` 1회. `all`은 `PPL_ALL_TARGETS`로 펼쳐 5개 프로세스로 쪼갠다
  - `50-eval.py` (수정): `run_vllm`/`run_transformers`/`generate`/`unsupported_vllm_error`/`is_unsupported_vllm_error`/`release_memory` + `--backend {auto,vllm,transformers}`(`$RQ_EVAL_BACKEND`) + `--max-tokens` + `--benchmarks` + 벤치별 `outputs/eval-<mode>/bench/<bench>/<target>.jsonl`
  - `50-eval.sh` (수정): `RQ_EVAL_BACKEND`(기본 auto) 전파 + `all`을 `EVAL_ALL_TARGETS`로 펼쳐 타겟별 프로세스로 실행
  - `54-score.py` (수정): `score_answer`(kmmlu_redux/gsm8k/humaneval) + `benchmarks` 블록을 `score.json`에 추가 (완결율·avg_len·think_rate는 유지)
  - `run-smoke.sh`, `run-full.sh` (수정): `_targets`/`_eval_targets`(base 선행) 정규화 후 ppl·eval·prep에 전달, `QUANT_CPU_ONLY=1`(양자화만 `env CUDA_VISIBLE_DEVICES=` 로 GPU 숨김)
  - `tests/test_ppl_and_eval.py` (신규): 23건 — 타겟 파싱/집계/집계모드, 백엔드 선택·폴백 판정, 채점 함수, 그리고 issue-4 §4가 요구한 **종단 간 체인**(`test_run_transformers_to_score_json_chain`: 합성 CausalLM → `run_transformers` → 벤치 JSONL 3종 → `54-score` → `score.json` 정확도, `test_auto_fallback_to_transformers_chain`: auto 폴백 경로가 같은 JSONL을 만듦)
  - `regression-tests/verify-issue-4.sh` (신규)
  - `cheatsheet.md`, `PROCEDURE.md` (수정)
  - `issues/issue-4__agent-stats.json` (신규)
- **계획 대비 변경 사항**:
  1. 타겟 별칭: 이슈는 기본값을 `bf16`라고 쓰지만 기존 산출물 파일명은 `base.json`이고 50-eval/54-score 관례도 `base`라, `base`를 정식 이름으로 두고 `bf16`을 **입력 별칭**으로만 받는다(`TARGET_ALIASES`). `parse_targets("bf16,awq-w5")` → `["base", "awq-w5"]`. 없으면 캐시된 `base.json`이 깨져 `summary.json`의 기준선이 사라진다.
  2. `--targets` 기본값은 `all`(=전 5타겟) 대신 `$TARGETS`를 먼저 보고, 없으면 전 타겟. `48-measure-ppl.sh`가 이 값을 넘겨준다.
  3. `run-{smoke,full}.sh`에서 PPL/eval 타겟은 `base,$_targets`로 고정했다. base를 빼면 `summary.json`/`score.json`에 기준선이 없어 issue-5의 모델 카드 비교표를 만들 수 없다.
  4. vLLM 폴백 판정은 예외 문자열 매칭(`Raon1MiniForCausalLM` / `unsupported architecture` / `not supported` / `unrecognized model` 등)이다. **CUDA OOM·NCCL 같은 진짜 실패는 폴백하지 않고 그대로 올린다** (auto 모드에서도). 판단 근거를 `is_unsupported_vllm_error` 한 곳에 모았다.
  5. `50-eval.py`가 생성 응답을 `outputs/eval-<mode>/<target>.jsonl`(기존 관례, 54-score가 집계)에 쓰고, 벤치별 상세는 `outputs/eval-<mode>/bench/<bench>/<target>.jsonl`에 쓴다. 54-score는 최상위만 glob하므로 중복 집계되지 않는다.
  6. `54-score.py`의 `score.json`에 `benchmarks.<bench>.acc`를 추가했다. 이슈-5가 이 값을 읽어 모델 카드 표를 만들어야 하며, 기존 `done_rate`/`avg_len`/`think_rate`는 하위 호환을 위해 그대로 둔다. humaneval은 gold의 `test` 하네스를 격리 exec해 통과 여부를 판정하고, kmmlu_redux은 명시 마커(`정답은 B`/`답: C`)만 신뢰한다.
  7. `QUANT_CPU_ONLY=1`은 부모 셸의 `CUDA_VISIBLE_DEVICES`를 바꾸지 않고 **양자화 자식 프로세스에만** `env CUDA_VISIBLE_DEVICES=` 를 붙인다. 셸을 바꾸면 뒤따르는 PPL/eval까지 GPU를 잃는 문제(요구사항 3)가 그대로 재현된다.
  8. `all`을 명시 목록으로 펼쳤다 — **초기 구현 결함 수정**. 첫 구현은 `TARGETS=all`(스크립트를 직접 실행할 때의 기본값)을 타겟 1개로 그대로 넘겨, 29B 모델 5개를 한 프로세스에서 로드했다. 이는 승인 기준 2(프로세스 격리)가 막으려는 바로 그 OOM 시나리오였다. `48-measure-ppl.sh`/`50-eval.sh` 모두 `all`을 5개 명시 목록으로 펼치게 고쳤고, `verify-issue-4.sh`에 실제 루프가 몇 프로세스를 띄우는지 계산하는 검사를 넣었다(결함을 되살리면 FAIL — negative test로 확인).
  9. `54-score.py`가 `SHARED`를 직접 이어붙이던 걸 `OUTDIR(mode)` 헬퍼로 통일했다. eval/ppl 세 스크립트가 경로 관례를 한 곳에 모으는 편이 낫고, 테스트도 모듈 함수만 갈아끼우면 되게 된다.
  10. 이 저장소에는 `step_enabled` 헬퍼가 없고 각 `run-*.sh`에 지역 `want_step`가 있으므로, 이슈 2.3의 `step_enabled 56-prep` 대신 기존 관례의 `want_step prep && rq_step "56-prep" ...` 로 통일했다.
- **검증 결과**:
  - `tests/test_ppl_and_eval.py`: 25 passed (종단 간 체인 2건 및 vLLM ModelRegistry 사전 체크·벤치별 슬라이싱 포함)
  - `regression-tests/verify-issue-4.sh`: 통과(rc=0). `verify-issue-1/2/3/5.sh`도 모두 통과(rc=0)
  - `all` 펼침 검사 negative test: 결함을 되살리면 `FAIL: 48-measure-ppl.sh가 'all'을 단일 타겟으로 통과시킴`으로 검출됨(rc=1) 확인
  - `ruff check`: All checks passed. `pyright`(변경 스크립트): 0 errors. `compileall`·`--help`·`bash -n` 전부 통과
  - **미실행**: 실제 29B 모델 로드/PPL/eval(GB10 128GB 통합 메모리 수 시간). 검증은 합성 픽스처 seam(타겟 파싱·집계·백엔드 선택·폴백·채점)에 한정했다. 다음 실행에서 확인할 것: (a) vLLM 미지원 예외 문자열이 실제 vllm 0.31.0 출력과 `is_unsupported_vllm_error`의 힌트 목록에 걸리는지, (b) `outputs/eval-<mode>/prompts-<bench>.jsonl`이 없으면 calib 텍스트 fallback으로 3개 벤치에 순환 배치되는 상태로 채점이 0이 되는지(정식 벤치 데이터를 두면 해결됨).
