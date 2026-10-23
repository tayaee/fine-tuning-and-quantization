# issue-1: `60-serve-vllm.sh` 및 `50-eval.{sh,py}` 스크립트 인자·양자화 방식 매핑 버그 수정

agent-tier: local-ok

## 1. 배경 및 문제점

양자화 파이프라인의 서빙(`60-serve-vllm.sh`) 및 평가(`50-eval.sh`, `50-eval.py`) 스크립트에 외부 의존성 없이 즉시 수정 가능한 설정/쉘 파싱 버그가 존재한다.

1. **`60-serve-vllm.sh`의 `awq-w5` / `awq-w3` 양자화 방식 오지정**:
   - `32-quant-awq-w5.py` 및 `33-quant-awq-w3.py`는 `llmcompressor`를 사용하여 모델을 저장하므로, 결과 `config.json`의 `quantization_config.quant_method`는 `"compressed-tensors"`이다.
   - 그러나 `60-serve-vllm.sh` (28~29행)와 `50-eval.py` (16행 `QUANT_OF`)에서 `awq-w5`, `awq-w3`를 `"gptq"`로 매핑하고 있어 vLLM 로드 시 `quant_method` 불일치 오류가 발생한다.
2. **`60-serve-vllm.sh`의 `-smoke` 접미사 중복 버그**:
   - `60-serve-vllm.sh` 36~37행에 `[ "$MODE" = smoke ] && REPO="$REPO-smoke"`가 두 줄 연속 중복 작성되어 있어, `MODE=smoke` 실행 시 모델 경로가 `raon-<target>-smoke-smoke`로 잘못 조합된다.
3. **`50-eval.sh`의 콤마 구분 `TARGETS` 미분리 버그**:
   - `run-smoke.sh` 및 `run-full.sh`는 `TARGETS="nvfp4-awq,nvfp4-gptq"` 형태의 콤마 구분 문자열을 `$(echo "$TARGETS" | tr ',' ' ')`로 순회하지만, `50-eval.sh` (17행)는 `for t in $TARGETS; do`로 순회하여 콤마 구분 `TARGETS`가 전달될 때 단일 토큰(`nvfp4-awq,nvfp4-gptq`)으로 처리되어 실패한다.

---

## 2. 요구사항

1. **`60-serve-vllm.sh` 수정**:
   - `awq-w5` 및 `awq-w3` 케이스의 `QUANT="gptq"`를 `QUANT="compressed-tensors"`로 변경한다.
   - 중복된 `[ "$MODE" = smoke ] && REPO="$REPO-smoke"` 줄을 1회만 남기고 제거한다.
2. **`50-eval.py` 수정**:
   - `QUANT_OF` 딕셔너리에서 `"awq-w5"`와 `"awq-w3"`의 값을 `"gptq"`에서 `"compressed-tensors"`로 변경한다.
3. **`50-eval.sh` 수정**:
   - `for t in $TARGETS; do` 루프를 `for t in $(echo "$TARGETS" | tr ',' ' '); do`로 변경하여 공백 구분과 콤마 구분 `TARGETS` 모두 정상 동작하도록 한다.

---

## 3. 승인 기준

- [ ] `60-serve-vllm.sh`에서 `awq-w5`, `awq-w3`, `nvfp4-awq`, `nvfp4-gptq` 모두 `QUANT="compressed-tensors"`로 설정됨
- [ ] `60-serve-vllm.sh`에서 `[ "$MODE" = smoke ] && REPO="$REPO-smoke"`가 정확히 1번만 실행되어 `MODE=smoke`일 때 `raon-<target>-smoke` 경로를 가리킴
- [ ] `50-eval.py`의 `QUANT_OF["awq-w5"]` 및 `QUANT_OF["awq-w3"]` 값이 `"compressed-tensors"`임
- [ ] `50-eval.sh`가 `TARGETS="awq-w5,awq-w3"`와 같은 콤마 구분 입력을 개별 타겟으로 분리하여 순회함

---

## 4. 검증

- `regression-tests/verify-issue-1.sh` 작성 및 실행:
  - `60-serve-vllm.sh` 내 `REPO-smoke` 할당 횟수가 1회인지 및 `QUANT="gptq"`가 제거되었는지 확인
  - `50-eval.py`를 import하거나 AST/실행으로 `QUANT_OF` 매핑 검증
  - `50-eval.sh` 내 `tr ',' ' '` 적용 여부 및 문법 검사(`bash -n`) 통과 확인

---

## 5. 의존성 / 관계

- 없음 (독립 수행 가능, 가장 먼저 처리할 단순 스크립트 버그 수정)

---

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-23T06:05:00-04:00
- **변경 파일 목록**: `60-serve-vllm.sh`, `50-eval.py`, `50-eval.sh`, `regression-tests/verify-issue-1.sh` (신규), `issues/issue-1__agent-stats.json` (신규)
- **계획 대비 변경 사항**: 없음 (요구사항 3건 그대로 적용)
- **검증 결과**: `regression-tests/verify-issue-1.sh` 통과 (8/8 OK), `bash -n` 양 스크립트 통과, `compileall` 통과. ruff `FURB122` 1건은 수정 전부터 존재하는 범위 외 이슈로 미수정. pyright 미설치로 미실행. 기존 회귀 스크립트 없음 (본 이슈가 첫 번째).
