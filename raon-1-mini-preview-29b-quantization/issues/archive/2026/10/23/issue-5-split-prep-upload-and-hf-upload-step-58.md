# issue-5: 업로드 전 모델 메타데이터/README 생성 스크립트(`56-prep-upload`) 신설 및 `58-upload-hf` 업로드 전담 분리

agent-tier: local-ok

## 1. 배경 및 문제점

1. **업로드 전 모델 디렉터리 준비(`README.md` 및 아티팩트 검증) 단계 부재**:
   - 현재 양자화 완료 직후의 모델 디렉터리(`$RQ_MODELS/raon-<target>-<mode>`)에는 HuggingFace 모델 카드(`README.md`)가 없으며, 필수 파일(`chat_template.jinja`, `modeling_raon1_mini.py`, `configuration_raon1_mini.py`, 토크나이저, 압축 포맷 등)이 모두 정상 준비되었는지 사전 검증하는 단계가 없다.
2. **`58-upload-hf.py`의 책임 혼재 및 모델 카드 정보 부족**:
   - 기존 `58-upload-hf.py`는 업로드 직전에 즉석에서 매우 빈약한 16줄짜리 `README.md`를 생성한 뒤 곧바로 `HfApi.upload_folder`를 호출한다.
   - 또한 기존 템플릿에는 HuggingFace Hub 표준 YAML frontmatter (`license: apache-2.0`, `base_model`, `base_model_relation: quantized`, `library_name: transformers`, `tags` 등), 타겟별 상세 양자화 스펙(그룹 크기, 비트폭, 보호 레이어, 캘리브레이션 조건), `transformers` / `vLLM` 실행 예제 코드, 라이선스 고지가 누락되어 있다.
3. **사용자 요구사항**:
   - `58-upload-hf` 실행 전에 필요한 모든 정보를 모델 폴더에 채워 넣고 검증하는 별도 스크립트(`56-prep-upload.py` / `56-prep-upload.sh`)를 추가하고, `58-upload-hf.py` / `58-upload-hf.sh`는 이미 준비된 폴더를 HuggingFace Hub에 업로드하는 역할에만 전념하도록 분리한다.

---

## 2. 요구사항

### 2.1. `56-prep-upload.py` 및 `56-prep-upload.sh` 신설
1. **CLI 인터페이스**:
   - `56-prep-upload.py --mode {smoke,full} [--targets awq-w5,awq-w3,nvfp4-awq,nvfp4-gptq] [--allow-missing-metrics]`
   - `56-prep-upload.sh [smoke|full]`: `config/common.env`를 로드하고 `56-prep-upload.py`를 호출한다.
2. **모델 디렉터리 무결성 검증 및 누락 보조 파일 보충**:
   - 각 타겟 디렉터리(`$RQ_MODELS/raon-<target>-<mode>`)에 대해:
     - `quant_common.verify_compressed_output(model_dir, expected_format)`을 호출하여 `"format": "dense"`가 아닌 정상 패킹 포맷(`pack-quantized` 또는 `nvfp4-pack-quantized`)인지 확인한다.
     - `chat_template.jinja`, `configuration_raon1_mini.py`, `modeling_raon1_mini.py`가 누락된 경우 `$RQ_MODELS/raon-bf16`에서 복사하여 보충한다.
     - `config.json`, `tokenizer.json`, `tokenizer_config.json` 존재 여부를 확인한다.
3. **벤치마크 지표 로드 및 HuggingFace 모델 카드(`README.md`) 생성**:
   - `outputs/ppl-<mode>/summary.json`과 `outputs/eval-<mode>/score.json`에서 베이스 모델(`bf16`) 및 각 양자화 타겟의 측정값(`WikiText-2 PPL`, `KMMLU-redux`, `GSM8K`, `HumanEval`)을 읽는다.
   - 각 모델 폴더에 풍부한 메타데이터를 담은 `README.md`를 생성한다:
     - **YAML Frontmatter**: `license: apache-2.0`, `base_model: KRAFTON/Raon-1-mini-preview`, `base_model_relation: quantized`, `library_name: transformers`, `pipeline_tag: text-generation`, `language: [ko, en]`, `tags: [raon, moe, compressed-tensors, llm-compressor, <target-tags>]` (주의: 원본 KRAFTON 레포의 `extra_gated_*` 필드는 절대 포함하지 않음).
     - **양자화 상세 사양 표**: 원본 모델 리비전(`da4cf162c8613993f2189ebb438a5f9a23f3e5ca`), 양자화 알고리즘(AWQ / GPTQ), 가중치/활성화 정밀도(W5A16, W3A16, NVFP4 W4A4, NVFP4 W4A16), 그룹 크기(`group_size=128` 또는 `16`), 양자화 제외 레이어(`lm_head`, `mlp.gate`, `self_attn`, `shared_experts`), 캘리브레이션 데이터셋(256 samples, seq_len 2048), 체크포인트 총 용량(GB).
     - **성능 비교 표**: `bf16` 대비 4종 양자화 모델의 WikiText-2 PPL 및 KMMLU-redux / GSM8K / HumanEval 정확도 비교표 (현재 모델 행 강조).
     - **실행 예제**: `transformers` (`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`) 및 `vLLM` (`--trust-remote-code --quantization compressed-tensors`) 사용법.
     - **라이선스 및 출처**: Apache 2.0 라이선스 및 KRAFTON 원본 모델 출처 명시.
   - 준비 완료 요약 매니페스트를 `outputs/prep-upload-<mode>.json`에 기록한다.

### 2.2. `58-upload-hf.py` 및 `58-upload-hf.sh` 업로드 전담화
1. **메타데이터 생성 로직 제거**:
   - `58-upload-hf.py`에서 `MODEL_CARD_TMPL`, `ppl`/`score` JSON 파싱, `README.md` 작성 코드를 모두 제거한다.
2. **사전 준비 상태 확인 후 업로드만 수행**:
   - `58-upload-hf.py`는 대상 모델 디렉터리에 `config.json`과 `README.md`(비어 있지 않은 파일)가 이미 존재하는지 확인한다. 만약 `README.md`가 없으면 `"README.md missing in <model_dir>; run ./56-prep-upload.sh first"` 오류를 내고 중단한다.
   - `--targets` 인자 및 `--dry-run` 플래그를 지원하여, `--dry-run` 시에는 준비 상태 검증과 업로드 대상 레포지토리 ID(`tayaee/Raon-1-mini-preview-<target>[-smoke]`)만 출력하고 종료한다.
   - 실제 업로드 시에는 `HfApi(token=token)`으로 `create_repo(..., exist_ok=True)` 및 `upload_folder(folder_path=str(model_dir), repo_id=repo, commit_message=...)`만 수행한다.

### 2.3. `run-smoke.sh` 및 `run-full.sh` 연동
- `run-smoke.sh`와 `run-full.sh`에서 `50-eval` 단계 다음, `58-upload` 단계 이전에 `step_enabled 56-prep && TARGETS="$TARGETS" ./56-prep-upload.sh <mode>`를 추가한다.

---

## 3. 승인 기준

- [ ] `56-prep-upload.py` 및 `56-prep-upload.sh`(실행 권한 포함)가 존재하고, 모델 디렉터리 검증 및 `README.md` 생성을 수행함
- [ ] 생성된 `README.md`에 YAML frontmatter(`license: apache-2.0`, `base_model: KRAFTON/Raon-1-mini-preview`, `extra_gated_*` 미포함), 양자화 사양, 벤치마크 표, `transformers`/`vLLM` 사용법이 모두 포함됨
- [ ] `58-upload-hf.py`에 `README.md` 생성 로직이 없으며, `README.md`가 없는 모델 폴더에 대해 즉시 오류를 발생시킴
- [ ] `58-upload-hf.py --dry-run`이 네트워크 호출 없이 준비된 모델 폴더의 업로드 가능 여부를 검증함
- [ ] `run-smoke.sh` 및 `run-full.sh`에 `56-prep` 단계가 포함됨

---

## 4. 검증

- 단위 테스트 (`tests/test_prep_and_upload.py`):
  - 가짜 압축 모델 픽스처와 샘플 `summary.json`/`score.json`을 사용해 `56-prep-upload.py` 실행 시 `README.md`와 `outputs/prep-upload-<mode>.json`이 정상 생성되는지 검증
  - `README.md`가 없는 폴더에서 `58-upload-hf.py`가 실패하는지, `56-prep-upload.py` 실행 후 `58-upload-hf.py --dry-run`이 성공하는지 검증
- 회귀 스크립트 (`regression-tests/verify-issue-5.sh`) 작성 및 실행

---

## 5. 의존성 / 관계

- 선행 이슈: `issue-3` (`verify_compressed_output`), `issue-4` (`48-measure-ppl` 및 `50-eval`/`54-score` 출력 포맷)

---

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-23T09:16:35-04:00
- **변경 파일 목록**:
  - `56-prep-upload.py` (신규): `SPECS`(타겟별 on-disk 포맷/비트폭/그룹 크기/제외 레이어) + `prepare_target`(검증·보충·카드 생성, 실패는 예외 대신 `ok=False`+`errors` 보고서) + `render_readme`(YAML frontmatter + 양자화 사양 표 + bf16 대비 벤치마크 표 + transformers/vLLM 예제 + 라이선스) + `_copy_missing_custom_code` + `_ppl_summary`/`_eval_scores`/`_dir_size_gb` + `BF16_DIR` + `run`(전 타겟 + `outputs/prep-upload-<mode>.json` 매니페스트) + `--targets`/`--allow-missing-metrics`/`--mode`
  - `56-prep-upload.sh` (신규, 실행 권한 포함)
  - `58-upload-hf.py` (수정): `MODEL_CARD_TMPL`/ppl·score 파싱/README 작성 코드 **전부 삭제** → `check_ready`(config.json + 비어 있지 않은 README.md 확인, 없으면 `56-prep-upload.sh first` 오류) + `--dry-run`(네트워크 0회 호출) + `HfApi.create_repo`/`upload_folder`
  - `58-upload-hf.sh` (수정): `DRY_RUN=1` 지원
  - `run-smoke.sh`, `run-full.sh` (수정): `want_step prep && rq_step "56-prep" ./56-prep-upload.sh "$MODE" "$_targets"` 추가, validate 목록에 `56-prep-upload` 포함(17 scripts)
  - `tests/test_prep_and_upload.py` (신규): 16건 — dense 폴백 거부 / custom_code 보충 / 토크나이저 누락 거부 / 카드 섹션·금지 필드 / 지표 필수·선택 / 매니페스트 / README 없을 때 오류 / 빈 README 거부 / dry-run 무네트워크 / smoke 거부 / 모델 없음 SKIP, 그리고 issue-5 §4가 요구한 **순서 검증**(`test_prep_then_dry_run_succeeds`: prep 전 dry-run은 실패 → prep 실행 → dry-run 성공, 그 사이 네트워크 호출 0회 보장)
  - `regression-tests/verify-issue-5.sh` (신규)
  - `cheatsheet.md`, `PROCEDURE.md` (수정)
  - `issues/issue-5__agent-stats.json` (신규)
- **계획 대비 변경 사항**:
  1. 보충(복사) 단계를 `verify_compressed_output` **앞에** 뒀다. `quant_common.verify_compressed_output`(issue-3)도 `CUSTOM_CODE_FILES` 존재를 강제하므로 순서를 반대로 하면 “custom code 누락”으로 먼저 죽어 복구 자체가 불가능하다.
  2. README 카드에 이슈가 예시한 `group_size=128` 대신 실제 레시피 값(`awq-w3/w5`는 `group_size=64, symmetric=False`, `nvfp4-*`는 `block_size=16`)을 `SPECS`에서 읽어 넣었다. 카드에 박히는 값은 파이프라인이 실제로 만든 값이어야 한다. 캘리브레이션 스펙(256/2048)은 `full` 기준이며 `smoke`에서는 config/common.env의 `CALIB_N`/`SEQ_LEN`으로 `$CALIB_N`/`$SEQ_LEN`를 넘길 수 있게 했다(기본 256/2048).
  3. `extra_gated_*` 금지는 **frontmatter 구간만** 검사한다(`readme.split("---", 2)[1]`). 본문에 KRAFTON 저장소가 게이트되어 있다는 설명 문장은 남긴다.
  4. `config.json`이 없는 타겟은 `58-upload-hf.py`에서 `SKIP` 후 다음 타겟을 계속 처리한다(기존 동작 유지). `README.md`가 없거나 빈 파일일 때만 즉시 `SystemExit`한다 — 허브에 빈 카드가 올라가는 게 업로드 실패보다 나쁘다.
  5. 실행 순서 기준: 카드의 “현재 모델” 강조는 bf16 대비 표에서 해당 타겟 행을 굵게 하는 것으로 구현했고, 성능 비교 표는 `outputs/ppl-<mode>/summary.json`의 **전체** 타겟 행을 읽어 만들어 내기 때문에 지표가 없는 타겟은 `—`로 표시된다.
  6. `_copy_missing_custom_code`가 보충 원본 경로를 `MODELS_DIR('*')`로 뽑았다. `MODELS_DIR`은 `mode`를 인자로 받는 계약인데 `raon-bf16`은 모드 무관이라 가짜 모드 문자열을 끼워 넣은 것이라, `BF16_DIR()` 전용 헬퍼로 뺐다. `MODELS_DIR`의 시그니처가 나중에 바뀌면 그대로 깨질 뻔한 코드였다.
  7. 이 저장소에는 `step_enabled` 헬퍼가 없고 각 `run-*.sh`에 지역 `want_step`가 있으므로, 이슈 2.3의 `step_enabled 56-prep` 대신 기존 관례의 `want_step prep` 로 통일했다.
- **검증 결과**:
  - `tests/test_prep_and_upload.py`: 17 passed (dense 폴백 거부 / custom_code 보충 / 토크나이저 누락 거부 / 카드 섹션·금지 필드 / 지표 필수·선택 / 매니페스트 / README 없을 때 오류 / 빈 README 거부 / dry-run 무네트워크 / smoke 거부 / 모델 없음 SKIP / prep→dry-run 순서 / 공백·환경변수 TARGETS 파싱)
  - `regression-tests/verify-issue-5.sh`: 통과(rc=0). `verify-issue-1/2/3/4.sh`도 모두 통과(rc=0)
  - `ruff check`: All checks passed. `pyright`(변경 스크립트): 0 errors. `compileall`·`--help`·`bash -n` 전부 통과
  - **미실행**: 실제 모델 디렉터리에 대한 `56-prep-upload.py` 실행과 Hub 업로드는 하지 않았다(네트워크·토큰 필요). 검증은 합성 압축 체크포인트 픽스처에 한정했다. 실제 업로드 전 `DRY_RUN=1 ./58-upload-hf.sh full`로 준비 상태를 확인할 것.
