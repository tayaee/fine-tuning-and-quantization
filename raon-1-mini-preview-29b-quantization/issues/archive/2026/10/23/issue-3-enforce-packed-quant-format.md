# issue-3: PTQ 스크립트(`32`, `33`, `35`, `36`)의 패킹 압축 포맷(`pack-quantized` / `nvfp4-pack-quantized`) 강제 검증 추가

agent-tier: local-ok

## 1. 배경 및 문제점

- `llmcompressor` (`oneshot(..., save_compressed=True)`) 실행 시 커스텀 모듈 타입 불일치 등의 이유로 압축기가 `can_compress == False`를 반환하면, 에러를 발생시키지 않고 조용히 `DenseCompressor`(`"format": "dense"`)로 폴백한다.
- 이로 인해 `32-quant-awq-w5.py`, `33-quant-awq-w3.py`, `35-quant-nvfp4-awq.py`, `36-quant-nvfp4-gptq.py`가 실제로는 가중치를 전혀 패킹하지 않은 59GB BF16 체크포인트(`"format": "dense"`, 미제거 `weight_zero_point` 포함)를 정상 완료된 것처럼 `commit_dir(tmp, out)`로 커밋하는 문제가 발생했다.
- 향후 동일한 폴백이나 불완전한 저장이 발생하더라도 `commit_dir` 이전에 즉시 감지하여 실패 처리하는 공통 검증 함수가 필요하다.

---

## 2. 요구사항

1. **`quant_common.py`에 `verify_compressed_output(model_dir: Path, expected_format: str)` 구현**:
   - `model_dir / "config.json"`을 읽어 `quantization_config`가 존재하는지, `quantization_config.get("format") == expected_format`인지 확인한다 (`"dense"`이거나 불일치하면 `RuntimeError` 발생).
   - `model_dir` 내 `.safetensors` 파일들의 텐서 헤더(`safetensors.safe_open`)를 스캔하여 다음을 검증한다:
     - `weight_packed` 텐서가 최소 1개 이상 존재할 것.
     - `expected_format == "nvfp4-pack-quantized"`인 경우: `weight_packed`의 dtype이 `U8`이고, `weight_scale`(`F8_E4M3`) 및 `weight_global_scale`(`F32`) 텐서가 존재하며, `weight_zero_point` 텐서가 남아 있지 않을 것.
     - `expected_format == "pack-quantized"`인 경우: `weight_packed`의 dtype이 `I32`이고, `weight_scale` 텐서가 존재하며, 대칭 양자화(symmetric) 설정 시 불필요한 `weight_zero_point` 텐서가 남아 있지 않을 것.
   - `model_dir`에 커스텀 모델 코드(`configuration_raon1_mini.py`, `modeling_raon1_mini.py`)가 복사되어 존재하는지 확인한다.
2. **네 개의 PTQ 스크립트에 검증 호출 연동**:
   - `32-quant-awq-w5.py`: `save_custom_code(tmp)` 직후, `commit_dir(tmp, out)` 직전에 `verify_compressed_output(tmp, "pack-quantized")` 호출.
   - `33-quant-awq-w3.py`: `save_custom_code(tmp)` 직후, `commit_dir(tmp, out)` 직전에 `verify_compressed_output(tmp, "pack-quantized")` 호출.
   - `35-quant-nvfp4-awq.py`: `save_custom_code(tmp)` 직후, `commit_dir(tmp, out)` 직전에 `verify_compressed_output(tmp, "nvfp4-pack-quantized")` 호출.
   - `36-quant-nvfp4-gptq.py`: `save_custom_code(tmp)` 직후, `commit_dir(tmp, out)` 직전에 `verify_compressed_output(tmp, "nvfp4-pack-quantized")` 호출.

---

## 3. 승인 기준

- [ ] `quant_common.verify_compressed_output`이 `"format": "dense"`인 디렉터리에 대해 `RuntimeError`를 발생시킴
- [ ] `quant_common.verify_compressed_output`이 `weight_packed` 텐서가 없거나 잘못된 dtype일 때 `RuntimeError`를 발생시킴
- [ ] 정상적인 `pack-quantized` (`I32` `weight_packed`) 및 `nvfp4-pack-quantized` (`U8` `weight_packed`, `F8_E4M3` `weight_scale`, `F32` `weight_global_scale`) 픽스처에 대해 검증을 통과함
- [ ] `32-quant-awq-w5.py`, `33-quant-awq-w3.py`, `35-quant-nvfp4-awq.py`, `36-quant-nvfp4-gptq.py` 모두 `commit_dir` 호출 전에 `verify_compressed_output`을 호출함

---

## 4. 검증

- 단위 테스트 (`tests/test_verify_compressed_output.py`):
  - `"format": "dense"` 가짜 체크포인트 거부 테스트
  - 정상 `nvfp4-pack-quantized` 및 `pack-quantized` 최소 `.safetensors` 픽스처 통과 테스트
  - 커스텀 코드 파일 누락 시 거부 테스트
- 회귀 스크립트 (`regression-tests/verify-issue-3.sh`) 작성 및 실행

---

## 5. 의존성 / 관계

- 선행 이슈: `issue-2` (`30-dequant-mxfp8.py`의 `nn.Linear` shim이 적용되어야 실제 PTQ 실행 시 압축 포맷이 생성됨)
- 후속 이슈: `issue-5` (`56-prep-upload.py`에서도 동일한 `verify_compressed_output`을 재사용하여 업로드 전 모델을 검증함)

---

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-23T06:24:00-04:00
- **변경 파일 목록**:
  - `quant_common.py` (수정): `verify_compressed_output(model_dir, expected_format)` 추가 + `_PackedContract`/`PACKED_CONTRACTS`/`_scan_shard_dtypes`/`_weights_symmetric` 헬퍼
  - `32-quant-awq-w5.py`, `33-quant-awq-w3.py` (수정): `save_custom_code` 직후 `verify_compressed_output(tmp, PACK_QUANTIZED)` 호출
  - `35-quant-nvfp4-awq.py`, `36-quant-nvfp4-gptq.py` (수정): `save_custom_code` 직후 `verify_compressed_output(tmp, NVFP4_PACK_QUANTIZED)` 호출
  - `tests/test_verify_compressed_output.py` (신규): 단위 테스트 20건 (pytest/단독 실행 모두 지원)
  - `regression-tests/verify-issue-3.sh` (신규)
  - `issues/issue-3__agent-stats.json` (신규)
- **계획 대비 변경 사항**:
  1. 이슈 본문에는 문자열 리터럴(`"pack-quantized"` / `"nvfp4-pack-quantized"`)로 호출하도록 적혀 있으나, 오타를 원천 차단하려고 `quant_common`의 `PACK_QUANTIZED` / `NVFP4_PACK_QUANTIZED` 상수를 쓰고 인자를 상수로 넘겼다(값은 동일).
  2. 대칭/비대칭 판정을 위해 `quantization_config.config_groups.*.weights.symmetric`(기본 `True`)를 읽어, 대칭일 때만 `weight_zero_point` 잔존을 거부한다. W3/W5 레시피는 `symmetric=False`라 패킹된 zero_point가 정상 잔존하므로 통과시킨다(요구사항의 "대칭 양자화 시" 조건 그대로).
  3. 검증 대상을 `*.safetensors` 전체 shard로 확장하고 헤더만 읽는다(`get_slice().get_dtype()`). 59GB 체크포인트에서 텐서 본문을 읽지 않는다. 멀티 shard 픽스처 테스트로 검증.
  4. 반환값 `{"format", "packed", "shards"}`를 두고 성공 시 한 줄 로그를 출력한다(기존 헬퍼 스타일).
  5. 검증 실패는 32/33/35의 폴백 체인 안에서 `RuntimeError`로 발생해 다음 레시피를 시도하고, 최종 실패 시 `SystemExit("all recipes failed, last: RuntimeError: ...")`로 표면화된다. 기본 설정은 레시피가 1개라 즉시 실패한다.
- **검증 결과**:
  - `tests/test_verify_compressed_output.py`: 20 passed (pytest `python -m pytest tests/ -q` 및 단독 실행 모두)
  - `regression-tests/verify-issue-3.sh`: 통과(6개 항목 OK, rc=0)
  - 기존 회귀 스크립트 `regression-tests/verify-issue-1.sh`: 통과(rc=0)
  - `ruff check` (변경 파일 6개): All checks passed (`TRY004`, `SIM118`는 계약상 필요해 사유 주석과 함께 `# noqa` 처리)
  - `pyright`: 변경 코드 신규 오류 0건. 기존 오류만 잔존(`oneshot` private import, `config_groups` dict, `actorder="static"`, `resolve_trust_remote_code` private import)
  - `python -m compileall` 통과, PTQ 4종 `--help` 통과(run-smoke `validate` 스텝과 동일 경로)
  - 참고: ruff/pytest/pyright는 이 로컬 검증 편의를 위해 `.venv`에 `uv pip install`했다(`.venv/`는 gitignore 대상). 실행 환경에 이미 있던 툴이 아닌 임시 설치이며 커밋 대상 아님
