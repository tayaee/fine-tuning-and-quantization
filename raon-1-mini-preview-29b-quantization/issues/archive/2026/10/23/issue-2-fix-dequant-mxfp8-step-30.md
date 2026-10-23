# issue-2: `30-dequant-mxfp8.py` MXFP8 E8M0 스케일 디퀀타이즈 버그 및 `Raon1MiniRoutedLinear` 압축 호환성 수정

agent-tier: local-ok

## 1. 배경 및 문제점

4비트/3비트/5비트 PTQ(`32-quant-awq-w5`, `33-quant-awq-w3`, `35-quant-nvfp4-awq`, `36-quant-nvfp4-gptq`)의 공통 입력인 `models/raon-bf16`을 생성하는 `30-dequant-mxfp8.py`에 두 가지 치명적인 문제가 존재한다.

### 문제 1: MXFP8 `weight_scale_inv` (E8M0 지수 바이트) 오해석으로 인한 가중치 폭주
- 원본 모델 `KRAFTON/Raon-1-mini-preview` (rev `da4cf162c8613993f2189ebb438a5f9a23f3e5ca`)는 9,216개의 MoE expert 선형 레이어 가중치(`model.layers.*.mlp.experts.*.{gate,up,down}_proj.weight`)를 MXFP8 (`F8_E4M3`, 블록 크기 `(1, 32)`)로 저장하며, `weight_scale_inv` 텐서는 `uint8` 타입의 **IEEE 754 E8M0 지수 바이트**(`true_scale = 2^(byte - 127.0)`)이다 (`modeling_raon1_mini.py:503-511`의 `dequantize_mxfp8` 참조).
- 그러나 현재 `30-dequant-mxfp8.py`는 `compressed_tensors`의 기본 `FP8BlockDequantizer`를 그대로 사용한다. `FP8BlockDequantizer._create_dequantized_weight`는 `weight.float() * scale_inv.float()`를 수행하므로, `2^(115 - 127) = 2^-12`를 곱하는 대신 원시 정수값 `115.0`을 그대로 곱한다.
- 결과적으로 `models/raon-bf16`의 모든 expert 가중치가 정상 범위(`[-0.203, +0.219]`)가 아닌 `[-51456.0, +51968.0]`으로 약 $4.7 \times 10^5$배 폭주하여 모델이 완전히 파손된다.

### 문제 2: `Raon1MiniRoutedLinear` 커스텀 서브클래스로 인한 `compressed-tensors` 패킹 누락 (`format: "dense"`) 및 AWQ 스무딩 실패
- `modeling_raon1_mini.py:637`의 `_routed_projection` 헬퍼는 `Raon1MiniRoutedLinear(in_features, out_features, bias=False)` (`nn.Linear`의 서브클래스)를 반환한다.
- `compressed-tensors==0.19.0`의 모든 압축기(`NVFP4PackedCompressor`, `PackedQuantizationCompressor` 등)는 `COMPRESSIBLE_MODULE_TYPES = (torch.nn.Linear, torch.nn.Embedding)`에 대해 `issubclass`가 아닌 정확 타입 멤버십(`module_type in COMPRESSIBLE_MODULE_TYPES`)을 검사한다.
- 이 때문에 9,216개 expert 모듈에 대해 `can_compress`가 `False`를 반환하여 `DenseCompressor`로 폴백되며, 모든 양자화 결과물(`32`, `33`, `35`, `36`)이 `"format": "dense"` 및 59GB 비압축 BF16 가중치로 저장되었다.
- 또한 `Raon1MiniRoutedLinear.forward(self, x)`는 추가 키워드 인자를 받지 않아 `AWQModifier`가 스무딩 시 `hidden=...` 키워드 인자를 전달할 때 `TypeError: Raon1MiniRoutedLinear.forward() got an unexpected keyword argument 'hidden'` 경고를 발생시키며 스무딩을 건너뛴다.
- `30-dequant-mxfp8.py`에서 이미 모든 가중치를 BF16으로 디퀀타이즈하므로, 디퀀타이즈된 모델에서 `_routed_projection`이 표준 `nn.Linear(in_features, out_features, bias=False)`를 반환하도록 패치하면 수식·가중치 키·동작이 100% 동일하면서 압축 패킹과 AWQ 스무딩, `AutoModelForCausalLM.from_pretrained` 재로드가 모두 정상 동작한다.

---

## 2. 요구사항

1. **E8M0 스케일 변환을 수행하는 `MXFP8BlockDequantizer` 구현**:
   - `30-dequant-mxfp8.py`에서 `FP8BlockDequantizer`를 상속한 `MXFP8BlockDequantizer`를 정의하고 `_create_dequantized_weight(self, weight, scale_inv, dtype)`를 오버라이드한다.
   - `scale_inv`가 `torch.uint8` (또는 E8M0 바이트)일 때 `true_scale = torch.exp2(scale_inv.view(torch.uint8).to(torch.float32) - 127.0)`로 변환한 뒤 블록 크기 `(1, 32)` 브로드캐스팅 곱셈을 수행하여 `modeling_raon1_mini.py`의 `dequantize_mxfp8`와 수치적으로 일치시킨다.
2. **`_apply_tracing_shim`에 `_routed_projection` → `nn.Linear` 치환 추가**:
   - `30-dequant-mxfp8.py`의 `_apply_tracing_shim(model_dir: Path)`에서 기존 attention 마스크 2곳 치환에 더해, `return Raon1MiniRoutedLinear(in_features, out_features, bias=False)`를 `return nn.Linear(in_features, out_features, bias=False)`로 치환한다.
   - 원본 HF 캐시(`src`)는 실행 종료 시 복원하되, 출력 디렉터리(`tmp` → `models/raon-bf16`)에 복사된 `modeling_raon1_mini.py`에는 해당 shim이 영구 보존되도록 한다 (`save_custom_code`를 통해 모든 하위 양자화 모델에도 전파됨).
3. **디퀀타이즈 결과 자동 검증(`verify_dequantized_bf16`) 추가**:
   - `commit_dir(tmp, out)` 호출 전에 다음을 검증하고, 하나라도 실패하면 예외를 발생시켜 잘못된 `raon-bf16`이 커밋되지 않게 한다:
     - 출력 `.safetensors` 파일들에 `F8_E4M3` dtype 텐서나 `weight_scale_inv` 키가 남아 있지 않을 것.
     - 샘플 expert 가중치(예: `model.layers.0.mlp.experts.0.gate_proj.weight`)를 원본 체크포인트의 `(weight, weight_scale_inv)`로부터 `2^(scale_inv - 127)` 공식으로 직접 계산한 결과와 비교하여 `torch.allclose(..., atol=1e-3, rtol=1e-3)`를 만족하고 `abs().max() < 10.0`일 것 (NaN/Inf 0개).
     - `tmp / "modeling_raon1_mini.py"`에 `return nn.Linear(in_features, out_features, bias=False)`가 포함되어 있을 것.

---

## 3. 승인 기준

- [ ] `MXFP8BlockDequantizer`가 `uint8` E8M0 `weight_scale_inv` 텐서에 대해 `2^(byte - 127.0)` 스케일을 적용하여 `dequantize_mxfp8`와 동일한 BF16 텐서를 생성함 (단위 테스트로 검증)
- [ ] `_apply_tracing_shim` 적용 후 생성된 모델에서 `model.layers[0].mlp.experts[0].gate_proj`의 타입이 `torch.nn.Linear` (`type(m) is torch.nn.Linear`)임
- [ ] `verify_dequantized_bf16` 함수가 폭주한 가중치(`abs().max() >= 10.0`) 또는 잔존 `weight_scale_inv`가 있을 때 즉시 `RuntimeError`/`AssertionError`를 발생시킴
- [ ] 기존 손상된 `/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared/models/raon-bf16` 및 잘못 생성된 파생 양자화 모델 디렉터리 정리 또는 재생성 준비가 가능함

---

## 4. 검증

- 단위 테스트 (`tests/test_dequant_mxfp8.py`):
  - 합성 MXFP8(`float8_e4m3fn`) 가중치 및 `uint8` E8M0 `weight_scale_inv` 텐서(예: exponent byte `115` → $2^{-12}$)에 대해 `MXFP8BlockDequantizer` 출력이 $w \times 2^{\text{byte}-127}$과 정확히 일치하는지 검증
  - `_apply_tracing_shim`을 임시 `modeling_raon1_mini.py` 복사본에 적용했을 때 `_routed_projection`이 `nn.Linear`를 반환하도록 변환되는지 검증
  - `verify_dequantized_bf16`의 정상 통과 및 비정상 가중치 거부 동작 검증
- 회귀 스크립트 (`regression-tests/verify-issue-2.sh`) 실행 및 통과 확인

---

## 5. 의존성 / 관계

- 선행 이슈: `issue-1` (권장 순서)
- 후속 이슈: `issue-3` (압축 포맷 검증), `issue-4` (PPL/Eval 실행), `issue-5` (업로드 준비)의 기반이 됨

---

## 6. 구현 결과

- **구현 완료 일시**: 2026-10-23T09:16:35-04:00
- **변경 파일 목록**:
  - `30-dequant-mxfp8.py` (수정): `MXFP8BlockDequantizer`(FP8BlockDequantizer 서브클래스, E8M0 `2**(byte-127)`) + `_e8m0_scale_bytes` + `_make_mxfp8_dequantizer_cls`(지연 import) + `_LazyMXFP8Dequantizer`(모듈 최상위 import 회피) + `_ROUTED_PROJECTION_SUB` 치환 + `_shard_map`/`_pick_sample_expert_key`/`_reference_expert_weight` + `verify_dequantized_bf16(tmp, src, sample_key=)` + `_apply_tracing_shim(Path)` 반환값화/치환 검증
  - `tests/test_dequant_mxfp8.py` (신규): 단위 테스트 21건 (pytest/단독 실행 모두 지원). `test_vendor_model_experts_become_plain_linear`은 실제 벤더 `modeling_raon1_mini.py` + 축소 config로 `Raon1MiniModel`을 구성해 승인 기준을 그대로 검증한다 (벤더 스냅샷이 없으면 skip)
  - `regression-tests/verify-issue-2.sh` (신규)
  - `issues/issue-2__agent-stats.json` (신규)
- **계획 대비 변경 사항**:
  1. `_create_dequantized_weight(self, weight, scale_inv, dtype=None)` — `dtype`에 기본값을 뱌다. `compressed_tensors`의 `FP8BlockDequantizer.process()`는 인자를 2개(위치)만 넘기므로 기본값 없이는 base 호출 경로가 깨진다. 이슈의 3인자 호출은 그대로 지원한다. (`scale_inv` 이름은 base의 `weight_scale_inv`와 달라 pyright `reportIncompatibleMethodOverride`가 뜨므로, 시그니처를 이슈대로 유지하고 해당 줄에만 ignore를 붙였다.)
  2. 샘플 expert 키를 `model.layers.0...` 하드코딩 대신 `model.safetensors.index.json`에서 실제 expert weight를 찾아 선택한다. 이 체크포인트는 `first_k_dense_replace=1`이라 0층이 dense(`model.layers.0.mlp.*`)이고 MoE expert는 1층부터라, 이슈 본문 예시 키가 존재하지 않는다. 인덱스에 없으면 `model.layers.1.moe.experts.0.gate_proj.weight`로 폴백.
  3. `_apply_tracing_shim`이 반환값 `{"applied", "total"}` dict를 주고, 치환 대상(`return Raon1MiniRoutedLinear(...)`)이 없으면 `RuntimeError`로 즉시 실패한다. 조용한 no-op으로 넘어가면 압축 패킹이 다시 dense로 폴백하므로, 계약이 이 단계에서 강제되어야 한다. 검증 항복 실패(로그만 남기고 진행)도 같은 이유로 `RuntimeError`로 바꾼다.
  4. 검증 대상을 `tmp/*.safetensors` 전체 shard로 확장하고 헤더만 읽는다(`safe_open().get_dtype()`). 59GB 체크포인트에서 텐서 본문을 읽지 않는다.
  5. `verify_dequantized_bf16`는 폭주(`max_abs >= 10`)·기준 불일치(`allclose(atol=rtol=1e-3)` 실패)·NaN/Inf·잔존 `weight_scale_inv`·잔존 `F8_E4M3`·shim 미적용을 각각 다른 `RuntimeError` 메시지로 구분해 낸다(`main()`의 기존 `SystemExit`는 dtype 잔류용으로 그대로 둠).
  6. shim 전파 경로는 관건 32/33/35/36이 `save_custom_code(tmp, SRC)`로 `SRC=$RQ_SRC ?? models/raon-bf16`의 `*.py`를 복사하는 기존 구조를 그대로 쓴다. 즉 shim을 `tmp`(=커밋 후 `models/raon-bf16`)에 영구 쓰면 하위 4종에 자동 전파된다. 벤더 Hub 캐시(`snap`)는 읽기 전용이며 건드리지 않는다.
  7. 승인 기준은 `model.layers[0]`을 예시하지만 이 체크포인트는 0층이 dense(`first_k_dense_replace=1`)이고 expert는 `layers.N.moe.experts.0.{gate,up,down}_proj`에 있다. 테스트는 1층(첫 MoE층)의 `gate/up/down_proj` 세 개를 전부 검사한다.
- **검증 결과**:
  - `tests/test_dequant_mxfp8.py`: 21 passed (pytest `python -m pytest tests/ -q` 및 단독 실행 모두)
  - 실측: 벤더 rev `da4cf16…` 파일로 축소 모델을 구성해 `layers[1].moe.experts[0].gate_proj/up_proj/down_proj` 전부 `type(...) is torch.nn.Linear` 확인. `_apply_tracing_shim`은 실제 파일에 `applied=4/6` (attention 마법 3 + routed 1, 구-스냅샷 마이그레이션 2개는 해당 패턴 부재로 미적용 — 방어적 목록)
  - 현재 디스크의 `models/raon-bf16` 및 파생 4종은 **구 코드로 생성되어 shim이 없다** (`Routed=1, nn.Linear=0` 확인). shim 유무는 30-dequant 재실행으로만 고쳐지므로 재실행이 선행이다.
  - `regression-tests/verify-issue-2.sh`: 통과(rc=0)
  - 기존 회귀 스크립트 `verify-issue-1.sh` / `verify-issue-3.sh`: 통과(rc=0), `verify-issue-4.sh` / `verify-issue-5.sh` 통과
  - `ruff check`: All checks passed. `pyright`: 변경 스크립트 6개 전부 `0 errors`. `python -m compileall`: 통과
  - 6개 PTQ/PPL/eval/prep/upload 스크립트 `--help` 통과, `bash -n` 전 스크립트 통과
  - **미실행**: 실제 `30-dequant-mxfp8.sh`(59GB 체크포인트 다운로드·변환)는 GPU/디스크 비용이 커서 미실행. 기존 손상된 `models/raon-bf16` 정리는 이 이슈의 구현이 아니라 실행 시점의 작업이며, `30-dequant-mxfp8.py`가 `clean_tmp(tmp_path(OUT))`으로 덮어쓴다. 실행 후 `verify_dequantized_bf16`가 통과해야 후속 PTQ 단계를 진행할 것.
