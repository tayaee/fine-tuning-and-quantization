# run-smoke.results.md — 파이프라인 테스트 결과 (PIPELINE_TEST=1, 기본값)

> 범위: 다운로드/모델로딩 제외 5분 목표의 파이프라인 테스트.
> 양자화·PPL·eval은 실행하지 않음 (`PIPELINE_TEST=0 ./run-smoke.sh`가 실측).
> 실행일: 2026-10-22, INFRA=dgx-spark-1x (GB10 sm_12.1).

## 총괄

- 결과: **PASS (rc=0, wall 12s)** — 캐시 적중 상태 기준.
- 산출된 양자화 모델: 없음 (설계상 미실행).

## 스크립트별 결과

| 스크립트 | 결과 | 시간 | 산출물 |
|---|---|---|---|
| `s0-10-setup.sh` | OK | ~2s | venv sync, llmcompressor 0.14.0, compressed-tensors 0.19.0 확인 |
| `12-prep-calibration.sh smoke` | OK | ~0s | `datasets/raon-calibration-2.jsonl` (2행, KoAlpaca+ultrachat-ko) |
| `20-compat-check.sh smoke` | OK | 10s | COMPAT-OK (arch Raon1MiniForCausalLM, 128 experts, 9496 Linear, NVFP4 가능) |
| `validate-helps` | OK | 1s | 15개 스크립트 `--help`/`bash -n` 통과 |

## 모델별 결과

- 해당 없음 (양자화 미실행). 4종(`awq-w3-transformers`, `awq-w5-transformers`, `nvfp4-awq-transformers`, `nvfp4-gptq-transformers`)은 full에서 생성.

## 중간 수정분 (본 실행 중)

1. **에러**: `30-dequant-mxfp8.py`에 argparse 없음 → validate의 `--help` 호출이 본체를 실행해버림.
   **수정**: argparse 추가 (이제 `--help`는 실행 없이 종료).
2. **에러**: dequant 타깃이 `moe.gate`(router, 스케일 없음)까지 포함 → `KeyError: ...moe.gate.weight_scale_inv`.
   **수정**: `EXPERT_TARGETS`로 expert linear만 지정 (`re:.*\.moe\.experts\.[0-9]+\.(gate|up|down)_proj`).
   (dequant 본 실행은 full에서 검증 예정.)

## 경고 처리

| 경고 | 조치 |
|---|---|
| `torch_dtype is deprecated` | 수정済み (`dtype=`, 20/48) |
| `TOKENIZERS_PARALLELISM` 스레딩 충돌 | 수정済み (`common.env`에 `export TOKENIZERS_PARALLELISM=false`) |
| `loading using fp8 but no linear modules` | 수정 불가 → 포기. 원인: expert linear가 `Raon1MiniRoutedLinear` 서브클래스라 transformers exact-type 교체 미적용 + 벤더가 attn/mlp 제외. forward 정상 확인됨 (`logs/forward-probe.log`: FORWARD-OK 1s). PPL은 동일 로더 상대비교라 유효. |
