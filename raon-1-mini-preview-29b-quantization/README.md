# raon-1-mini-preview-29b-quantization — KRAFTON Raon-1-mini PTQ (SFT 없음, calibration-only)

> 템플릿: `llama-3.2-1b-to-math-model/` (스크립트 구조·freshness/lock/timing 관례 계승, 학습 Stage 제거).
> 베이스: `KRAFTON/Raon-1-mini-preview` (28.8B MoE, 4.3B active, custom_code, BF16).
> 참고 경로: `65-serve-infer-all.results.txt` 30B 플랜 — 30B는 FFT 불가·QLoRA→merge→양자화가 정석이나,
> 본 디렉토리는 **추가학습 없이 원본에 직접 PTQ**만 수행한다.

## 4종 타깃

| # | 스크립트 | 타깃 | 레시피 (llm-compressor) | 서빙 `--quantization` | 비고 |
|---|---|---|---|---|---|
| 22 | `22-quant-awq-w3` | `awq-w3` (W3A16-g128) | AWQ smoothing + GPTQ W3 | `gptq` | 품질 붕괴 위험 최대, smoke 우선 |
| 25 | `25-quant-awq-w5` | `awq-w5` (W5A16-g128) | AWQ smoothing + GPTQ W5 | `gptq` | 비표준 비트, 서빙 백엔드 확인 필요 |
| 28 | `28-quant-nvfp4-awq` | `nvfp4-awq` (NVFP4) | AWQ smoothing + NVFP4 | `compressed-tensors` | Blackwell 필수, calib 필수 |
| 31 | `31-quant-nvfp4-gptq` | `nvfp4-gptq` (NVFP4) | GPTQModifier NVFP4 | `compressed-tensors` | smoke 마지막 순서 |

출력: `$RQ_MODELS/raon-<target>-<mode>` (예: `raon-nvfp4-gptq-full`).
업로드명: `tayaee/Raon-1-mini-preview-<target>[-smoke]` (상세 `output-models-tree.txt`).
타깃 기본값 experts-only (벤더 MXFP4 recipe와 동일 범위, `RQ_TARGETS=all`로 전 Linear).

## 캘리브레이션 데이터 추천 (리서치 결과)

Raon은 한국어 대화·thinking 모델이므로 GSM8K 영어 슬라이스 재사용은 부적합.
한국어 instruction + 추론 혼합을 권장한다 (근거: 한국어 GPTQ는 형태소-다양 한국어 calib가
영어 calib 대비 유지율 97.4% vs 91.7%, K-EXAONE-236B-AWQ 실전 레시피, COLA의 reasoning-chain 권고).

| 데이터셋 (HF ID) | 용도 | 비고 |
|---|---|---|
| `beomi/KoAlpaca-RealQA` | 한국어 instruction 주력 | K-EXAONE W4A16-AWQ 실사용 (512, seq 2048) |
| `ChuGyouk/HFH4_ultrachat_200k_ko` | 한국어 멀티턴 대화 | K-EXAONE 혼합 구성 요소 (512) |
| `HuggingFaceH4/ultrachat_200k` | 영어 유지 (소량) | NVFP4 스킬 기본값, EN 보존용 |
| `eouya2/KGuard-Korean-MoE-Calibration-v1` | MoE 특화 대안 (24,576 한국어 calib) | MoE expert routing 관찰 포함, 대안 |

기본 혼합 (`13-prep-calibration.sh`): smoke = KoAlpaca 8 (seq 512, 10분 목표),
full = KoAlpaca 128 + ultrachat-ko 64 + ultrachat-en 64 (seq 2048).
chat template 적용 + thinking 포함 (Raon은 thinking mode 권장).

## 실행

```bash
./10-setup.sh                        # 양 노드 1회 (env + HF gated 로그인)
./07-acceptance.sh smoke             # Stage 1: 수용 기준(acceptance.toml) 검증 + 계약 스냅샷
./13-prep-calibration.sh smoke      # calib jsonl 생성
./16-compat-check.sh                 # custom_arch 지원 사전 점검 (필수)
./19-dequant-mxfp8.sh                # BF16 복원 (32~36의 선행 조건, 1회)
./run-smoke.sh                       # 첫 실행=파이프라인 테스트 (5분 목표)
PIPELINE_TEST=0 ./run-smoke.sh       # smoke 4종 실측
./run-full.sh                        # full 4종 (밤샘 권장)

# 단일 타깃만 (mode × target = 8종) — dequant→quant→PPL→eval + 게이트
./10-16,19,28-40-build-smoke-nvfp4-awq.sh  # smoke × nvfp4-awq
./10-16,19,28-40-build-full-nvfp4-awq.sh   # full  × nvfp4-awq
# 게이트: quant 직후 패킹 계약·global_scale, PPL 직후 유한값 검증 → 실패 시 즉시 중단
```

상세: `PROCEDURE.md`, 명령 모음: `cheatsheet.md`.

## SDLC (9단계)

파이프라인은 9단계 SDLC로 관리한다 — 단계마다 산출물과 **게이트(검증 관문)**가 있고,
게이트가 실패하면 다음 단계로 넘어가지 않는다.

| # | 단계 | 산출물 | 게이트 |
|---|---|---|---|
| 0 | 환경 | `.venv/.rq-setup-stamp` | - |
| 1 | 수용 기준 | `outputs/acceptance-<mode>.json` | gate-acceptance |
| 2 | 데이터 | `datasets/raon-calibration-<N>.jsonl` | - |
| 3 | 호환성 | `outputs/compat.stamp` | - |
| 4 | 베이스 복원 | `models/raon-bf16` | - |
| 5 | 양자화 | `models/raon-<target>-<mode>` | gate-quant |
| 6 | PPL | `outputs/ppl-<mode>/<target>.json` | gate-ppl |
| 7 | 평가 | `outputs/eval-<mode>/<target>.jsonl` | gate-eval |
| 8 | 릴리스·배포 | `outputs/release-notes-<mode>.md` | gate-release / gate-serving |

- 계약(수용 기준): `config/acceptance.toml` — Δ 판정 임계값·용량 범위·포맷의 단일 소스
- 게이트 구현: `gates.inc` — `run-{smoke,full}.sh`와 빌드 경로가 **같은 함수**를 사용
- 증빙: `outputs/stage-ledger-<mode>.json` → `61-release-notes.py` → 릴리스 노트
- 정의 문서: `docs/SDLC.md`

## 알려진 위험 (스캐폴딩 시점 미검증)

0. **베이스가 MXFP8 (2026-10-22 실측)**: 메인 repo 텐서가 F8_E4M3 9216 + 스케일.
   llmcompressor가 기양자화 모델을 거부하므로 `19-dequant-mxfp8`으로 BF16 복원 후 PTQ.
   복원본(`raon-bf16`)이 34·37의 base 기준. 복원 수치 검증은 19 스크립트 내장.
1. **custom_code MoE 지원 미검증**: RDA/MLA/MoE-128expert 구조에 AWQModifier/GPTQModifier가
   그대로 동작하는지 smoke 첫 실행으로 확인. gate/router/lm_head는 양자화 제외.
   expert linear가 `Raon1MiniRoutedLinear` 서브클래스라 transformers FP8 교체는 미적용
   (경고)되나 forward는 정상 (`logs/forward-probe.log`).
2. **공식 양자화 존재**: KRAFTON이 MXFP8·NVFP4를 이미 공개함. full 후 PPL 게이트에서 공식 NVFP4와 비교 권장.
3. **W5/W3 서빙**: vLLM GPTQ 백엔드의 3/5bit 지원 여부를 smoke에서 확인. 안 되면 HF 직접 배포용으로만 사용.
4. **10분은 smoke 목표**: 28.8B MoE 로드+양자화 4종을 10분에 끝내는 것은 calib 8·seq 512 전제.
   느린 순서: nvfp4-gptq < nvfp4-awq < awq-w5 < awq-w3 (run-smoke는 빠른 순 실행).
   expert 9216 모듈 대상이라 AWQ smoothing은 수 시간 가능.
5. **NVFP4 하드웨어**: Blackwell (SM100+, DGX Spark GB10 해당)에서만 가속. 서빙은 TP=2 (2x 기준).
