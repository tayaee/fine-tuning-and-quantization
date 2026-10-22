# raon-1-mini-preview-29b-quantization — KRAFTON Raon-1-mini PTQ (SFT 없음, calibration-only)

> 템플릿: `llama-3.2-1b-to-math-model/` (스크립트 구조·freshness/lock/timing 관례 계승, 학습 Stage 제거).
> 베이스: `KRAFTON/Raon-1-mini-preview` (28.8B MoE, 4.3B active, custom_code, BF16).
> 참고 경로: `65-serve-infer-all.results.txt` 30B 플랜 — 30B는 FFT 불가·QLoRA→merge→양자화가 정석이나,
> 본 디렉토리는 **추가학습 없이 원본에 직접 PTQ**만 수행한다.

## 4종 타깃

| # | 스크립트 | 타깃 | 레시피 (llm-compressor) | 서빙 `--quantization` | 비고 |
|---|---|---|---|---|---|
| 33 | `33-quant-awq-w3` | `awq-w3` (W3A16-g128) | AWQ smoothing + GPTQ W3 | `gptq` | 품질 붕괴 위험 최대, smoke 우선 |
| 32 | `32-quant-awq-w5` | `awq-w5` (W5A16-g128) | AWQ smoothing + GPTQ W5 | `gptq` | 비표준 비트, 서빙 백엔드 확인 필요 |
| 35 | `35-quant-nvfp4-awq` | `nvfp4-awq` (NVFP4) | AWQ smoothing + NVFP4 | `compressed-tensors` | Blackwell 필수, calib 필수 |
| 36 | `36-quant-nvfp4-gptq` | `nvfp4-gptq` (NVFP4) | GPTQModifier NVFP4 | `compressed-tensors` | smoke 마지막 순서 |

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

기본 혼합 (`12-prep-calibration.sh`): smoke = KoAlpaca 8 (seq 512, 10분 목표),
full = KoAlpaca 128 + ultrachat-ko 64 + ultrachat-en 64 (seq 2048).
chat template 적용 + thinking 포함 (Raon은 thinking mode 권장).

## 실행

```bash
./10-setup.sh                        # 양 노드 1회 (env + HF gated 로그인)
./12-prep-calibration.sh smoke      # calib jsonl 생성
./20-compat-check.sh                 # custom_arch 지원 사전 점검 (필수)
./30-dequant-mxfp8.sh                # BF16 복원 (32~36의 선행 조건, 1회)
./run-smoke.sh                       # 첫 실행=파이프라인 테스트 (5분 목표)
PIPELINE_TEST=0 ./run-smoke.sh       # smoke 4종 실측
./run-full.sh                        # full 4종 (밤샘 권장)
```

상세: `PROCEDURE.md`, 명령 모음: `cheatsheet.md`.

## 알려진 위험 (스캐폴딩 시점 미검증)

0. **베이스가 MXFP8 (2026-10-22 실측)**: 메인 repo 텐서가 F8_E4M3 9216 + 스케일.
   llmcompressor가 기양자화 모델을 거부하므로 `30-dequant-mxfp8`으로 BF16 복원 후 PTQ.
   복원본(`raon-bf16`)이 48·50의 base 기준. 복원 수치 검증은 30 스크립트 내장.
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
