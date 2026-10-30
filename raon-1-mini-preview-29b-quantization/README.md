# raon-1-mini-preview-29b-quantization — KRAFTON Raon-1-mini PTQ (SFT 없음, calibration-only)

> 템플릿: `llama-3.2-1b-to-math-model/` (스크립트 구조·freshness/lock/timing 관례 계승, 학습 Stage 제거).
> 베이스: `KRAFTON/Raon-1-mini-preview` (28.8B MoE, 4.3B active, custom_code, BF16).
> 참고 경로: `65-serve-infer-all.results.txt` 30B 플랜 — 30B는 FFT 불가·QLoRA→merge→양자화가 정석이나,
> 본 디렉토리는 **추가학습 없이 원본에 직접 PTQ**만 수행한다.

## 4종 타깃

| # | 스크립트 | 타깃 | 레시피 (llm-compressor) | 서빙 `--quantization` | 비고 |
|---|---|---|---|---|---|
| s3-01 | `s3-00-quant-awq-w3-transformers` | `awq-w3-transformers` (W3A16-g128) | AWQ smoothing + GPTQ W3 | `gptq` | 품질 붕괴 위험 최대, smoke 우선 |
| s3-01 | `s3-05-quant-awq-w5-transformers` | `awq-w5-transformers` (W5A16-g128) | AWQ smoothing + GPTQ W5 | `gptq` | 비표준 비트, 서빙 백엔드 확인 필요 |
| s3-01 | `s3-10-quant-nvfp4-awq-transformers` | `nvfp4-awq-transformers` (NVFP4) | AWQ smoothing + NVFP4 | `compressed-tensors` | Blackwell 필수, calib 필수 |
| s3-01 | `s3-20-quant-nvfp4-gptq-transformers` | `nvfp4-gptq-transformers` (NVFP4) | GPTQModifier NVFP4 | `compressed-tensors` | smoke 마지막 순서 |

출력: `$RQ_MODELS/raon-<target>-<mode>` (예: `raon-nvfp4-gptq-transformers-full`).
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

기본 혼합 (`s1-15-prep-calibration.sh`): smoke = KoAlpaca 8 (seq 512, 10분 목표),
full = KoAlpaca 128 + ultrachat-ko 64 + ultrachat-en 64 (seq 2048).
chat template 적용 + thinking 포함 (Raon은 thinking mode 권장).

## 실행

모든 stage 스크립트는 모드(smoke/full)를 인자로 받는다. **인자를 생략하면 환경변수 `MODE`를 보고**,
그것도 없으면 smoke다. 타깃을 받는 스크립트는 타깃도 필수다 (미지정 시 사용법 출력 후 종료,
`TARGETS=all`로 전체) — 예: `MODE=full ./s4-70-update-readme.sh full nvfp4-awq-transformers`.

```bash
./s0-00-serve-base-from-hf-with-transformers.sh full       # s0 베이스 검증: 접근+메타데이터 (gated)
./s0-05-chat-with-transformers.sh full       # s0: chat 생성 검증 (GPU)
./s0-25-serve-base-from-hf-with-vllm-fork.sh full               # s0: 포크로 베이스 서빙 (이후 s0-16/19, s0-22가 정리)
./s1-00-setup-environment.sh                        # 양 노드 1회 (env + HF gated 로그인)
./s1-05-setup-acceptance-criteria.sh smoke             # Stage 1: 수용 기준(acceptance.toml) 검증 + 계약 스냅샷
./s1-15-prep-calibration.sh smoke      # calib jsonl 생성
./s1-15-preflight-check.sh                 # custom_arch 지원 사전 점검 (필수)
./s4-05-prep-gsm8k.sh full            # 표준 GSM8K 평가 세트(프롬프트+매니페스트) 준비
./s4-25-prep-kmmluko.sh full          # 표준 한국어 KMMLU 평가 세트 준비
./s4-35-check-format__transformers_vllm_both.sh full --prepare      # 지시/형식 준수 고정 10케이스 준비
./s2-00-dequant-mxfp8-into-bf16.sh                # BF16 복원 (s3 양자화의 선행 조건, 1회)

# 단일 타깃 빌드 (mode × target = 8종) — dequant→quant→PPL→eval + 게이트
./run-s1-to-s4-smoke-nvfp4-awq-transformers.sh  # smoke × nvfp4-awq-transformers (TARGETS 불필요, 1종 고정)
./run-s1-to-s4-full-nvfp4-awq-transformers.sh   # full  × nvfp4-awq-transformers
# 게이트: quant 직후 패킹 계약·global_scale, PPL 직후 유한값 검증 → 실패 시 즉시 중단
# 구 모놀리식 러너(run-{smoke,full}.sh, TARGETS/STEPS 다종 실행)는 archive/로 은퇴.
# 다종 실행은 타깃별 개별 스크립트에 TARGETS로 지정한다 (예: 아래 s4-01).
```

상세: `PROCEDURE.md`, 명령 모음: `cheatsheet.md`.

## SDLC (8개 그룹: s0·s1·s2·s3·s4·s5·s6·s7)

파이프라인은 8개 그룹 SDLC로 관리한다 — 그룹마다 산출물과 **게이트(검증 관문)**가 있고,
게이트가 실패하면 다음 그룹으로 넘어가지 않는다.
(두 번째 번호는 그룹 내 순서로 01부터 3단위 — 각 그룹의 첫 스크립트는 항상 `s#-01-`.)

| # | 그룹 | 산출물 | 게이트 |
|---|---|---|---|
| 0 | 베이스모델 검증 | `outputs/s0-<mode>.json` + 벤치 하네스 | chat/pi 응답 존재 |
| 1 | 환경 체인 (setup·수용기준·calib·compat) | `outputs/compat.stamp` | gate-acceptance |
| 2 | 베이스 복원 | `models/raon-bf16` | - |
| 3 | 양자화 | `models/raon-<target>-<mode>` | gate-quant |
| 4 | 측정·평가 (PPL+GSM8K+KMMLU+형식+카드+감사+릴리스노트) | `outputs/eval-<mode>/<target>.jsonl` + `score.json`·`checks.json` + `outputs/release-notes-<mode>.md` | gate-ppl / gate-eval / gate-kmmluko / gate-format / gate-release |
| 5 | 업로드 | Hub 업로드 | gate-release |
| 6 | 고객인수(transformers: Hub→로딩→chat→pi) | `outputs/s6-<mode>/` | chat/pi 응답 존재 |
| 7 | 고객인수(vLLM 서빙: frozen 래퍼→chat→pi) | `outputs/serve-<mode>.json` | gate-serving |

> 모델 카드는 **업로드 대상(repo 1개 = 모델 1개)** 기준이라 base + 그 모델만 비교한다.
> 같은 모드의 **전체 양자화 버전 비교**는 `models/README-{smoke,full}.md`에 모인다.

- 계약(수용 기준): `config/acceptance.toml` — Δ 판정 임계값·용량 범위·포맷·GSM8K/KMMLU 허용폭·형식 통과율의 단일 소스
- 평가: **5개 검사** — ① 표준 GSM8K(`openai/gsm8k` test, revision·seed 고정) ② 표준 한국어 KMMLU(`HAERAE-HUB/KMMLU`) ③ 지시/형식 준수(고정 10케이스) ④ PPL(Stage 4) ⑤ 서빙 처리량
  - 프롬프트 생성: `s4-05-prep-gsm8k.sh` · `s4-25-prep-kmmluko.sh` · `s4-35-check-format__transformers_vllm_both.sh` (모두 결정론·멱등)
- 모델 카드: **Check 1 PPL** · **Check 2 GSM8K** · **Check 3 KMMLU** · **Check 4 형식 준수** (참고: 서빙 성능)
- 게이트 구현: `gates.inc` — `build-ppl-eval.inc` 단일 경로가 **같은 함수**를 사용
- 증빙: `outputs/stage-ledger-<mode>.json` → `s4-80-release-notes.py` → 릴리스 노트
- 정의 문서: `docs/SDLC.md`

## 알려진 위험 (스캐폴딩 시점 미검증)

0. **베이스가 MXFP8 (2026-10-22 실측)**: 메인 repo 텐서가 F8_E4M3 9216 + 스케일.
   llmcompressor가 기양자화 모델을 거부하므로 `s2-00-dequant-mxfp8`으로 BF16 복원 후 PTQ.
   복원본(`raon-bf16`)이 s4-01·s4-07의 base 기준. 복원 수치 검증은 19 스크립트 내장.
1. **custom_code MoE 지원 미검증**: RDA/MLA/MoE-128expert 구조에 AWQModifier/GPTQModifier가
   그대로 동작하는지 smoke 첫 실행으로 확인. gate/router/lm_head는 양자화 제외.
   expert linear가 `Raon1MiniRoutedLinear` 서브클래스라 transformers FP8 교체는 미적용
   (경고)되나 forward는 정상 (`logs/forward-probe.log`).
2. **공식 양자화 존재**: KRAFTON이 MXFP8·NVFP4를 이미 공개함. full 후 PPL 게이트에서 공식 NVFP4와 비교 권장.
3. **W5/W3 서빙**: vLLM GPTQ 백엔드의 3/5bit 지원 여부를 smoke에서 확인. 안 되면 HF 직접 배포용으로만 사용.
4. **10분은 smoke 목표**: 28.8B MoE 로드+양자화 4종을 10분에 끝내는 것은 calib 8·seq 512 전제.
   느린 순서: nvfp4-gptq-transformers < nvfp4-awq-transformers < awq-w5-transformers < awq-w3-transformers.
   expert 9216 모듈 대상이라 AWQ smoothing은 수 시간 가능.
5. **NVFP4 하드웨어**: Blackwell (SM100+, DGX Spark GB10 해당)에서만 가속. 서빙은 TP=2 (2x 기준).
