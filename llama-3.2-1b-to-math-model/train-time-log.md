# runlog — math-1b-sft-and-quantization 실행 시간 기록 (logs/timing-*.log 실측 집계)

## run-*.sh 요약

| 스크립트 | 범위 | 실행 위치 | 실행시간 |
|---|---|---|---|
| run-all-all.sh | 아래 6종 고정 순서로 순차 실행 | 양 노드 (같은 명령) | 약 32h 00m (추정, 미실측) |
| run-mini-single.sh | mini 전 파이프라인 (Stage 0→7) | spark1 단독 | 약 0h 55m (실측 합계) |
| run-mini-ddp.sh | mini 2노드 DDP (학습만 분산, 나머지는 rank0) | 양 노드 | 약 0h 35m (실측, 30/31 수동분 제외) |
| run-mini-fsdp.sh | mini FSDP 2노드 (ddp와 동일 구조) | 양 노드 | 약 0h 38m (실측, 32 수동분 제외) |
| run-full-single.sh | full 전 파이프라인 (Stage 0→7) | spark1 단독 | 약 10h 08m (순수 성공) / 약 15h 09m (재실행 누적) / 약 24h 55m (합성생성 포함) |
| run-full-ddp.sh | full 2노드 DDP | 양 노드 | 약 13h 56m (순수 성공, 캐시 적중) / 약 23h 40m (합성생성 포함) / 약 23h 42m (재실행 누적) |
| run-full-fsdp.sh | full FSDP (동일) | 1x spark1 / 2x 양 노드 | 약 14h 12m (순수 성공, 캐시 적중) / 약 15h 16m (재실행 누적) |

- 공통: 스텝 타이밍(stdout + `logs/timing-*.log`), `STEPS=` 부분 실행,
  `MASTER_ADDR`·`NODE_RANK` 자동 검출, `SKIP_SETUP=1` 생략.
  run-all-all 전용: `RUNS=` 일부만 실행, single 2종 rank0 전용 게이팅.
- 하류(merge→평가)는 같은 전략 산출물 기준으로 진행 (콤보별 독립).

- 시간 표기: 분 단위 (1분 미만은 `<1m`). 대표 성공런 기준.
- full 계열: run-full-single.sh / run-full-ddp.sh / run-full-fsdp.sh 모두 완주 완료
  (single 약 10h 08m, ddp 약 13h 56m, fsdp 약 14h 12m — 아래 상세 참조).

## run-mini-single.sh (spark1 단독, 완주)

| 스텝 | 시간 | 비고 |
|---|---|---|
| 10-setup | <1m | 2회目以降 (초회는 수 분) |
| 10-baseline-single | 3m | 256행·4스텝 |
| 18-select-candidates | <1m | 임베딩 캐시 적중 시 |
| 20-build-prompts | <1m | |
| 22-teacher-generate | 8m | 64프롬프트, TP=1, 8B 상주 후 |
| 24-postprocess | <1m | |
| 30-fft-single | 3m | |
| 31-qlora-single | 3m | |
| 30-merge-lora | 1m | |
| 32-quant-fft-awq | 1m | llmcompressor 0.14.0 + compressed-tensors 0.19.0 |
| 34-quant-fft-fp8 | <1m | |
| 36-quant-fft-gguf | 1m | convert+quantize (Q8_0 1종 실측, 5종 전량 ~2분, f16 기변환·q8_0 skip 시) |
| 38-quant-fft-gptq | 2m | calib 2 |
| 40-quant-qlora-awq | 1m | |
| 42-quant-qlora-fp8 | <1m | |
| 44-quant-qlora-gguf | 2m | Q8_0 1종 실측, 5종 전량 ~2분 (f16 기변환·q8_0 skip 시) |
| 46-quant-qlora-gptq | 2m | |
| 48-measure-ppl | 2m | 9타깃×32텍스트 실측, 19타깃 시 증가 (GGUF 10종은 llama-perplexity) |
| 50-eval | 11m | 9타깃×10문항 (타깃당 ~1m) |
| 52-eval-gguf | 미측정 | 10타깃×10문항, llama-completion CPU (PROCEDURE 추정 ~30분) |
| 54-score | <1m | |
| **합계** | **약 43m** | PROCEDURE §3의 약 30분보다 김 (eval·teacher 실측 반영) |

PPL 게이트 (mini): base/fft/qlora 9.867 동일, quant Δ +0.12~+0.99
→ fft-fp8·qlora-fp8 Accept, 나머지 Conditional, Discard 없음.

## run-full-single.sh (spark1 단독, 완주)

| 스텝 | 시간 | 비고 |
|---|---|---|
| 10-setup | <1m (4s) | 의존성 및 환경 검증 |
| 10-baseline-single | 65m (3,913s) | GSM8K 7,473행 SFT (초회 14m 중단 후 재실행 완주) |
| 18-select-candidates | <1m (0s*) | Alpaca 52k 중 10k 선별 (*full-ddp 선행 실측: 97s) |
| 20-build-prompts | <1m (0s*) | GSM8K 스타일 합성 프롬프트 생성 (*선행 실측: 2s) |
| 22-teacher-generate | <1m (1s*) | Llama-3.1-8B-Instruct 10k 생성 (*선행 실측: 35,022s ≈ 9h 44m, valid 9,466행) |
| 24-postprocess | <1m (0s*) | synthetic-full.jsonl 추출 (*선행 실측: 1s) |
| 30-fft-single | 104m (6,258s) | 합성데이터 9,466행 전체 3 epoch FFT (~1h 44m) |
| 31-qlora-single | 144m (8,615s) | 합성데이터 9,466행 전체 3 epoch QLoRA (~2h 24m) |
| 30-merge-lora | 1m (83s) | QLoRA 어댑터 가중치 병합 (bf16) |
| 32-quant-fft-awq | 3m (207s) | FFT 모델 AWQ 4-bit 양자화 |
| 34-quant-fft-fp8 | <1m (50s) | FFT 모델 FP8 양자화 |
| 36-quant-fft-gguf | 2m (113s) | FFT F16 변환 + GGUF 5종 양자화 (q8_0, q6_k, q5_k_m, q4_k_m, q3_k_m) |
| 38-quant-fft-gptq | 2m (130s) | FFT 모델 GPTQ 4-bit 양자화 (calib 256) |
| 40-quant-qlora-awq | 3m (209s) | QLoRA-merged 모델 AWQ 4-bit 양자화 |
| 42-quant-qlora-fp8 | <1m (58s) | QLoRA-merged 모델 FP8 양자화 |
| 44-quant-qlora-gguf | 2m (132s) | QLoRA-merged F16 변환 + GGUF 5종 양자화 (5종 전량) |
| 46-quant-qlora-gptq | 2m (136s) | QLoRA-merged 모델 GPTQ 4-bit 양자화 (calib 256) |
| 48-measure-ppl | 3m (184s) | 21개 타깃 PPL 측정 및 게이트 판정 (Accept 10, Conditional 3, Ref 8) |
| 50-eval | 59m (3,543s) | vLLM 기반 9개 모델 GSM8K test 1,319문항 평가 (~1h) |
| 52-eval-gguf | 215m (12,875s) | llama-completion CPU 기반 GGUF 10개 모델 각 1,319문항 평가 (~3h 35m, 초회 실패 4h 47m 후 재실행 완주) |
| 54-score | <1m (1s) | 19개 타깃 정확도 집계 (score.json) |
| **순수 완주 실측 (성공 스텝 합계)** | **약 10h 08m** | **36,512s** (기생성 10k 합성데이터 캐시 적중 기준) |
| **중단/재실행 포함 누적 실행시간** | **약 15h 09m** | **54,549s** (baseline 중단 14m + eval-gguf 1회차 실패 4h 47m 포함) |
| **합성생성 포함 총 파이프라인 연산** | **약 24h 55m** | **89,670s** (선행 full-ddp 10k 합성생성 9h 45m까지 전 과정 합산) |

- **중단 및 재실행 상세 타임라인**:
  1. **1차 중단 (10/10 19:10Z)**: `10-baseline-single` 840s(14m) 진행 중 SIGINT(rc=130)로 중단 → 19:10Z 즉시 재실행하여 3,913s(65m) 완주 (baseline 누적 연산: 79m).
  2. **연속 완주 (10/10 20:16Z ~ 10/11 01:44Z)**:
     - Stage 2 합성데이터(10k): 선행 생성된 캐시 감지로 1초 만에 스킵.
     - 30-fft (1h 44m) → 31-qlora (2h 24m) → 32-merge (1m) → PTQ 8종 (17m) → 60-ppl (3m) → 50-eval (59m) 무중단 순차 완주.
  3. **2차 중단 (10/11 06:31Z)**: `52-eval-gguf` (llama-completion CPU 10타깃×1,319문항) 17,197s(4h 46m 37s) 실행 중 프로세스 중단/실패(rc=1).
  4. **3차 짧은 시도 및 재부팅 (10/11 06:38Z ~ 08:24Z)**: 06:38Z 짧은 재실행 시도 후 중단, 08:24Z(04:24 EDT) 시스템 리부팅.
  5. **최종 재실행 완주 (10/11 09:19Z ~ 12:54Z)**:
     - 신선도 검사(freshness check) 메커니즘으로 이미 완료된 전 단계(00~71)는 재연산 없이 즉시 패스 (0~1s).
     - `52-eval-gguf`가 12,875s(3h 34m 35s) 완주에 성공 (72단계 누적 연산: 8h 21m).
     - `54-score` 즉시 집계 (1s)로 최종 파이프라인 정상 종결.
  - **벽시계(Wall-Clock) 총 경과 시간**: 10/10 18:56Z 시작 ~ 10/11 12:54Z 최종 완료 (약 17시간 58분, 대기 및 재부팅 포함).

- PPL 게이트 (full):
  - transformers 계열 (기준 base 9.867):
    - FFT 계열: fft 4.549 (ref), fft-awq 4.489 (Accept, Δ -0.060), fft-fp8 4.570 (Accept, Δ +0.021), fft-gptq 4.604 (Accept, Δ +0.055)
    - QLoRA 계열: qlora 4.990 (ref), qlora-awq 5.840 (Conditional, Δ +0.850), qlora-fp8 5.342 (Conditional, Δ +0.352), qlora-gptq 5.045 (Accept, Δ +0.055)
  - llama.cpp 계열 (기준 f16: fft-gguf 2.781, qlora-gguf 2.749):
    - GGUF 10종 전량 Accept (fft Δ -0.001 ~ +0.121, qlora Δ -0.009 ~ +0.192, Discard 없음).
- GSM8K 최종 정확도 (full-single 1,319문항):
  - base: acc 0.000 (invalid 99.1%)
  - fft: acc 0.101 (invalid 7.2%)
  - qlora: acc 0.072 (invalid 24.1%)
  - 양자화 모델 상위: fft-gguf-q4_k_m (0.094), fft-gguf-q5_k_m (0.088), fft-gguf-q8_0 (0.087), fft-fp8 (0.086), fft-gguf-q6_k (0.086), fft-gptq (0.084)

## run-full-ddp.sh (2노드 DDP, 완주 — logs/timing-full-ddp.log)

| 스텝 | 시간 | 비고 |
|---|---|---|
| 10-setup | <1m (4s) | |
| 10-baseline-ddp | 38m (2,286s) | 2노드 DDP |
| 18-select-candidates | 2m (97s) | Alpaca 52k 중 10k 선별 |
| 20-build-prompts | <1m (2s) | |
| 22-teacher-generate | 583m (35,022s, rc=2 기록) | 10k 생성, valid 9,466행 산출 — 후속 전 단계 캐시 원본 (single·fsdp 재사용) |
| 24-postprocess | <1m (1s) | synthetic-full.jsonl 추출 |
| 30-fft-ddp | 62m (3,690s) | 합성 9,466행 3 epoch FFT (중단 41s rc=131 1회 후 재실행 완주) |
| 31-qlora-ddp | 83m (4,995s, rc=2 기록) | 어댑터 저장·후속 정상 진행 (내부 train_runtime 4,863s) |
| 30-merge-lora | 2m (116s) | 중단 18s rc=130 1회 후 재실행 완주 |
| 32-quant-fft-awq | 4m (212s) | 실패 13s rc=1 1회 후 재실행 완주 |
| 34-quant-fft-fp8 | <1m (53s) | |
| 36-quant-fft-gguf | 2m (110s) | F16 변환 + 5종 |
| 38-quant-fft-gptq | 2m (132s) | calib 256 |
| 40-quant-qlora-awq | 4m (226s) | |
| 42-quant-qlora-fp8 | 1m (60s) | |
| 44-quant-qlora-gguf | 2m (129s) | F16 변환 + 5종 |
| 46-quant-qlora-gptq | 2m (134s) | calib 256 |
| 48-measure-ppl | 3m (190s) | 21개 타깃 (transformers: Accept 4, Conditional 1, Discard 1(qlora-awq), Ref 3 + GGUF 10종 전량 Accept) |
| 50-eval | 61m (3,686s) | vLLM 9개 모델 × 1,319문항 |
| 52-eval-gguf | 567m (34,005s ≈ 9h 27m) | llama-completion CPU 10개 모델 × 1,319문항 |
| 54-score | <1m (timing 미기록) | score.json 10-12 15:36Z 생성 (직접 실행분) |
| **순수 성공 합계 (캐시 적중 기준)** | **약 13h 56m** | **50,151s** (22-teacher 1s 캐시 적중 가정) |
| **합성생성 포함 총 연산** | **약 23h 40m** | **85,172s** (35,022s 합성생성 합산) |
| **재실행 누적 (timing 합계)** | **약 23h 42m** | **85,297s** (실패분 41s+18s+13s 포함) |

- PPL 게이트 (full-ddp, 기준 base 9.867):
  - fft 4.542 (ref) → fft-awq 4.608 / fft-fp8 4.565 / fft-gptq 4.559 전량 Accept
  - qlora 4.991 (ref) → qlora-awq 6.006 (Discard, Δ +1.015), qlora-fp8 5.500 (Conditional, Δ +0.509), qlora-gptq 5.084 (Accept, Δ +0.093)
  - GGUF 10종 전량 Accept.
- GSM8K 정확도 (full-ddp 1,319문항, score.json):
  - fft 0.092 / qlora 0.071, 상위: fft-gguf-q5_k_m 0.093, fft-gguf-q8_0 0.086, fft-gptq 0.080, fft-fp8 0.075.

## run-full-fsdp.sh (2노드 FSDP, 완주 — logs/timing-full-fsdp.log)

| 스텝 | 시간 | 비고 |
|---|---|---|
| 10-setup | <1m (누적 40s, 재시도분 포함) | |
| 10-baseline-fsdp | 46m (2,782s) | 중단·실패 누적 3,767s (324s rc=130 ×2, 1,364s rc=130, 1,529s rc=1 등 10회) 후 완주 |
| 20~23 syndata | 스킵 (캐시 적중) | DDP 생성 synthetic-full.jsonl 재사용 |
| 30-fft-fsdp | 72m (4,343s) | 합성 9,466행 3 epoch FFT |
| 31-qlora-fsdp | 84m (5,011s) | FSDP+QLoRA 미지원 → DDP 폴백 실행 (내부 train_runtime 4,895s, 실패 10s·5s 후 완주) |
| 30-merge-lora | 1m (85s) | |
| 32-quant-fft-awq | 4m (215s) | |
| 34-quant-fft-fp8 | <1m (53s) | |
| 36-quant-fft-gguf | 2m (111s) | F16 변환 + 5종 |
| 38-quant-fft-gptq | 2m (134s) | calib 256 |
| 40-quant-qlora-awq | 4m (227s) | |
| 42-quant-qlora-fp8 | 1m (62s) | |
| 44-quant-qlora-gguf | 2m (129s) | F16 변환 + 5종 |
| 46-quant-qlora-gptq | 2m (134s) | calib 256 |
| 48-measure-ppl | 3m (182s) | 21개 타깃 (transformers: Accept 5, Discard 1(qlora-awq), Ref 3 + GGUF 10종 전량 Accept) |
| 50-eval | 60m (3,574s) | vLLM 9개 모델 × 1,319문항 |
| 52-eval-gguf | 567m (34,035s ≈ 9h 27m) | llama-completion CPU 10개 모델 × 1,319문항 |
| 54-score | <1m (timing 미기록) | score.json 10-13 21:57Z 생성 (직접 실행분) |
| **순수 성공 합계 (rc=0 합)** | **약 14h 13m** | **51,165s** (재시도 setup 포함) / 대표 성공런 약 14h 12m (51,104s) |
| **재실행 누적 (timing 합계)** | **약 15h 16m** | **54,932s** (baseline 실패 3,767s 포함) |

- PPL 게이트 (full-fsdp, 기준 base 9.867):
  - fft 4.555 (ref) → fft-awq 4.472 / fft-fp8 4.583 / fft-gptq 4.551 전량 Accept
  - qlora 4.980 (ref) → qlora-awq 6.033 (Discard, Δ +1.053), qlora-fp8 5.278 (Accept, Δ +0.298), qlora-gptq 5.103 (Accept, Δ +0.123)
  - GGUF 10종 전량 Accept.
- GSM8K 정확도 (full-fsdp 1,319문항, score.json):
  - fft 0.094 / qlora 0.062, 상위: fft-gguf-q6_k 0.094, fft-gguf-q8_0 0.093, fft-gguf-q5_k_m 0.083, fft-gptq 0.072.

## 분산 baseline (STEPS=baseline 실측)

| 스크립트·스텝 | spark1 (rank0) | spark2 (rank1) |
|---|---|---|
| run-mini-ddp.sh / 10-baseline-ddp | 6m (rank1 대기 ~3m 포함) | 4m |
| run-mini-fsdp.sh / 10-baseline-fsdp (2노드) | 3m | 3m |
| run-mini-fsdp.sh / 10-baseline-fsdp (1x) | — | 4m (spark2 단독) |

## 실패 기록 (원인→수정, 재실행 성공)

| 시각(Z) | 현상 | 원인 | 수정 |
|---|---|---|---|
| 10:24 | 20-select 실패 | detached 실행 `uv` 미해결 | common.env PATH 보완 |
| 10:28 | baseline 실패 | TRL `trackio` import 충돌 (hub 1.33) | `create_model_card` no-op |
| 10:37 | 22-teacher 실패 (<1m) | TP=2 단일GPU 불가 | TP=1 고정 |
| 10:50 | 22-teacher 실패 (4m) | vLLM이 `torchvision` 필수인데 setup에서 제거 | 10-setup 제거→설치로 전환 |
| 11:16 | 32-merge 실패 (<1m) | `optimum` 1.16.2 vs gptqmodel 요구 ≥1.24 | requirements 핀 + 양 venv 2.3.0 |
| 11:52 | 42-awq 실패 (<1m) | llmcompressor 0.14.0 vs compressed-tensors 0.17.0 | 0.19.0 고정 (1m 성공) |
| 12:15 | 60-ppl 실패 (2m) | GGUF SKIP행 `delta: None` 포맷 크래시 | None 가드 (2m 성공) |
| 12:54 | mini-ddp baseline 실패 (15m) | rank1 합류 실패 (1/2 join 타임아웃) | 조사 중 — 아래 참조 |
| 10/10 19:10 | full-single baseline 중단 (14m) | 사용자 인터럽트 (rc=130) | 즉시 재실행 (65m 완주 성공) |
| 10/11 06:31 | full-single 52-eval-gguf 실패 (4h 47m) | llama-completion CPU 추론 중단 (rc=1) | 09:19 재실행 (3h 35m 완주 성공) |

## 랑데부 이슈 (2026-10-08 13:4xZ 현재진행)

- spark2 rank1이 spark1:29500 store에 합류 못함. TCP 연결 자체는 즉시 성공, loopback rendezvous 정상, GPU·NCCL 정상.
- rank1은 무출력 재시도, rank0는 15m 대기 후 타임아웃. strace상 `ECONNREFUSED` 재시도 루프 관측.
- GLOO 단독 테스트도 동일 증상 → NCCL/IB 문제가 아니라 join 경로 문제로 좁혀짐.
- 다음 단계: 클린 포트(2960x)에서 즉시 페어링 재시험.

## pending

- run-full-ddp.sh / run-full-fsdp.sh — 완주 완료 (위 상세 참조, 54-score는 timing 미기록·score.json 직접 생성)
- run-mini-ddp.sh 전체 (baseline 이후 미진행 — 랑데부 해결 후)
- run-mini-fsdp.sh 전체 (DCP 저장 검증됨: 양 노드 rc=0, 3m/3m)
