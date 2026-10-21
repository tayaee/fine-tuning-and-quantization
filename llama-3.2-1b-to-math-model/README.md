# math-1b-sft-and-quantization — Capstone: GSM8K SFT + PTQ (DGX Spark 1x/2x)

DGX Spark에서 실행하는 파이프라인.
실행에 필요한 것은 이 디렉토리(`data/`, `assets/`, 스크립트)에 다 들어 있다.
상세 절차는 `PROCEDURE.md`.

## 실습 목표

1. 합성 데이터 생성 — teacher 모델(vLLM 서빙)로 GSM8K 스타일 Q/A 합성
   (선별 → 프롬프트 → 생성 → 후처리, Stage 2).
2. FFT vs QLoRA — 전파라미터 파인튜닝과 4bit 양자화 저랭크 적응(QLoRA) 비교,
   어댑터 병합으로 추론용 풀모델 복원 (Stage 3).
3. 양자화 기법 비교 — FP8 / AWQ / GPTQ / GGUF(Q8_0·K-quant 4종) 4종을
   FFT 결과와 QLoRA-merged 결과에 각각 적용 (Stage 4·5, 양자화 총 16종).
4. 모델 평가 — PPL 게이트(수용/조건부/기각 판정) + GSM8K greedy 평가·채점·종합 리포트
   (Stage 6·7).
5. Hub 업로드와 서빙 — Hugging Face Hub(`tayaee/*`) 업로드,
   vLLM/llama-server OpenAI-호환 서빙과 추론 예제 (Stage 8).
6. 단일 vs 분산 학습 — single(1 GPU) vs DDP(2노드 데이터 병렬)
   vs FSDP(완전 샤딩) 전략 비교 (동일 effective batch, 해시 대조).

## 실행 규칙 (repo 관례 준수)

- 원샷 우선: `./run-mini-single.sh` (mini 전 파이프라인, 약 30분).
  `./run-<mode>-<strat>.sh` (`mode=mini|full`, `strat=single|ddp|fsdp`, `all` 포함),
  `STEPS=setup,baseline,syndata,fft,qlora,merge,quant,ppl,eval`로 부분 실행,
  `SKIP_SETUP=1`로 셋업 생략. 수동 재현은 `source ./run-<mode>-<strat>.inc`.
- 인자 규칙: 학습 3종(`14/26/28`)은 `[mini|full] [single|ddp|fsdp]`,
  `16`은 `[mini|full] [single|ddp|fsdp|all]`. 나머지 `.sh`는 `[mini|full]`만 받고
  전략은 `STRAT` env(기본 `single`). 예: `STRAT=single ./48-measure-ppl.sh mini`.
  산출물은 `...-<mode>-<strat>`로 분리된다 (`ppl-<mode>-<strat>/`, `eval-<mode>-<strat>/`).
- `.sh` 래퍼 권장 (fresh-skip/lock/타이밍). `uv run *.py --mode` 직접 실행은 디버그용.
- infra 선택: 명령 앞에 `INFRA=dgx-spark-2x` (기본 1x, `PROFILE=1x|2x` 약칭).
  셸 전체에 export했으면 추론계(`22/50/52/60/61`) 실행 시 `TP=1`을 앞에 붙인다.
  `ddp/fsdp`는 2x + `MASTER_ADDR`(spark1 IP) 필수, `single`은 1x.
- 이 repo는 spark1·spark2 양쪽에 같은 경로(`~/git/math-llm-sft-and-quantization`)로
  clone 하고, `/rosenas/data/AIML/`을 양쪽에서 같이 본다.
- 공용 산출물: `/rosenas/data/AIML/math-1b-shared/{datasets,models,outputs,hf-cache,vllm-cache}`
- HF 업로드는 `tayaee/*`.
- Python 3.12, `uv`, `UV_TORCH_BACKEND=cu130` (root `mise.toml`/`pyproject.toml` 준수).

```bash
# 0) 원샷 (권장, 양 노드에서 1회 셋업 후): mini 전 파이프라인 약 30분
./run-mini-single.sh
# STEPS=setup,baseline ./run-mini-single.sh  # 부분 실행
# SKIP_SETUP=1 ./run-mini-single.sh          # 10-setup 생략

# 아래는 단계별 수동 실행 (원샷과 동등, .sh 래퍼 권장)
# 0) 환경 (양 노드에서 1회)
./10-setup.sh

# 1) 베이스라인 FFT (GSM8K, Llama-3.2-1B) — 전략 3종 비교
./14-baseline-train.sh mini single          # 기준 (하류 입력용)
INFRA=dgx-spark-2x ./14-baseline-train.sh mini ddp     # 2노드 DDP (MASTER_ADDR 필요)
./14-baseline-train.sh mini fsdp            # 1GPU FSDP (학습용)
INFRA=dgx-spark-2x ./14-baseline-train.sh mini fsdp    # 2노드 FSDP (실분산)
./16-compare-strategies.sh mini             # 해시 대조 (기본 all)
# 26/28도 동일하게 [mini|full] [single|ddp|fsdp]를 받는다 (출력 ...-<mode>-<strat>)

# 2) 합성데이터: 선별 → 프롬프트 → 생성 → 후처리
./18-select-candidates.sh mini
./20-build-prompts.sh mini
./22-teacher-to-generate-syn-data.sh mini
./24-postprocess.sh mini

# 3) 합성 SFT: FFT vs QLoRA + merge
./26-fft-train.sh mini single
./28-qlora-train.sh mini single
./30-merge-lora.sh mini              # → synthetic-qlora-mini-single-merged/ (bf16)

# 4) PTQ-FFT (택1~전체, 입력=single): AWQ / FP8 / GGUF 5종 / GPTQ
./32-quant-fft-awq.sh mini
./34-quant-fft-fp8.sh mini
./36-quant-fft-gguf.sh mini          # QTYPES="Q8_0 Q4_K_M" 부분 실행 가능
./38-quant-fft-gptq.sh mini          # → HF 3종 + GGUF 5종

# 5) PTQ-QLoRA (택1~전체, 입력=merged): 동일 4종 → HF 3종 + GGUF 5종, 양자화 총 16종
./40-quant-qlora-awq.sh mini
./42-quant-qlora-fp8.sh mini
./44-quant-qlora-gguf.sh mini
./46-quant-qlora-gptq.sh mini

# 6) PPL 게이트 (base + FP 2종 + 양자화 16종, Δ<0.3 Accept / 0.3~1.0 Conditional / ≥1.0 Discard)
./48-measure-ppl.sh mini                 # → outputs/ppl-mini-single/ (GGUF 10종은 llama-perplexity)

# 7) 평가 (HF 9종 + GGUF 10종) → 채점 → 리포트
./50-eval.sh mini
./52-eval-gguf.sh mini   # llama-completion 필요, 미빌드 시 SKIP
./54-score.sh mini
./56-report.sh mini      # PPL+score 종합 (→ outputs/report-mini-single/)

# 8) HF 업로드 → vLLM/llama-server 서빙 → 추론 예제
./58-upload-hf.sh mini            # → tayaee/Llama-3.2-1B-math-*-mini (targets 지정 가능)
./60-serve-vllm.sh mini fft      # 터미널1: OpenAI-호환 서버 (:8000)
./62-infer-examples.sh tayaee/Llama-3.2-1B-math-fft-mini  # 터미널2
# SOURCE=hf ./60-serve-vllm.sh mini fft-gptq  # Hub repo 직접 서빙 예시
# ./61-serve-gguf.sh mini fft-gguf            # GGUF는 llama-server (vLLM 미지원)
# ./64-serve-infer-all.sh 3                   # 전 모델 순회 서빙+추론 (단일 터미널)
```

## 분산 학습 (ray/k3s)

- L1 torchrun: 위 스크립트가 `INFRA=dgx-spark-2x`일 때
  `--nnodes=2 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR` 로 분기
  (rank0=spark1, `MASTER_ADDR`는 spark1 IP).
- L2 ray: `distributed/ray-10-baseline.py` (workers=1|2).
- L3 k3s: `distributed/k8s-job-torchrun.yaml` (스켈레톤, 공용경로 마운트).

## 학습 전략 3종 (single / ddp / fsdp)

- `single`: 단일 GPU 단일 프로세스. 산출물 `...-<mode>-single` — **Stage 4·5·7 하류 입력**.
- `ddp`: 2노드 DDP. `INFRA=dgx-spark-2x` + `MASTER_ADDR`(spark1 IP) 필수.
- `fsdp`: FSDP full_shard. 1x에서는 1proc으로 동작만 학습(메모리 이득 없음),
  2x에서는 2proc 실분산.
- effective batch는 월드 크기에 맞춰 자동 동일화
  (micro 2 × accum × world = 64: single accum 32, 2proc accum 16).
- **결과가 똑같은가?** effective batch·seed가 같으면 정확도는 거의 같지만
  bit-identical은 아니다. DDP allreduce와 FSDP reduce-scatter+allgather의
  부동소수점 합산 순서가 다르고, dataloader 샤딩으로 배치 구성도 달라진다.
  `./16-compare-strategies.sh`로 해시를 직접 대조해 확인할 것.
  ddp/fsdp 산출물은 비교용으로 보관하고 하류에서는 single만 쓴다.
