이 파이프라인이 무엇을 하고 있는가

범용 베이스 모델 unsloth/Llama-3.2-1B
    수학 파인 튜닝 base-gsm8k-mini-single [10] (하류에서 로드 안함, 12번 해시 대조용)
		수학 풀 파인튜닝 모델 synthetic-fft-mini-single [30] (하류 입력은 single만)
			awq 양자화 모델 synthetic-fft-mini-single-awq [40]
			fp8 양자화 모델 synthetic-fft-mini-single-fp8 [41]
			gguf 양자화 모델 synthetic-fft-mini-single-gguf [42]
			gptq 양자화 모델 synthetic-fft-mini-single-gptq [43]
		수학 qlora 어댑터 synthetic-qlora-mini-single [31]
			수학 qlora 모델 synthetic-qlora-mini-single-merged [32]
				awq 양자화 모델 synthetic-qlora-mini-single-merged-awq [50]
				fp8 양자화 모델 synthetic-qlora-mini-single-merged-fp8 [51]
				gguf 양자화 모델 synthetic-qlora-mini-single-merged-gguf [52]
				gptq 양자화 모델 synthetic-qlora-mini-single-merged-gptq [53]

Notes:
	- mini: 최소 데이터를 사용한 파이프라인 점검용 (다른 옵션: full)
	- single: 싱글 노드 싱글 GPU 사용 파이프라인 (다른 옵션: ddp, fsdp)

데이터 파이프라인 중간 정리
	unsloth/Llama-3.2-1B 범용 모델
	$M1B_SHARED/datasets/gsm8k-train.jsonl 7473개의 수학 문제 ({"source":[q], "target":[a]} 리스트형)
	$M1B_SHARED/datasets/alpaca-embeddings.parquet 52002개 일반 문장 (257MB, 모드 공용 캐시)
	data/·assets/ 원본을 10-setup.sh가 $M1B_SHARED/datasets/로 한번 복사, 하류는 data/가 아니라 $M1B_SHARED를 읽는다

10 베이스라인 학습 (순수 FFT bf16, --peft 없음, --peft는 31에서만)
	unsloth/Llama-3.2-1B 베이스모델을
	$M1B_SHARED/datasets/gsm8k-train.jsonl 로 풀 파인튜닝 (train-entry.py 공통 진입점)
	$M1B_SHARED/models/base-gsm8k-mini-single/ 아래에 저장 (7개 파일 + checkpoint-4 최신, checkpoint-117 구버전 잔존)
	lr 1e-5 cosine, warmup 0.03, micro 2 x accum 32 = effective 64, max_len 1024, seed 1, mini 256행 4스텝 / full 3 epoch
	이 산출물은 하류(30/31/40~43/50~53)에서 로드하지 않는다. 30/31은 원본 pretrained에서 다시 학습, 50-eval --target base도 원본을 평가

20 ~ 23 까지는 파인튜닝 학습 데이터 생성

20
	tatsu-lab/alpaca의 train 전체를 all-mpnet-base-v2로 500개 배치 인코딩하여 instruction, input, output, embedding 을
	$M1B_SHARED/datasets/alpaca-embeddings.parquet 에 저장함 (없을 때만 빌드, full 52k 인코딩 원가 한번만)

	$M1B_SHARED/datasets/gsm8k-train.jsonl 을 읽어서 source + target의 임베딩 배열을 구한 후 (mini는 GSM 앵커 256개만, full은 전량 7473개, Alpaca 52k는 항상 전체와 비교)
	$M1B_SHARED/datasets/alpaca-embeddings.parquet 의 embedding 과 cosine_similarity(Gx52k, full시 1.5GB)를 구한다.

	유사도 기준 GSM 문제마다 상위 1000개 안에서 100개를 무작위 샘플링 (top-100이 아니라 다양성 샘플링) 하여 gsm_idx, dg_instruction, dg_input, dg_output 필드를
	$M1B_SHARED/datasets/candidates-mini.parquet에 저장 (mini 256x100=25600행, full 747300행 중 21이 10000만 취함, 20은 HF push 없음)

21
	$M1B_SHARED/datasets/candidates-mini.parquet 에서 dg_instruction, dg_input, dg_output 을 읽고
	assets/template.txt (단수)에 dg_instruct, dg_output 을 치환한 결과를 prompt 필드로 생성하여 (dg_instruct = instruction + "\n" + input, Example 1~3 고정 few-shot + Example 4 자리, {{ }} 이스케이프)
	$M1B_SHARED/datasets/prompts-mini.parquet 에 저장 (shuffle sample(frac=1) 후 head(SYNTH_N), mini 64 / full 10000, prompt 1컬럼)

22 (teacher 생성, 증류 구조)
	$M1B_SHARED/datasets/prompts-mini.parquet 를 읽어서 teacher meta-llama/Llama-3.1-8B-Instruct (student는 1B)를 apply_chat_template([{role:user}]) 후 vLLM으로 문장 생성 (SamplingParams temp 0.5, top_p 0.8, top_k 5, repetition_penalty 1.05, max_tokens 2048, max_model_len 8192, tp=INFRA 1x->1/2x->2)
	invalid행만 최대 3회 재생성하고 0이면 조기 종료 (valid = Transformed Domain Question + Transformed Domain Answer + The answer is 3마커 포함) 후 prompt, generated 2컬럼 전행을
	$M1B_SHARED/datasets/generated-mini.csv 에 저장 (현 200행은 이전 SYNTH_N=200 유산, 현재 64와 불일치)

23
	$M1B_SHARED/datasets/generated-mini.csv 을 열어 두 마커가 각각 정확히 1회 있는 행만 남기고 (count>1 drop) 질문/답 분리 후 잔여 마커·**·:·-·번호 정규식 제거 + strip + 길이<10 drop 하고, 정상 레코드를 source, target 리스트형 필드로 분리하여 (GSM8K 동일 스키마)
	$M1B_SHARED/datasets/synthetic-mini.jsonl 파일에 저장 (mini kept=188 / dropped=12, 200행 기준)

30 full 파인튜닝 (출력 bf16)
	unsloth/Llama-3.2-1B 베이스 모델을
	$M1B_SHARED/datasets/synthetic-mini.jsonl (source, target) 로 full 파인튜닝하여 (no --peft, --max_rows $GSM_ROWS mini 256 / full 0=무제한, mini는 188<256이라 무영향)
	$M1B_SHARED/models/synthetic-fft-mini-single/ 아래에 저장 (7개 파일, checkpoint-3, 188/64=3스텝 undertraining이라 mini 평가 acc 0.000)
	토크나이저에 [PAD]/<mask> 추가 + resize_token_embeddings, 실측 vocab 128257 (<mask>는 원래 있어 [PAD]만 추가, 신규 임베딩은 평균값 초기화), mask_rate 0 = masking off

31 qlora 파인튜닝 (작은 어댑터 아님, 4비트 가중치 저장 안함)
	unsloth/Llama-3.2-1B 베이스 모델을
	$M1B_SHARED/datasets/synthetic-mini.jsonl (source, target) 로 qlora 학습 (--peft, bnb nf4 + double_quant + compute bf16, LoRA r16/a32/dropout0.05, target q,k,v,o,gate,down,up, modules_to_save=["embed_tokens","lm_head"]) 하여
	$M1B_SHARED/models/synthetic-qlora-mini-single/ 아래에 저장 (6개 어댑터 파일, 1개 체크포인트, adapter_model.safetensors 4.0GB = embed/lm_head 원본+신규 사본 230 tensors 1.59B)

40 awq 양자화 (full 파인튜닝 출력 bf16 이 여기의 입력)
	$M1B_SHARED/models/synthetic-fft-mini-single/ full 파인튜닝 모델을 로딩하고
	$M1B_SHARED/datasets/synthetic-mini.jsonl 에서 CALIB_N개만 캘리브레이션에 사용 (mini 2 / full 10)
	AWQModifier(scheme="W4A16", targets=["Linear"], ignore=["lm_head"]) 양자화 레시피를 만든 후, oneshot(dataset, num_calibration_samples=calib, max_seq_length=1024, output_dir=out) 호출로 양자화하여
	$M1B_SHARED/models/synthetic-fft-mini-single-awq/ 에 저장 (6개 파일, 980MB, compressed-tensors라 평가시 quantization="compressed-tensors", gptq와 달리 llmcompressor oneshot)

41 fp8 양자화 (full 파인튜닝 출력 bf16 이 여기의 입력)
	$M1B_SHARED/models/synthetic-fft-mini-single/ full 파인튜닝 모델을 로딩하고
	$M1B_SHARED/datasets/synthetic-mini.jsonl 에서 CALIB_N개만 캘리브레이션에 사용 (mini 2 / full 10, 로딩 상한만 max(calib,64))
	QuantizationModifier(scheme="FP8") 양자화 레시피 (W8A8 FP8_E4M3, 활성화 스케일 동적이라 캘리브 의존도 낮음, AWQ와 oneshot 동일 계열, GPTQ만 라이브러리 다름) 만든 후, oneshot() 호출로 양자화 한 후 저장하여
	$M1B_SHARED/models/synthetic-fft-mini-single-fp8/ 에 저장 (6개 파일, 1.43GB, compressed-tensors).

42 gguf 양자화 (full 파인튜닝 출력 bf16 이 여기의 입력, Q8_0/Q6_K/Q5_K_M/Q4_K_M/Q3_K_M 5종 루프)
	$M1B_SHARED/models/synthetic-fft-mini-single/ 입력을 convert_hf_to_gguf.py를 사용하여
	$M1B_SHARED/models/synthetic-fft-mini-single-gguf/model-f16.gguf 로 1차 변환 저장함 (bf16 -> f16 2.37GB).
	$M1B_SHARED/models/synthetic-fft-mini-single-gguf/model-f16.gguf 를 llama-quantize 도구를 사용하여
	model-{q8_0,q6_k,q5_k_m,q4_k_m,q3_k_m}.gguf 5종으로 양자화 저장 (mini 실측 q8_0 1.3G / q6_k 975M / q5_k_m 870M / q4_k_m 771M / q3_k_m 659M, 추가 데이터 불필요, QTYPES로 부분 실행)

43 gptq 양자화 (full 파인튜닝 출력 bf16 이 여기의 입력)
	$M1B_SHARED/models/synthetic-fft-mini-single/ full 파인튜닝 모델을 로딩하고
	$M1B_SHARED/datasets/synthetic-mini.jsonl 에서 CALIB_N개만 캘리브레이션에 사용 (mini 2 / full 10, 로딩 상한 max(calib,10), 188개 전부가 아님)
	QuantizeConfig(bits=4, group_size=128) 양자화 구성 이용하여, load(src, qc), quantize(calibration=texts[:calib], tokenizer) 호출로 양자화 한 후 save() + tokenizer 저장하여
	$M1B_SHARED/models/synthetic-fft-mini-single-gptq/ 에 저장 (7개 파일, quant_log.csv 포함, 985MB)
	실측 용량: FFT bf16 2.47GB -> AWQ 980MB / FP8 1.43GB / GGUF 5종 1.3G~659M / GPTQ 985MB

32 어댑터 머지 (31 직후 실행, 50~53의 입력)
	31 출력 $M1B_SHARED/models/synthetic-qlora-mini-single/ 어댑터를 로딩하여
	학습 때와 동일한 base id (unsloth/Llama-3.2-1B)로 bf16 로딩 전에 resize_token_embeddings(len(tok)) 먼저 호출 (생략시 size mismatch), PAD/[PAD]·<mask>는 vocab에 없으면만 추가하고
	merge_and_unload() 호출하여 합치고, 모델/토크나이저에 대해 save_pretrained() 각각 호출하여
	$M1B_SHARED/models/synthetic-qlora-mini-single-merged/ 아래에 모델과 토크나이저를 모두 저장함 (5개 파일, 2.99GB 1498M params, tie 풀려 FFT 1235.7M보다 큼)

50~53 qlora-merged 양자화 (입력만 merged, 레시피는 40~43과 동일)
	$M1B_SHARED/models/synthetic-qlora-mini-single-merged/ 를 입력으로 50-awq / 51-fp8 / 52-gguf / 53-gptq 동일하게 뽑음 (CALIB_N mini 2 / full 10)

60 ppl 측정
	48-measure-ppl.sh/py 로 모델별 in-domain perplexity 측정 + base 대비 Δ판정 (GGUF 10종은 llama-perplexity로 측정)

71 평가
	50-eval.sh/py 로 vLLM greedy(temp 0, max_tokens 512) GSM8K 추론, 타깃 base/fft/qlora/fft-gptq/fft-awq/fft-fp8/qlora-gptq/qlora-awq/qlora-fp8 (기본 all, GGUF 10종은 52-eval-gguf.sh로 llama-completion 평가)
	평가 프롬프트는 prompt_no_input 고정 템플릿, gsm8k-test 앞 EVAL_N개 (mini 10 / full 0=전체), 출력 $M1B_SHARED/outputs/eval-<mode>/<target>.jsonl, 양자화 타깃은 quantization flag 부여

72 스코어
	54-score.py 로 #### <숫자> 추출, 없으면 The answer is: X 보조 추출, 둘 다 없으면 [invalid] -> 타깃별 acc / invalid율 표 (mini는 3스텝 undertraining이라 전 타깃 acc 0.000 / invalid 0.98~1.00)

73 HF 업로드
	58-upload-hf.sh/py 로 TARGETS 맵대로 tayaee/<slug>-math-<target>-<mode> 업로드, 없는 산출물 SKIP, --dry-run 지원

80 서빙
	60-serve-vllm.sh 로 vLLM OpenAI-호환 서버(:8000), --served-model-name tayaee/..., 타깃별 --quantization (gptq vs compressed-tensors vs gguf), SOURCE=hf면 Hub 직접 서빙, VLLM_ATTENTION_BACKEND=FLASH_ATTN + VLLM_USE_FLASHINFER_SAMPLER=0

81 추론 예시
	62-infer-examples.py 로 gsm8k-test 앞 N개 + 한국어 문제 1개를 completions API로 요청 (chat API는 chat template 없어 400)

12 전략 비교
	16-compare-strategies.sh 로 single/ddp/fsdp 체크포인트 safetensors sha256 대조 (effective batch 64 고정해도 bit-identical 아님, allreduce vs reduce-scatter 합산 순서 + dataloader 샤딩 차이)

3개 축
	mini|full: SYNTH_N 64/10000, EPOCHS 1/3, CALIB_N 2/10, EVAL_N 10/전체, GSM_ROWS 256/전체, 산출물 -<mode>- 로 구분
	single|ddp|fsdp: ACCUM = 64 / (micro 2 x world) 로 effective 64 고정 (single 32, 2proc 16), 하류(4x·32·5x·7x)는 -single만 사용
	INFRA=dgx-spark-1x|2x: TP 1<->2, NNODES, MASTER_ADDR 필요
