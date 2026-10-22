# cheatsheet — 자주 쓰는 명령 모음 (4종: nvfp4-gptq, nvfp4-awq, awq-w5, awq-w3)

```bash
# 0. 환경 (양 노드 1회) + gated 로그인
./10-setup.sh
uv run hf auth login

# 1. calib (smoke 8 / full 256)
./12-prep-calibration.sh smoke
./12-prep-calibration.sh full

# 2. compat (양자화 전 필수)
./20-compat-check.sh

# 3. dequant (32~36의 선행 조건, 1회)
./30-dequant-mxfp8.sh

# 3. PTQ 개별 (smoke 예시, calib 수 직접 지정 가능)
./33-quant-awq-w3.sh smoke
./32-quant-awq-w5.sh smoke
./35-quant-nvfp4-awq.sh smoke
./36-quant-nvfp4-gptq.sh smoke
RQ_AWQ_FIRST=0 ./33-quant-awq-w3.sh smoke   # custom_arch AWQ 실패 시 pure-GPTQ 폴백
RQ_TARGETS=all ./36-quant-nvfp4-gptq.sh smoke  # experts-only 대신 전 Linear

# 4. PPL (일부 타깃만)
./48-measure-ppl.sh smoke
./48-measure-ppl.sh smoke nvfp4-gptq,nvfp4-awq

# 5. eval + score
./50-eval.sh smoke nvfp4-gptq
./54-score.sh smoke

# 6. 업로드·서빙 (full만 업로드)
./58-upload-hf.sh full
./60-serve-vllm.sh smoke nvfp4-gptq  # 터미널1
./62-infer-examples.sh 3             # 터미널2
./64-serve-infer-all.sh smoke 3      # 순회

# 원샷
./run-smoke.sh
SKIP_SETUP=1 STEPS=quant,ppl ./run-smoke.sh
./run-full.sh
```

env 핀: `INFRA=dgx-spark-2x TP=2 MASTER_ADDR=<spark1-ip> ./60-serve-vllm.sh smoke nvfp4-gptq`
