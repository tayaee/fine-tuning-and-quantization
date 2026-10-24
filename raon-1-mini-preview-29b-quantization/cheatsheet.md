# cheatsheet — 자주 쓰는 명령 모음 (4종: nvfp4-gptq, nvfp4-awq, awq-w5, awq-w3)

```bash
# 0. 환경 (양 노드 1회) + gated 로그인
./10-setup.sh
uv run hf auth login

# 1. calib (smoke 8 / full 256)
./13-prep-calibration.sh smoke
./13-prep-calibration.sh full

# 2. compat (양자화 전 필수)
./16-compat-check.sh

# 3. dequant (22~31의 선행 조건, 1회)
./19-dequant-mxfp8.sh

# 3. PTQ 개별 (smoke 예시, calib 수 직접 지정 가능)
./22-quant-awq-w3.sh smoke
./25-quant-awq-w5.sh smoke
./28-quant-nvfp4-awq.sh smoke
./31-quant-nvfp4-gptq.sh smoke
RQ_AWQ_FIRST=0 ./22-quant-awq-w3.sh smoke   # custom_arch AWQ 실패 시 pure-GPTQ 폴백
RQ_TARGETS=all ./31-quant-nvfp4-gptq.sh smoke  # experts-only 대신 전 Linear

# 4. PPL (일부 타깃만 — 타겟마다 별도 프로세스, 마지막에 캐시 집계)
./34-measure-ppl.sh smoke
./34-measure-ppl.sh smoke nvfp4-gptq,nvfp4-awq

# 5. eval + score (백엔드: auto = vLLM → 커스텀 아키텍처 미지원 시 transformers)
./37-eval.sh smoke nvfp4-gptq
./40-score.sh smoke
RQ_EVAL_BACKEND=transformers ./37-eval.sh smoke nvfp4-gptq

# 6. 모델카드 및 업로드 전 감사 (README.md 생성 + 디렉터리·모델사이즈·vLLM·지표 전수 검증)
./43-prep-upload.sh smoke
./43-prep-upload.sh smoke nvfp4-gptq,nvfp4-awq
./46-verify-upload.sh smoke nvfp4-awq
./46-verify-upload.sh full nvfp4-awq

# 7. 업로드·서빙 (full만 업로드, 43-prep 산출물이 있어야 함)
DRY_RUN=1 ./49-upload-hf.sh full     # 네트워크 없이 준비 상태만 검증
./49-upload-hf.sh full
./52-serve-vllm.sh smoke nvfp4-gptq  # 터미널1
./55-infer-examples.sh 3             # 터미널2
./58-serve-infer-all.sh smoke 3      # 순회

# 단일 타깃 원샷 (mode × target = 8종) — dequant→quant→PPL→eval (+게이트)
#   10-16,19,{22|25|28|31}-40-build-{smoke,full}-{awq-w3|awq-w5|nvfp4-awq|nvfp4-gptq}.sh
#   (파일명 앞 숫자 = 실행 스텝 번호: 10-setup·13-calib·16-compat·19-dequant·<quant>·34-ppl·37-eval·40-score)
./10-16,19,28-40-build-smoke-nvfp4-awq.sh    # smoke × nvfp4-awq
./10-16,19,28-40-build-full-nvfp4-awq.sh     # full  × nvfp4-awq
STEPS=quant,ppl ./10-16,19,28-40-build-smoke-nvfp4-awq.sh      # 일부 스텝만
STEPS=prep,verify ./10-16,19,28-40-build-smoke-nvfp4-awq.sh    # 모델카드(43)+감사(46)까지
FORCE=1 ./10-16,19,28-40-build-smoke-nvfp4-awq.sh               # freshness 무시(강제 재계산)
# 게이트: gate-quant(패킹 계약·global_scale) → gate-ppl(유한 PPL) → gate-eval(응답 생성)
#        게이트 실패 시 즉시 중단 — NaN 모델을 eval에 흘려보내지 않는다.
# 공유 접두(10/13/16/19)는 lock+freshness/stamp로 중복 실행이 자동 회피된다.
#   (2개 build 동시 실행 → 먼저 들어온 쪽만 수행, 나머지는 대기 후 skip)

# 원샷
./run-smoke.sh
SKIP_SETUP=1 STEPS=quant,ppl ./run-smoke.sh
QUANT_CPU_ONLY=1 ./run-smoke.sh      # 양자화만 CPU 강제, PPL/eval은 GPU
./run-full.sh

# 2노드 분산 (smoke/full 동일 분할, dequant 공유·PPL/eval은 1곳)
TARGETS=awq-w3,awq-w5 STEPS=dequant,quant ./run-smoke.sh  # spark1
TARGETS=nvfp4-awq,nvfp4-gptq STEPS=dequant,quant ./run-smoke.sh  # spark2
STEPS=ppl,eval ./run-smoke.sh  # 전 타깃 모인 뒤 1곳에서
# full도 동일: MODE만 full (run-full.sh + 같은 TARGETS/STEPS)
```

env 핀: `INFRA=dgx-spark-2x TP=2 MASTER_ADDR=<spark1-ip> ./52-serve-vllm.sh smoke nvfp4-gptq`
