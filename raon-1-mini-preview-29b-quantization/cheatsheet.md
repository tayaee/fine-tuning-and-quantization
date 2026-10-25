# cheatsheet — 자주 쓰는 명령 모음 (4종: nvfp4-gptq, nvfp4-awq, awq-w5, awq-w3)

## 모드 지정 (smoke / full)

모든 stage 스크립트는 **인자로 모드를 받는다**. 인자를 생략하면 환경변수 `MODE`를 보고,
그것도 없으면 `smoke`다.

```bash
./s8-43-prep-upload.sh full              # 인자 우선
MODE=full ./s8-43-prep-upload.sh         # 환경변수 (아무 인자 없이)
MODE=full TARGETS=nvfp4-awq ./s8-43-prep-upload.sh   # 타깃도 함께
```

> `export MODE=full` 후 인자 없이 실행하면 **full**로 돈다.
> (옛 wrapper는 `"${1:-smoke}"`로 환경변수를 덮어써서 항상 smoke였음 — issue-18에서 수정)

```bash
# 0. 환경 (양 노드 1회) + gated 로그인
./s0-10-setup.sh
uv run hf auth login

# 1. 수용 기준 (Stage 1) — config/acceptance.toml을 검증해 계약 스냅샷을 만든다
./s1-07-acceptance.sh smoke
./s1-07-acceptance.sh full

# 1. calib (smoke 8 / full 256)
./s2-13-prep-calibration.sh smoke
./s2-13-prep-calibration.sh full

# 2. compat (양자화 전 필수)
./s3-16-compat-check.sh

# 3. dequant (22~31의 선행 조건, 1회)
./s4-19-dequant-mxfp8.sh

# 3. PTQ 개별 (smoke 예시, calib 수 직접 지정 가능)
./s5-22-quant-awq-w3.sh smoke
./s5-25-quant-awq-w5.sh smoke
./s5-28-quant-nvfp4-awq.sh smoke
./s5-31-quant-nvfp4-gptq.sh smoke
RQ_AWQ_FIRST=0 ./s5-22-quant-awq-w3.sh smoke   # custom_arch AWQ 실패 시 pure-GPTQ 폴백
RQ_TARGETS=all ./s5-31-quant-nvfp4-gptq.sh smoke  # experts-only 대신 전 Linear

# 4. PPL (일부 타깃만 — 타겟마다 별도 프로세스, 마지막에 캐시 집계)
./s6-34-measure-ppl.sh smoke
./s6-34-measure-ppl.sh smoke nvfp4-gptq,nvfp4-awq

# 게이트 (run-{smoke,full}.sh가 자동 호출 — 수동으로 확인할 때)
bash -c 'source config/common.env smoke; source gates.inc; rq_gate_quant "$RQ_MODELS/raon-nvfp4-awq-smoke" nvfp4-pack-quantized'
bash -c 'source config/common.env smoke; source gates.inc; rq_gate_ppl "$RQ_OUTPUTS/ppl-smoke/nvfp4-awq.json" nvfp4-awq'

# 4.5 평가 세트 준비 (36 GSM8K / 43 한국어 KMMLU / 49 형식지시) — 결정론, 멱등
./s7-36-prep-gsm8k.sh smoke
./s7-36-prep-gsm8k.sh full                # GSM8K_N=0 이면 표준 테스트셋 전체 1319
GSM8K_N=300 EVAL_N=200 ./s7-36-prep-gsm8k.sh full   # 표본 300 중 200문제만 풀기
./s7-43-prep-kmmluko.sh smoke             # 표준 한국어 KMMLU (KMMLU_N=40, 과목 8종)
./s7-43-prep-kmmluko.sh full              # KMMLU_N=200 으로 확대
./s7-49-format.sh smoke --prepare         # 형식/지시 준수 고정 10케이스(eval 전에 1회)

# 5. eval + score (백엔드: auto = vLLM → 커스텀 아키텍처 미지원 시 transformers)
#    EVAL_BATCH=8 로 배치 생성(표준 GSM8K 실측에 필수)
#    EVAL_BENCHMARKS 기본값 = gsm8k,kmmluko,format (3개 벤치 동시 평가)
./s7-37-eval.sh smoke nvfp4-gptq
./s7-40-gsm8k.sh smoke
RQ_EVAL_BACKEND=transformers ./s7-37-eval.sh smoke nvfp4-gptq

# 7.5 검사 채점 (40 GSM8K / 46 한국어 / 49 형식) + 서빙 처리량(52, 서버 필요)
./s7-40-gsm8k.sh smoke        # → outputs/eval-<mode>/score.json
./s7-46-kmmluko.sh smoke     # → checks.json "kmmluko"
./s7-49-format.sh smoke      # → checks.json "format"
STEPS=serve ./run-smoke.sh   # 서빙 처리량 측정(s7-52, 서버가 떠 있어야)

# 6. 모델카드 및 업로드 전 감사 (README.md 생성 + 디렉터리·모델사이즈·vLLM·지표 전수 검증)
#    - 모델 디렉터리 README.md = base + 그 모델 1개 비교 (Hub 업로드 대상)
#    - models/README-<mode>.md      = 그 모드의 전체 양자화 버전 비교 리포트 (업로드 아님)
./s8-43-prep-upload.sh smoke
./s8-43-prep-upload.sh smoke nvfp4-gptq,nvfp4-awq
./s8-46-verify-upload.sh smoke nvfp4-awq
./s8-46-verify-upload.sh full nvfp4-awq

# 7. 업로드·서빙 (full만 업로드, 43-prep 산출물이 있어야 함)
DRY_RUN=1 ./s8-49-upload-hf.sh full     # 네트워크 없이 준비 상태만 검증
./s8-49-upload-hf.sh full
./s8-52-serve-vllm.sh smoke nvfp4-gptq  # 터미널1
./s8-55-infer-examples.sh 3             # 터미널2
./s8-58-serve-infer-all.sh smoke 3      # 순회

# 8. 릴리스 증빙 — 단계 원장 → 릴리스 노트 / 배포 후 서빙 검증
python s8-61-release-notes.py --mode smoke     # outputs/release-notes-smoke.md
python s8-64-verify-serving.py --mode smoke   # logs/55.*.log → outputs/serve-smoke.json
STEPS=verify TARGETS=all ./run-smoke.sh    # 46-verify + gate-release (명시 시에만)

# 단일 타깃 원샷 (mode × target = 8종) — dequant→quant→PPL→eval (+게이트)
#   s0-s7-build-{smoke,full}-{awq-w3|awq-w5|nvfp4-awq|nvfp4-gptq}.sh
#   (파일명 앞 숫자 = 실행 스텝 번호: s0-10-setup·s2-13-calib·s3-16-compat·s4-19-dequant·s5-<quant>·s6-34-ppl·s7-37-eval·s7-40-gsm8k)
./s0-s7-build-smoke-nvfp4-awq.sh    # smoke × nvfp4-awq
./s0-s7-build-full-nvfp4-awq.sh     # full  × nvfp4-awq
STEPS=quant,ppl ./s0-s7-build-smoke-nvfp4-awq.sh      # 일부 스텝만
STEPS=prep,verify ./s0-s7-build-smoke-nvfp4-awq.sh    # 모델카드(43)+감사(46)까지
FORCE=1 ./s0-s7-build-smoke-nvfp4-awq.sh               # freshness 무시(강제 재계산)
# 게이트: gate-quant(패킹 계약·global_scale) → gate-ppl(유한 PPL) → gate-eval(응답 생성)
#        게이트 실패 시 즉시 중단 — NaN 모델을 eval에 흘려보내지 않는다.
# 공유 접두(10/13/16/19)는 lock+freshness/stamp로 중복 실행이 자동 회피된다.
#   (2개 build 동시 실행 → 먼저 들어온 쪽만 수행, 나머지는 대기 후 skip)

# 원샷
# TARGETS는 반드시 명시 (미지정하면 가능한 값을 찍고 종료 — 자동 전체 실행 없음)
TARGETS=all ./run-smoke.sh
TARGETS=nvfp4-awq ./run-smoke.sh      # 1종만
SKIP_SETUP=1 STEPS=quant,ppl TARGETS=all ./run-smoke.sh
QUANT_CPU_ONLY=1 TARGETS=all ./run-smoke.sh   # 양자화만 CPU 강제, PPL/eval은 GPU
TARGETS=all ./run-full.sh

# 2노드 분산 (smoke/full 동일 분할, dequant 공유·PPL/eval은 1곳)
TARGETS=awq-w3,awq-w5 STEPS=dequant,quant ./run-smoke.sh  # spark1
TARGETS=nvfp4-awq,nvfp4-gptq STEPS=dequant,quant ./run-smoke.sh  # spark2
STEPS=ppl,eval TARGETS=all ./run-smoke.sh  # 전 타깃 모인 뒤 1곳에서
# full도 동일: MODE만 full (run-full.sh + 같은 TARGETS/STEPS)
```

env 핀: `INFRA=dgx-spark-2x TP=2 MASTER_ADDR=<spark1-ip> ./s8-52-serve-vllm.sh smoke nvfp4-gptq`
