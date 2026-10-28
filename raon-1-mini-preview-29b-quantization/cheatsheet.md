# cheatsheet — 자주 쓰는 명령 모음 (4종: nvfp4-gptq-transformers, nvfp4-awq-transformers, awq-w5-transformers, awq-w3-transformers)

## 모드 지정 (smoke / full)

모든 stage 스크립트는 **인자로 모드를 받는다**. 인자를 생략하면 환경변수 `MODE`를 보고,
그것도 없으면 `smoke`다.

```bash
./s4-31-prep-upload.sh full nvfp4-awq-transformers     # 인자 우선 (targets 필수)
MODE=full ./s4-31-prep-upload.sh full nvfp4-awq-transformers   # 환경변수 MODE (타깃 인자는 그대로 필수)
MODE=full TARGETS=nvfp4-awq-transformers ./s4-31-prep-upload.sh   # 타깃도 함께
```

> `export MODE=full` 후 인자 없이 실행하면 **full**로 돈다.
> (옛 wrapper는 `"${1:-smoke}"`로 환경변수를 덮어써서 항상 smoke였음 — issue-18에서 수정)

```bash
# s0. 베이스모델 검증 (transformers half + vLLM half, 각 chat·pi·벤치)
./s0-01-load-transformers.sh full      # transformers 서버 기동 (endpoint 상주)
./s0-04-chat-transformers.sh full       # endpoint chat 검증
./s0-07-picall-transformers.sh full     # endpoint pi 툴콜 검증 (고정 질문)
./s0-10-bench-transformers.sh full      # endpoint 벤치 측정 + 서버 정리
./s0-11-stop-transformers.sh            # 잔존 transformers 서버 수동 정리 (3단위 예외)
./s0-13-load-vllm.sh full               # 포크로 베이스 서빙 (서버 상주)
./s0-16-chat-vllm.sh full               # 서빙 chat 검증
./s0-19-picall-vllm.sh full             # 서빙 모델 pi 툴콜 검증 (고정 질문)
./s0-22-bench-vllm.sh full              # vLLM endpoint 벤치 측정 + 서버 정리
./s0-21-stop-vllm.sh                    # 잔존 vLLM 서버 수동 정리 (3단위 예외)

# s1. 환경 체인 (setup·수용기준·calib·compat — 양 노드 setup 1회 + gated 로그인)
./s1-01-setup.sh
uv run hf auth login

# s1. 환경 체인 — 수용 기준: config/acceptance.toml을 검증해 계약 스냅샷을 만든다
./s1-04-acceptance.sh smoke
./s1-04-acceptance.sh full

# s1. 환경 체인 — calib (smoke 8 / full 256)
./s1-07-prep-calibration.sh smoke
./s1-07-prep-calibration.sh full

# s1. 환경 체인 — compat (양자화 전 필수)
./s1-10-compat-check.sh

# s2. dequant (s3 양자화의 선행 조건, 1회)
./s2-01-dequant-mxfp8.sh

# s3. PTQ 개별 (smoke 예시, calib 수 직접 지정 가능)
./s3-01-quant-awq-w3-transformers.sh smoke
./s3-01-quant-awq-w5-transformers.sh smoke
./s3-01-quant-nvfp4-awq-transformers.sh smoke
./s3-01-quant-nvfp4-gptq-transformers.sh smoke
RQ_AWQ_FIRST=0 ./s3-01-quant-awq-w3-transformers.sh smoke   # custom_arch AWQ 실패 시 pure-GPTQ 폴백
RQ_TARGETS=all ./s3-01-quant-nvfp4-gptq-transformers.sh smoke  # experts-only 대신 전 Linear

# s4. 측정·평가 — PPL (일부 타깃만 — 타겟마다 별도 프로세스, 마지막에 캐시 집계, targets 필수)
TARGETS=all ./s4-01-measure-ppl.sh smoke
./s4-01-measure-ppl.sh smoke nvfp4-gptq-transformers,nvfp4-awq-transformers

# 게이트 (build-ppl-eval.inc가 자동 호출 — 수동으로 확인할 때)
bash -c 'source config/common.env smoke; source gates.inc; rq_gate_quant "$RQ_MODELS/raon-nvfp4-awq-transformers-smoke" nvfp4-pack-quantized'
bash -c 'source config/common.env smoke; source gates.inc; rq_gate_ppl "$RQ_OUTPUTS/ppl-smoke/nvfp4-awq-transformers.json" nvfp4-awq-transformers'

# s4. 측정·평가 — 평가 세트 준비 (s4-04 GSM8K / s4-16 한국어 KMMLU / s4-22 형식지시) — 결정론, 멱등
./s4-04-prep-gsm8k.sh smoke
./s4-04-prep-gsm8k.sh full                # GSM8K_N=0 이면 표준 테스트셋 전체 1319
GSM8K_N=300 EVAL_N=200 ./s4-04-prep-gsm8k.sh full   # 표본 300 중 200문제만 풀기
./s4-16-prep-kmmluko.sh smoke             # 표준 한국어 KMMLU (KMMLU_N=40, 과목 8종)
./s4-16-prep-kmmluko.sh full              # KMMLU_N=200 으로 확대
./s4-22-format.sh smoke --prepare         # 형식/지시 준수 고정 10케이스(eval 전에 1회)

# s4. 측정·평가 — eval + score (백엔드: auto = vLLM → 커스텀 아키텍처 미지원 시 transformers)
#    EVAL_BATCH=8 로 배치 생성(표준 GSM8K 실측에 필수)
#    EVAL_BENCHMARKS 기본값 = gsm8k,kmmluko,format (3개 벤치 동시 평가)
./s4-10-eval-quantized.sh smoke nvfp4-gptq-transformers
./s4-13-gsm8k.sh smoke
RQ_EVAL_BACKEND=transformers ./s4-10-eval-quantized.sh smoke nvfp4-gptq-transformers

# s4. 측정·평가 — 검사 채점 (s4-13 GSM8K / s4-19 한국어 / s4-22 형식) + 서빙 처리량(s4-25, 서버 필요)
./s4-13-gsm8k.sh smoke        # → outputs/eval-<mode>/score.json
./s4-19-kmmluko.sh smoke     # → checks.json "kmmluko"
./s4-22-format.sh smoke      # → checks.json "format"
STEPS=serve ./run-s1-to-s4-smoke-nvfp4-awq-transformers.sh   # 서빙 처리량 측정(s4-25, 서버가 떠 있어야)
# transformers 인프로세스 처리량 (서버 불필요, Raon 정석 스택 — stock vLLM은 커스텀 아키 미지원)
./s4-28-bench-transformers.sh smoke base,nvfp4-awq-transformers   # → bench-transformers-<mode>.json + checks.json servebench

# s4. 측정·평가 — 모델카드 및 업로드 전 감사 (README.md 생성 + 디렉터리·모델사이즈·vLLM·지표 전수 검증, targets 필수)
#    - 모델 디렉터리 README.md = base + 그 모델 1개 비교 (Hub 업로드 대상)
#    - models/README-<mode>.md      = 그 모드의 전체 양자화 버전 비교 리포트 (업로드 아님)
TARGETS=all ./s4-31-prep-upload.sh smoke
./s4-31-prep-upload.sh smoke nvfp4-gptq-transformers,nvfp4-awq-transformers
./s4-34-verify-upload.sh smoke nvfp4-awq-transformers
./s4-34-verify-upload.sh full nvfp4-awq-transformers

# s5. 업로드 (full만 업로드, s4-31 산출물이 있어야 함, targets 필수)
TARGETS=all DRY_RUN=1 ./s5-01-upload-hf.sh full     # 네트워크 없이 준비 상태만 검증
TARGETS=all ./s5-01-upload-hf.sh full

# s6. 고객 시나리오 (transformers) — Hub 다운로드 → 로딩 → chat → pi
./s6-01-test-nvfp4-awq-transformers-with-transformers.sh full   # 고객 절차 그대로 (미발행 repo면 FAIL)
./s6-13-chat.sh full tayaee/Raon-1-mini-preview-nvfp4-awq-transformers   # HF 모델명 입력도 가능
./s6-13-chat.sh full nvfp4-awq-transformers       # chat 검증 (all이면 4종 순차)
./s6-16-pi-tool-call-test.sh full nvfp4-awq-transformers         # 임시 서버+pi 툴콜 후 정리

# s7. 고객 시나리오 (vLLM 서빙) — frozen 래퍼 + chat + pi
./s7-01-test-nvfp4-awq-transformers-with-vllm-fork.sh   # 터미널1 (상주, 카드 명령 as-is)
./s7-13-chat.sh full nvfp4-awq-transformers       # 서버 기동+chat 검증 (서버 상주)
./s7-16-pi-tool-call-test.sh full nvfp4-awq-transformers         # pi 검증 후 서버 정리
./s7-19-verify-serving-log.sh full   # s7-13/s7-16 로그 → outputs/serve-full.json

# s4/s7. 릴리스 증빙 — 단계 원장 → 릴리스 노트 / CAT 서빙 검증
python s4-37-release-notes.py --mode smoke     # outputs/release-notes-smoke.md
python s7-19-verify-serving-log.py --mode smoke   # logs/s7-*.log → outputs/serve-smoke.json

# 단일 타깃 원샷 (mode × target = 8종) — dequant→quant→PPL→eval (+게이트)
#   run-s1-to-s4-{smoke,full}-{awq-w3-transformers|awq-w5-transformers|nvfp4-awq-transformers|nvfp4-gptq-transformers}.sh
#   (두 번째 번호 = 그룹 내 순서, 01부터 3단위: s1-01-setup·s1-07-calib·s1-10-compat·s2-01-dequant·s3-<quant>·s4-01-ppl·s4-07-eval·s4-13-gsm8k)
./run-s1-to-s4-smoke-nvfp4-awq-transformers.sh    # smoke × nvfp4-awq-transformers
./run-s1-to-s4-full-nvfp4-awq-transformers.sh     # full  × nvfp4-awq-transformers
STEPS=quant,ppl ./run-s1-to-s4-smoke-nvfp4-awq-transformers.sh      # 일부 스텝만
STEPS=prep,verify ./run-s1-to-s4-smoke-nvfp4-awq-transformers.sh    # 모델카드(s4-31)+감사(s4-34)까지
FORCE=1 ./run-s1-to-s4-smoke-nvfp4-awq-transformers.sh               # freshness 무시(강제 재계산)
# 게이트: gate-quant(패킹 계약·global_scale) → gate-ppl(유한 PPL) → gate-eval(응답 생성)
#        게이트 실패 시 즉시 중단 — NaN 모델을 eval에 흘려보내지 않는다.
# 공유 접두(s1-01/07/10·s2-01)는 lock+freshness/stamp로 중복 실행이 자동 회피된다.
#   (2개 build 동시 실행 → 먼저 들어온 쪽만 수행, 나머지는 대기 후 skip)

# 빌드 (mode × target = 8종 원샷 + STEPS 부분 실행)
# TARGETS는 개별 스크립트에 직접 지정 (미지정하면 가능한 값을 찍고 종료 — 자동 전체 실행 없음)
./s4-01-measure-ppl.sh smoke base,nvfp4-awq-transformers
SKIP_SETUP=1 STEPS=quant,ppl ./run-s1-to-s4-smoke-nvfp4-awq-transformers.sh
QUANT_CPU_ONLY=1 ./run-s1-to-s4-smoke-nvfp4-awq-transformers.sh   # 양자화만 CPU 강제, PPL/eval은 GPU

# 2노드 분산 (타깃별 원샷을 노드에 나눔, dequant 공유·PPL/eval은 1곳)
./run-s1-to-s4-smoke-awq-w3-transformers.sh ./run-s1-to-s4-smoke-awq-w5-transformers.sh  # spark1 (순차)
./run-s1-to-s4-smoke-nvfp4-awq-transformers.sh ./run-s1-to-s4-smoke-nvfp4-gptq-transformers.sh  # spark2 (순차)
# full도 동일: run-s1-to-s4-full-*.sh 8종
# 구 모놀리식 러너(run-{smoke,full}.sh, TARGETS/STEPS 다종 실행)는 archive/로 은퇴.
```

env 핀: `INFRA=dgx-spark-2x TP=2 MASTER_ADDR=<spark1-ip> ./s7-13-chat.sh smoke nvfp4-awq-transformers`
