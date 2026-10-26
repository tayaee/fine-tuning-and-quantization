# PROCEDURE — Raon-1-mini PTQ only (DGX Spark 1x 기본, 서빙은 2x/TP=2)

> 학습 Stage 없음 (SFT 제외). MXFP8 베이스 → BF16 복원 → 한국어 calib → 4종 PTQ
> → PPL 게이트 → vLLM eval → 업로드·서빙.
> 30B 플랜(65 결과 159~190행) 대비 VRAM 참고: BF16 서빙 32~36GB, AWQ 18~20GB.
> 베이스가 MXFP8이라 4종 전부 이중양자화 전제 (FP8 타깃 제외됨).
> SDLC 정의(9단계·게이트·증빙)는 `docs/SDLC.md` 참고.

## Stage (9단계)

```
[0. 환경 10-setup] → [1. 수용기준 07-acceptance] → [2. calib 13-prep] → [3. compat 16-check]
  → [4. dequant 19] → [5. PTQ 22·25·28·31] → [6. PPL 34] → [7. eval 37·40]
  → [8. 릴리스·배포 43·46·49·52·55·58]
```

> **파일 명명 규칙**: `s<stage>-##-<slug>.{sh,py}` — `s#`은 9단계 중 몇 번째 단계인지
> (0=환경, 1=수용기준, 2=데이터, 3=호환성, 4=복원, 5=양자화, 6=PPL, 7=평가, 8=배포),
> `##`은 그 Stage 안에서의 스텝 번호(실행 순서), `<slug>`은 작업 이름이다.
> 예) `s5-28-quant-nvfp4-awq.sh` = 5단계(양자화)의 28번 스텝.
> 빌드 wrapper(`run-s0-to-s7-<mode>-<target>.sh`)는 **Stage 0~7을 한 번에** 돌리는
> 진입점이므로 `s0-s7-` 접두사를 쓴다(Stage 8의 카드·감사·업로드는 `STEPS=prep,verify`로 선택).

| Stage | 스크립트 | 입력 → 출력 | Exit 게이트 |
|---|---|---|---|
| 0. 환경 | `s0-10-setup.sh` | env sync + HF gated 로그인 + GPU/NVFP4 가능 점검 | stamp(uv.lock 해시 + 핀 버전) |
| 1. 수용기준 | `s1-07-acceptance.sh [smoke\|full]` | `config/acceptance.toml` → `outputs/acceptance-<mode>.json` | 계약 스키마 정상(타깃 4종 기준 존재) |
| 2. calib | `s2-13-prep-calibration.sh [smoke\|full]` | HF 한국어 혼합 → `datasets/raon-calibration-<N>.jsonl` (`text` chat-formatted) | 산출물 존재·최신 |
| 3. compat | `s3-16-compat-check.sh` | trust_remote_code 로드, expert 타깃, MoE 제외층, SM≥10, vLLM 버전 | stamp(probe 해시 + GPU cc + 버전) |
| 4. dequant | `s4-19-dequant-mxfp8.sh` | MXFP8 repo → `models/raon-bf16` (22~31·34·37의 입력, 1회) | 수치 대조 + MoE 전-expert 캘리브레이션 심 존재 |
| 5. PTQ | `22-awq-w3` `25-awq-w5` `28-nvfp4-awq` `31-nvfp4-gptq` | 복원본 → `models/raon-<target>-<mode>` 4종 (experts-only 기본) | **gate-quant**: 패킹 포맷·global_scale·dense 폴백 차단 |
| 6. PPL | `s6-34-measure-ppl.sh [smoke\|full]` | base(raon-bf16) + 4종 in-domain PPL | **gate-ppl**: 유한값 + 수용 기준 Δ 판정 일치 |
| 6.5 평가세트 | `s7-36-prep-gsm8k.sh [smoke\|full]` | 표준 GSM8K test → `outputs/eval-<mode>/prompts-gsm8k.jsonl` + 매니페스트 | 프롬프트 파일 존재(없으면 37-eval이 실패) |
| 7. eval | `s7-36-prep-gsm8k` · `s7-43-prep-kmmluko` · `s7-49-format` → `s7-37-eval` → `s7-40-gsm8k` · `s7-46-kmmluko` · `s7-49-format` · `s7-55-bench-transformers` → `s7-56-prep-upload` → `s7-57-verify-upload` | **5개 검사** + 모델카드 + 업로드 전 감사: ① 표준 GSM8K(영어 수학) ② 표준 한국어 KMMLU ③ 지시/형식 준수 ④ 서빙 처리량(Tier3) ⑤ 카드·감사 — 전부 결정론 생성 | **gate-eval** · **gate-kmmluko** · **gate-format**: 응답 생성 + 정확도/통과율 계약 대조 |
| 8. 릴리스·배포 | `s8-49-upload-hf` → `s8-52`(개별 서빙)/`53`(전 타깃 서빙)/`55`(추론)/`58`(순회) | Hub 업로드 + `outputs/release-notes-<mode>.md` + 서빙 | **gate-release**: 모델카드·지표·업로드 전 감사 / **gate-serving**: 서빙 응답 |

> 판정 기준(Δ<0.3 Accept / 0.3~1.0 Conditional / ≥1.0 Discard), 용량 범위,
> GSM8K 정확도 허용폭(base 대비 하락 ≤ 2%p)은 코드 하드코딩이 아니라
> `config/acceptance.toml`(Stage 1)이 단일 소스다.
>
> **평가는 5개 검사**로 구성한다(각각 표준/결정론):
> ① 표준 GSM8K(`openai/gsm8k` main/test) — 영어 수학 추론 정확도
> ② 표준 한국어 KMMLU(`HAERAE-HUB/KMMLU`) — 한국어 능력(GSM8K가 못 재는 축)
> ③ 지시/형식 준수 스위트(고정 10케이스) — 언어지시·JSON·마커·길이·bullet
> ④ PPL(Stage 6) — teacher-forced 회귀
> ⑤ 서빙 처리량(`s7-52-servebench`, 서버 필요) — 양자화의 이유(속도)를 같은 축에서
> 프롬프트는 `s7-36-prep-gsm8k.sh`가 결정론적으로 만든다 — 수작업 표본 금지.

## 단계 원장 · 릴리스 노트

- 원장: `outputs/stage-ledger-<mode>.json` (각 단계의 status/artifact/verdict, 타깃별 집계)
- 릴리스 노트: `python s8-73-release-notes.py --mode smoke` → `outputs/release-notes-<mode>.md`
  (원장 + 수용 기준 + 실측 지표 대조표)

## TARGETS는 필수

`run-{smoke,full}.sh`는 **양자화 대상을 명시해야** 실행한다. 미지정 시 가능한 값
(`awq-w3`/`awq-w5`/`nvfp4-awq`/`nvfp4-gptq`/`all`)을 찍고 exit 2 한다 —
"인수 깜빡하고 4종을 2시간씩 돌리는" 사고를 막기 위한 것이다.

같은 규칙이 타깃을 받는 모든 개별 스크립트에도 적용된다
(`s6-34-measure-ppl` · `s7-37-eval` · `s7-55-bench-transformers` ·
`s7-56-prep-upload` · `s7-57-verify-upload` · `s8-49-upload-hf` · `s8-70-serve-infer-all`):
`$2` 인자 또는 `TARGETS` 환경변수가 없으면 사용법을 찍고 exit 2, 전체 실행은 `TARGETS=all`로만 된다.
(`s8-58`은 `[mode] [n]` 형태라 `TARGETS` 환경변수로만 받는다.)

```bash
TARGETS=nvfp4-awq ./run-smoke.sh        # 1종
TARGETS=awq-w3,awq-w5 ./run-full.sh     # 2노드 분담
TARGETS=all ./run-full.sh               # 4종 전부
./s6-34-measure-ppl.sh smoke base,nvfp4-awq   # 개별 스크립트도 동일 ($2 또는 TARGETS)
```

## smoke/full 스케일 (`config/common.env`)

| 항목 | smoke (full 동일 단계, 작은 규모) | full |
|---|---|---|
| CALIB_N | 8 | 256 |
| SEQ_LEN | 512 | 2048 |
| EVAL_N / PPL_N | 8 / 8 | 200 / 32 |
| 실행순 | awq-w3 → awq-w5 → nvfp4-awq → nvfp4-gptq (알파벳순) | 동일 (밤샘 권장) |
| 2노드 분할 | `TARGETS=awq-w3,awq-w5` (spark1) + `TARGETS=nvfp4-awq,nvfp4-gptq` (spark2), `STEPS=dequant,quant`로 양쪽 실행 후 1곳에서 `STEPS=ppl,eval` | 동일 분할 (`MODE=full`) |

## 단일 타깃 실행 (`run-s0-to-s7-*.sh`)

`run-{smoke,full}.sh`의 TARGETS/STEPS 조합을 한 타깃으로 좁힌 진입점 8종.
**파일명 앞 숫자 = 실행되는 스텝 번호** (s0-10-setup·s2-13-calib·s3-16-compat·s4-19-dequant·s5-<quant>·s6-34-ppl·s7-37-eval·s7-40-gsm8k):

| 파일 | 스텝 번호 | 양자화 스크립트 |
|---|---|---|
| `run-s0-to-s7-{smoke,full}-awq-w5.sh` | 0~7 (setup~eval) | s5-25-quant-awq-w5 |
| `run-s0-to-s7-{smoke,full}-awq-w3.sh` | 0~7 (setup~eval) | s5-22-quant-awq-w3 |
| `run-s0-to-s7-{smoke,full}-nvfp4-awq.sh` | 0~7 (setup~eval) | s5-28-quant-nvfp4-awq |
| `run-s0-to-s7-{smoke,full}-nvfp4-gptq.sh` | 0~7 (setup~eval) | s5-31-quant-nvfp4-gptq |

(engine: `build-ppl-eval.inc`)

| 스텝 | 내용 |
|---|---|
| setup calib compat dequant | run-*.sh와 동일 (freshness로 건너뜀) |
| quant | 해당 타깃의 32/33/35/36만 실행 (`QUANT_CPU_ONLY=1` 지원) |
| **gate-quant** | 패킹 계약 + `global_scale` 값 검증 — dense 폴백·미보정 expert 차단 |
| ppl | 48(base 대비 Δ) → **gate-ppl** (유한 PPL 아니면 중단) |
| eval | 50 + 54 → **gate-eval** (응답 미생성이면 중단) |
| prep verify (기본) | 56 모델카드 / 57 업로드 전 감사 — 빌드가 끝나면 검증된 모델 + 검증결과가 완료된다 |

게이트는 산출물을 **먼저 검사**한다: quant 직후 패킹 계약, PPL 직후 유한값.
이전처럼 NaN 모델을 eval까지 흘려보내 0점 벤치만 남는 일이 없다.
timing은 `logs/timing-build-<mode>.log`에(run-*.sh와 별도 파일).

## 공유 스텝의 중복 실행 회피 (8종 build 병렬 실행 대비)

`10-20, 30`은 8종 build가 공통으로 쓰는 접두 스텝이다. 같은 스텝을 2개 build가
동시에 돌면 **먼저 lock을 잡은 쪽이 수행하고, 나머지는 flock 대기 후 freshness/stamp로 skip**한다.

| 스텝 | lock | freshness/stamp | 동시 실행 시 |
|---|---|---|---|
| 10-setup | `rq_lock $RQ_ROOT` (venv 변조 직렬화) | stamp `.venv/.rq-setup-stamp` (uv.lock 해시 + llmcompressor/compressed-tensors 버전) | 1번만 `uv sync`·`uv pip install`, 2번째는 skip (0.1s) |
| s2-13-prep-calibration | `rq_lock $RQ_DATASETS` | `rq_fresh` (calib jsonl vs 스크립트 mtime) | 1번만 생성, 2번째는 skip |
| 16-compat-check | `rq_lock $RQ_OUTPUTS/compat` | stamp `outputs/compat.stamp` (probe 해시 + GPU cc + llmcompressor/vllm 버전) | 1번만 probe(119s), 2번째는 skip + 마지막 리포트 재생 |
| 19-dequant | `rq_lock $OUT` | `rq_fresh` (+ config.json mtime + shim grep) | 1번만 복원(999s), 2번째는 skip |
| 32/33/35/36 quant | `rq_lock $OUT` (타깃별) | `rq_fresh` | 타깃이 다르면 동시 실행 가능, 같으면 직렬+skip |
| 34-measure-ppl | `rq_lock $RQ_OUTPUTS/ppl-$MODE` | 타깃별 `rq_fresh` | 직렬(29B 모델 동시 로드 방지) |
| s7-37-eval | `rq_lock $RQ_OUTPUTS/eval-$MODE` | 타깃별 `rq_fresh` | 직렬(동일) |
| 40-gsm8k | `rq_lock $RQ_OUTPUTS/eval-$MODE` | — | 37-eval과 같은 lock → jsonl 쓰는 중엔 대기 |

stamp는 `FORCE=1`로 무시된다. GPU/라이브러리 버전이 바뀌면 stamp가 불일치해
자동으로 다시 검증된다.

## 30B 경로와의 관계 (65 결과 파일 기준)

- 30B FFT-FSDP/DDP는 OOM으로 불가 판정 → 본 파이프는 학습 자체를 안 하므로 해당 없음.
- 30B 퍼블리싱 3종(FP8/AWQ/GGUF) 중 GGUF는 본 스코프 제외. FP8도 MXFP8 베이스와 겹쳐 제외.
- QLoRA-merged 이중양자화 비교도 제외 (SFT 제외 결정 따름).
