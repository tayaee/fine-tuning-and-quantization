# PROCEDURE — Raon-1-mini PTQ only (DGX Spark 1x 기본, 서빙은 2x/TP=2)

> 학습 Stage 없음 (SFT 제외). 베이스 검증 → 환경 체인 → MXFP8 베이스 → BF16 복원
> → 한국어 calib → 4종 PTQ → PPL 게이트 → vLLM eval → 업로드·CAT.
> 30B 플랜(65 결과 159~190행) 대비 VRAM 참고: BF16 서빙 32~36GB, AWQ 18~20GB.
> 베이스가 MXFP8이라 4종 전부 이중양자화 전제 (FP8 타깃 제외됨).
> SDLC 정의(8개 그룹·게이트·증빙)는 `docs/SDLC.md` 참고.

## 그룹 (8개: s0·s1·s2·s3·s4·s5·s6·s7)

```
[s0 베이스검증 01·04·07·10·13·16·19·22]
  → [s1 환경체인 01-setup·04-acceptance·07-calib·10-compat]
  → [s2 복원 01-dequant] → [s3 양자화 01·04·07·10]
  → [s4 측정·평가 01·04·07·10·13·16·19·22·25·28·31·34·37]
  → [s5 업로드 01-upload]
  → [s6 고객-transformers 01·04·07·10·13·16] → [s7 고객-vLLM 01·04·07·10·13·16·19]
```

> **파일 명명 규칙**: `s<group>-##-<slug>.{sh,py}` — `s#`은 8개 그룹 중 어디인지
> (0=베이스검증, 1=환경체인, 2=복원, 3=양자화, 4=측정·평가, 5=업로드, 6=고객-transformers, 7=고객-vLLM),
> `##`은 그룹 안 순서(**01부터 3단위**, 각 그룹의 첫 스크립트는 항상 `s#-01-`),
> `<slug>`은 작업 이름이다.
> 예) `s3-07-quant-nvfp4-awq.sh` = 3그룹(양자화)의 07번 스텝.
> 빌드 wrapper(`run-s1-to-s4-<mode>-<target>.sh`)는 **s1~s4를 한 번에** 돌리는
> 진입점이므로 `run-s1-to-s4-` 접두사를 쓴다(s0 베이스검증·s5 업로드·s6/s7 CAT는 별도 실행).

| 그룹 | 스크립트 | 입력 → 출력 | Exit 게이트 |
|---|---|---|---|
| 0. 베이스검증 | `s0-01-serve`(상주) → `04/07/10` endpoint 검증 → `s0-13-serve`(상주) → `16/19/22` endpoint 검증 (+단독 정리 `s0-11/s0-21`, 3단위 예외) | BASE_MODEL → `outputs/s0-<mode>.json` + 벤치 결과 2종 | chat/pi/bench 응답 존재 |
| 1. 환경체인 | `s1-01-setup` → `s1-04-acceptance` → `s1-07-calib` → `s1-10-compat` | env sync + `acceptance-<mode>.json` + calib jsonl + compat stamp | gate-acceptance(계약 스키마) |
| 2. 복원 | `s2-01-dequant-mxfp8` | MXFP8 repo → `models/raon-bf16` (1회) | 수치 대조 + MoE 전-expert 캘리브레이션 심 존재 |
| 3. 양자화 | `s3-01/04/07/10-quant-*` | 복원본 → `models/raon-<target>-<mode>` 4종 (experts-only 기본) | **gate-quant**: 패킹 포맷·global_scale·dense 폴백 차단 |
| 4. 측정·평가 | `s4-01-measure-ppl` → `04/07/10/13/16/19/22/25/28` → `s4-31-prep-upload` → `s4-34-verify-upload` → `s4-37-release-notes` | base + 4종 PPL·GSM8K·KMMLU·형식·처리량 + 모델카드 + 감사 + 릴리스노트 | **gate-ppl**·**gate-eval**·**gate-kmmluko**·**gate-format**·**gate-release** |
| 5. 업로드 | `s5-01-upload-hf` | 준비된 산출물 → Hub (`tayaee/*`, full만) | 카드·감사 선행 (s4-31/34 없으면 중단) |
| 6. 고객-transformers | `s6-01/04/07/10-transformers-test-*` (Hub→로딩) / `s6-13-chat` / `s6-16-pi`(임시 서버) | Hub 다운로드 + `outputs/s6-<mode>/` | chat/pi 응답 존재 (미발행 repo면 FAIL) |
| 7. 고객-vLLM | `s7-01/04/07/10-vllm-fork-test-*` (frozen 서빙 래퍼) / `s7-13-chat`(서버 상주) / `s7-16-pi`(검증 후 정리) / `s7-19`(서빙감사) | Hub·로컬 서빙 + `logs/s7-13·s7-16.<slug>.log` + `outputs/serve-<mode>.json` | **gate-serving**: 서빙 응답 (현재 PTQ 4종 전체 실패 — 포크 한계, 아래 참조) |

> 판정 기준(Δ<0.3 Accept / 0.3~1.0 Conditional / ≥1.0 Discard), 용량 범위,
> GSM8K 정확도 허용폭(base 대비 하락 ≤ 2%p)은 코드 하드코딩이 아니라
> `config/acceptance.toml`(s1-04)이 단일 소스다.
>
> **평가는 5개 검사**로 구성한다(각각 표준/결정론):
> ① 표준 GSM8K(`openai/gsm8k` main/test) — 영어 수학 추론 정확도
> ② 표준 한국어 KMMLU(`HAERAE-HUB/KMMLU`) — 한국어 능력(GSM8K가 못 재는 축)
> ③ 지시/형식 준수 스위트(고정 10케이스) — 언어지시·JSON·마커·길이·bullet
> ④ PPL(s4-01) — teacher-forced 회귀
> ⑤ 서빙 처리량(`s4-25-servebench`, 서버 필요) — 양자화의 이유(속도)를 같은 축에서
> 프롬프트는 `s4-04-prep-gsm8k.sh`가 결정론적으로 만든다 — 수작업 표본 금지.

## s0 베이스검증 (8절차 — transformers half + vLLM half)

베이스모델(`BASE_MODEL`, 기본 `KRAFTON/Raon-1-mini-preview`)이 두 런타임에서
올바르게 동작하는지 양자화 전에 고정한다. 양자화본의 이상이 베이스 탓인지
구분하는 기준선이다.

| # | 스크립트 | 내용 |
|---|---|---|
| 01 | `s0-01-load-transformers.sh` | transformers OpenAI 서버 기동(BASE_MODEL, endpoint 상주, s0-10이 정리) |
| 04 | `s0-04-chat-transformers.sh` | endpoint chat 검증 (고정 질문, 응답 존재 → `outputs/s0-<mode>.json`) |
| 07 | `s0-07-picall-transformers.sh` | endpoint pi 툴콜 검증 (고정 질문, 서버 TOOL-CALL 증거 + 답 포함 판정) |
| 10 | `s0-10-bench-transformers.sh` | endpoint 벤치 8문항 측정(results.json, 최초 baseline 충전) + 서버 정리 |
| 13 | `s0-13-load-vllm.sh` | 포크로 베이스 서빙(백그라운드 상주, `logs/s0-serve-base.pid`) — MXFP8 릴리스는 포크 지원 프로파일. 단 검증 GPU(sm_103/sm_120) 외에서는 불가 체인(vsplit·MLA·prefill·MoE tcgen05+MODE=4 강제)이 확정돼 시도 없이 SKIP, 16/19/22 연쇄 SKIP |
| 16 | `s0-16-chat-vllm.sh` | 서빙 endpoint로 chat 검증 (OpenAI chat API) |
| 19 | `s0-19-picall-vllm.sh` | 서빙 모델을 pi에 등록하고 동일 고정 질문으로 툴콜 왕복 검증 |
| 22 | `s0-22-bench-vllm.sh` | vLLM endpoint 벤치 8문항 측정 + s0-13 서버 정리 |

pi 고정 질문의 기대값은 `outputs/s0-pi-probe/` (파일 5개 고정)의 실제 파일 수와
대조한다. 답이 단일 숫자가 아니면 FAIL(도구 호출 미동작).

## s6/s7 CAT (고객인수 — s7 vLLM half는 현재 총체적 실패 상태)

s6(고객-transformers)과 s7(고객-vLLM)은 고객 시나리오 그대로 검증한다.
s7 vLLM half는 2026-10-26 실측 기준 **PTQ 4종 전부 실패**한다:

- 핀된 포크(`krafton-ai/vllm@raon-1-mini-preview`)의 `Raon1MiniForCausalLMConfig` 후크가
  ModelOpt가 아닌 체크포인트의 `quantization_config`를 삭제 →
  `TypeError: CompressedTensorsConfig.__init__() missing ...` (`logs/13.<slug>.serve.log` 보존).
- 업스트림 vLLM(0.31)도 RDA `linear_attention` 미구현 + MLA에 transformers≥5.15 요구로 불가.
- 체크포인트 자체는 건강하다 (transformers 로드·PPL·eval·Hub 업로드 모두 통과).

s7이 green이 되는 조건: 포크의 compressed-tensors 지원 추가, 또는 ModelOpt-NVFP4
레시피 추가. 그 전까지 s7 실패는 파이프라인(s1~s4·s5)을 막지 않는다
(`rq_gate_serving`은 미수행·실패를 실패가 아닌 기록으로 남긴다 — 단 `serve-*.json`
`ok:false`는 릴리스 노트에 그대로 노출된다).
구 s9-13/16/19 디스패처·순회는 `archive/`로 은퇴했다 (s9-25 감사는 s7-19로 유지).

## 단계 원장 · 릴리스 노트

- 원장: `outputs/stage-ledger-<mode>.json` (각 그룹의 status/artifact/verdict, 타깃별 집계)
- 릴리스 노트: `python s4-37-release-notes.py --mode smoke` → `outputs/release-notes-<mode>.md`
  (원장 + 수용 기준 + 실측 지표 대조표)

## TARGETS는 필수

타깃을 받는 개별 스크립트는 **양자화 대상을 명시해야** 실행한다. 미지정 시 가능한 값
(`awq-w3`/`awq-w5`/`nvfp4-awq`/`nvfp4-gptq`/`all`)을 찍고 exit 2 한다 —
"인수 깜빡하고 4종을 2시간씩 돌리는" 사고를 막기 위한 것이다.

대상 스크립트
(`s4-01-measure-ppl` · `s4-10-eval-quantized` · `s4-28-bench-transformers` ·
`s4-31-prep-upload` · `s4-34-verify-upload` · `s5-01-upload-hf`):
`$2` 인자 또는 `TARGETS` 환경변수가 없으면 사용법을 찍고 exit 2, 전체 실행은 `TARGETS=all`로만 된다.
(s6/s7 검증 스크립트는 `[mode] [HF repo id|short target|all]` 형태다.)

```bash
./s4-01-measure-ppl.sh smoke base,nvfp4-awq   # 개별 스크립트 ($2 또는 TARGETS)
TARGETS=all ./s5-01-upload-hf.sh full        # 전체 실행은 TARGETS=all로만
```

> 구 모놀리식 러너(`run-{smoke,full}.sh`, TARGETS/STEPS 다종 실행)는 `archive/`로
> 은퇴했다. 다종 실행은 타깃별 개별 스크립트 + `run-s1-to-s4-*` 원샷 8종으로 나눠서 한다.

## smoke/full 스케일 (`config/common.env`)

| 항목 | smoke (full 동일 단계, 작은 규모) | full |
|---|---|---|
| CALIB_N | 8 | 256 |
| SEQ_LEN | 512 | 2048 |
| EVAL_N / PPL_N | 8 / 8 | 200 / 32 |
| 실행순 | awq-w3 → awq-w5 → nvfp4-awq → nvfp4-gptq (알파벳순) | 동일 (밤샘 권장) |
| 2노드 분할 | 타깃별 원샷을 노드에 나눔: `run-s1-to-s4-smoke-awq-{w3,w5}.sh` (spark1) + `run-s1-to-s4-smoke-nvfp4-{awq,gptq}.sh` (spark2). dequant는 lock+freshness로 1회만 실행된다 | 동일 분할 (`run-s1-to-s4-full-*.sh`) |

## 단일 타깃 실행 (`run-s1-to-s4-*.sh`)

전체 빌드를 한 타깃으로 좁힌 진입점 8종 (STEPS 부분 실행 지원).
**두 번째 번호 = 그룹 내 순서** (s1-01-setup·s1-07-calib·s1-10-compat·s2-01-dequant·s3-<quant>·s4-01-ppl·s4-07-eval·s4-13-gsm8k):

| 파일 | 그룹 범위 | 양자화 스크립트 |
|---|---|---|
| `run-s1-to-s4-{smoke,full}-awq-w5.sh` | s1~s4 (env체인~측정평가) | s3-04-quant-awq-w5 |
| `run-s1-to-s4-{smoke,full}-awq-w3.sh` | s1~s4 (env체인~측정평가) | s3-01-quant-awq-w3 |
| `run-s1-to-s4-{smoke,full}-nvfp4-awq.sh` | s1~s4 (env체인~측정평가) | s3-07-quant-nvfp4-awq |
| `run-s1-to-s4-{smoke,full}-nvfp4-gptq.sh` | s1~s4 (env체인~측정평가) | s3-10-quant-nvfp4-gptq |

(engine: `build-ppl-eval.inc`)

| 스텝 | 내용 |
|---|---|
| setup calib compat dequant | run-*.sh와 동일 (freshness로 건너뜀) |
| quant | 해당 타깃의 s3-01/04/07/10만 실행 (`QUANT_CPU_ONLY=1` 지원) |
| **gate-quant** | 패킹 계약 + `global_scale` 값 검증 — dense 폴백·미보정 expert 차단 |
| ppl | s4-01(base 대비 Δ) → **gate-ppl** (유한 PPL 아니면 중단) |
| eval | s4-07 + s4-13 → **gate-eval** (응답 미생성이면 중단) |
| prep verify (기본) | s4-31 모델카드 / s4-34 업로드 전 감사 — 빌드가 끝나면 검증된 모델 + 검증결과가 완료된다 |

게이트는 산출물을 **먼저 검사**한다: quant 직후 패킹 계약, PPL 직후 유한값.
이전처럼 NaN 모델을 eval까지 흘려보내 0점 벤치만 남는 일이 없다.
timing은 `logs/timing-build-<mode>.log`에(run-*.sh와 별도 파일).

## 공유 스텝의 중복 실행 회피 (8종 build 병렬 실행 대비)

`s1-01, s1-07, s1-10, s2-01`은 8종 build가 공통으로 쓰는 접두 스텝이다. 같은 스텝을 2개 build가
동시에 돌면 **먼저 lock을 잡은 쪽이 수행하고, 나머지는 flock 대기 후 freshness/stamp로 skip**한다.

| 스텝 | lock | freshness/stamp | 동시 실행 시 |
|---|---|---|---|
| s1-01-setup | `rq_lock $RQ_ROOT` (venv 변조 직렬화) | stamp `.venv/.rq-setup-stamp` (uv.lock 해시 + llmcompressor/compressed-tensors 버전) | 1번만 `uv sync`·`uv pip install`, 2번째는 skip (0.1s) |
| s1-07-calib | `rq_lock $RQ_DATASETS` | `rq_fresh` (calib jsonl vs 스크립트 mtime) | 1번만 생성, 2번째는 skip |
| s1-10-compat | `rq_lock $RQ_OUTPUTS/compat` | stamp `outputs/compat.stamp` (probe 해시 + GPU cc + llmcompressor/vllm 버전) | 1번만 probe(119s), 2번째는 skip + 마지막 리포트 재생 |
| s2-01-dequant | `rq_lock $OUT` | `rq_fresh` (+ config.json mtime + shim grep) | 1번만 복원(999s), 2번째는 skip |
| s3 quant | `rq_lock $OUT` (타깃별) | `rq_fresh` | 타깃이 다르면 동시 실행 가능, 같으면 직렬+skip |
| s4-01-ppl | `rq_lock $RQ_OUTPUTS/ppl-$MODE` | 타깃별 `rq_fresh` | 직렬(29B 모델 동시 로드 방지) |
| s4-07-eval | `rq_lock $RQ_OUTPUTS/eval-$MODE` | 타깃별 `rq_fresh` | 직렬(동일) |
| s4-13-gsm8k | `rq_lock $RQ_OUTPUTS/eval-$MODE` | — | s4-07과 같은 lock → jsonl 쓰는 중엔 대기 |

stamp는 `FORCE=1`로 무시된다. GPU/라이브러리 버전이 바뀌면 stamp가 불일치해
자동으로 다시 검증된다.

## 30B 경로와의 관계 (65 결과 파일 기준)

- 30B FFT-FSDP/DDP는 OOM으로 불가 판정 → 본 파이프는 학습 자체를 안 하므로 해당 없음.
- 30B 퍼블리싱 3종(FP8/AWQ/GGUF) 중 GGUF는 본 스코프 제외. FP8도 MXFP8 베이스와 겹쳐 제외.
- QLoRA-merged 이중양자화 비교도 제외 (SFT 제외 결정 따름).
