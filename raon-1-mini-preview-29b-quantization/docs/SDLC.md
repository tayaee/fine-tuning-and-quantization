# SDLC — 모델 양자화·배포 8개 그룹

> 이 파이프라인의 소프트웨어 개발 라이프사이클(SDLC)을 **8개 그룹**으로 정의한다.
> 그룹은 "실행 단위"가 아니라 **게이트(검증 관문)** 단위다 — 각 그룹은
> 산출물을 남기고, 다음 그룹으로 넘어가기 전에 계약(contract)을 대조한다.
> 그룹 번호는 s0·s1·s2·s3·s4·s5·s6·s7이다.

```
[0 베이스검증] → [1 환경체인] → [2 베이스복원] → [3 양자화]
→ [4 측정·평가] → [5 업로드] → [6 고객인수(transformers)] → [7 고객인수(vLLM)]
```

| # | 그룹 | 스크립트 | 산출물 | Exit 게이트 (다음으로 가려면) |
|---:|---|---|---|---|
| 0 | 베이스모델 검증 | `s0-01-serve`(상주) → `04/07/10` endpoint 검증 → `s0-13-serve`(상주) → `16/19/22` endpoint 검증 | `outputs/s0-<mode>.json` + 벤치 결과 2종 | chat/pi/bench 응답 존재 (13 vLLM half는 포크 검증 GPU sm_103/sm_120에서만) |
| 1 | 환경 체인 | `s1-01-setup.sh` → `s1-04-acceptance.sh` → `s1-07-prep-calibration.sh` → `s1-10-compat-check.sh` | `.venv/.rq-setup-stamp`, `outputs/acceptance-<mode>.json`, calib jsonl, `outputs/compat.stamp` | 의존성 핀·계약 스키마·GPU(NVFP4=Blackwell) 확인 |
| 2 | 베이스 복원 | `s2-01-dequant-mxfp8.sh` | `models/raon-bf16` | MXFP8→BF16 수치 대조·`quantization_config` 제거·MoE 전-expert 캘리브레이션 심 |
| 3 | 양자화 실행 | `01/04/07/10-quant-*.sh` | `models/raon-<target>-<mode>` | **gate-quant**: 패킹 포맷·`global_scale` 값·dense 폴백 차단 |
| 4 | 측정·평가·카드·감사 | `s4-01-measure-ppl.sh` → `04/07/10/13/16/19/22/25/28` → `s4-31-prep-upload.sh` → `s4-34-verify-upload.sh` → `s4-37-release-notes.sh` | `outputs/eval-<mode>/<target>.jsonl`, `score.json`, `checks.json`, 모델카드 `README.md`, `prep-upload-<mode>.json`, `verify-upload-<mode>.json`, `outputs/release-notes-<mode>.md` | **gate-ppl**·**gate-eval**: 응답이 실제로 생성됨(빈 응답 차단), **gate-kmmluko**/**gate-format**: 정확도·통과율 계약 대조, **gate-release**: 카드·지표·감사 |
| 5 | 업로드 | `s5-01-upload-hf.sh` | Hub 업로드 | 카드·감사 선행 (없으면 중단) |
| 6 | 고객인수(transformers) | `s6-*-test-*-with-transformers.sh` (Hub→로딩) / `s6-13-chat.sh` / `s6-16-pi-tool-call-test.sh`(임시 서버) | Hub 다운로드 + `outputs/s6-<mode>/` | chat/pi 응답 존재 (미발행 repo면 FAIL) |
| 7 | 고객인수(vLLM) | `s7-*-test-*-with-vllm-fork.sh` frozen 래퍼 / `s7-13-chat.sh`(서버 상주) / `s7-16-pi-tool-call-test.sh`(검증 후 정리) / `s7-19-verify-serving-log.sh` | Hub·로컬 서빙 + `logs/s7-13·s7-16.<slug>.log` + `outputs/serve-<mode>.json` | **gate-serving**: 서빙 응답 확인 (PTQ 4종 현재 실패 — 포크 한계, PROCEDURE.md 참조) |

## 0. 파일 명명 규칙

```
s<group>-##-<slug>.{sh,py}
 │      │   └── 작업 이름
 │      └────── 그룹 내 순서 (01부터 3단위 — 각 그룹의 첫 스크립트는 항상 s#-01-)
 └───────────── 8개 그룹 중 어디인지 (0·1·2·3·4·5·6·7, 단독 정리 `-stop-` 2종은 3단위 예외)
```

| group | 그룹 | 파일 예시 |
|---:|---|---|
| 0 | 베이스검증 | `s0-01-load-transformers.sh` … `s0-22-bench-vllm.sh` |
| 1 | 환경 체인 | `s1-01-setup.sh`, `s1-04-acceptance.{sh,py}`, `s1-07-prep-calibration.{sh,py}`, `s1-10-compat-check.{sh,py}` |
| 2 | 베이스 복원 | `s2-01-dequant-mxfp8.{sh,py}` |
| 3 | 양자화 | `s3-01-quant-awq-w3.{sh,py}` … `s3-10-quant-nvfp4-gptq.{sh,py}` |
| 4 | 측정·평가 | `s4-01-measure-ppl.{sh,py}`, `s4-07-eval.{sh,py}`, `s4-13-gsm8k.{sh,py}`, `s4-16-prep-kmmluko.{sh,py}`, `s4-19-kmmluko.{sh,py}`, `s4-22-format.{sh,py}`, `s4-25-servebench.{sh,py}`, `s4-28-bench-transformers.{sh,py}`, `s4-31-prep-upload.{sh,py}`, `s4-34-verify-upload.{sh,py}`, `s4-37-release-notes.{sh,py}` |
| 5 | 업로드 | `s5-01-upload-hf.{sh,py}` |
| 6 | 고객-transformers | `s6-*-test-*-with-transformers.sh`, `s6-13-chat.sh`, `s6-16-pi-tool-call-test.sh` |
| 7 | 고객-vLLM | `s7-*-test-*-with-vllm-fork.sh`, `s7-13-chat.sh`, `s7-16-pi-tool-call-test.sh`, `s7-19-verify-serving-log.{sh,py}` |

빌드 wrapper(`run-s1-to-s4-<mode>-<target>.sh`)는 **s1~s4를 한 번에** 돌리는 진입점이라
`run-s1-to-s4-` 접두사를 쓴다(s0 베이스검증·카드·감사까지 기본 포함, s5 업로드·s6/s7 CAT는 별도 실행).

## 0-1. s0 베이스검증 (8절차)

베이스모델(`BASE_MODEL`)이 transformers half와 vLLM half 양쪽에서 동작하는지
양자화 전에 고정한다. 절차와 스크립트는 PROCEDURE.md의 s0 표를 따른다.
pi 검증은 고정 질문(`현재 디렉토리 파일이 몇 개니? 숫자 하나로 답해.`) 하나로
도구 호출 왕복을 판정한다 (서버 TOOL-CALL 증거 + 답 포함).
벤치(s0-10/22)는 고정 8프롬프트를 endpoint로 측정해 results.json을 남기고
최초 1회 baseline을 충전한 뒤 서버를 내린다.

## 1. Stage 1 — 수용 기준 (Acceptance Criteria)

수용 기준은 **한 파일**에 산다: `config/acceptance.toml`.
이 파일이 s4-01(Δ 판정) · s4-31(모델 카드) · s4-34(용량 범위) · s4-37(릴리스 노트)의
판정 기준을 모두 결정한다. 기준이 코드에 흩어져 있으면 "무엇이 계약인지"를
다음 사람이 읽을 수 없다.

```toml
[targets.nvfp4-awq]
quant_format   = "nvfp4-pack-quantized"
size_gb        = [15.0, 28.0]
ppl_delta_warn = 0.3    # Δ < 0.3 → Accept
ppl_delta_max  = 1.0    # Δ < 1.0 → Conditional, 이상 → Discard
```

TOML을 쓰는 이유: python 3.12 표준 `tomllib`로 읽혀 **새 의존성이 없다**
(`uv.lock`을 건드리지 않으므로 s1-01-setup 스탬프가 무효화되지 않는다).

## 1-1. Stage 4 — 표준 GSM8K 평가 (측정 프로토콜)

평가 입력은 **표준 벤치마크만** 쓴다. 프롬프트는 `s4-04-prep-gsm8k.sh`가 만든다.

| 항목 | 값 |
|---|---|
| 데이터셋 | `openai/gsm8k`, config `main`, **test split** (1319문제) |
| 리비전 | `740312add88f781978c0658806c59bc2815b9866` (고정) |
| 샘플링 | `random.Random(seed)` — seed `20261024` 고정, `GSM8K_N`(0=전체) |
| 프롬프트 | Raon chat template(`user\n…\nassistant\n<think></think>\n\n`) + `#### <number>` 지시 |
| gold | 데이터셋 정답의 `#### N` (콤마 제거) |
| 재현성 | `outputs/eval-<mode>/evalset-<mode>.json`에 데이터셋·리비전·seed·n·**sha256** 기록 |
| 생성 | `s4-07-eval.py --batch-size` (기본 `EVAL_BATCH=8`, 좌측 패딩) |

**배치 생성 주의**: 배치가 바뀌면 어휘는 달라질 수 있고(패딩/배치 모양에 따른 수치 차이),
**채점 결과도 달라질 수 있다** — 실측 20문제 대조에서 bs=1 19/20, bs=8 20/20(1문제 차이).
따라서 **base와 양자화본은 반드시 같은 프로토콜**(동일 배치 크기·동일 max_new_tokens)로
재야 비교가 공정하다. 허용폭 2%p는 n=200 기준 약 1σ 수준이니 표본 수와 함께 해석한다.

## 1-1b. 검사 5종 (무엇을 얼마나 검증하는가)

| # | 검사 | 스크립트 | 데이터 | 잡는 것 | 잡지 못하는 것 |
|---:|---|---|---|---|---|
| 1 | in-domain PPL | `s4-01-measure-ppl` | 캘리브 코퍼스 | 가중치 손상·스케일 붕괴(NaN) | 생성 품질·한국어 |
| 2 | 표준 GSM8K | `s4-04-prep-gsm8k` → `s4-13-gsm8k` | `openai/gsm8k` test | 영어 수학 추론 붕괴 | 코드·지식·한국어 |
| 3 | 표준 한국어 KMMLU | `s4-16-prep-kmmluko` → `s4-19-kmmluko` | `HAERAE-HUB/KMMLU` test | 한국어 능력·토크나이저 손상 | 영어 앵커링 |
| 4 | 지시/형식 준수 | `s4-22-format` | 고정 10케이스 | 언어지시·JSON·마커·길이·bullet | 정확성 자체 |
| 5 | 서빙 처리량 | `s4-25-servebench` | 실서빙(없으면 skip) | 양자화의 대가(속도) | 품질 |

- 1~4는 **품질 회귀**(base 대비 하락 허용폭이 계약 `config/acceptance.toml`에 있음).
- 5는 **배포 증거**(KRAFTON 카드에 공개된 처리량과 같은 축 — 하드웨어가 달라 방향성 참조).
- 표본 수: GSM8K n=200이면 σ≈1.6%p → 허용폭 2%p는 ≈1.2σ. 릴리스는 1319 전체(σ≈0.6%p) 권장.
- **결과 위치가 다르다**: 1·2는 `outputs/eval-<mode>/score.json`, 3·4·5는 `outputs/eval-<mode>/checks.json`.

## 1-2. 모델 카드의 검사 표

`s4-31-prep-upload`가 만드는 카드에는 **검사 2종**이 실측과 함께 박힌다.

| 검사 | 지표 | 기준 | 출처 |
|---|---|---|---|
| Check 1 — PPL | in-domain PPL, base 대비 Δ | `ppl_delta_warn` / `ppl_delta_max` | `s4-01-measure-ppl.sh` |
| Check 2 — GSM8K | 표준 테스트셋 정확도 (+답 마커율·절단률) | base 대비 하락 ≤ `gsm8k_acc_drop_max` | `s4-07-eval-base.sh` + `s4-10-eval-quantized.sh` + `s4-13-gsm8k.sh` |
| Check 3 — KMMLU | 표준 한국어 정확도 | base 대비 하락 ≤ `kmmluko_acc_drop_max` | `s4-16-prep-kmmluko.sh` + `s4-19-kmmluko.sh` |
| Check 4 — 형식/지시 준수 | 고정 10케이스 통과율 | ≥ `format_pass_rate_min` | `s4-22-format.sh` |
| (참고) 서빙 성능 | 출력 tok/s · 요청 지연 | 방향성(공개 처리량과 같은 축) | `s4-25-servebench.sh` |

## 2. 게이트 (그룹 간 계약)

게이트 구현은 `gates.inc` 하나에 있고, `build-ppl-eval.inc`(구 `run-{smoke,full}.sh`는
`archive/`로 은퇴)가 **같은 함수**를 호출한다(경로별로 정책이 갈리지 않게).

| 게이트 | 함수 | 막는 것 |
|---|---|---|
| gate-acceptance | `rq_gate_acceptance` | 기준 없이 진행 (Stage 1 산출물 누락) |
| gate-quant | `rq_gate_quant` | dense 폴백·미보정 `global_scale`(MoE 캘리브레이션 누락) |
| gate-ppl | `rq_gate_ppl` | NaN/미측정 PPL, 기준과 어긋난 verdict |
| gate-eval | `rq_gate_eval` | 응답 0건(모델이 아무것도 못 냄), GSM8K 정확도가 계약 허용폭 초과 |
| gate-kmmluko | `rq_gate_kmmluko` | 한국어 KMMLU 정확도가 계약 허용폭 초과 |
| gate-format | `rq_gate_format` | 지시/형식 준수 통과율이 계약 하한 미달 |
| gate-release | `rq_gate_release` | 모델카드/지표/업로드 전 감사 미통과 |
| gate-serving | `rq_gate_serving` | 서빙 실패 (로그 미수행은 skip) |

게이트는 **산출물을 먼저 검사**한다. 다음 그룹으로 넘어가기 전에
"패킹 안 된 체크포인트 / NaN PPL / 빈 응답"을 여기서 막는다.

## 3. 단계 원장 (Ledger) — 증빙

각 그룹은 `outputs/stage-ledger-<mode>.json`에 상태를 남긴다.

```json
{ "mode": "smoke", "stages": { "3": { "id": 3, "name": "quant", "status": "pass",
  "targets": { "nvfp4-awq": { "status": "pass" } } } } }
```

`s4-37-release-notes.py`가 원장 + 수용 기준 + 실측 지표를 합쳐
`outputs/release-notes-<mode>.md`를 만든다 — 이게 SDLC의 "릴리스 노트"다.

## 4. 멱등성 (2회 실행이 즉시 끝나야 한다)

재실행이 즉시 끝나려면 각 그룹의 재작업 판정이 **계약 기준**이어야 한다.

| 그룹 | 재작업 판정 | 비고 |
|---|---|---|
| 0 베이스검증 | 서버 기동 여부 + 산출물 존재 | s0-13은 포트 점유 시 기존 서버 재사용, s0-22가 정리 |
| 1 환경체인 | setup stamp / acceptance 존재 / calib `rq_fresh` / compat stamp | 의존성이 그대로면 skip |
| 2 복원 | `rq_fresh` + shim/마커 grep | **MoE 캘리브레이션 심이 없으면 재복원** |
| 3 양자화 | `rq_fresh`(산출물 > calib/스크립트) | |
| 4 측정·평가 | 산출물 > 모델 `config.json` | **스크립트 mtime이 아니라 산출물 기준** |
| 5 업로드 | s4-31/34 산출물 존재 | |
| 6 CAT-tf | `outputs/s6-<mode>/` 존재 시 skip(타깃별) | s6-13/16 (실패 로그도 skip — 지우고 재실행) |
| 7 CAT-vllm | `logs/s7-13·s7-16.<slug>.log` 존재 시 skip | s7-13/16 (실패 로그도 skip — 지우고 재실행) |

원칙: **스크립트를 고쳐도 이미 얻은 측정값은 무효가 되지 않는다.**
측정값이 무효가 되는 경우는 (a) 모델이 바뀌었거나 (b) 판정 계약이 바뀌었을 때뿐이다.
코드를 고쳤다는 이유로 30분짜리 측정을 다시 하는 것은 SDLC가 아니라 시간 낭비다.

## 5. 단계 ↔ issue

- **Stage** = 릴리스 트레인(재현 가능한 파이프라인). 한 번의 파이프라인 실행 단위.
- **issue** = Stage 내부의 개발 단위. `issues/issue-N-*.md` → 구현 → 테스트 →
  `regression-tests/verify-issue-N.sh` → `issues/archive/` 아카이브.
- 하나의 issue는 대개 한 그룹에만 속한다. 그룹을 건드리는 issue는
  `regression-tests/verify-issue-N.sh`에 그 그룹의 게이트 회귀 검사를 함께 넣는다.

## 6. 11단계 확장 (학습 포함)

SFT까지 포함한 전체 SDLC는 형제 프로젝트(`llama-3.2-1b-to-math-model`) 기준
**11단계**다: 0 환경 → 1 베이스라인 → 2 합성데이터 → 3 SFT+merge →
4 PTQ(FFT) → 5 PTQ(QLoRA-merged) → 6 PPL → 7 평가 → 8 배포 (+9·10 학습 계측).
현재 저장소는 SFT를 의도적으로 뺀 **8개 그룹 축소판**이다.
