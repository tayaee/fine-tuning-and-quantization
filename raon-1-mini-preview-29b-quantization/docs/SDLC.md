# SDLC — 모델 양자화·배포 9단계

> 이 파이프라인의 소프트웨어 개발 라이프사이클(SDLC)을 **9단계**로 정의한다.
> 단계는 "실행 단위"가 아니라 **게이트(검증 관문)** 단위다 — 각 단계는
> 산출물을 남기고, 다음 단계로 넘어가기 전에 계약(contract)을 대조한다.

```
[0 환경] → [1 수용기준] → [2 데이터] → [3 호환성] → [4 베이스복원]
→ [5 양자화] → [6 PPL게이트] → [7 평가] → [8 릴리스·배포]
```

| # | 단계 | 스크립트 | 산출물 | Exit 게이트 (다음 단계로 가려면) |
|---:|---|---|---|---|
| 0 | 환경·접근성 | `s0-10-setup.sh` | `.venv/.rq-setup-stamp` | 의존성 핀·llmcompressor/compressed-tensors 버전·GPU(NVFP4=Blackwell) 확인 |
| 1 | 요구사항·수용 기준 | `s1-07-acceptance.sh` | `outputs/acceptance-<mode>.json` | `config/acceptance.toml` 스키마 정상, 타깃 4종 기준 존재 |
| 2 | 캘리브레이션 데이터 | `s2-13-prep-calibration.sh` | `datasets/raon-calibration-<N>.jsonl` | chat template 적용·용량·한국어 혼합 비율 |
| 3 | 호환성 사전 점검 | `s3-16-compat-check.sh` | `outputs/compat.stamp` | custom_code 로드·expert 타깃·SM≥10·vLLM/transformers 버전 |
| 4 | 베이스 복원 | `s4-19-dequant-mxfp8.sh` | `models/raon-bf16` | MXFP8→BF16 수치 대조·`quantization_config` 제거·MoE 전-expert 캘리브레이션 심 |
| 5 | 양자화 실행 | `22/25/28/31-quant-*.sh` | `models/raon-<target>-<mode>` | **gate-quant**: 패킹 포맷·`global_scale` 값·dense 폴백 차단 |
| 6 | PPL 품질 게이트 | `s6-34-measure-ppl.sh` | `outputs/ppl-<mode>/<target>.json` | **gate-ppl**: PPL 유한값 + 수용 기준 Δ 판정 일치 |
| 7 | 벤치마크 평가·카드·감사 | `s7-37-eval-base.sh` + `s7-38-eval-quantized.sh` + `s7-40-gsm8k.sh` + `s7-46-kmmluko.sh` + `s7-49-format.sh` + `s7-55-bench-transformers.sh` → `s7-56-prep-upload.sh` → `s7-57-verify-upload.sh` | `outputs/eval-<mode>/<target>.jsonl`, `score.json`, `checks.json`, 모델카드 `README.md`, `prep-upload-<mode>.json`, `verify-upload-<mode>.json` | **gate-eval**: 응답이 실제로 생성됨(빈 응답 차단), **gate-kmmluko**/**gate-format**: 정확도·통과율 계약 대조 |
| 8 | 릴리스·배포 | `s8-49-upload-hf.sh` → `s8-52-vllm-serve-*.sh`(개별) / `s8-64-vllm-serve-one.sh`(전 타깃) / `s8-67-inference-test.sh` / `s8-70-vllm-serve-all-models.sh` | Hub 업로드, `outputs/release-notes-<mode>.md` | **gate-release**: 모델카드·지표·업로드 전 감사 통과, **gate-serving**: 서빙 응답 확인 |

## 0. 파일 명명 규칙

```
s<stage>-##-<slug>.{sh,py}
 │      │   └── 작업 이름
 │      └────── Stage 내 스텝 번호 (실행 순서, 10 단위 간격)
 └───────────── 9단계 중 몇 번째 단계 (0~8, 한 자리)
```

| stage | 단계 | 파일 예시 |
|---:|---|---|
| 0 | 환경 | `s0-10-setup.sh` |
| 1 | 수용 기준 | `s1-07-acceptance.{sh,py}` |
| 2 | 데이터 | `s2-13-prep-calibration.{sh,py}` |
| 3 | 호환성 | `s3-16-compat-check.{sh,py}` |
| 4 | 베이스 복원 | `s4-19-dequant-mxfp8.{sh,py}` |
| 5 | 양자화 | `s5-22-quant-awq-w3.{sh,py}` … `s5-31-quant-nvfp4-gptq.{sh,py}` |
| 6 | PPL 게이트 | `s6-34-measure-ppl.{sh,py}` |
| 7 | 평가 | `s7-36-prep-gsm8k.{sh,py}`, `s7-37-eval.{sh,py}`, `s7-40-gsm8k.{sh,py}`, `s7-43-prep-kmmluko.{sh,py}`, `s7-46-kmmluko.{sh,py}`, `s7-49-format.{sh,py}`, `s7-52-servebench.{sh,py}`, `s7-55-bench-transformers.{sh,py}`, `s7-56-prep-upload.{sh,py}`, `s7-57-verify-upload.{sh,py}` |
| 8 | 릴리스·배포 | `s8-49-upload-hf.{sh,py}`, `s8-64-vllm-serve-one.sh`, `s8-67-inference-test.{sh,py}`, `s8-70-vllm-serve-all-models.sh`, `s8-73-release-notes.{sh,py}`, `s8-76-verify-serving-log.{sh,py}` |

빌드 wrapper(`run-s0-to-s7-<mode>-<target>.sh`)는 **Stage 0~7을 한 번에** 돌리는 진입점이라
`s0-s7-` 접두사를 쓴다(카드·감사까지 기본 포함, 업로드 등 Stage 8은 별도 실행).

## 1. Stage 1 — 수용 기준 (Acceptance Criteria)

수용 기준은 **한 파일**에 산다: `config/acceptance.toml`.
이 파일이 34(Δ 판정) · 56(모델 카드) · 57(용량 범위) · 61(릴리스 노트)의
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
(`uv.lock`을 건드리지 않으므로 s0-10-setup 스탬프가 무효화되지 않는다).

## 1-1. Stage 7 — 표준 GSM8K 평가 (측정 프로토콜)

평가 입력은 **표준 벤치마크만** 쓴다. 프롬프트는 `s7-36-prep-gsm8k.sh`가 만든다.

| 항목 | 값 |
|---|---|
| 데이터셋 | `openai/gsm8k`, config `main`, **test split** (1319문제) |
| 리비전 | `740312add88f781978c0658806c59bc2815b9866` (고정) |
| 샘플링 | `random.Random(seed)` — seed `20261024` 고정, `GSM8K_N`(0=전체) |
| 프롬프트 | Raon chat template(`user\n…\nassistant\n<think></think>\n\n`) + `#### <number>` 지시 |
| gold | 데이터셋 정답의 `#### N` (콤마 제거) |
| 재현성 | `outputs/eval-<mode>/evalset-<mode>.json`에 데이터셋·리비전·seed·n·**sha256** 기록 |
| 생성 | `s7-37-eval.py --batch-size` (기본 `EVAL_BATCH=8`, 좌측 패딩) |

**배치 생성 주의**: 배치가 바뀌면 어휘는 달라질 수 있고(패딩/배치 모양에 따른 수치 차이),
**채점 결과도 달라질 수 있다** — 실측 20문제 대조에서 bs=1 19/20, bs=8 20/20(1문제 차이).
따라서 **base와 양자화본은 반드시 같은 프로토콜**(동일 배치 크기·동일 max_new_tokens)로
재야 비교가 공정하다. 허용폭 2%p는 n=200 기준 약 1σ 수준이니 표본 수와 함께 해석한다.

## 1-1b. 검사 5종 (무엇을 얼마나 검증하는가)

| # | 검사 | 스크립트 | 데이터 | 잡는 것 | 잡지 못하는 것 |
|---:|---|---|---|---|---|
| 1 | in-domain PPL | `s6-34-measure-ppl` | 캘리브 코퍼스 | 가중치 손상·스케일 붕괴(NaN) | 생성 품질·한국어 |
| 2 | 표준 GSM8K | `s7-36-prep-gsm8k` → `s7-40-gsm8k` | `openai/gsm8k` test | 영어 수학 추론 붕괴 | 코드·지식·한국어 |
| 3 | 표준 한국어 KMMLU | `s7-43-prep-kmmluko` → `s7-46-kmmluko` | `HAERAE-HUB/KMMLU` test | 한국어 능력·토크나이저 손상 | 영어 앵커링 |
| 4 | 지시/형식 준수 | `s7-49-format` | 고정 10케이스 | 언어지시·JSON·마커·길이·bullet | 정확성 자체 |
| 5 | 서빙 처리량 | `s7-52-servebench` | 실서빙(없으면 skip) | 양자화의 대가(속도) | 품질 |

- 1~4는 **품질 회귀**(base 대비 하락 허용폭이 계약 `config/acceptance.toml`에 있음).
- 5는 **배포 증거**(KRAFTON 카드에 공개된 처리량과 같은 축 — 하드웨어가 달라 방향성 참조).
- 표본 수: GSM8K n=200이면 σ≈1.6%p → 허용폭 2%p는 ≈1.2σ. 릴리스는 1319 전체(σ≈0.6%p) 권장.
- **결과 위치가 다르다**: 1·2는 `outputs/eval-<mode>/score.json`, 3·4·5는 `outputs/eval-<mode>/checks.json`.

## 1-2. 모델 카드의 검사 표

`s7-56-prep-upload`가 만드는 카드에는 **검사 2종**이 실측과 함께 박힌다.

| 검사 | 지표 | 기준 | 출처 |
|---|---|---|---|
| Check 1 — PPL | in-domain PPL, base 대비 Δ | `ppl_delta_warn` / `ppl_delta_max` | `s6-34-measure-ppl.sh` |
| Check 2 — GSM8K | 표준 테스트셋 정확도 (+답 마커율·절단률) | base 대비 하락 ≤ `gsm8k_acc_drop_max` | `s7-37-eval-base.sh` + `s7-38-eval-quantized.sh` + `s7-40-gsm8k.sh` |
| Check 3 — KMMLU | 표준 한국어 정확도 | base 대비 하락 ≤ `kmmluko_acc_drop_max` | `s7-43-prep-kmmluko.sh` + `s7-46-kmmluko.sh` |
| Check 4 — 형식/지시 준수 | 고정 10케이스 통과율 | ≥ `format_pass_rate_min` | `s7-49-format.sh` |
| (참고) 서빙 성능 | 출력 tok/s · 요청 지연 | 방향성(공개 처리량과 같은 축) | `s7-52-servebench.sh` |

## 2. 게이트 (단계 간 계약)

게이트 구현은 `gates.inc` 하나에 있고, `run-{smoke,full}.sh`와
`build-ppl-eval.inc`가 **같은 함수**를 호출한다(경로별로 정책이 갈리지 않게).

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

게이트는 **산출물을 먼저 검사**한다. 다음 단계로 넘어가기 전에
"패킹 안 된 체크포인트 / NaN PPL / 빈 응답"을 여기서 막는다.

## 3. 단계 원장 (Ledger) — 증빙

각 단계는 `outputs/stage-ledger-<mode>.json`에 상태를 남긴다.

```json
{ "mode": "smoke", "stages": { "5": { "id": 5, "name": "quant", "status": "pass",
  "targets": { "nvfp4-awq": { "status": "pass" } } } } }
```

`s8-73-release-notes.py`가 원장 + 수용 기준 + 실측 지표를 합쳐
`outputs/release-notes-<mode>.md`를 만든다 — 이게 SDLC의 "릴리스 노트"다.

## 4. 멱등성 (2회 실행이 즉시 끝나야 한다)

재실행이 즉시 끝나려면 각 단계의 재작업 판정이 **계약 기준**이어야 한다.

| 단계 | 재작업 판정 | 비고 |
|---|---|---|
| 0 환경 | stamp(uv.lock 해시 + 핀 버전) | 의존성이 그대로면 skip |
| 2 데이터 | `rq_fresh`(산출물 > 스크립트) | |
| 3 호환성 | stamp(probe 해시 + GPU cc + 버전) | |
| 4 복원 | `rq_fresh` + shim/마커 grep | **MoE 캘리브레이션 심이 없으면 재복원** |
| 5 양자화 | `rq_fresh`(산출물 > calib/스크립트) | |
| 6 PPL | 산출물 > 모델 `config.json` | **스크립트 mtime이 아니라 산출물 기준** |
| 7 평가 | 산출물 > 모델 `config.json` | 동일 |
| 8 릴리스 | `43/46` 산출물 존재 | |

원칙: **스크립트를 고쳐도 이미 얻은 측정값은 무효가 되지 않는다.**
측정값이 무효가 되는 경우는 (a) 모델이 바뀌었거나 (b) 판정 계약이 바뀌었을 때뿐이다.
코드를 고쳤다는 이유로 30분짜리 측정을 다시 하는 것은 SDLC가 아니라 시간 낭비다.

## 5. 단계 ↔ issue

- **Stage** = 릴리스 트레인(재현 가능한 파이프라인). 한 번의 파이프라인 실행 단위.
- **issue** = Stage 내부의 개발 단위. `issues/issue-N-*.md` → 구현 → 테스트 →
  `regression-tests/verify-issue-N.sh` → `issues/archive/` 아카이브.
- 하나의 issue는 대개 한 Stage에만 속한다. Stage를 건드리는 issue는
  `regression-tests/verify-issue-N.sh`에 그 Stage의 게이트 회귀 검사를 함께 넣는다.

## 6. 11단계 확장 (학습 포함)

SFT까지 포함한 전체 SDLC는 형제 프로젝트(`llama-3.2-1b-to-math-model`) 기준
**11단계**다: 0 환경 → 1 베이스라인 → 2 합성데이터 → 3 SFT+merge →
4 PTQ(FFT) → 5 PTQ(QLoRA-merged) → 6 PPL → 7 평가 → 8 배포 (+9·10 학습 계측).
현재 저장소는 SFT를 의도적으로 뺀 **9단계 축소판**이다.
