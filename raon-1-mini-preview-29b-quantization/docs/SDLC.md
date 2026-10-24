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
| 0 | 환경·접근성 | `10-setup.sh` | `.venv/.rq-setup-stamp` | 의존성 핀·llmcompressor/compressed-tensors 버전·GPU(NVFP4=Blackwell) 확인 |
| 1 | 요구사항·수용 기준 | `07-acceptance.sh` | `outputs/acceptance-<mode>.json` | `config/acceptance.toml` 스키마 정상, 타깃 4종 기준 존재 |
| 2 | 캘리브레이션 데이터 | `13-prep-calibration.sh` | `datasets/raon-calibration-<N>.jsonl` | chat template 적용·용량·한국어 혼합 비율 |
| 3 | 호환성 사전 점검 | `16-compat-check.sh` | `outputs/compat.stamp` | custom_code 로드·expert 타깃·SM≥10·vLLM/transformers 버전 |
| 4 | 베이스 복원 | `19-dequant-mxfp8.sh` | `models/raon-bf16` | MXFP8→BF16 수치 대조·`quantization_config` 제거·MoE 전-expert 캘리브레이션 심 |
| 5 | 양자화 실행 | `22/25/28/31-quant-*.sh` | `models/raon-<target>-<mode>` | **gate-quant**: 패킹 포맷·`global_scale` 값·dense 폴백 차단 |
| 6 | PPL 품질 게이트 | `34-measure-ppl.sh` | `outputs/ppl-<mode>/<target>.json` | **gate-ppl**: PPL 유한값 + 수용 기준 Δ 판정 일치 |
| 7 | 벤치마크 평가 | `37-eval.sh` + `40-score.sh` | `outputs/eval-<mode>/<target>.jsonl`, `score.json` | **gate-eval**: 응답이 실제로 생성됨(빈 응답 차단) |
| 8 | 릴리스·배포 | `43-prep-upload.sh` → `46-verify-upload.sh` → `49-upload-hf.sh` → `52/55/58` | 모델카드 `README.md`, `outputs/release-notes-<mode>.md` | **gate-release**: 모델카드·지표·업로드 전 감사 통과, **gate-serving**: 서빙 응답 확인 |

## 1. Stage 1 — 수용 기준 (Acceptance Criteria)

수용 기준은 **한 파일**에 산다: `config/acceptance.toml`.
이 파일이 34(Δ 판정) · 43(모델 카드) · 46(용량 범위) · 61(릴리스 노트)의
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
(`uv.lock`을 건드리지 않으므로 10-setup 스탬프가 무효화되지 않는다).

## 1-1. Stage 7 — 표준 GSM8K 평가 (측정 프로토콜)

평가 입력은 **표준 벤치마크만** 쓴다. 프롬프트는 `36-prep-evalset.sh`가 만든다.

| 항목 | 값 |
|---|---|
| 데이터셋 | `openai/gsm8k`, config `main`, **test split** (1319문제) |
| 리비전 | `740312add88f781978c0658806c59bc2815b9866` (고정) |
| 샘플링 | `random.Random(seed)` — seed `20261024` 고정, `GSM8K_N`(0=전체) |
| 프롬프트 | Raon chat template(`user\n…\nassistant\n<think></think>\n\n`) + `#### <number>` 지시 |
| gold | 데이터셋 정답의 `#### N` (콤마 제거) |
| 재현성 | `outputs/eval-<mode>/evalset-<mode>.json`에 데이터셋·리비전·seed·n·**sha256** 기록 |
| 생성 | `37-eval.py --batch-size` (기본 `EVAL_BATCH=8`, 좌측 패딩) |

**배치 생성 주의**: 배치가 바뀌면 어휘는 달라질 수 있고(패딩/배치 모양에 따른 수치 차이),
**채점 결과도 달라질 수 있다** — 실측 20문제 대조에서 bs=1 19/20, bs=8 20/20(1문제 차이).
따라서 **base와 양자화본은 반드시 같은 프로토콜**(동일 배치 크기·동일 max_new_tokens)로
재야 비교가 공정하다. 허용폭 2%p는 n=200 기준 약 1σ 수준이니 표본 수와 함께 해석한다.

## 1-2. 모델 카드의 두 검사

`43-prep-upload`가 만드는 카드에는 **검사 2종**이 실측과 함께 박힌다.

| 검사 | 지표 | 기준 | 출처 |
|---|---|---|---|
| 검사 1 — PPL | in-domain PPL, base 대비 Δ | `ppl_delta_warn` / `ppl_delta_max` | `34-measure-ppl.sh` |
| 검사 2 — GSM8K | 표준 테스트셋 정확도 | base 대비 하락 ≤ `gsm8k_acc_drop_max` | `37-eval.sh` + `40-score.sh` |

## 2. 게이트 (단계 간 계약)

게이트 구현은 `gates.inc` 하나에 있고, `run-{smoke,full}.sh`와
`build-ppl-eval.inc`가 **같은 함수**를 호출한다(경로별로 정책이 갈리지 않게).

| 게이트 | 함수 | 막는 것 |
|---|---|---|
| gate-acceptance | `rq_gate_acceptance` | 기준 없이 진행 (Stage 1 산출물 누락) |
| gate-quant | `rq_gate_quant` | dense 폴백·미보정 `global_scale`(MoE 캘리브레이션 누락) |
| gate-ppl | `rq_gate_ppl` | NaN/미측정 PPL, 기준과 어긋난 verdict |
| gate-eval | `rq_gate_eval` | 응답 0건(모델이 아무것도 못 냄), GSM8K 정확도가 계약 허용폭 초과 |
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

`61-release-notes.py`가 원장 + 수용 기준 + 실측 지표를 합쳐
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
