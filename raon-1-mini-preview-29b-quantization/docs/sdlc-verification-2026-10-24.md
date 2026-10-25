# SDLC 9단계 도입 검증 기록 (2026-10-24)

> 이 문서는 "9단계 SDLC 도입" 작업의 **검증 결과**와, 그 과정에서 발견·수정한
> 실제 결함 3건을 기록한다. (코드 변경의 근거는 `issues/archive/2026/10/24/`의 issue 문서)

## 1. 왜 다시 돌려야 했는가 (재실행이 3분을 넘긴 원인)

사용자 요구: "nvfp4-awq smoke/full을 한 번 더 실행. 이미 수행돼서 순식간에 끝나야 정상.
3분 이상 걸린다면 그것도 수정해. 2회 실행할 경우 즉시 끝나도록."

재실행이 느린 원인은 3가지였다(모두 **freshness 판정**과 관련):

| # | 원인 | 증상 | 근본 | 조치 |
|---|---|---|---|---|
| A | dequant 산출물에 MoE 전-expert 캘리브레이션 심(`get_calibrate_all_experts_flag`)이 없음 | `s4-19-dequant-mxfp8.sh`의 freshness grep이 실패 → **매 실행마다 999s(16.6분) 재복원** | 심 적용본이 커밋된 적 없음(구 코드에선 치환 미적용) | `s4-19-dequant-mxfp8.py` 재실행 → 심 적용(5/7) → 이후 skip |
| B | 파일 개명(commit `1e573cc`)으로 스크립트 mtime이 산출물보다 최신 | `s5-28-quant-nvfp4-awq.sh` freshness 실패 → **매 실행마다 1190s(20분) 재양자화** | freshness가 산출물↔스크립트 mtime 비교에 의존 | 산출물 재생성(스크립트 mtime보다 최신) → 이후 skip |
| C | 측정 산출물 자체가 미생성 | `ppl-smoke/nvfp4-awq.json`, `eval-full/*.jsonl` 부재 → 1회 실행이 수십 분~수 시간 | 이전 실행이 중단됨 | Phase B로 재측정 |

## 2. 실측 결함(게이트가 잡아낸 것)

`global_scale` 값 계약 위반 — 2026-10-24 이전 산출물은 **모아 produces garbage**:

```
raon-nvfp4-awq-smoke: 비정상/미보정 global_scale 267개
  (예: model.layers.1.moe.experts.118.down_proj.input_global_scale=0.000e+00)
```

- 원인: 미라우팅 expert의 `input_global_scale`이 미초기화 상태(0 또는 쓰레기값)로 저장.
- 결과: 해당 expert가 라우팅되면 출력이 garbage → **평가 점수 전부 0** (`eval-smoke/score.json`:
  nvfp4-awq gsm8k 0.0 / humaneval 0.0 / kmmlu 0.0, avg_len=256.0 = 토큰 상한 도달).
- 조치: dequant의 MoE 전-expert 캘리브레이션 심을 적용한 복원본으로 재양자화 →
  `input_global_scale` 9,216개 전부 정상(bad=0), 평가 점수가 base와 동일해짐.

> 이 결함은 `run-{smoke,full}.sh` 경로에 게이트가 없어서 end-to-end로 드러나지 않았다.
> 게이트를 경로에 연결한 뒤(issue-7) 즉시 검출 가능한 상태가 됐다.

## 3. `46-verify-upload` 오탐(감사 스크립트 결함)

릴리스 게이트를 실제 돌리니 compliant 모델이 실패했다:

```
config.json: vLLM/생성 필수 필드 'eos_token_id' 누락
config.json: vLLM/생성 필수 필드 'pad_token_id' 누락
```

조사 결과 **KRAFTON 원본 config.json에도 두 필드가 없다**(토크나이저 설정만 보유).
즉 검사가 "원본에 없는 필드"를 요구해 원본을 충실히 따르는 모델을 막고 있었다.
→ **베이스 모델에 있을 때만** 양자화본에도 요구하도록 수정(`_base_config()` 기준).
수정 후 smoke/full 모두 `ok=true`.

## 4. 환경 드리프트(별제 결함)

`compressed-tensors`가 0.19.0 → 0.17.0으로 내려가 있어 llmcompressor 0.14.0 import가
실패(`cannot import name 'exec_jobs_dynamic'`)했다. `s0-10-setup.sh`의 **스탬프 판정**
(uv.lock 해시 + 핀 버전)이 이를 감지해 자동 복구했다 — stamp 설계가 실제로 작동한 사례.

## 5. 재실행 측정 결과

| 대상 | 1회차 | 2회차 | 비고 |
|---|---|---|---|
| `10-16,19,28-40-build-smoke-nvfp4-awq.sh` | **3s** | **2s** | 전 단계 skip + 게이트 3종 통과 |
| `10-16,19,28-40-build-full-nvfp4-awq.sh` | 3211s | **2s** | 1회차는 캘리브 재생성 → 양자화 무효화 cascade(아래) |
| (2026-10-24 재측정) smoke / full | 2s / 3s | **2s / 2s** | 표준 GSM8K 평가 도입 후에도 즉시 종료 유지 |

full 1회차가 53분 걸린 이유(정상적 재작업):
`13-calib`이 캘리브 세트를 **재생성**했다(스크립트 mtime이 데이터보다 최신 — 개명 커밋의 부작용).
캘리브 데이터는 양자화의 입력이므로 `28-quant-nvfp4-awq`의 freshness가 깨져
**양자화(2176s) → PPL(341s) → eval(655s)** 이 순차 재실행됐다.
2회차부터는 캘리브·양자화·측정 모두 최신이라 전부 skip → **2초**.

> 원칙: **입력이 바뀌면 산출물은 무효화된다**(이건 맞다). 다만 입력이 *내용 변경 없이*
> 재생성되는 경우(스크립트 mtime 변화)는 불필요한 36분 재작업이 된다.
> 규칙: "캘리브 데이터가 재생성되면 양자화 산출물은 무효" — 무효화는 옳고 재사용은 위험하다.
> 재작업을 없애려면 캘리브 스텝에 "데이터 내용 해시" stamp를 두어야 한다(향후 과제).

개별 스텝 skip 확인(2회차 로그):

```
10-setup        skip (stamp)                     0s
07-acceptance   acceptance-<mode>.json 재생성    0s
13-calib        skip (fresh)                     0s
16-compat       COMPAT-OK (stamp)                0s
19-dequant      skip (fresh + MoE 심 확인)        0s
28-nvfp4-awq    skip (fresh)                     0s
gate-quant      verify_compressed_output OK      1s
34-ppl          skip (산출물 > 모델 config.json)  0s
gate-ppl        PPL OK + 수용 기준 판정 일치      0s
37-eval         skip (산출물 > 모델 config.json)  0s
gate-eval       eval OK (9 lines)                0s
40-score        재집계                            0s
```

## 6. 측정 결과 (재수집)

| 모드 | 대상 | PPL | Δ vs bf16 | 판정 | 표준 GSM8K | done_rate |
|---|---|---|---|---|---|---|
| smoke | base | 6.0550 | - | ref | 0.900 (n=20) | 1.0 |
| smoke | nvfp4-awq | 6.0825 | +0.0275 | **Accept** | 0.950 (n=20) | 1.0 |
| full | base | 7.7022 | - | ref | 0.945 (n=200) | 1.0 |
| full | nvfp4-awq | 7.7011 | −0.0010 | **Accept** | 0.935 (n=200) | 1.0 |

> GSM8K는 `openai/gsm8k` main/test(1319문제)에서 seed 20261024로 셔플한 표본의 앞 n개.
> 전체 1319를 돌리려면 `GSM8K_N=0 EVAL_N=1319 ./run-*.sh` (배치 8 기준 약 2.2시간).

수집 비용(재수집): 양자화 smoke 1190s / full 2227s · PPL smoke 947s / full 1029s ·
eval smoke ~1130s / full ~1650s · dequant 958s.

Stage 8 릴리스 게이트 실측: `43-prep-upload`·`46-verify-upload` smoke/full 모두 `ok=true`,
`rq_gate_release` 통과. 모델 카드에 "수용 기준(Stage 1 계약) 대비 판정" 표가 실측과 함께 박힘.

## 7. 판정 원칙 (코드 고정)

1. **스크립트를 고쳐도 이미 얻은 측정값은 무효가 되지 않는다.**
   측정값이 무효가 되는 경우는 (a) 모델이 바뀌었거나 (b) 판정 계약이 바뀌었을 때뿐.
   (PPL/eval freshness는 "산출물 > 모델 config.json" 기준 — 스크립트 mtime 아님)
2. **Stage 1 계약이 기준의 단일 소스**: `config/acceptance.toml`
   (Δ 임계값·용량 범위·포맷). 코드 하드코딩 없음.
3. **게이트는 산출물을 먼저 검사**: 다음 단계로 넘어가기 전에
   패킹 계약 위반·NaN PPL·빈 응답을 차단.
4. **증빙은 원장**: `outputs/stage-ledger-<mode>.json` → `s8-61-release-notes.py` → 릴리스 노트.
