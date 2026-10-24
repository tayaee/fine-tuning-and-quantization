# troubleshooting-awq-w3.md — 22-awq-w3 smoke 8연패 → 성공 기록

> 대상: `22-quant-awq-w3.sh smoke` (pure-GPTQ W3A16-g64, experts-only, calib 8·seq 512)
> 기간: 2026-10-22 08:41Z~16:08Z (timing: `logs/timing-smoke.log`)
> 결과: 8회 실패 후 9번째 시도에서 성공 (`22-awq-w3 7248s rc=0`)
> 콘솔 로그: `logs/run-smoke-console*.log` (prev = 시도별 보관분)

## 시도 일람

| # | 시각(Z) | 소요 | rc | 실패 시그니처 | 원인 | 수정 |
|---|---|---|---|---|---|---|
| F1 | 08:41 | 3s | 1 | `oneshot does not support models already quantized (mxfp8)` | 베이스가 MXFP8 저장이라 oneshot 직접 불가 | `19-dequant-mxfp8` 스테이지 신설 (BF16 복원 후 양자화) |
| F2 | 10:01 | 1170s | 1 | (a) `columns not divisible by group_size` (down_proj 1088 % 128) (b) `'dict' object has no attribute 'last_hidden_state'` (26/26 블록) | (a) g128이 down_proj 불가 (b) fx trace가 ModelOutput→plain dict로 바꿔 attribute 접근 실패 | (a) 전 expert g64 (b) `modeling_raon1_mini.py` shim: `outputs.x` → `outputs["x"]` (`19-dequant-mxfp8.py:_apply_tracing_shim`) |
| F3 | 10:42 | 166s | 1 | (a) `AWQ needs to match a single smoothlayer` (input_layernorm ×22 vs q/k/v 0개) (b) `Cannot initialize a modifier that has already been initialized` | (a) AWQ auto-mapping이 q/k/v 전제, Raon은 해당 구조 없음 (b) 실패한 modifier 객체 재사용 | (a) Raon 전용 AWQ 매핑 (`quant_common.py:RAON_AWQ_MAPPINGS`) (b) 시도마다 fresh 인스턴스 생성 |
| F4 | 11:21 | 1146s | 1 | (a) 동일 single-smoothlayer (post_attention_layernorm ×25 한 그룹) (b) `KeyError: 'past_key_values'` | (a) dense 0층 norm이 balance 없이 먼저 매칭 → parent_context가 "model.layers"로 굳어 per-layer 분리 실패 (b) trace된 dict에 past_key_values 키 없음 | (a) smooth를 MoE층(1~24)으로 제한 (b) `outputs.get(...)` |
| F5 | 12:34 | 1263s | 137 | (a) `Raon1MiniRoutedLinear.forward() got an unexpected keyword argument 'hidden'` (b) fallback이 compression 31%에서 SIGKILL | (a) AWQ replay가 bound kwargs(`hidden=`)로 linear 호출, 커스텀 forward와 충돌 (결정적) (b) `last_err = e`가 exception→traceback→58GB 모델을 붙잡아 다음 시도와 2중 적재 OOM | (a) `RQ_AWQ_FIRST` 기본값 `0` (pure 경로 직행, 폴백과 동일 결과) (b) `last_err`는 str만 보관 + `del recipe` + `gc.collect()` |
| F6 | 13:06 | 1386s | 137 | compression 79%에서 SIGKILL (it/s 17→4 thrash) | 단일 모델로도 ~2~3GB/min 증가 → 121GB 초과 (원인 후보: 단편화 vs 진짜 누수 vs 중복 적재) | `MALLOC_MMAP_THRESHOLD_`로 단편화 가설 검증 시도 |
| F7 | 13:33 | 1314s | 137 | compression ~0%에서 SIGKILL (mmap 적용 후에도 동일) | 단편화 아님 (가설 기각) → 진짜 retention | buff/cache·RSS 동시 로깅으로 seasoning 분석 |
| F8 | 14:03 | 1314s | 137 | 사망 시점 RSS 67GB + buff/cache 0.8GB + used 124GB → 56GB 행방불명 | unified 메모리: GPU 측에 2GB/layer씩 누적 (RSS에 안 잡힘, nvidia-smi는 N/A로 은폐) | `CUDA_VISIBLE_DEVICES=""` 진단 → CPU-only로 bounding 확인 |
| OK | 16:08 | 7248s | 0 | — (pure-GPTQ W3-g64 완주, 9216 experts I8 packed 확인) | — | — |

참고: 서버 재시작으로 강제 종료된 1회(로그 미기록)는 번호에서 제외. `raon-awq-w3-full/` 잔해, `raon-awq-w3-smoke/` 빈 껍데기는 이후 성공분이 덮어씀.

## 결정적 디버깅 포인트 3개

1. **tail dict crash (F2)** — `probe_trace.py`로 `trace_subgraphs`를 직접 돌려 26/26번 subgraph 생성 코드를 출력.
   원인 코드 (생성된 코드): `wrapped_34 = wrapped_11({'last_hidden_state': ...}, True)` 뒤에
   `getattr_2 = wrapped_34.last_hidden_state` → plain dict라 AttributeError.
   Llama류는 `outputs[0]` (getitem)이라 무사한 반면 Raon은 attribute 접근이라 깨짐.
   수정: `outputs["last_hidden_state"]` / `.get(...)` (ModelOutput도 getitem 지원이라 런타임 동일).
2. **AWQ grouping 붕괴 (F4)** — `probe_match.py`로 `match_modules_set` 그룹 수를 직접 확인.
   `N_GROUPS=1, lens=[25, 3072, 3072]` → dense 0층 norm이 balance 없이 먼저 매칭되면
   parent_context가 "model.layers"로 일반화되어 이후 per-layer 분기가 영원히 안 일어남
   (`compressed_tensors/utils/match.py` 상태머신). smooth 패턴을 MoE층(1~24)으로 제한하자
   `N_GROUPS=24, lens=[1,128,128]` 로 정상 분리됨을 probe로 먼저 확인 후 본런.
3. **OOM 정체 규명 (F6→F8)** — `memlog.sh` (RSS + mem_used + buff/cache + swap 30초 로깅)로
   F6(17→4 it/s thrash), F7(mmap 무효→단편화 기각), F8(RSS 67GB인데 used 124GB→56GB HDF→GPU unified)
   순서로 가설을 좁힘. `nvidia-smi`는 GB10에서 memory N/A라 early check에서 놓쳤음.

## 최종 성공 조건 (재현용)

- `SRC=raon-bf16` (shim 적용済み: `outputs.get(...)` 2곳 + `outputs["last_hidden_state"]` 1곳)
- recipe = pure-GPTQ (`RQ_AWQ_FIRST=0` 기본): g64 전 expert, `dampening` 기본
- `CUDA_VISIBLE_DEVICES=""` (CPU-only, GB10 unified 메모리 GPU 누적 회피. 느림: smoke 2시간/모델)
- 산출물 검증: `I8 9216` packed + BF16 scales (dtype census), 55GB/3 shards

## Lessons learned (교육용)

1. **smoke는 full의 축소판이어야지, 다른 파이프라인이면 안 된다.**
   `PIPELINE_TEST=1` 기본값이 양자화·PPL·eval을 통째로 skip해서 모든 에러가 full에서 처음 터졌다.
   수정: 기본값을 전 단계 실행(`PIPELINE_TEST=0`)으로 바꾸고 빠른 점검만 opt-in으로 (`run-smoke.sh:17`).
2. **핀된 venv를 깨뜨리지 마라.**
   `uv run` 한 번이 lockfile 기준으로 `llmcompressor/compressed-tensors`를 되돌려
   `ImportError: exec_jobs_dynamic`를 만들었다. 이 repo는 `10-setup.sh`가 의도적으로 pip override를 얹는
   구조라, 모든 실행은 `.sh` 래퍼(= `UV_NO_SYNC=1`) 경유 + probe도 `UV_NO_SYNC=1` 필수.
3. **20분짜리 본런 전에 3분짜리 probe로 가설을 검증해라.**
   fx-trace probe(2분), match-grouping probe(3분), dtype census(수초)가 없었다면 시행착오가 배로 늘었다.
   특히 framework 내부를(서브그래프 생성 코드, 그룹 분할) 눈으로 보지 않고는 못 잡는 버그가 있다.
4. **커스텀 아키텍처 세금표를 미리 써라.**
   `outputs.last_hidden_state`(attribute) vs `outputs[0]`(getitem),
   `Raon1MiniRoutedLinear` 서브클래스, dict 반환 cache, `([1-9]|1[0-9]|2[0-4])` 같은 layer-index 가정 —
   벤더 custom code는 표준(Llama) 경로를 벗어나는 곳마다 터진다. 처음부터 목록화하고 shim은
   산출물(`19-dequant`)에 코드로 남겨라(일회성 수기 패치 금지).
5. **framework의 매칭/그룹 상태머신을 이해하고 써라.**
   `match_modules_set`은 순서 의존적이다: balance 없이 먼저 매칭된 smooth가 parent_context를
   오염시키면(dense 0층) 이후 분기가 전부 무너진다. auto-mapping이 있더라도 custom arch에선
   explicit mapping + probe 검증을 기본 절차로.
6. **재시도 체인은 실패작의 참조를 끊어라.**
   `last_err = e` 한 줄이 58GB 모델을 붙잡아 다음 시도를 OOM(`rc=137`)으로 죽였다.
   str만 보관 + `del recipe` + `gc.collect()`, modifier는 시도마다 fresh 생성.
   (`Cannot initialize a modifier that has already been initialized`도 같은 뿌리)
7. **결정적 실패는 기본값에서 빼라 (doomed-attempt tax).**
   AWQ replay 실패가 100% 재현되자 `RQ_AWQ_FIRST` 기본을 `0`으로 바꿨다.
   결과물은 폴백과 동일하고, 모델당 ~7분 + 2중 적재 OOM 리스크를 아낀다. 나중에 고치면 flag만 뒤집으면 된다.
8. **리소스 모니터링은 OOM 난 뒤가 아니라 처음부터.**
   RSS만 보면 안 된다. unified 메모리 박스에선 `free` total vs RSS vs buff/cache(+GPU 사용량)를
   함께 봐야 한다. `nvidia-smi` N/A 같은 사각지대를 알았으면 `CUDA_VISIBLE_DEVICES=""` 같은
   격리 실험으로 조기에 가설을 좁혀라. F6→F8 세 번의 OOM은 로거 하나로 한 번에 끝낼 수 있었다.
9. **unattended 원칙: traceback은 삼키지 말고 로그에 남겨라.**
   폴백 체인의 `except: print`는 `traceback.print_exc()`와 함께 (32/33/35 적용).
   사람이 안 보고 있을 때 원인 특정 가능한 유일한 수단이다.
10. **한 번에 한 변수, 타이밍 로그는 진실의 원천.**
    `logs/timing-<mode>.log`의 `(이름, 소요, rc)` 행이 8연패를 한눈에 세게 해줬다.
    수정→재실행은 실패 지점부터 resume되게(fresh-skip) 짜여 있으니, 전체 재실행 공포 없이
    가설 하나씩 검증할 것. `STEPS=`/`FORCE=`는 그ための 것.
