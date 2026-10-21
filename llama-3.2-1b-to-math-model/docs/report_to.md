# report_to — wandb + trackio 2종 리포팅

full 잡에 `wandb,trackio` 2개를 켜기 위한 문서.
개별 옵션 상세는 `docs/reporting.wandb.howto.md`, `docs/reporting.trackio.howto.md` 참조.

> 2026-10-11 결정: tensorboard/mlflow/dvclive는 활성 셋업에서 제외.
> mlflow는 file-store가 maintenance 모드라 `MLFLOW_ALLOW_FILE_STORE=true` 없이는
> Trainer 생성 시점에 잡 전체를 죽임 (trackio 파손과 동일 패턴). 관련 howto 3종 삭제됨.

## 0. 사전 체크 (학습 전)

```bash
uv run --no-sync check_report_to.py --report_to wandb,trackio --out $OUT
./check_report_to.sh   # 기본 2종 (install/config/output 섹션)
```

`--no-sync` 필수 (plain `uv run`은 `uv sync`로 .venv를 갈아엎음 — 학습 중 실행 금지).
종료 코드 0=전체 활성, 1=MISSING/BROKEN/UNKNOWN 있음.

## 1. 코드 변경 (완료)

* `report_check.py`: 검증 단일 진실 공급원. `parse_report_to` + `summarize`/`gate_report_to`.
  Trainer와 동일한 함수(`get_reporting_integration_callbacks` + 콜백 인스턴스화)로 검증하므로
  체커 OK ⇔ 잡 통과가 보장된다. mlflow file-store는 임시 디렉토리로 setup 모사 검증.
* `train-entry.py`: `--report_to` CSV 인자 + dry_run 직후 `gate_report_to()` 기동 게이트.
  문제 있으면 데이터 적재 전에 exit=2로 즉시 실패 (NCCL 연쇄 사망 없음).
  `SFTConfig(project="trackio-math-1b-sft-and-quantization")`.
* `10-baseline/30-fft/28-qlora-train.sh`: `--report_to "${REPORT_TO:-trackio}"` 전달.
  기본값 `none`이라 기존 동작 불변. `WANDB_PROJECT`/`MLFLOW_EXPERIMENT_NAME` 기본값 적용.

## 2. 설치 (spark1/spark2 양쪽)

```bash
uv pip install "trackio==0.42.0" wandb   # 버전은 math-1b-sft-and-quantization/pyproject.toml에 고정 (uv sync가 설치)
```

## 3. 환경변수 (양 노드 동일)

```bash
export REPORT_TO="wandb,trackio"   # 미설정 시 common.env 기본값 trackio. wandb 추가용.
export WANDB_PROJECT="wandb-math-1b-sft-and-quantization"   # 미설정 시 학습 스크립트가 기본값 적용
wandb login   # wandb만 유일하게 인증 필요. offline 비교면 WANDB_MODE=offline
```

구분명 정리: trackio=`trackio-math-1b-sft-and-quantization`(SFTConfig project), wandb=`WANDB_PROJECT`.

## 4. 중단 후 재실행

`report_to`는 Trainer 생성 시 고정이라 진행 중 잡에는 적용 불가. 중단 후 재실행 필수.

## 5. 보고서 위치 대조표 (동일 잡)

| 연동 | 산출물 | 확인 명령 |
|---|---|---|
| trackio | `$HF_HOME/trackio/trackio-math-1b-sft-and-quantization.db` | `uv run --no-sync trackio show --project trackio-math-1b-sft-and-quantization --host <IP>` |
| wandb | cloud/offline `wandb/` | wandb.ai 대시보드 또는 `wandb sync --sync-all` |

trackio 대시보드는 학습과 같은 `HF_HOME`으로 띄울 것 (다르면 빈 DB를 봄).

## 6. 트러블슈팅

* 설치 안 된 연동은 조용히 스킵됨 (`get_available_reporting_integrations()`에 없음).
* **설치됐는데 import가 깨지는 연동은 Trainer 생성 시점에 잡 전체를 죽임.** `check_report_to.py`가 실제 import까지 검증해 `[BROKEN]`으로 표시. 처방: 해당 연동을 report_to에서 제외하고 재실행.
* wandb 미로그인 상태에서 online 모드면 잡 실패 가능 — 먼저 `wandb login` 또는 `WANDB_MODE=offline`.
* 패키지 추가 후에도 `--report_to` 미지정(`none`)이면 아무 파일도 안 생김. 정상.
