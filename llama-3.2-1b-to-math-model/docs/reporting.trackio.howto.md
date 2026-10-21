# reporting.trackio — HF 경량 대시보드 (로컬)

## 1. 설치 (양 노드 동일하게)

```bash
uv pip install "trackio>=0.21.1"
```

> 주의 (실측): trackio 0.19.0 + huggingface_hub>=1 조합은 `import trackio` 자체가
> `CommitOperationAdd` ImportError로 깨지며, Trainer 생성 시점에 분산잡 전체가 죽는다.
> 반드시 `check_report_to.py`로 `[OK]` 확인 후 투입. 깨지면 report_to에서 trackio 제외.

## 2. 학습에 전달

```bash
REPORT_TO=trackio ./28-qlora-train.sh full ddp
```

외부 Space 공유를 쓰지 않는 한 로그인·키 불필요. 프로젝트명=`trackio-math-1b-sft-and-quantization`
(`train-entry.py`의 `SFTConfig(project=...)`).

저장 위치: `$HF_HOME/trackio/trackio-math-1b-sft-and-quantization.db` (SQLite).
`common.env`가 `HF_HOME=$M1B_SHARED/hf-cache`로 지정하므로 실측 경로는
`/rosenas/data/AIML/math-1b-shared/hf-cache/trackio/`. `./trackio/`가 아님에 유의.

## 3. 보고서 확인

```bash
uv run --no-sync trackio show --project trackio-math-1b-sft-and-quantization --host <IP>
```

> 경고: `--no-sync` 없이 `uv run trackio show`를 실행하면 `uv sync`가 돌아가며
> `uv.lock` 기준 trackio==0.19.0으로 **되돌아가** 학습이 깨진다 (2026-10-11 실측).
> 대시보드는 반드시 `--no-sync`으로 띄울 것.

로컬 `./trackio/` 아래 run 기록. Hub Space로 실시간 공유하려면 `SFTConfig(trackio_space_id=...)` 추가가 필요 — 현재 `train-entry.py`는 미지원, 로컬 비교용으로 사용.

## 4. DDP 주의

* rank0만 기록. 양 노드 동일 버전 설치 권장.
