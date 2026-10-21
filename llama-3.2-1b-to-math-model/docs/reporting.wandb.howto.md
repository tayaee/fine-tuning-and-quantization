# reporting.wandb — SaaS 실험추적 (유일한 외부망 필요)

## 1. 설치 (양 노드 동일하게)

```bash
uv pip install wandb
```

## 2. 로그인 (양 노드)

```bash
wandb login            # API 키 입력 (https://wandb.ai/authorize)
# 또는 비상호작용
wandb login "$WANDB_API_KEY"
```

프로젝트 지정 (미지정 시 학습 스크립트가 `wandb-math-1b-sft-and-quantization` 적용):

```bash
export WANDB_PROJECT="wandb-math-1b-sft-and-quantization"
```

외부망 차단 환경에서 먼저 기록만 하려면:

```bash
export WANDB_MODE=offline   # 나중에 wandb sync --sync-all 로 업로드
```

## 3. 학습에 전달

```bash
REPORT_TO=wandb,trackio ./28-qlora-train.sh full ddp
```

## 4. 보고서 확인

* online: `https://wandb.ai/<entity>/<WANDB_PROJECT>/runs` 대시보드.
* offline: `wandb sync --sync-all` 후 동일 URL.

## 5. DDP 주의

* rank0만 기록. `WANDB_API_KEY`·`WANDB_PROJECT`·`WANDB_MODE`는 양 노드 동일 export.
* wandb 인증 실패는 잡 실패로 이어질 수 있음 — 재실행 전 `wandb login` 상태 확인.
