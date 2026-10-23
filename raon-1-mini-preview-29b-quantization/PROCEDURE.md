# PROCEDURE — Raon-1-mini PTQ only (DGX Spark 1x 기본, 서빙은 2x/TP=2)

> 학습 Stage 없음 (SFT 제외). MXFP8 베이스 → BF16 복원 → 한국어 calib → 4종 PTQ
> → PPL 게이트 → vLLM eval → 업로드·서빙.
> 30B 플랜(65 결과 159~190행) 대비 VRAM 참고: BF16 서빙 32~36GB, AWQ 18~20GB.
> 베이스가 MXFP8이라 4종 전부 이중양자화 전제 (FP8 타깃 제외됨).

## Stage

```
[0. 환경 10-setup] → [1. calib 12-prep] → [2. compat 20-check] → [3. dequant 30]
  → [4. PTQ 33·32·35·36] → [5. PPL 48] → [6. eval 50·54] → [7. 배포 58·60·62·64]
```

| Stage | 스크립트 | 입력 → 출력 |
|---|---|---|
| 0. 환경 | `10-setup.sh` | env sync + HF gated 로그인 + GPU/NVFP4 가능 점검 |
| 1. calib | `12-prep-calibration.sh [smoke\|full]` | HF 한국어 혼합 → `datasets/raon-calibration-<N>.jsonl` (`text` chat-formatted) |
| 2. compat | `20-compat-check.sh` | trust_remote_code 로드, expert 타깃, MoE 제외층, SM≥10, vLLM 버전 |
| 3. dequant | `30-dequant-mxfp8.sh` | MXFP8 repo → `models/raon-bf16` (32~36·48·50의 입력, 1회) |
| 4. PTQ | `33-awq-w3` `32-awq-w5` `35-nvfp4-awq` `36-nvfp4-gptq` | 복원본 → `models/raon-<target>-<mode>` 4종 (experts-only 기본) |
| 5. PPL | `48-measure-ppl.sh [smoke\|full]` | base(raon-bf16) + 4종 in-domain PPL, Δ<0.3 Accept / 0.3~1.0 Conditional / ≥1.0 Discard |
| 6. eval | `50-eval.sh` + `54-score.sh` | vLLM offline 한국어 smoke 생성 + 완결율·thinking 포함률 리포트 |
| 7. 배포 | `58-upload-hf.sh`, `60-serve-vllm.sh`, `62-infer-examples.sh`, `64-serve-infer-all.sh` | `tayaee/Raon-1-mini-preview-<target>` 업로드, TP=2 서빙 순회 |

## smoke/full 스케일 (`config/common.env`)

| 항목 | smoke (full 동일 단계, 작은 규모) | full |
|---|---|---|
| CALIB_N | 8 | 256 |
| SEQ_LEN | 512 | 2048 |
| EVAL_N / PPL_N | 8 / 8 | 200 / 32 |
| 실행순 | awq-w3 → awq-w5 → nvfp4-awq → nvfp4-gptq (알파벳순) | 동일 (밤샘 권장) |
| 2노드 분할 | `TARGETS=awq-w3,awq-w5` (spark1) + `TARGETS=nvfp4-awq,nvfp4-gptq` (spark2), `STEPS=dequant,quant`로 양쪽 실행 후 1곳에서 `STEPS=ppl,eval` | 동일 분할 (`MODE=full`) |

## 30B 경로와의 관계 (65 결과 파일 기준)

- 30B FFT-FSDP/DDP는 OOM으로 불가 판정 → 본 파이프는 학습 자체를 안 하므로 해당 없음.
- 30B 퍼블리싱 3종(FP8/AWQ/GGUF) 중 GGUF는 본 스코프 제외. FP8도 MXFP8 베이스와 겹쳐 제외.
- QLoRA-merged 이중양자화 비교도 제외 (SFT 제외 결정 따름).
