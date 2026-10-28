# handoff — 2026-10-25 429 중단 복구 (issue-21 후속)

## 발생한 일
- `EVAL_BENCHMARKS=gsm8k ./s7-37-eval.sh full base,nvfp4-awq-transformers` 백그라운드 실행 중
  Individual quota 429로 세션 비정상 종료. 로그(`logs/eval-gsm8k-refill-full.log`)는
  base 가중치 로딩 100% 지점에서 멈춤. 출력 파일 미변경, 프로세스 잔존 없음.
- 요청했던 handoff.md는 429 전에 생성되지 못했음 → 본 파일이 그 기록을 겸함.

## 점검 결과 (2026-10-25 09:24 EDT)
- `eval-full/base.jsonl`, `nvfp4-awq-transformers.jsonl`: gsm8k 0행 (kmmluko 200 + format 10 = 210행).
- `eval-full/backup-0903/`: 소실 이후에 뜬 백업이라 gsm8k 없음 (복구 불가).
- **`eval-full/bench/gsm8k/{base,nvfp4-awq-transformers}.jsonl`: 각 200행 온전** (10-25 00:06/00:58 생성).
  prompts-gsm8k.jsonl 선두 200행과 prompt·gold 전수 일치, 가중치 불변
  (raon-bf16 10-24 21:46, raon-nvfp4-awq-transformers-full safetensors 10-24 23:19 이후 README만 추가).
  → GPU 재실행(예상 1h40m) 불필요, bench 사본 병합으로 복구 결정.

## 복구 수행
1. 현 top-level 4종을 `backup-1025-prerestore/`에 백업.
2. `bench/gsm8k` 200행 + 기존 kept(kmmluko+format) 순서로 top-level 재작성
   (s7-37-eval.py run_target과 동일 순서: 신규행 먼저, kept 나중).
   → 각 410행 (gsm8k 200 / kmmluko 200 / format 10).
3. `./s7-40-gsm8k.sh full` → `./s7-46-kmmluko.sh full` → `./s7-49-format.sh full`
4. `./s8-43-prep-upload.sh full nvfp4-awq-transformers` → `./s8-46-verify-upload.sh full nvfp4-awq-transformers` (ok:true)

## 결과 수치 (checks.json)
- gsm8k: base 0.945 (189/200) ref / nvfp4-awq-transformers 0.935 (187/200) PASS, drop_max 0.02
- kmmluko: base 0.565 / nvfp4-awq-transformers 0.535 PASS, drop_max 0.03
- format: base 0.900 / nvfp4-awq-transformers 0.900 PASS (실패 케이스: empty_question, 양쪽 동일)
- README(디스크 `raon-nvfp4-awq-transformers-full/README.md`, 6015B)에 Check 2 표 포함 확인.

## 검증
- `uv run pytest -q tests/`: 136 passed
- `regression-tests/verify-issue-*.sh`: 전부 OK (14·15·16 스크립트는 존재하지 않음)
- 코드 변경 없음 → HEAD 64851c7 유지. `git status` clean (untracked `d1` 제외, 커밋 안 함).

## 다음 할 일
- `d1` 메모는 사용자 파일이므로 그대로 둠. 필요시 삭제 요망.
- 업로드(`s8-49-upload-hf.sh`)는 미실행 — 지시 시 진행.
