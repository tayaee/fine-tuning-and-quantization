"""train-entry.py — Stage 1/3 공용 학습 진입점 (TRL SFTTrainer).

원본: main.py + trainer436 + train_basic/FFT/QLoRA.sh → 2026 스택으로 이식.
- 데이터: {"source":[...], "target":[...]} jsonl → 학습 프롬프트 템플릿
  (원본 TokenizeProcessor/quant 노트북과 동일 문구).
- 토크나이저: "[PAD]"+ "<mask>" 추가 후 신규 임베딩 평균 초기화 (main.py resize()).
- QLoRA(--peft): bnb 4bit nf4 double_quant + LoRA r16/a32/drop0.05.
- 전략: single=플레인, ddp=torchrun DDP(자동), fsdp=True+auto_wrap.
- 강의 기본값: lr 1e-5, cosine, warmup 0.03, micro 2, tok/tgt 512, seed 1,
  mask_rate 0 (masking off).
- DGX 적응: gradient_checkpointing=True (원본 False), bf16, tf32 허용.

사용: torchrun ... train-entry.py --model ... --train ... --out ... [--peft]
검증: python3 train-entry.py --train <jsonl> --out /tmp/x --dry_run
"""

import argparse
import json
import math
import os
import random
import warnings

# torch 2.13 내부(FSDP/accelerate 경로)가 아직 구 별칭
# all_gather_into_tensor를 호출해 FutureWarning을 띄운다. 우리 코드 직접 호출은
# 없으므로(리포지토리 내 all_gather 호출 없음) 해당 경고만 필터한다.
warnings.filterwarnings(
    "ignore",
    message=".*all_gather_into_tensor.*",
    category=FutureWarning,
)

from io_common import clean_tmp, commit_dir, tmp_path
from report_check import gate_report_to, parse_report_to

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context.\n"
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\nGive a response as the assistant with the input conversation history\n\n"
    "### Response:\n{response}"
)
MASK_TOKEN = "<mask>"
PAD_TOKEN = "[PAD]"


def first(v):
    return v[0] if isinstance(v, list) else v


def load_texts(path):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(
                PROMPT_TEMPLATE.format(
                    instruction=first(row["source"]), response=first(row["target"])
                )
            )
    return texts


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # dry_run 환경 (torch 없음)에서는 stdlib만


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Llama-3.2-1B")
    p.add_argument(
        "--revision",
        default=os.environ.get("BASE_REVISION", ""),
        help="HF 리비전 고정 (기본 $BASE_REVISION, 1년 후 재현용). 빈 값이면 floating.",
    )
    p.add_argument("--train", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--mode", default="mini")
    p.add_argument("--peft", action="store_true", help="QLoRA 경로 (31에서 사용)")
    p.add_argument("--strategy", default="single", choices=["single", "ddp", "fsdp"])
    p.add_argument(
        "--expect-world",
        type=int,
        default=0,
        help="기대 월드 크기 (0=검사 생략). .sh가 $WORLD로 전달 — "
        "분산 의도와 실제 런치가 다르면 시작 즉시 실패시킨다.",
    )
    p.add_argument(
        "--accum", type=int, default=32, help="grad accum (effective batch 64 유지용)"
    )
    p.add_argument("--micro", type=int, default=2)
    p.add_argument("--max_len", type=int, default=1024, help="tok 512 + tgt 512")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=float, default=0.03, help="warmup 비율 (내부에서 warmup_steps로 환산)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--mask_rate", type=float, default=0.0, help="원본 max_mask_rate (기본 0=off)"
    )
    p.add_argument(
        "--max_rows", type=int, default=0, help="학습 행 상한 (0=전체, mini 경량화용)"
    )
    p.add_argument("--dry_run", action="store_true", help="데이터 포맷만 확인 후 종료")
    p.add_argument(
        "--report_to",
        default="trackio",
        help='리포팅 연동 CSV (예: "trackio", "wandb,trackio"). '
        '"none"=미사용, "all"=설치된 연동 전체.',
    )
    return p.parse_args()


def _save_fsdp_dcp(fsdp_model, model_cfg, tok, out_dir, dtype):
    """2노드 FSDP 전용 저장: sharded DCP + rank0 디스크 consolidation.

    full-model gather를 fabric에 던지지 않는다 (CX7 RoCE 버스트 손실로
    rank0 최종 저장이 deterministic하게 실패). fabric을 타는 것은 barrier
    (수 바이트)뿐이며, 노드 속도 차는 barrier가 흡수한다.
    shard 파일은 out_dir/shards/ 에 남긴다 (재조립·디버그용).
    """
    import torch.distributed as dist
    from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
    from torch.distributed.checkpoint import load as dcp_load
    from torch.distributed.checkpoint import save as dcp_save
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    rank = dist.get_rank()
    shard_dir = os.path.join(out_dir, "shards")
    if rank == 0:
        os.makedirs(
            shard_dir, exist_ok=True
        )  # 생성은 rank0만 (공유FS makedirs race 회피)
    dist.barrier()

    # 1) 각 rank 자기 shard만 디스크에 기록 (대형 fabric 전송 없음).
    # get_model_state_dict은 FSDP 래핑을 자동 감지해 sharded로 반환한다
    # (FSDP.state_dict_type 컨텍스트는 deprecated).
    local_sd = get_model_state_dict(fsdp_model)
    dcp_save(local_sd, storage_writer=FileSystemWriter(shard_dir))
    dist.barrier()  # 전 rank 쓰기 완료 후 rank0 읽기 시작

    # 2) rank0만: 디스크 shard → CPU full 모델 조립 (fabric 무관).
    # no_dist=True: load 플래너의 collective를 끄고 단일 프로세스로 읽음.
    if rank == 0:
        from transformers import AutoModelForCausalLM

        full = AutoModelForCausalLM.from_config(model_cfg, dtype=dtype)
        full.resize_token_embeddings(len(tok))
        full_sd = get_model_state_dict(
            full, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        dcp_load(full_sd, storage_reader=FileSystemReader(shard_dir), no_dist=True)
        missing, _ = set_model_state_dict(
            full, full_sd, options=StateDictOptions(full_state_dict=True)
        )
        assert not missing, f"DCP consolidate missing keys: {sorted(missing)[:5]}"
        tmp = clean_tmp(tmp_path(out_dir))
        full.save_pretrained(tmp, safe_serialization=True)
        tok.save_pretrained(tmp)
        commit_dir(tmp, out_dir)
    dist.barrier()


def main():
    a = parse_args()
    # 1차: 런치 환경 검사 (무거운 작업 전, 수 초 내 실패).
    # 분산 의도(expect-world=2)인데 WORLD_SIZE=1이면 solo 오인 성공이므로 즉시 거부.
    if a.expect_world:
        _w = int(os.environ.get("WORLD_SIZE", "1"))
        if _w != a.expect_world:
            raise SystemExit(
                f"REFUSE: WORLD_SIZE={_w} != expected {a.expect_world} "
                f"(RANK={os.environ.get('RANK', '?')}, strategy={a.strategy}). "
                f"2노드 실행이 맞나?"
            )
    set_seed(a.seed)
    texts = load_texts(a.train)
    if a.max_rows > 0:
        texts = texts[: a.max_rows]
    print(f"[{a.mode}][{a.strategy}] n={len(texts)} sample_chars={len(texts[0])}")
    print("---- sample ----")
    print(texts[0][:600])
    print("----------------")
    if a.dry_run:
        return
    # 리포팅 기동 게이트: 데이터 적재·토크나이즈 전에 실패 확정.
    # (dry_run은 torch 없는 환경용이라 게이트를 타지 않는다.)
    gate_report_to(a.report_to)

    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision or None)
    added = []
    if tok.pad_token is None:
        added.append(PAD_TOKEN)
    added.append(MASK_TOKEN)
    n_new = tok.add_special_tokens({"additional_special_tokens": added})
    if tok.pad_token is None:
        tok.pad_token = PAD_TOKEN
    tok.padding_side = "right"

    dtype = torch.bfloat16
    if a.peft:
        # Compat shim: peft<=0.19는 임포트 시 BloomPreTrainedModel을 참조하지만
        # transformers 5.x에서 삭제됨. Bloom prefix-tuning 전용이라 LoRA 경로에
        # 영향 없음 (hasattr 체크에서 False → 매핑 스킵).
        try:
            from transformers import BloomPreTrainedModel  # noqa
        except ImportError:
            import transformers as _tf

            class _BloomStub:
                pass

            _tf.BloomPreTrainedModel = _BloomStub
        if a.strategy == "fsdp":
            print(
                "WARN: FSDP+QLoRA unsupported, training as DDP "
                "(same 2-proc run, rank0 saves adapter, OUT stays *-fsdp)."
            )
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            a.model, quantization_config=bnb, dtype=dtype, revision=a.revision or None
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=dtype, revision=a.revision or None
        )

    # main.py resize_at_begin: PEFT 래핑보다 먼저 수행 (래핑 후 resize 불가).
    # 신규 토큰 임베딩을 기존 평균으로 초기화.
    model.resize_token_embeddings(len(tok))
    if n_new > 0:
        with torch.no_grad():
            for emb in {model.get_input_embeddings(), model.get_output_embeddings()}:
                w = emb.weight.data
                w[-n_new:] = w[:-n_new].mean(dim=0, keepdim=True)

    if a.peft:
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
            ],
            modules_to_save=["embed_tokens", "lm_head"],
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    ds = Dataset.from_list([{"text": t} for t in texts])

    # trl은 map 배치를 먼저 패딩한 텐서(input_ids/attention_mask/labels 포함)로
    # collator를 호출하므로, 배치는 여기서 최종 max에 맞춰 다시 패딩한다.
    # mask_rate>0일 때만 <mask> 치환을 적용한다 (기본 0=강의와 동일하게 off).
    mask_id = tok.convert_tokens_to_ids(MASK_TOKEN)
    special = set(tok.all_special_ids)

    def collator(features):
        ids = [torch.as_tensor(f["input_ids"]) for f in features]
        pad = torch.nn.utils.rnn.pad_sequence
        input_ids = pad(ids, batch_first=True, padding_value=tok.pad_token_id)
        if "attention_mask" in features[0]:
            ams = [torch.as_tensor(f["attention_mask"]) for f in features]
            attention_mask = pad(ams, batch_first=True, padding_value=0)
        else:
            attention_mask = (input_ids != tok.pad_token_id).long()
        if "labels" in features[0]:
            lbs = [torch.as_tensor(f["labels"]) for f in features]
            labels = pad(lbs, batch_first=True, padding_value=-100)
        else:
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
        if a.mask_rate > 0:
            prob = torch.rand(input_ids.shape)
            is_special = torch.zeros_like(prob, dtype=torch.bool)
            for s in special:
                is_special |= input_ids == s
            do_mask = (prob < a.mask_rate) & ~is_special & (labels != -100)
            input_ids[do_mask] = mask_id
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # fsdp=True + fsdp_config (문자열 "full_shard auto_wrap"은 deprecated).
    # full_shard는 FSDP2 기본(reshard_after_forward=True), auto_wrap은 기본 정책이라
    # transformer_layer_cls_to_wrap만 지정하면 동등하다.
    # FSDP+QLoRA는 미지원 → DDP로 동작 (peft면 FSDP 래핑·DCP 둘 다 끈다).
    # DCP consolidate는 full-model 전제라 어댑터 샤드에 깨진다
    # (Missing key lm_head.weight 실측).
    fsdp = a.strategy == "fsdp" and not a.peft
    fsdp_cfg = (
        {
            "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            # gradient_checkpointing 대신 FSDP 네이티브 (backward 冗長 AllGather 회피).
            "activation_checkpointing": True,
        }
        if fsdp
        else {}
    )
    # 2노드 FSDP는 중간 체크포인트(full gather)를 건너뛰고 최종 DCP 저장만 수행.
    # _dcp_2node에서도 M1B_DCP_SAVE=0이면 기존 경로(trainer.save_model) 사용.
    _world = int(os.environ.get("WORLD_SIZE", "1"))
    _dcp_2node = (
        fsdp
        and _world > 1
        and os.environ.get("M1B_DCP_SAVE", "1") == "1"
    )
    # warmup_ratio deprecated → warmup_steps로 환산 (transformers 내부 공식과 동일:
    # global_batch = micro*accum*world, total = ceil(N/global_batch)*epochs).
    # mini(256행/64eff/1ep)=4스텝 → round(4*0.03)=0으로 기존 동작과 일치.
    _gbs = a.micro * a.accum * _world
    _warmup_steps = round(math.ceil(len(ds) / _gbs) * a.epochs * a.warmup)
    # report_to CSV 파싱 + 기동 게이트 (report_check.py와 단일 규칙 공유).
    # 게이트는 Trainer와 동일한 해석+인스턴스화 경로로 검증하므로,
    # check_report_to.py OK면 여기도 통과한다. 무거운 작업 전 즉시 실패.
    _report_to = parse_report_to(a.report_to)
    cfg = SFTConfig(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.micro,
        gradient_accumulation_steps=a.accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=_warmup_steps,
        logging_steps=200,
        save_strategy="no" if _dcp_2node else "epoch",
        save_total_limit=2,
        bf16=True,
        seed=a.seed,
        dataset_text_field="text",
        max_length=a.max_len,
        packing=False,
        gradient_checkpointing=not fsdp,
        gradient_checkpointing_kwargs={"use_reentrant": False} if a.peft else {},
        fsdp=fsdp,
        fsdp_config=fsdp_cfg,
        report_to=_report_to,
        project="trackio-math-1b-sft-and-quantization",  # trackio 프로젝트명 (TrainingArguments 기본값 huggingface 대신)
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        data_collator=collator,
    )
    # TRL _save_checkpoint은 매번 create_model_card → trackio import를 타는데,
    # huggingface_hub>=1(CommitOperationAdd 삭제)과 충돌해 ImportError로 죽는다.
    # 모델 카드는 파이프라인 산출물이 아니므로 no-op (버전 조합과 무관하게 동작).
    trainer.create_model_card = lambda *a, **k: {}
    # 2차: 런타임 진실 검사 (dist 초기화 후 — env 속임수까지 잡는다).
    import torch.distributed as dist

    if (
        a.expect_world
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() != a.expect_world
    ):
        raise RuntimeError(
            f"REFUSE: dist world={dist.get_world_size()} != expected {a.expect_world}"
        )
    trainer.train()
    if _dcp_2node:
        _save_fsdp_dcp(trainer.model, model.config, tok, a.out, dtype)
    else:
        # 분산 저장 레이스 방지: 공유FS의 <out>.tmp에 양 rank가 동시 쓰기+rmtree하면
        # ENOTEMPTY 등으로 한쪽이 죽는다 (DCP 경로는 이미 rank0-only). 저장은 rank0만,
        # 나머지는 barrier로 대기 후 함께 종료 (elastic agent 조기 종료 오판 방지).
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            _rank = dist.get_rank()
        else:
            _rank = 0
        if _rank == 0:
            tmp = clean_tmp(tmp_path(a.out))
            trainer.save_model(tmp)
            # DDP 래핑 시 trainer.model이 PreTrainedModel이 아니라 save_model이
            # config.json을 생략한다 (가중치는 저장됨) → .module 언랩해 직접 보완.
            # 없으면 vLLM 로드·m1b_fresh가 깨진다.
            _unwrapped = getattr(trainer.model, "module", trainer.model)
            if _unwrapped is not trainer.model:
                _unwrapped.config.save_pretrained(tmp)
            tok.save_pretrained(tmp)
            commit_dir(tmp, a.out)
            print(f"saved -> {a.out}")
        else:
            print(f"rank {_rank} done (model saved by rank 0)")
        if dist.is_available() and dist.is_initialized():
            import torch

            # device_ids 명시 (없으면 "using the device under current context" 경고).
            if torch.cuda.is_available():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()


if __name__ == "__main__":
    main()
