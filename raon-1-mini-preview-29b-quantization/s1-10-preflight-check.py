"""s1-10-preflight-check.py — 양자화 전제조건 점검 (가중치 변경 없음).
- trust_remote_code 로드 + 아키텍처/MoE 구조 출력
- Linear 타깃 수, 제외층(gate/router/lm_head) 존재 확인
- GPU compute capability (NVFP4는 SM>=10.0)
- llm-compressor / vLLM 버전

  uv run s1-10-preflight-check.py
"""

import os

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")
BASE_MODEL = os.environ.get("BASE_MODEL", "KRAFTON/Raon-1-mini-preview")
BASE_REVISION = os.environ.get("BASE_REVISION") or None


def main():
    import torch
    from transformers import AutoConfig, AutoTokenizer

    print(f"base={BASE_MODEL} rev={BASE_REVISION or 'floating (고정 요망)'}")
    cfg = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True, revision=BASE_REVISION)
    print(f"arch={cfg.architectures} model_type={getattr(cfg, 'model_type', '?')}")
    for k in ("num_hidden_layers", "hidden_size", "num_experts", "num_activated_experts",
              "vocab_size", "max_position_embeddings"):
        if hasattr(cfg, k):
            print(f"  {k}={getattr(cfg, k)}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, revision=BASE_REVISION)
    print(f"tokenizer_vocab={len(tok)} chat_template={'yes' if getattr(tok, 'chat_template', None) else 'NO'}")

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=True, revision=BASE_REVISION,
        dtype=torch.bfloat16, device_map="cpu",
    )
    linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    print(f"linear_modules={len(linears)}")
    for pat in ("gate", "router", "lm_head", "expert"):
        hit = [n for n in model.named_modules() if pat in n[0].lower()]
        print(f"  ~{pat}: {len(hit)} (예: {[n for n, _ in hit[:3]]})")
    del model

    print(f"cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        print(f"  gpu{i}={torch.cuda.get_device_name(i)} sm={cap[0]}.{cap[1]} nvfp4_ok={cap >= (10, 0)}")

    import llmcompressor
    import vllm
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier

    print(f"llmcompressor={llmcompressor.__version__} vllm={vllm.__version__}")
    print(f"modifiers_ok={AWQModifier.__name__},{GPTQModifier.__name__},{QuantizationModifier.__name__}")
    print("COMPAT-OK")


if __name__ == "__main__":
    main()
