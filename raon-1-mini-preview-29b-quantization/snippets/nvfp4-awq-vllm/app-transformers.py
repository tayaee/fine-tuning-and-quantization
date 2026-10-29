# app.py - copied from the base model
import subprocess
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "__REPO_ID__"

# This checkpoint is stored in the ModelOpt `modelopt_fp4` format (NVFP4-packed weights
# plus ModelOpt scales). Transformers has no `modelopt_fp4` quantizer, so it cannot read
# the checkpoint directly. Convert it to BF16 once with the converter shipped in this
# repository, then load the converted copy. (The vLLM fork reads `modelopt_fp4`
# natively — see "Usage - vLLM" if you just want to serve it.)
local_dir = "./bf16"
subprocess.run([sys.executable, "nvfp4_to_bf16.py", model_id, "--out", local_dir], check=True)

tokenizer = AutoTokenizer.from_pretrained(
    local_dir,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {"role": "user", "content": "Tell me about KRAFTON AI."}
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=True,
    truncate_history_thinking=False,
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=32768,
    do_sample=True,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    repetition_penalty=1.0,
    presence_penalty=1.5,
)

# The reasoning ends at </think>; the answer follows it.
response = outputs[0][inputs["input_ids"].shape[-1]:].tolist()
think_end = tokenizer.convert_tokens_to_ids("</think>")
split = response.index(think_end) + 1 if think_end in response else 0
reasoning = tokenizer.decode(response[:split], skip_special_tokens=True).strip()
answer = tokenizer.decode(response[split:], skip_special_tokens=True).strip()
print("Reasoning:\n" + reasoning)
print("\nAnswer:\n" + answer)
