# `nvfp4-awq-vllm` — NVFP4 + AWQ Lite vLLM model

[← All targets](../README.md) · [← Root README](../../../README.md)

## Status

✅ **Published** — Hugging Face model:
[`tayaee/Raon-1-mini-preview-NVFP4-AWQ-GB10-vLLM`](https://huggingface.co/tayaee/Raon-1-mini-preview-NVFP4-AWQ-GB10-vLLM)

Observed timing evidence: [wall-clock durations from collected logs](timing-log.md).

However, **publication and quality-gate completion are separate states**. The current
published checkpoint has a known quality issue that must be resolved through the
[revalidation procedure](#known-issue-and-revalidation) below.

## Goal

Quantize the MoE expert weights of `KRAFTON/Raon-1-mini-preview` with ModelOpt NVFP4 and
produce a `modelopt_fp4` checkpoint that can be served by the GB10-specific vLLM fork.

| Property | Value |
|---|---|
| Target ID | `nvfp4-awq-vllm` |
| Runtime | vLLM fork for NVIDIA GB10 (`sm_121`) |
| Storage format | `modelopt_fp4` |
| Weight recipe | ModelOpt `NVFP4_EXPERTS_ONLY` + `awq_lite`, with a pure-NVFP4 fallback |
| Quantization scope | MoE experts only; router, attention, shared experts, and LM head remain in BF16 |
| Full calibration | 256 samples, sequence length 2048 |
| Checkpoint size | 15.0–28.0 GB (current published artifact: 19.25 GB) |
| Published model | [`tayaee/Raon-1-mini-preview-NVFP4-AWQ-GB10-vLLM`](https://huggingface.co/tayaee/Raon-1-mini-preview-NVFP4-AWQ-GB10-vLLM) |

`transformers` cannot load this format directly. Transformers-only metrics such as PPL
are therefore omitted. GSM8K, KMMLU, format compliance, and throughput are measured on
the actual vLLM serving stack.

## Prerequisites

- A GB10 (`sm_121`) GPU and a CUDA 13-class environment are required.
- vLLM must be built from the `gb10-support` branch of
  [`tayaee/raon-vllm-spark`](https://github.com/tayaee/raon-vllm-spark/tree/gb10-support).
- The base model is gated, so Hugging Face authentication is required.
- `$RQ_SHARED` must have enough space for the BF16 base and the quantized output.

## Recommended procedure: smoke → full → serving metrics → upload → customer acceptance

### 1. Prepare the vLLM fork

First, perform the same source build used by the model card. The exact commands are in
[`snippets/nvfp4-awq-vllm/build-vllm-gb10.sh`](../../../snippets/nvfp4-awq-vllm/build-vllm-gb10.sh).
The key condition is:

```bash
MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="12.1a" \
  uv pip install --python "$VENV/bin/python" --no-build-isolation \
  "vllm @ git+https://github.com/tayaee/raon-vllm-spark.git@gb10-support"
```

### 2. Run the smoke pipeline

```bash
./run-s1-to-s4-smoke-nvfp4-awq-vllm.sh
```

The wrapper performs environment, acceptance, calibration, and compatibility checks;
restores the MXFP8 base to BF16; quantizes with ModelOpt NVFP4; validates the
`modelopt_fp4` contract; and then serves the model with the vLLM fork to measure quality
and throughput. It also regenerates the model card and audit files.

### 3. Run the full pipeline

```bash
./run-s1-to-s4-full-nvfp4-awq-vllm.sh
```

The full build uses 256 calibration samples and sequence length 2048. Common artifacts are
reused through freshness stamps.

### 4. Re-measure quality metrics on the serving stack

Because `modelopt_fp4` cannot be loaded by Transformers, re-measure its quality metrics
on the actual serving stack:

```bash
./run-s4-metrics-via-serving.sh full nvfp4-awq-vllm
```

The wrapper runs these stages in order:

1. `s4-45-serve-local-weight-with-vllm-fork.sh` — start the server with the GB10 source
   build
2. `s4-50-answer-via-server.sh` — generate evaluation responses through the OpenAI-compatible
   API
3. `s4-55-servebench__vllm-only.sh` — measure throughput
4. `s4-60-stop-serve-local-weight.sh` — stop the server
5. Aggregate GSM8K, KMMLU, and format results and run their gates
6. `s4-70-update-readme.sh` — regenerate the model card
7. `s4-75-verify-upload.sh` — perform the pre-upload audit

To repeat an existing server benchmark, set `FORCE=1`:

```bash
FORCE=1 ./run-s4-metrics-via-serving.sh full nvfp4-awq-vllm
```

### 5. Upload or republish to the Hub

```bash
./s5-00-upload-to-hf.sh full nvfp4-awq-vllm
```

### 6. Run vLLM customer acceptance

```bash
./s7-00-clean-serving-workspace.sh full
./s7-20-verify-readme-with-nvfp4-awq-vllm-and-vllm-fork.sh full
```

`s7-20` compares the Hub README's `Usage - vLLM` block byte for byte and then runs the
fork build and server startup in the order shown in the card. Continue with the chat and
tool-call checks:

```bash
./s7-30-chat.sh full nvfp4-awq-vllm
./s7-35-pi-tool-call-test.sh full nvfp4-awq-vllm
```

## Acceptance criteria

The single source of truth is
[`config/acceptance.toml`](../../../config/acceptance.toml).

| Check | Criterion |
|---|---|
| Compression format | `modelopt_fp4` |
| Checkpoint size | 15.0–28.0 GB |
| GSM8K | Accuracy drop from the base `<= 0.02` |
| KMMLU | Accuracy drop from the base `<= 0.03` |
| Instruction and format compliance | Pass rate `>= 0.90` |
| Release | Model card and pre-upload audit pass |
| Serving | Successful vLLM endpoint response and tool-call round trip |

## Outputs

- Checkpoint: `$RQ_SHARED/models/raon-nvfp4-awq-vllm-full`
- Evaluation: `$RQ_SHARED/outputs/eval-full/nvfp4-awq-vllm.jsonl`
- Score summary: `$RQ_SHARED/outputs/eval-full/score.json`
- Throughput: `$RQ_SHARED/outputs/eval-full/servebench-full.json`
- Upload preparation and audit: `$RQ_SHARED/outputs/prep-upload-full.json`,
  `$RQ_SHARED/outputs/verify-upload-full.json`
- Customer acceptance: `$RQ_SHARED/outputs/s7-full/`

## Known issue and revalidation

The current published checkpoint has the following quality measurements:

| Test | Base | Published vLLM checkpoint | Verdict |
|---|---:|---:|---|
| GSM8K | 0.945 | 0.170 | FAIL |
| KMMLU | 0.565 | 0.300 | FAIL |
| Instruction and format compliance | 0.900 | 0.900 | PASS |

The cause has been narrowed down to an **export/loader format mismatch**, not the
quantization math. The official KRAFTON NVFP4 checkpoint scored 0.940 on GSM8K with the
same fork, flags, and harness, and the cosine similarity between our expert weights and
the BF16 base was 0.9936. The fork logs show that the official checkpoint uses a mixed
`MIXED_PRECISION` layout containing FP8, NVFP4, W4A16, and MXFP8 tensors. Our checkpoint
instead keeps attention, shared experts, and dense layer 0 in BF16 in an experts-only
layout, which the loader interprets differently.

Revalidate in this order:

1. Compare the `quantization_config` in `config.json` field by field with the official
   KRAFTON NVFP4 checkpoint.
2. Compare the exported and loaded expert tensor names, scale dtypes, global scales, and
   dense-layer scope.
3. Rebuild the checkpoint with the corrected format.
4. Re-measure GSM8K and KMMLU with
   `./run-s4-metrics-via-serving.sh full nvfp4-awq-vllm`.
5. Republish with `./s5-00-upload-to-hf.sh full nvfp4-awq-vllm` only after both quality
   gates are within the allowed drop from the base.

The **Published** status means that the Hub artifact exists; it does not mean that the
release has passed every quality gate. Do not approve a release based on publication
status alone while this issue remains unresolved.
