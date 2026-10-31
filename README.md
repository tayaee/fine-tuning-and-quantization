# Fine-tuning and Quantization Experiments

This repository studies how fine-tuning and post-training quantization affect model
quality, and publishes the resulting artifacts for reproducible evaluation.

## Goals

1. Learn and validate an end-to-end quantization pipeline using Llama 3.2.
2. Quantify the model-quality loss introduced by quantization.
3. Quantify how FFT and QLoRA change model quality relative to the base model.

The project builds the corresponding model combinations, publishes them to the Hugging
Face Hub, and validates the published checkpoints against the baseline.

## Workstreams

### Llama 3.2 1B math models

[`llama-3.2-1b-to-math-model`](llama-3.2-1b-to-math-model/) explores synthetic-data SFT,
FFT versus QLoRA, and subsequent AWQ, FP8, GPTQ, and GGUF quantization. See its
[README](llama-3.2-1b-to-math-model/README.md) and
[procedure](llama-3.2-1b-to-math-model/PROCEDURE.md).

### Raon-1-mini-preview 29B quantization

[`raon-1-mini-preview-29b-quantization`](raon-1-mini-preview-29b-quantization/) targets
KRAFTON's 29B
[`Raon-1-mini-preview`](https://huggingface.co/KRAFTON/Raon-1-mini-preview), released in
October 2026. Its goal is to build 4-bit quantized variants and measure their performance
against the base model. See its [README](raon-1-mini-preview-29b-quantization/README.md)
and [target procedures](raon-1-mini-preview-29b-quantization/docs/targets/README.md).

## Published models

The public model repositories below were verified through the Hugging Face API on
2026-10-31. Only repositories currently present on the Hub and attributable to this
repository are listed; local-only artifacts are omitted.

### Llama 3.2 — FFT lineage

- [`Llama-3.2-1B-math-fft`](https://huggingface.co/tayaee/Llama-3.2-1B-math-fft) — FFT model
- [`Llama-3.2-1B-math-fft-awq`](https://huggingface.co/tayaee/Llama-3.2-1B-math-fft-awq) — AWQ quantized
- [`Llama-3.2-1B-math-fft-fp8`](https://huggingface.co/tayaee/Llama-3.2-1B-math-fft-fp8) — FP8 quantized
- [`Llama-3.2-1B-math-fft-gptq`](https://huggingface.co/tayaee/Llama-3.2-1B-math-fft-gptq) — GPTQ quantized
- [`Llama-3.2-1B-math-fft-gguf`](https://huggingface.co/tayaee/Llama-3.2-1B-math-fft-gguf) — GGUF Q8_0
- [`Llama-3.2-1B-math-fft-gguf-q3_k_m`](https://huggingface.co/tayaee/Llama-3.2-1B-math-fft-gguf-q3_k_m) — GGUF Q3_K_M
- [`Llama-3.2-1B-math-fft-gguf-q4_k_m`](https://huggingface.co/tayaee/Llama-3.2-1B-math-fft-gguf-q4_k_m) — GGUF Q4_K_M
- [`Llama-3.2-1B-math-fft-gguf-q5_k_m`](https://huggingface.co/tayaee/Llama-3.2-1B-math-fft-gguf-q5_k_m) — GGUF Q5_K_M
- [`Llama-3.2-1B-math-fft-gguf-q6_k`](https://huggingface.co/tayaee/Llama-3.2-1B-math-fft-gguf-q6_k) — GGUF Q6_K
- [`Llama-3.2-1B-math-fft-gguf-q8_0`](https://huggingface.co/tayaee/Llama-3.2-1B-math-fft-gguf-q8_0) — GGUF Q8_0

### Llama 3.2 — QLoRA lineage

- [`Llama-3.2-1B-math-qlora`](https://huggingface.co/tayaee/Llama-3.2-1B-math-qlora) — merged QLoRA model
- [`Llama-3.2-1B-math-qlora-awq`](https://huggingface.co/tayaee/Llama-3.2-1B-math-qlora-awq) — AWQ quantized
- [`Llama-3.2-1B-math-qlora-fp8`](https://huggingface.co/tayaee/Llama-3.2-1B-math-qlora-fp8) — FP8 quantized
- [`Llama-3.2-1B-math-qlora-gptq`](https://huggingface.co/tayaee/Llama-3.2-1B-math-qlora-gptq) — GPTQ quantized
- [`Llama-3.2-1B-math-qlora-gguf`](https://huggingface.co/tayaee/Llama-3.2-1B-math-qlora-gguf) — GGUF Q8_0
- [`Llama-3.2-1B-math-qlora-gguf-q4_k_m`](https://huggingface.co/tayaee/Llama-3.2-1B-math-qlora-gguf-q4_k_m) — GGUF Q4_K_M
- [`Llama-3.2-1B-math-qlora-gguf-q5_k_m`](https://huggingface.co/tayaee/Llama-3.2-1B-math-qlora-gguf-q5_k_m) — GGUF Q5_K_M
- [`Llama-3.2-1B-math-qlora-gguf-q6_k`](https://huggingface.co/tayaee/Llama-3.2-1B-math-qlora-gguf-q6_k) — GGUF Q6_K
- [`Llama-3.2-1B-math-qlora-gguf-q8_0`](https://huggingface.co/tayaee/Llama-3.2-1B-math-qlora-gguf-q8_0) — GGUF Q8_0

### Earlier Llama mini artifact

- [`p5-1B-math-fft-mini`](https://huggingface.co/tayaee/p5-1B-math-fft-mini) — early mini FFT checkpoint

### Raon-1-mini-preview

- [`Raon-1-mini-preview-NVFP4-AWQ-GB10-vLLM`](https://huggingface.co/tayaee/Raon-1-mini-preview-NVFP4-AWQ-GB10-vLLM) — NVFP4/AWQ ModelOpt checkpoint for the GB10 vLLM fork; published for experimentation because the current checkpoint does not pass the quality gates
