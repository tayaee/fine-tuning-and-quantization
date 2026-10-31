# raon-1-mini-preview-29b-quantization — KRAFTON Raon-1-mini PTQ

Post-training quantization of
[`KRAFTON/Raon-1-mini-preview`](https://huggingface.co/KRAFTON/Raon-1-mini-preview)
(28.8B MoE, 4.3B active). This project uses calibration only; no fine-tuning is performed.
The original weights are stored in MXFP8, so they are restored to BF16 once and then
quantized into five targets.

## Current status (2026-10-31)

The project plans to produce **five quantized models**, each built, evaluated, and released
independently. One vLLM target is published, one Transformers target is in progress, and
three Transformers targets are planned.

| # | Quantized model | Runtime / storage format | Status | Detailed procedure |
|---:|---|---|---|---|
| 1 | `awq-w3-transformers` | Transformers / `pack-quantized` | 📋 **Planned** | [Build, validate, and deploy](docs/targets/awq-w3-transformers/README.md) |
| 2 | `awq-w5-transformers` | Transformers / `pack-quantized` | 📋 **Planned** | [Build, validate, and deploy](docs/targets/awq-w5-transformers/README.md) |
| 3 | `nvfp4-awq-transformers` | Transformers / `nvfp4-pack-quantized` | 🚧 **In progress** | [Build, validate, and deploy](docs/targets/nvfp4-awq-transformers/README.md) |
| 4 | `nvfp4-gptq-transformers` | Transformers / `nvfp4-pack-quantized` | 📋 **Planned** | [Build, validate, and deploy](docs/targets/nvfp4-gptq-transformers/README.md) |
| 5 | `nvfp4-awq-vllm` | GB10 vLLM fork / `modelopt_fp4` | ✅ **Published** | [Build, validate, and deploy](docs/targets/nvfp4-awq-vllm/README.md) |

Published vLLM model:
[`tayaee/Raon-1-mini-preview-NVFP4-AWQ-GB10-vLLM`](https://huggingface.co/tayaee/Raon-1-mini-preview-NVFP4-AWQ-GB10-vLLM)

> **Planned** means that the target has not started yet. **In progress** means that work
> has started, but final quality and deployment validation may not be complete. See each
> target's **sub-README** for its exact commands, gates, and outputs. This root README
> documents the shared workflow for completing one target at a time instead of attempting
> all five at once.

## How the five models are built

All targets follow the same scope:

1. Restore the original MXFP8 model to BF16 (`raon-bf16`).
2. Quantize **only the MoE expert weights** with Korean calibration data. The router,
   attention, shared experts, dense layer 0, and LM head remain in BF16.
3. Validate the compression format and global scales (`gate-quant`).
4. Measure perplexity, GSM8K, KMMLU, instruction-format compliance, and serving throughput.
5. Generate the model card and pre-upload audit, publish to the Hugging Face Hub, and run
   the published card's examples verbatim as the customer acceptance test.

The recommended order is **smoke → full → upload → customer acceptance**. Target-specific
wrappers reuse common artifacts through freshness checks and locks, so preparing another
target does not require repeating completed setup work. There is no need to build all five
targets concurrently.

```bash
# Run every command from the repository root.
./run-s1-to-s4-smoke-<target>.sh     # Validate the pipeline and gates
./run-s1-to-s4-full-<target>.sh      # Build the full artifact
./s5-00-upload-to-hf.sh full <target>
```

See the [target README index](docs/targets/README.md) for the exact wrapper, manual steps,
and acceptance criteria of each target.

## SDLC

The pipeline has eight groups. Each group must pass its gates before the next group starts.

| # | Group | Outputs |
|---|---|---|
| 0 | Base verification | `outputs/s0-<mode>.json` |
| 1 | Environment (setup · acceptance · calibration · compatibility) | `outputs/acceptance-<mode>.json` · `outputs/compat.stamp` |
| 2 | Base dequantization (MXFP8 → BF16) | `models/raon-bf16` |
| 3 | Quantization | `models/raon-<target>-<mode>` |
| 4 | Measurement · model card · audit · release notes | `outputs/eval-<mode>/<target>.jsonl` · `score.json` · `checks.json` · `outputs/release-notes-<mode>.md` |
| 5 | Upload | Hugging Face Hub repository |
| 6 | Customer acceptance (Transformers) | `outputs/s6-<mode>/` |
| 7 | Customer acceptance (vLLM) | `outputs/serve-<mode>.json` |

Common artifacts are stored under `$RQ_SHARED` by default:
`/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared`.

## Known issue in the published vLLM model

> [!WARNING]
> `nvfp4-awq-vllm` is published, but the current checkpoint does not pass the quality gates.
> Its GSM8K accuracy is `0.170` versus `0.945` for the base model, and its KMMLU accuracy
> is `0.300` versus `0.565`. The issue has been narrowed down to an export/loader format
> mismatch rather than the quantization math. See
> [Known issue and revalidation procedure](docs/targets/nvfp4-awq-vllm/README.md#known-issue-and-revalidation)
> for details.

## Further reading

- [Target-specific READMEs](docs/targets/README.md) — complete procedures for building each
  of the five models
- [`PROCEDURE.md`](PROCEDURE.md) — shared stages, gates, and evidence rules
- [`cheatsheet.md`](cheatsheet.md) — command cookbook
- [`docs/SDLC.md`](docs/SDLC.md) — detailed group definitions and gates
