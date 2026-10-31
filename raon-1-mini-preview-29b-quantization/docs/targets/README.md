# Target-specific READMEs

This repository completes five quantized models **one at a time**, rather than building all
of them concurrently. Select one target from the table below and follow only that target's
README. Run every command from the repository root.

| # | Target | Runtime / format | Status | Detailed procedure |
|---:|---|---|---|---|
| 1 | [`awq-w3-transformers`](awq-w3-transformers/README.md) | Transformers / `pack-quantized` | 📋 Planned | [README](awq-w3-transformers/README.md) |
| 2 | [`awq-w5-transformers`](awq-w5-transformers/README.md) | Transformers / `pack-quantized` | 📋 Planned | [README](awq-w5-transformers/README.md) |
| 3 | [`nvfp4-awq-transformers`](nvfp4-awq-transformers/README.md) | Transformers / `nvfp4-pack-quantized` | 🚧 In progress | [README](nvfp4-awq-transformers/README.md) |
| 4 | [`nvfp4-gptq-transformers`](nvfp4-gptq-transformers/README.md) | Transformers / `nvfp4-pack-quantized` | 📋 Planned | [README](nvfp4-gptq-transformers/README.md) |
| 5 | [`nvfp4-awq-vllm`](nvfp4-awq-vllm/README.md) | GB10 vLLM fork / `modelopt_fp4` | ✅ Published | [README](nvfp4-awq-vllm/README.md) |

## Shared principles

1. Run `smoke` first to validate the environment, compression format, and PPL/evaluation
   gates.
2. Run `full` only for a target whose smoke gates passed.
3. Upload only after every full-build gate for that target has passed.
4. Run the examples from the published model card verbatim to complete customer acceptance.
5. Finish one target's procedure and evidence before starting the next one.

For the shared SDLC and gate definitions, see the root
[`PROCEDURE.md`](../../PROCEDURE.md), [`docs/SDLC.md`](../SDLC.md), and
[`config/acceptance.toml`](../../config/acceptance.toml).
