# raon-1-mini-preview-29b-quantization — KRAFTON Raon-1-mini PTQ

Post-training quantization of [`KRAFTON/Raon-1-mini-preview`](https://huggingface.co/KRAFTON/Raon-1-mini-preview)
(28.8B MoE, 4.3B active) — no fine-tuning, calibration only. The base ships in MXFP8, so it is
dequantized to BF16 once, then quantized.

## Status: only one checkpoint is published

**Published:** [`tayaee/Raon-1-mini-preview-NVFP4-AWQ-GB10-vLLM`](https://huggingface.co/tayaee/Raon-1-mini-preview-NVFP4-AWQ-GB10-vLLM)
(`modelopt_fp4` / NVFP4, 19.25 GB, served with the GB10 vLLM fork).

**Not published:** the four `transformers`-hosted builds. Served in-process they are *slower than
the unquantized base* (1.7 tok/s vs 4.5 tok/s at C=1) and ~7× slower than the fork-served
checkpoint (11.5 tok/s) — quantizing them buys users nothing, so they are not shipped. They remain
here as experiment artifacts.

> The published checkpoint **currently fails its quality gate** (GSM8K 0.170 vs base 0.945,
> KMMLU 0.300 vs base 0.565). Its model card carries a warning. Cause is narrowed down to an
> export↔loader mismatch, not the quantization math — see *Known issue* below.

## The five planned targets

| # | Target | Format | Published | Note |
|---|---|---|---|---|
| 1 | `awq-w3-transformers` | `pack-quantized` (W3A16-g128) | no | quality-collapse risk; not built here |
| 2 | `awq-w5-transformers` | `pack-quantized` (W5A16-g128) | no | non-standard bit width |
| 3 | `nvfp4-awq-transformers` | `nvfp4-pack-quantized` | no | built and measured — too slow to ship |
| 4 | `nvfp4-gptq-transformers` | `nvfp4-pack-quantized` | no | not built here |
| 5 | `nvfp4-awq-vllm` | `modelopt_fp4` (NVFP4, AWQ Lite) | **yes** | vLLM fork only, GB10 (sm_121) only |

All five quantize the MoE experts only (router, attention, shared experts, LM head stay BF16).
The `transformers` builds are usable on any GPU; the vLLM build needs the GB10 fork
([`tayaee/raon-vllm-spark`](https://github.com/tayaee/raon-vllm-spark/tree/gb10-support)).

## SDLC

8 groups, each gated before the next one starts.

| # | Group | Artifact |
|---|---|---|
| 0 | Base verification | `outputs/s0-<mode>.json` |
| 1 | Environment (setup · acceptance · calib · compat) | `outputs/acceptance-<mode>.json` · `outputs/compat.stamp` |
| 2 | Base dequantization (MXFP8 → BF16) | `models/raon-bf16` |
| 3 | Quantization | `models/raon-<target>-<mode>` |
| 4 | Measurement · card · audit · release notes | `outputs/eval-<mode>/<target>.jsonl` · `score.json` · `checks.json` · `outputs/release-notes-<mode>.md` |
| 5 | Upload | Hub repo |
| 6 | Customer acceptance (transformers) | `outputs/s6-<mode>/` |
| 7 | Customer acceptance (vLLM) | `outputs/serve-<mode>.json` |

## Running it

Artifacts land in `$RQ_SHARED`
(`/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared`). Targets are always explicit
(`TARGETS=all` for everything).

```bash
# shared steps (once)
./s1-00-setup-environment.sh                      # env + HF login (both nodes)
./s1-05-setup-acceptance-criteria.sh smoke        # validate the acceptance contract
./s2-00-dequant-mxfp8-into-bf16.sh                # MXFP8 → BF16 base

# one-shot build + measure per target (mode × target = 10 wrappers; smoke = same with "smoke")
./run-s1-to-s4-full-awq-w3-transformers.sh
./run-s1-to-s4-full-awq-w5-transformers.sh
./run-s1-to-s4-full-nvfp4-awq-transformers.sh
./run-s1-to-s4-full-nvfp4-gptq-transformers.sh
./run-s1-to-s4-full-nvfp4-awq-vllm.sh            # GB10 (sm_121) only — needs the vLLM fork

# measure the vLLM-only target through the fork, then publish
./run-s4-metrics-via-serving.sh full nvfp4-awq-vllm   # GB10 (sm_121) only
./s5-00-upload-to-hf.sh full nvfp4-awq-vllm
```

## Known issue (published checkpoint fails its gate)

| Test | Result | Means |
|---|---|---|
| Official KRAFTON NVFP4, same fork + flags + harness | GSM8K **0.940** | serving stack is healthy |
| Our expert weights vs the BF16 base | cosine **0.9936** | PTQ math is healthy |
| Fork logs | official = FP8/NVFP4/W4A16/MXFP8 mix (`MIXED_PRECISION`); ours = experts-only `NVFP4` | export↔loader mismatch |

The fork's `modelopt_fp4` path is built for KRAFTON's mixed-precision layout; our checkpoint keeps
attention, shared experts and the dense layer 0 in BF16 and is silently mis-loaded. Next step:
align `config.json` with the official shape, otherwise rebuild with the official scope
(`RQ_AWQ_FIRST=0` for pure NVFP4).

## More

- `PROCEDURE.md` — stage-by-stage procedure, gates, parallel-run rules
- `cheatsheet.md` — command cookbook
- `docs/SDLC.md` — group definitions, gates, evidence
