# `nvfp4-awq-vllm` timing log

[← Target README](README.md) · [← All targets](../README.md)

## Scope and interpretation

- The measurements below come from wall-clock timing logs collected on the DGX Spark 1x
  (NVIDIA GB10, `sm_121`, one GPU, TP=1) build host.
- Values are rounded to the nearest minute. Raw elapsed seconds are retained for
  traceability.
- Only successful work (`rc=0`) is used for the planning figures. Freshness-cache hits
  that completed in zero or one second and failed experiments are excluded.
- The quantization logs predate the current script names: legacy
  `s3-06-quant-nvfp4-awq-modelopt` and `s3-01-quant-nvfp4-awq-vllm` both map to the
  current `nvfp4-awq-vllm` target.
- The raw source logs are local build artifacts under `logs/` and are excluded by
  `.gitignore`; the relevant entries are quoted below.

## Longest-running observed work

| Work | Scope | Raw elapsed | Approximate time |
|---|---|---:|---:|
| ModelOpt NVFP4 quantization | smoke | 1,194–1,583 s | **20–26 min** |
| Serving-metrics invocation (reached quality gate) | full | about 585 s wall-clock | **about 10 min** |
| MXFP8 → BF16 restoration | shared | 958–1,070 s | **16–18 min** |
| vLLM endpoint startup | full | 190 s | **about 3 min** |
| vLLM model loading | full/smoke | 125–236 s | **2–4 min** |
| vLLM engine initialization | full/smoke | 6–106 s | **under 2 min** |
| Compatibility probe | shared | 112–178 s | **2–3 min** |
| Calibration preparation | shared | 34–37 s | **about 1 min** |

The serving-metrics invocation includes roughly 3 minutes of server startup, 410
server-generated benchmark responses, servebench, aggregation, and the quality-gate
check. It stopped at the failed GSM8K gate before model-card/audit work. The duration is
estimated from the serving log's start time and the final output timestamp rather than a
separately instrumented timer.

## Shared preparation

The following work is shared by all targets and should normally be completed once:

| Stage | Observed duration | Notes |
|---|---:|---|
| Environment setup | 0–5 s after dependencies were cached | The first dependency installation is not represented by the cached timing |
| Calibration generation | 34–37 s | Small calibration set |
| Compatibility probe | 112–178 s | GPU and library checks |
| MXFP8 → BF16 restoration | 958–1,070 s | One-time 28.8B checkpoint conversion; later runs skip it through freshness checks |

## Evidence

- `logs/timing-build-smoke.log`:
  `s3-06-quant-nvfp4-awq-modelopt / 1194 s / rc=0` and
  `s3-01-quant-nvfp4-awq-vllm / 1583 s / rc=0`
- `logs/metrics-via-serving-full-20261030-074411.log`:
  `Endpoint ready ... (190s)` and the complete full serving-metrics invocation
- `logs/s4-45-serve-vllm-fork.log`:
  `Model loading took ... 137.338783 seconds` and engine initialization took `17.69 s`
- `logs/s7-13-serve-tayaee__raon-1-mini-preview-nvfp4-awq-vllm.log`:
  full-model loading took `124.619931 seconds` and engine initialization took `6.49 s`
- `logs/s7-13-serve-tayaee__raon-1-mini-preview-nvfp4-awq-vllm-smoke.log`:
  smoke-model loading took `236.449737 seconds` and engine initialization took `105.62 s`
- `logs/phase-a.log` and `logs/timing-full.log`: shared restoration timings

## Planning caveats

- The successful quantization timing is from the smoke profile. The collected logs do
  not contain a separately timed successful full ModelOpt quantization, so no full
  quantization estimate is inferred from the smoke value.
- The GB10 vLLM fork source build was not wrapped by a durable timing logger. Its
  duration is therefore not included; measure a clean `FORCE=1` build separately.
- Upload time was not captured.
- Server startup varies with cache state and whether the vLLM kernel cache is warm. The
  observed range is approximately 3–6 minutes after the vLLM process starts.
