# `nvfp4-gptq-transformers` timing log

[← Target README](README.md) · [← All targets](../README.md)

## Scope and interpretation

- The measurements below come from wall-clock timing logs collected on the DGX Spark 1x
  (NVIDIA GB10, one GPU, TP=1) build host.
- Values are rounded to the nearest minute. Raw elapsed seconds are retained for
  traceability.
- Only successful work (`rc=0`) is used for the planning figures. Freshness-cache hits
  that completed in zero or one second and failed experiments are excluded.
- The log predates the current script names: legacy `36-nvfp4-gptq` maps to the current
  `nvfp4-gptq-transformers` target.
- The raw source logs are local build artifacts under `logs/` and are excluded by
  `.gitignore`; the relevant entries are quoted below.

## Longest-running observed work

| Work | Mode | Raw elapsed | Approximate time |
|---|---|---:|---:|
| Target quantization | smoke | 9,391 s | **157 min** (about 2 h 37 min) |
| MXFP8 → BF16 restoration | shared | 958–1,070 s | **16–18 min** |
| Compatibility probe | shared | 112–178 s | **2–3 min** |
| Calibration preparation | shared | 34–37 s | **about 1 min** |

A fresh smoke build took at least about **173 minutes (2 h 53 min)** for restoration plus
quantization. There is no successful GPTQ-NVFP4 full-quantization record in the collected
logs, so the smoke duration is **not extrapolated into a full-build estimate**.

## Shared preparation

The following work is shared by all targets and should normally be completed once:

| Stage | Observed duration | Notes |
|---|---:|---|
| Environment setup | 0–5 s after dependencies were cached | The first dependency installation is not represented by the cached timing |
| Calibration generation | 34–37 s | Small calibration set |
| Compatibility probe | 112–178 s | GPU and library checks |
| MXFP8 → BF16 restoration | 958–1,070 s | One-time 28.8B checkpoint conversion; later runs skip it through freshness checks |

## Evidence

- `logs/timing-smoke.log`:
  `smoke / 36-nvfp4-gptq / 9391 s / rc=0`, successful compatibility probes, and successful
  dequantization runs
- `logs/run-smoke-console.log`:
  `step done: 36-nvfp4-gptq elapsed=9391s rc=0`
- `logs/probe-36-nvfp4-gptq.log` records the earlier MXFP8-input failure; it is excluded
  from the successful-run estimate

## Planning caveats

- Only a successful smoke quantization was recorded. Full GPTQ-NVFP4 uses a larger
  calibration set and longer sequence length, so its runtime must be measured separately.
- No reliable GPTQ-NVFP4-specific PPL, benchmark, upload, or customer-acceptance timing
  was found. Those stages must be added after a timed end-to-end run.
- The 157-minute figure is an observed successful run, not a guaranteed SLA.
