# `awq-w3-transformers` timing log

[← Target README](README.md) · [← All targets](../README.md)

## Scope and interpretation

- The measurements below come from wall-clock timing logs collected on the DGX Spark 1x
  (NVIDIA GB10, one GPU, TP=1) build host.
- Values are rounded to the nearest minute. Raw elapsed seconds are retained for
  traceability.
- Only successful work (`rc=0`) is used for the planning figures. Freshness-cache hits
  that completed in zero or one second and failed experiments are excluded.
- The log predates the current script names: legacy `33-awq-w3` maps to the current
  `awq-w3-transformers` target.
- The raw source logs are local build artifacts under `logs/` and are excluded by
  `.gitignore`; the relevant entries are quoted below.

## Longest-running observed work

| Work | Mode | Raw elapsed | Approximate time |
|---|---|---:|---:|
| Target quantization | full | 18,147 s | **302 min** (about 5 h 02 min) |
| Target quantization | smoke | 7,248 s | **121 min** (about 2 h 01 min) |
| MXFP8 → BF16 restoration | shared | 958–1,070 s | **16–18 min** |
| Compatibility probe | shared | 112–178 s | **2–3 min** |
| Calibration preparation | shared | 34–37 s | **about 1 min** |

A fresh full build therefore took at least about **320 minutes (5 h 20 min)** for
restoration plus quantization. This is a lower bound: the collected logs do not contain a
successful W3-specific PPL, benchmark, upload, or customer-acceptance timing.

## Shared preparation

The following work is shared by all targets and should normally be completed once:

| Stage | Observed duration | Notes |
|---|---:|---|
| Environment setup | 0–5 s after dependencies were cached | The first dependency installation is not represented by the cached timing |
| Calibration generation | 34–37 s | Small calibration set |
| Compatibility probe | 112–178 s | GPU and library checks |
| MXFP8 → BF16 restoration | 958–1,070 s | One-time 28.8B checkpoint conversion; later runs skip it through freshness checks |

## Evidence

- `logs/timing-full.log`:
  `full / 30-dequant / 1070 s / rc=0` and
  `full / 33-awq-w3 / 18147 s / rc=0`
- `logs/timing-smoke.log`:
  `smoke / 30-dequant / 1034 s / rc=0`,
  `smoke / 33-awq-w3 / 7248 s / rc=0`, and successful compatibility probes
- `logs/full-spark1-33.log`:
  `step done: 33-awq-w3 elapsed=18147s rc=0`
- `logs/run-smoke-console.log`:
  `step done: 33-awq-w3 elapsed=7248s rc=0`

## Planning caveats

- The 302-minute full quantization result is the longest successful W3 run in the
  collected logs, not a guaranteed SLA.
- The historical run used the recipe available at the time. The current default is
  pure-GPTQ because AWQ replay is incompatible with `RoutedLinear`; a current run may
  therefore differ.
- Add separate estimates for PPL, evaluation, model-card/audit work, upload, and
  customer acceptance until those stages are timed for this target.
