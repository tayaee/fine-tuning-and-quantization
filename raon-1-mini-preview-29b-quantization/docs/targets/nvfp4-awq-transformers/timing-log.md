# `nvfp4-awq-transformers` timing log

[← Target README](README.md) · [← All targets](../README.md)

## Scope and interpretation

- The measurements below come from wall-clock timing logs collected on the DGX Spark 1x
  (NVIDIA GB10, one GPU, TP=1) build host.
- Values are rounded to the nearest minute. Raw elapsed seconds are retained for
  traceability.
- Only successful work (`rc=0`) is used for the planning figures. Freshness-cache hits
  that completed in zero or one second and failed experiments are excluded.
- Older logs use `28-quant-nvfp4-awq` or `35-nvfp4-awq`; both map to the current
  `nvfp4-awq-transformers` target.
- The raw source logs are local build artifacts under `logs/` and are excluded by
  `.gitignore`; the relevant entries are quoted below.

## Longest-running observed work

| Work | Mode | Raw elapsed | Approximate time |
|---|---|---:|---:|
| Evaluation (longest successful run) | full | 4,850 s | **81 min** |
| Evaluation (isolated phase-B run) | full | 1,655 s | **28 min** |
| Evaluation (another successful run) | full | 655 s | **11 min** |
| Target quantization | full | 2,164–2,227 s | **36–37 min** |
| Evaluation | smoke | 1,131–1,283 s | **19–21 min** |
| Target quantization | smoke | 1,190–1,198 s | **about 20 min** |
| MXFP8 → BF16 restoration | shared | 958–1,070 s | **16–18 min** |
| PPL measurement | full | 1,026–1,029 s | **about 17 min** |
| PPL measurement | smoke | 946–947 s | **about 16 min** |
| Compatibility probe | shared | 112–178 s | **2–3 min** |
| Calibration preparation | shared | 34–37 s | **about 1 min** |

Using the isolated full-build measurements, restoration + quantization + PPL + evaluation
took about **98 minutes (1 h 38 min)**. Using the longest observed full evaluation, the
same measured portion took about **151 minutes (2 h 31 min)**. These totals exclude
model-card/audit work, upload, and customer acceptance.

## Why the evaluation time varies

Three successful full evaluation entries were recorded: 655, 1,655, and 4,850 seconds.
They used different runner revisions and surrounding execution conditions, including
whether base and quantized responses were already available. For capacity planning, use
the **11–81 minute observed range** until a current, isolated run supplies a more
repeatable value.

## Shared preparation

The following work is shared by all targets and should normally be completed once:

| Stage | Observed duration | Notes |
|---|---:|---|
| Environment setup | 0–5 s after dependencies were cached | The first dependency installation is not represented by the cached timing |
| Calibration generation | 34–37 s | Small calibration set |
| Compatibility probe | 112–178 s | GPU and library checks |
| MXFP8 → BF16 restoration | 958–1,070 s | One-time 28.8B checkpoint conversion; later runs skip it through freshness checks |

## Evidence

- `logs/phase-a.log`:
  `dequant rc=0 elapsed=958s`
- `logs/phase-a2.log`:
  `quant-smoke rc=0 elapsed=1190s` and `quant-full rc=0 elapsed=2227s`
- `logs/phase-b-smoke.log`:
  `ppl-smoke rc=0 elapsed=947s` and `eval-smoke rc=0 elapsed=1131s`
- `logs/phase-b-full.log`:
  `ppl-full rc=0 elapsed=1029s` and `eval-full rc=0 elapsed=1655s`
- `logs/timing-build-smoke.log`:
  successful quantization, PPL, and evaluation entries
- `logs/timing-build-full.log`:
  successful full quantization (`2164 s` and `2176 s`), PPL (`341 s` and `1026 s`), and
  evaluation (`655 s` and `4850 s`) entries
- `logs/timing-smoke.log`:
  earlier successful `35-nvfp4-awq` smoke quantization (`1344 s` and `1937 s`)

## Planning caveats

- The 81-minute evaluation is the longest observed successful run, not a guaranteed SLA.
- The historical recipe and runner changed during the collected period. Treat the ranges
  as capacity-planning evidence and record a fresh isolated timing before making a
  release schedule commitment.
- Upload and customer-acceptance timings were not captured for this target.
