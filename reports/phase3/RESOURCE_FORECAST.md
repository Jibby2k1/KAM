# Phase III Resource Forecast

## HiPerGator allocation

- Account: `uf-dsi`
- Accepted QOS: `uf-dsi` (the `Investment` label in `slurmInfo` is an allocation label, not a usable QOS string)
- Partition: `hpg-turin`
- GPU: L4
- Blue root: `/blue/uf-dsi/rvalle1`
- Verified allocation summary: 744 hours, 8 GPUs, 128 CPUs, 1000 GB memory
- Recommended overnight throttle: **4 concurrent GPUs**, leaving headroom under the 8-GPU allocation and current cluster utilization.

## Pilot measurements

| Scale | Rows | Median job duration (s) | Approx. p90 job duration (s) | Peak allocated VRAM (MB) | Parameter target |
|---|---:|---:|---:|---:|---:|
| XS | 4 | 5.6 | 6.3 | 21.1 | 40,000 |
| S | 1 | 4.8 | 4.8* | 26.3 | 250,000 |
| M | 1 | 5.2 | 5.2* | 38.2 | 1,000,000 |

`*`Single-observation estimates are provisional. Job duration includes prequential and causal diagnostics where enabled; training-only wall time was approximately 1.0, 1.4, and 2.3 seconds for the representative XS, S, and M rows.

## Queue forecast

The development manifest contains 1,728 independent rows: three tasks, three scales, four variants, 24 trials, and two training seeds. All rows are below the scheduler’s 3,000-row limit and are idempotent. At the pilot rate, expected GPU-hours are well below the 500--900 GPU-hour suggested ceiling; this is a pilot-resolution development search, not a confirmatory budget claim.

The queue writes all row outputs under Blue storage. The pilot footprint was approximately 1.55 MB per row when diagnostics were enabled; the full campaign is expected to remain comfortably below a few GB because only the first row per task/scale/variant stores long traces and deletion diagnostics.

## Guardrails

- One GPU per row; four concurrent rows.
- `cuda/12.8.1` and the preinstalled PyTorch 2.8 module.
- 32 GB/4 CPU bulk requests; 64 GB/8 CPU confirmation requests.
- Resume skips validated completed rows and reruns failed/missing rows.
- Gate A is dependent on the audit chain; development search is dependent on Gate A; confirmation is not submitted by the overnight command.
