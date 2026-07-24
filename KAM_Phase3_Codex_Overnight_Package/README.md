# KAM Phase III Codex Overnight Package

Start with `KAM_PHASE3_CODEX_OVERNIGHT_BRIEF.md`. It is the authoritative execution contract.

Contents:

- `KAM_PHASE3_CODEX_OVERNIGHT_BRIEF.md` — scientific, engineering, statistical, and cluster execution specification.
- `phase3_overview.yaml` — machine-readable summary of the intended experiment matrix and gates.
- `slurm/phase3_array_template.sbatch` — generic HiPerGator array-row template; account, QOS, partition, group, environment, and Blue path remain placeholders.
- `slurm/phase3_gate_template.sbatch` — aggregation and scientific-gate template.
- `scripts/local_overnight_template.sh` — local single-GPU overnight runner template.

The templates are intentionally not tied to a specific repository commit or UF allocation. Codex must audit the current repository, adapt the CLI paths, validate with smoke jobs, and generate the final manifests before submission.
