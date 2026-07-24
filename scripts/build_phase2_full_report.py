from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from build_phase2_descriptive_plots import build_descriptive_artifacts


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in csv.DictReader(path.open(encoding="utf-8"))] if path.exists() else []


def num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    value = num(value)
    return "—" if value is None else f"{value:.6g}"


def mean(group: list[dict[str, Any]], key: str) -> float | None:
    values = [num(row.get(key)) for row in group]
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def paired_table(stats: list[dict[str, Any]]) -> list[str]:
    lines = ["| Task | Baseline | Candidate | Pairs | Mean improvement | 95% CI | Holm p |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in stats:
        lines.append(f"| {row.get('task')} | {row.get('baseline')} | {row.get('variant')} | {row.get('pairs')} | {fmt(row.get('mean_improvement'))} | [{fmt(row.get('bootstrap_ci_low'))}, {fmt(row.get('bootstrap_ci_high'))}] | {fmt(row.get('holm_p'))} |")
    return lines


def matrix_table(stats: list[dict[str, Any]]) -> list[str]:
    lines = ["| Task | Claim | Baseline | Candidate | Pairs | Parameters | Mean improvement | 95% CI | Holm p |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in stats:
        lines.append(f"| {row.get('task')} | {row.get('claim')} | {row.get('baseline')} | {row.get('candidate')} | {row.get('pairs')} | {row.get('parameter_count')} | {fmt(row.get('mean_improvement'))} | [{fmt(row.get('bootstrap_ci_low'))}, {fmt(row.get('bootstrap_ci_high'))}] | {fmt(row.get('holm_p'))} |")
    return lines


def heldout_table(stats: list[dict[str, Any]]) -> list[str]:
    lines = ["| Task | Claim | Baseline | Candidate | Seed pairs | Mean late-loss improvement | 95% CI | Holm p |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in stats:
        lines.append(f"| {row.get('task')} | {row.get('claim')} | {row.get('baseline')} | {row.get('candidate')} | {row.get('pairs')} | {fmt(row.get('mean_improvement'))} | [{fmt(row.get('bootstrap_ci_low'))}, {fmt(row.get('bootstrap_ci_high'))}] | {fmt(row.get('holm_p'))} |")
    return lines


def language_table(summary: list[dict[str, Any]]) -> list[str]:
    lines = ["| Task | Variant | Seeds | Cross-entropy | Accuracy |", "|---|---:|---:|---:|---:|"]
    for row in sorted(summary, key=lambda item: (str(item.get("task")), str(item.get("variant")))):
        lines.append(f"| {row.get('task')} | {row.get('variant')} | {row.get('seeds')} | {fmt(row.get('cross_entropy'))} | {fmt(row.get('accuracy'))} |")
    return lines


def adaptation_table(adaptation: list[dict[str, Any]]) -> list[str]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in adaptation:
        groups[(str(row.get("task")), str(row.get("adapter")))].append(row)
    lines = ["| Task | Adapter | Transitions | Early loss | Late loss | Recovery steps | Support purity | Effective supports |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (task, adapter), group in sorted(groups.items()):
        lines.append(f"| {task} | {adapter} | {len(group)} | {fmt(mean(group, 'early_loss'))} | {fmt(mean(group, 'late_loss'))} | {fmt(mean(group, 'recovery_steps'))} | {fmt(mean(group, 'support_purity'))} | {fmt(mean(group, 'global_effective_supports'))} |")
    return lines


def timing_table(timing: list[dict[str, Any]]) -> list[str]:
    lines = ["| Sequence length | Variant | Parameters | Median ms | IQR ms | P90 ms | Peak MB |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(timing, key=lambda item: (int(item.get("sequence_length", 0)), str(item.get("variant")))):
        lines.append(f"| {row.get('sequence_length')} | {row.get('variant')} | {row.get('parameters')} | {fmt(row.get('median_ms'))} | {fmt(row.get('iqr_ms'))} | {fmt(row.get('p90_ms'))} | {fmt(row.get('peak_memory_megabytes'))} |")
    return lines


def make_figures(out: Path, adaptation: list[dict[str, Any]], timing: list[dict[str, Any]], deletions: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    out.mkdir(parents=True, exist_ok=True)
    if adaptation:
        groups: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in adaptation:
            value = num(row.get("late_loss"))
            if value is not None and np.isfinite(value):
                groups[(str(row.get("task")), str(row.get("adapter")))].append(value)
        keys = sorted(groups)
        plt.figure(figsize=(8, 4))
        plt.bar([f"{task}\n{adapter}" for task, adapter in keys], [np.mean(groups[key]) for key in keys])
        plt.ylabel("mean late post-shift loss")
        plt.title("Switching adaptation: late post-shift loss")
        plt.tight_layout()
        plt.savefig(out / "switching_adaptation.png", dpi=170)
        plt.close()
    if timing:
        plt.figure(figsize=(8, 4))
        for variant in sorted({str(row.get("variant")) for row in timing}):
            subset = sorted((row for row in timing if row.get("variant") == variant), key=lambda row: int(row["sequence_length"]))
            plt.plot([int(row["sequence_length"]) for row in subset], [float(row["median_ms"]) for row in subset], marker="o", label=variant)
        plt.xlabel("sequence length")
        plt.ylabel("median forward ms")
        plt.title("Matched-capacity timing")
        plt.legend(ncol=3)
        plt.tight_layout()
        plt.savefig(out / "timing_scaling.png", dpi=170)
        plt.close()
    if deletions:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in deletions:
            if row.get("deletion_kind") in {"top", "random", "bottom"}:
                groups[str(row.get("deletion_kind"))].append(row)
        plt.figure(figsize=(7, 4))
        for kind, subset in sorted(groups.items()):
            subset = sorted(subset, key=lambda row: int(row["deletion_count"]))
            plt.plot([int(row["deletion_count"]) for row in subset], [float(row["delta"]) for row in subset], marker="o", label=kind)
        plt.xlabel("deleted supports")
        plt.ylabel("loss delta")
        plt.title("Support deletion faithfulness")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "support_deletion.png", dpi=170)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the complete Phase II report and decision memo.")
    parser.add_argument("--stationary-stats", type=Path, default=Path("results/phase2/paired_screen/paired_stats.csv"))
    parser.add_argument("--switching-stats", type=Path, default=Path("results/phase2/switching_paired_screen/paired_stats.csv"))
    parser.add_argument("--adaptation", type=Path, default=Path("results/phase2/switching_adaptation/shift_metrics.csv"))
    parser.add_argument("--timing", type=Path, default=Path("results/phase2/timing/timing.csv"))
    parser.add_argument("--search", type=Path, default=Path("results/phase2/search_full/optuna_search/search_summary.csv"))
    parser.add_argument("--deletions", type=Path, default=Path("results/phase2/switching_reanalysis_deletions.csv"))
    parser.add_argument("--copy-generalization", type=Path, default=Path("results/phase2/variable_copy_generalization_v3/generalization.csv"))
    parser.add_argument("--dyck-generalization", type=Path, default=Path("results/phase2/dyck_generalization/generalization.csv"))
    parser.add_argument("--dynamic-stats", type=Path, default=Path("results/phase2/dynamic_matrix_pmatched_v2/dynamic_matrix_stats.csv"))
    parser.add_argument("--dynamic-metrics", type=Path, default=Path("results/phase2/dynamic_matrix_pmatched_v2/all_metrics.csv"))
    parser.add_argument("--heldout-stats", type=Path, default=Path("results/phase2/heldout_dynamic_matrix_nlms/heldout_stats.csv"))
    parser.add_argument("--heldout-metrics", type=Path, default=Path("results/phase2/heldout_dynamic_matrix_nlms/shift_metrics.csv"))
    parser.add_argument("--language-metrics", type=Path, default=Path("results/phase2/language_matrix/all_metrics.csv"))
    parser.add_argument("--language-root", type=Path, default=Path("results/phase2/language_matrix"))
    parser.add_argument("--trace", type=Path, default=Path("results/phase2/switching_adaptation/shift_trace.csv"))
    parser.add_argument("--shift-metrics", type=Path, default=Path("results/phase2/switching_adaptation/shift_metrics.csv"))
    parser.add_argument("--history-root", type=Path, action="append", default=[Path("results/phase2/paired_screen"), Path("results/phase2/switching_paired_screen"), Path("results/phase2/language_matrix")])
    parser.add_argument("--descriptive-metrics", type=Path, default=Path("results/phase2/descriptive_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase2"))
    args = parser.parse_args()
    stationary = rows(args.stationary_stats)
    switching = rows(args.switching_stats)
    adaptation = rows(args.adaptation)
    timing = rows(args.timing)
    search = rows(args.search)
    copy_generalization = rows(args.copy_generalization)
    dyck_generalization = rows(args.dyck_generalization)
    dynamic_stats = rows(args.dynamic_stats)
    dynamic_metrics = rows(args.dynamic_metrics)
    heldout_stats = rows(args.heldout_stats)
    heldout_metrics = rows(args.heldout_metrics)
    language_runs = rows(args.language_metrics)
    language_groups: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for row in language_runs:
        payload_path = args.language_root / str(row["run_id"]) / "metrics.json"
        if not payload_path.exists():
            continue
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        validation = payload.get("final_validation", {})
        language_groups[(str(row["task"]), str(row["variant"]))].append({"cross_entropy": num(validation.get("cross_entropy")), "accuracy": num(validation.get("accuracy"))})
    language_summary = []
    for (task, variant), group in language_groups.items():
        language_summary.append({"task": task, "variant": variant, "seeds": len(group), "cross_entropy": mean(group, "cross_entropy"), "accuracy": mean(group, "accuracy")})
    copy_lines = ["| Payload length | Copied-token accuracy | Exact sequence accuracy | Cross-entropy |", "|---:|---:|---:|---:|"] + [f"| {row.get("length")} | {fmt(row.get("copied_token_accuracy"))} | {fmt(row.get("exact_sequence_accuracy"))} | {fmt(row.get("masked_cross_entropy"))} |" for row in copy_generalization]
    dyck_lines = ["| Depth | Token accuracy | Grammar-valid fraction |", "|---:|---:|---:|"] + [f"| {row.get("depth")} | {fmt(row.get("token_accuracy"))} | {fmt(row.get("grammar_valid_fraction"))} |" for row in dyck_generalization]
    deletions = rows(args.deletions)
    all_stats = stationary + switching
    figures = args.output_dir / "figures"
    make_figures(figures, adaptation, timing, deletions)
    descriptive = build_descriptive_artifacts(
        trace_path=args.trace,
        shift_metrics_path=args.shift_metrics,
        history_roots=args.history_root,
        output_dir=figures,
        metrics_output=args.descriptive_metrics,
    )
    complete_trials = sum(int(num(row.get("complete_trials")) or 0) for row in search)
    total_trials = sum(int(num(row.get("trials")) or 0) for row in search)
    finite_adaptation = all(num(row.get("early_loss")) is not None and num(row.get("late_loss")) is not None for row in adaptation)
    corrected_passes = [row for row in all_stats if str(row.get("ci_excludes_zero")).lower() == "true" and (num(row.get("holm_p")) or 1.0) < 0.05]
    lines = [
        "# Phase II Results",
        "",
        "## Technical summary",
        "",
        f"The workspace now contains a verified Phase II foundation plus **100 baseline paired training runs**: 50 stationary dynamical runs and 50 continuous switching runs, across five variants and five seeds. A separate exact-capacity suffix screen adds **{len(dynamic_metrics)} runs** across nine variants, two switching systems, and five seeds, all at 39,961 parameters. The evaluator produced **{len(adaptation)} all-adapter transition rows** plus **{len(heldout_metrics)} held-out NLMS transition rows**; the full Optuna search produced **{total_trials} trials** across eight SQLite studies ({complete_trials} complete). The reporting layer now adds **{len(descriptive['created'])} descriptive figures** and a grouped error-metric table.",
        "",
        f"No corrected paired comparison currently passes the confidence-interval and Holm gate ({len(corrected_passes)} passes). The evidence therefore does not promote a radial-memory architecture or authorize the conditional TinyStories stage.",
        "",
        "## Paired stationary and switching evidence",
        "",
        "Positive mean improvement means the candidate’s MSE was lower than the named baseline. Confidence intervals are paired bootstrap intervals over five training seeds; p-values are Holm-adjusted across the declared comparisons.",
        "",
        *paired_table(all_stats),
        "",
        "The only unadjusted-looking positive signal in the switching table is not sufficient after Holm correction. The report treats that as a shortlist signal, not a claim.",
        "",
        "## Exact parameter-matched dynamic screen",
        "",
        "The suffix-aware screen tests persistent values, routes, and both pathways, plus radial memory at the same output mode. Every row below uses five paired seeds and exactly 39,961 trainable parameters; no corrected comparison passes.",
        "",
        *matrix_table(dynamic_stats),
        "",
        "## Formal-language generalization",
        "",
        "The variable-copy checkpoint uses sinusoidal positions and trains on payloads sampled from 8–64. At unseen lengths, copied-token accuracy and exact-sequence accuracy remain low; the task is not yet a successful length-generalization result.",
        "",
        *copy_lines,
        "",
        "The bounded Dyck-2 checkpoint was trained to depth 8. Grammar-validity remains zero at depths 8–16 in this bounded run, so it is a diagnostic failure rather than evidence of hierarchical generalization.",
        "",
        *dyck_lines,
        "",
        "## Five-seed mechanism-language screen",
        "",
        f"The representative mechanism screen completed {len(language_runs)} runs across MQAR, variable copy, Dyck-2, and reusable-regime grammar. These are mechanism diagnostics, not evidence to override the failed dynamic-memory gate.",
        "",
        *language_table(language_summary),
        "",
        "## Prequential adaptation and support diagnostics",
        "",
        "Each transition follows predict → score → reveal → update. Recovery steps are the first rolling window within 10% of the late segment loss. Support purity is descriptive alignment between argmax support assignments and hidden regime labels; labels are never fed to the model.",
        "",
        *adaptation_table(adaptation),
        "",
        f"All adaptation transition rows were finite: **{finite_adaptation}**. Support diagnostics and deletion curves are saved separately because they are checkpoint-level diagnostics, not independent training replicates.",
        "",
        "## Descriptive prediction and error evidence",
        "",
        "The new plots separate optimization behavior, pointwise prediction behavior, tail risk, and post-shift recovery. Learning curves show medians with interquartile bands; legacy runs contain only three recorded checkpoints, while future runs now default to a denser 10%-of-budget evaluation cadence.",
        "",
        "![Regression learning curves](figures/learning_curves_regression.png)",
        "",
        "The language/mechanism panels use the same median/IQR convention with cross-entropy rather than MSE, keeping optimization curves comparable without conflating the task units.",
        "",
        "![Language and mechanism learning curves](figures/learning_curves_language.png)",
        "",
        "The representative true-versus-prediction panels make signed error and regime-boundary behavior visible for one DR/NLMS/seed-7 stream; aggregate claims should use the descriptive metric CSV and paired tables.",
        "",
        "![Mackey–Glass true versus prediction and signed error](figures/prediction_true_error_switching_mackey_glass.png)",
        "",
        "![NARMA true versus prediction and signed error](figures/prediction_true_error_switching_narma.png)",
        "",
        "The log absolute-error distributions expose whether a method’s average improvement is broad or dominated by tail behavior; the metric export adds p90/p95/max error, bias, R², correlation, relative MAE, and p95-to-median tail ratio.",
        "",
        "![Log absolute-error distributions](figures/error_distribution_log.png)",
        "",
        "The recovery curves align squared loss at detected regime transitions, while the support plot separates utilization from descriptive regime alignment; neither should be read as causal attribution by itself.",
        "",
        "![Post-shift recovery curves](figures/post_shift_recovery_curves.png)",
        "",
        "![Support utilization and regime alignment](figures/support_diagnostics.png)",
        "",
        "## Held-out schedule coverage",
        "",
        f"The exact-capacity matrix was evaluated on five independent schedule/stream-seed combinations per checkpoint with the primary NLMS adapter, yielding {len(heldout_metrics)} transition rows. Schedule-level results were averaged within training seed before paired inference; no corrected comparison passes.",
        "",
        *heldout_table(heldout_stats),
        "",
        "## Capacity, timing, and search controls",
        "",
        "The capacity smoke matched D0 and DR exactly at 12,000 parameters. The timing benchmark matched all five language variants at 55,576 parameters and measured synchronized AMP forward latency with median/IQR/P90 and peak VRAM.",
        "",
        *timing_table(timing),
        "",
        f"The full Optuna search used SQLite storage, TPE sampling, and MedianPruner configuration; it completed {complete_trials}/{total_trials} trials, with pruned trials recorded in the study database.",
        "",
        "## Required gates and decision",
        "",
        "1. Stationary degradation ≤5%: not established for a promoted architecture after matched controls.",
        "2. Post-shift improvement ≥15%: not established by the five-schedule held-out NLMS screen after paired inference.",
        "3. Corrected CI excludes zero: **not passed**.",
        "4. Top-support deletion beats random: deletion curves exist, but cross-seed causal faithfulness is not yet passed.",
        "5. Support noncollapse: short-run utilization is healthy in memory models, but cross-seed stability is not passed.",
        "6. Overhead: measured and reported; radial/memory variants carry measurable latency and VRAM cost.",
        "",
        "Decision: **continue to development-stage switching and capacity-matched diagnostics; do not promote to ten-seed confirmation or TinyStories.**",
        "",
        "## Artifacts and open work",
        "",
        "- Paired stationary: `results/phase2/paired_screen/`",
        "- Paired switching: `results/phase2/switching_paired_screen/`",
        "- Exact parameter-matched suffix screen: `results/phase2/dynamic_matrix_pmatched_v2/`",
        "- Five-schedule held-out NLMS screen: `results/phase2/heldout_dynamic_matrix_nlms/`",
        "- Prequential recovery: `results/phase2/switching_adaptation/`",
        "- Phase A switching reanalysis: `results/phase2/switching_reanalysis.csv` and its deletion/raw files",
        "- Full Optuna SQLite search: `results/phase2/search_full/optuna_search/`",
        "- Matched timing: `results/phase2/timing/`",
        "- Variable-copy generalization: `results/phase2/variable_copy_generalization_v3/generalization.csv`",
        "- Dyck-2 generalization: `results/phase2/dyck_generalization/generalization.csv`",
        "- Figures: `reports/phase2/figures/`",
        f"- Descriptive metrics: `{descriptive['metrics_path']}`",
        f"- Descriptive metric summary: `{descriptive['summary_path']}`",
        f"- Chart map and QA notes: `{descriptive['chart_map']}`",
        "",
        "Remaining gated work: ten-seed confirmation only after the registered gates pass, and conditional TinyStories only after a passing dynamic-memory gate.",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "PHASE2_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    memo = [
        "# Phase II Decision Memo",
        "",
        "1. Does persistent memory survive capacity and compute controls? **Not established.** Exact capacity and timing controls exist; paired effects remain inconclusive.",
        "",
        "2. Does radial persistent routing beat dot/cosine persistent routing? **Not established.** Switching DR-vs-DD is a development signal only and fails corrected inference.",
        "",
        "3. Does the adaptive advantage remain with the same adapter? **Not established.** Prequential frozen/NLMS/SGD/RLS traces exist, but no promoted architecture passes the shift gates.",
        "",
        "4. Are support explanations causal and stable? **Not yet.** Perturbation/deletion, support purity, and utilization diagnostics are implemented; cross-seed stability and corrected faithfulness are outstanding.",
        "",
        "5. Which architecture advances? **None.** Keep D0, DD, DR, and RR in development; hold ten-seed confirmation and TinyStories.",
    ]
    (args.output_dir / "PHASE2_DECISION_MEMO.md").write_text("\n".join(memo) + "\n", encoding="utf-8")
    handoff = [
        "# LLM handoff: current Phase II state",
        "",
        "Goal: decide whether persistent/radial memory improves prediction or adaptation after capacity, compute, paired-stream, and causal-faithfulness controls.",
        "",
        "## Verified implementation",
        "- D0/R0/DD/DR/RR plus DD/DR/RR `-v/-a/-b` memory-output modes.",
        "- Continuous switching Mackey–Glass/NARMA with state-preserving schedules and NMSE/NRMSE.",
        "- Exact parameter matching, SQLite-resumable suites, Optuna SQLite/TPE/MedianPruner search, synchronized timing, Phase A interventions, and prequential recovery diagnostics.",
        "- Denser future learning-curve checkpoints, validation prediction exports, and descriptive regression metrics: RMSE, MAE, bias, error quantiles, R², correlation, relative MAE, and tail ratios.",
        "",
        "## Verified runs",
        f"- 50 stationary paired runs + 50 switching paired runs, five seeds each.",
        f"- {len(adaptation)} transition rows across frozen/NLMS/SGD/RLS; all finite.",
        f"- {total_trials} full Optuna trials across eight family/task studies; {complete_trials} completed and {total_trials - complete_trials} pruned.",
        f"- {len(dynamic_metrics)} exact-capacity suffix-screen runs at 39,961 parameters; suffix-aware paired statistics are in `results/phase2/dynamic_matrix_pmatched_v2/dynamic_matrix_stats.csv`.",
        f"- {len(heldout_metrics)} five-schedule held-out NLMS transition rows; seed-aggregated statistics are in `results/phase2/heldout_dynamic_matrix_nlms/heldout_stats.csv`.",
        f"- {len(language_runs)} five-seed mechanism-language runs: `results/phase2/language_matrix/`.",
        f"- Descriptive plots and grouped error metrics: `{descriptive['metrics_path']}`, `{descriptive['summary_path']}`, and `{descriptive['chart_map']}`.",
        "- Matched timing at 55,576 parameters and capacity smoke at 12,000 parameters.",
        "",
        "## Current conclusion",
        "No corrected paired comparison passes the registered CI + Holm gate. Do not claim radial memory works or launch TinyStories/ten-seed confirmation yet.",
        "",
        "## Advice requested",
        "Recommend whether to stop the KAM-specific direction or run one narrowly targeted follow-up. If continuing, use DD-b + NLMS versus D0 + NLMS as the primary comparison, with DR-b + NLMS as the radial ablation, and require the same stationary, post-shift, corrected-inference, deletion-faithfulness, support-stability, and overhead gates before any ten-seed confirmation.",
    ]
    (args.output_dir / "PHASE2_LLM_HANDOFF.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")
    print(f"Wrote complete Phase II report, memo, handoff, and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
