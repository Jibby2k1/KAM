#!/usr/bin/env python3
"""Audit and aggregate the follow-up dataset, noise, and model experiments."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from random import Random
from statistics import mean, median

from analyze_kam_fullwiki_support_links import paired


def read(path):
    return json.loads(path.read_text())


def runs(directory, pattern, expected):
    files = sorted(directory.glob(pattern))
    assert len(files) == expected, (directory, pattern, len(files), expected)
    data = [read(p) for p in files]
    for field in ("fixture_sha256", "training_sha256", "source_sha256", "weights_sha256", "heads_sha256", "ablations_sha256"):
        assert len({r[field] for r in data if field in r}) <= 1
    return data


def flattened(data):
    rows = [r for part in data for r in part["rows"]]
    assert len({r["id"] for r in rows}) == len(rows)
    return rows


def totals(rows, metrics):
    return {"n": len(rows), **{field: sum(r[field] for r in rows) for field in metrics}}


def ablation(directory):
    pos = "context_only"
    excluded = ("learned_single_shot", "learned_sequential", "one_pass_pair", "matched_plain", "support_link", "oracle")
    fit = read(directory / "fit.json")
    assert fit["full_refit_max_param_delta"] == 0
    output = {"fit": fit, "datasets": {}}
    for dataset, size in (("hotpot", 4), ("twowiki", 4), ("musique", 2), ("news", 2)):
        rows = flattened(runs(directory, f"{dataset}-?.json", size))
        arms = [name for name in rows[0]["arms"] if name not in excluded]
        assert all(r["arms"][pos] == r["arms"]["refit_full"] for r in rows)
        groups = {"all": rows, "linked": [r for r in rows if r["gold_linked"] is True],
                  "unlinked": [r for r in rows if r["gold_linked"] is False]}
        result = {}
        for group, members in groups.items():
            if not members:
                continue

            def pull(arm, metric):
                key = "both_support" if metric == "support" else "em"
                return [int(r["arms"][arm][key] and (metric == "support" or r["arms"][arm]["both_support"]))
                        for r in members]

            result[group] = {"n": len(members), "arms": {}}
            for arm in arms:
                result[group]["arms"][arm] = {
                    metric: {"hits": sum(y := pull(arm, metric)),
                             "vs_full": paired([a - b for a, b in zip(y, pull(pos, metric), strict=True)])}
                    for metric in ("support", "joint")
                }
        output["datasets"][dataset] = result
    return output


def link_noise(directory, trained):
    answer = {}
    for dataset, size in (("twowiki", 4), ("musique", 2)):
        shards = runs(directory, f"{dataset}-?.json", size)
        rows = flattened(shards)
        arms = list(rows[0]["scenarios"]["clean"])
        n = len(rows)
        assert n == sum(r["eligible"] for r in shards)
        for r in rows:
            assert r["target"] in r["gold"] and r["donor"] not in r["gold"]
            for arm in ("learned_single_shot", "learned_sequential"):
                assert r["scenarios"]["clean"][arm] == r["scenarios"]["graph_only"][arm]
            for records in r["scenarios"].values():
                for out in records.values():
                    assert out["both_support"] == (set(out["selection"]) == set(r["gold"]))
        metrics = ("both_support", "injected_pair", "em", "contains_donor")
        experiment = {"n": n, "target_questions": sum(r["total"] for r in shards),
                      "linked_gold": sum(r["gold_both_linked"] for r in rows),
                      "scenarios": {}, "effect_vs_sham": {}, "guards": {}}
        for s in rows[0]["scenarios"]:
            experiment["scenarios"][s] = {
                a: totals([r["scenarios"][s][a] for r in rows], metrics) |
                {"joint": sum(r["scenarios"][s][a]["both_support"] and r["scenarios"][s][a]["em"] for r in rows)}
                for a in arms
            }
        for arm in arms:
            experiment["effect_vs_sham"][arm] = {
                metric: paired([r["scenarios"]["linked_text"][arm][metric] - r["scenarios"]["sham_text"][arm][metric] for r in rows])
                for metric in ("both_support", "injected_pair", "em")
            }
        if trained:
            experiment["injection_template"] = shards[0]["injection_template"]
            for guard in ("link_noise_trained", "link_noise_without_graph"):
                experiment["guards"][guard] = {
                    s: {metric: paired([r["scenarios"][s][guard][metric] -
                                        r["scenarios"][s]["sham_trained"][metric] for r in rows])
                        for metric in ("both_support", "em")}
                    for s in ("clean", "sham_text", "linked_text", "graph_only")
                }
        answer[dataset] = experiment
    if trained:
        details = read(directory / "fit.json")
        assert all(part["injection_template"] == details["prespecified_test_template"] for part in answer.values())
        answer["fit"] = {"training_cases": len(details["training_cases"]), "templates": details["templates"],
                         "elapsed_seconds": details["elapsed_seconds"], "training_results": details["training_results"]}
    return answer


def news(directory):
    shards = runs(directory, "news-controls-?.json", 2)
    rows = flattened(shards)
    assert len(rows) == 222
    return {
        "n": len(rows),
        "all_facts_in_articles": sum(r["all_facts_in_source_articles"] for r in rows),
        "retrieved_articles": sum(r["both_articles_retrieved"] for r in rows),
        "retrieved_exact_spans": sum(r["both_fact_spans_retrieved"] for r in rows),
        "modes": {
            mode: {model: totals([r["controls"][mode][model] for r in rows if mode in r["controls"]],
                                 ("em", "token_overlap")) for model in ("small", "base")}
            for mode in rows[0]["controls"]
        },
    }


def tempreason(directory):
    answer = {}
    for level in (2, 3):
        rows = flattened(runs(directory, f"l{level}-?.json", 4))
        assert len(rows) == len({r["entity"] for r in rows}) == 384
        spans = [r["gold_sets"] for r in rows]
        answer[level] = {"n": len(rows), "parsed_facts": sum(r["parsed_facts"] for r in rows),
                         "facts": sum(r["n_facts"] for r in rows),
                         "all_annotated_spans_visible": sum(all(v) for v in spans),
                         "oracle": totals([r["oracle"] for r in rows], ("em", "at_least_one", "all")),
                         "scenarios": {}}
        for scenario in ("clean", "noise"):
            answer[level]["scenarios"][scenario] = {
                arm: totals([r["scenarios"][scenario][arm] for r in rows],
                            ("em", "at_least_one", "all", "noise_slots", "returned"))
                for arm in rows[0]["scenarios"][scenario]
            }
    return answer


def qasper(directory):
    rows = flattened(runs(directory, "eval-?.json", 4))
    assert len(rows) == len({r["paper"] for r in rows}) == 384
    groups = {}
    for kind in ("one", "two", "three_plus"):
        group = [r for r in rows if r["kind"] == kind]
        results = {"n": len(group), "oracle_exact": sum(r["oracle"]["answer_em"] for r in group),
                   "oracle_token_f1": mean(r["oracle"]["answer_f1"] for r in group), "scenarios": {}}
        for scenario in ("clean", "noise"):
            results["scenarios"][scenario] = {
                "pool_coverage": sum(r["scenarios"][scenario]["pool_can_cover"] for r in group),
                "arms": {a: {"complete": sum(r["scenarios"][scenario]["arms"][a]["complete"] for r in group),
                             "evidence_f1": mean(r["scenarios"][scenario]["arms"][a]["evidence_f1"] for r in group),
                             "answer_em": sum(r["scenarios"][scenario]["arms"][a]["answer_em"] for r in group),
                             "answer_f1": mean(r["scenarios"][scenario]["arms"][a]["answer_f1"] for r in group)}
                         for a in group[0]["scenarios"][scenario]["arms"]},
            }
        groups[kind] = results
    return groups


def timeqa_collision(directory):
    conditions = {
        variant: {r["id"]: r for r in flattened(runs(directory, f"{variant}-?.json", 4))}
        for variant in ("sham", "date_collision")
    }
    assert conditions["sham"].keys() == conditions["date_collision"].keys()
    target = [id_ for id_, sample in conditions["sham"].items() if sample["date_collision_eligible"]]
    assert set(target) == {id_ for id_, sample in conditions["date_collision"].items() if sample["date_collision_eligible"]}
    assessments = {}
    for variant, by_id in conditions.items():
        one = next(iter(by_id.values()))["questions"][0]["scenarios"]["matched_noise"]
        assessments[variant] = {
            arm: totals([q["scenarios"]["matched_noise"][arm] for id_ in target for q in by_id[id_]["questions"]],
                        ("em", "evidence_hit", "distractor", "opposite_period", "opposite_answer_em"))
            for arm in one
        }
    delta = {}
    base, new = conditions["sham"], conditions["date_collision"]
    for arm in assessments["sham"]:
        diffs = {m: [] for m in ("evidence_hit", "em", "distractor", "opposite_answer_em")}
        for id_ in target:
            for metric, d in diffs.items():
                result = [sum(q["scenarios"]["matched_noise"][arm][metric] for q in arm_data[id_]["questions"])
                          for arm_data in (base, new)]
                d.append((result[1] - result[0]) / 2)
        delta[arm] = {m: paired(d) for m, d in diffs.items()}
    return {"eligible_entity_groups": len(target), "total_groups": len(base),
            "outcomes": assessments, "paired_date_vs_sham": delta}


def vintage(directory):
    parts = runs(directory, "eval-?.json", 4)
    rows = [r for part in parts for r in part["rows"]]
    fit = read(directory / "fit-0.json")
    assert all(p["weight_sha256"] == fit["weight_sha256"] for p in parts)
    groups = defaultdict(list)
    for r in rows:
        groups[r["style"], r["known"]].append(r)
    result = {}
    for (style, known), subset in sorted(groups.items()):
        result[f"{style}:{'known' if known else 'unavailable'}"] = {
            "n": len(subset),
            "semantic_series": sum(r["semantic_series"] == r["series"] for r in subset),
            "scenarios": {
                sc: {arm: totals([r["scenarios"][sc][arm] for r in subset],
                                 ("record_exact", "value_exact", "abstained", "future", "wrong_series", "wrong_period"))
                     for arm in subset[0]["scenarios"][sc]}
                for sc in ("clean", "noise")
            },
        }
    contrasts = defaultdict(list)
    for r in rows:
        if r["known"]:
            contrasts[r["id"], r["style"]].append(r)
    assert all(len(g) == 2 for g in contrasts.values())
    for style in ("canonical", "paraphrase"):
        groups = [g for (id_, st), g in contrasts.items() if st == style]
        result[f"{style}:contrast"] = {s: {
            arm: sum(all(r["scenarios"][s][arm]["record_exact"] for r in g) for g in groups)
            for arm in rows[0]["scenarios"][s]} for s in ("clean", "noise")}
        result[f"{style}:contrast"]["n"] = len(groups)
    result["fit"] = fit["training"] | {"threshold": fit["threshold"]}
    result["series"] = {
        fin: {k: {
            "known": len(g := [r for r in rows if r["series"] == fin and r["style"] == k and r["known"]]),
            "semantic": sum(r["semantic_series"] == fin for r in g),
            "learned_noise": sum(r["scenarios"]["noise"]["learned"]["record_exact"] for r in g),
            "semantic_guard_noise": sum(r["scenarios"]["noise"]["semantic_guard"]["record_exact"] for r in g),
        } for k in ("canonical", "paraphrase")} for fin in sorted({r["series"] for r in rows})
    }
    return result


def procedural(directory):
    data = runs(directory, "seed-*.json", 8)
    seeds = {r["seed"] for r in data}
    assert len(seeds) == 8
    results = {}
    for i, spec in enumerate(data[0]["rows"]):
        paired_runs = [(run["seed"], run["rows"][i]) for run in data]
        for _, r in paired_runs:
            assert (r["profile"], r["depth_per_bank"]) == (spec["profile"], spec["depth_per_bank"])
            assert r["arms"]["exact"]["joint"] == r["count"]
        arms = {}
        for arm in spec["arms"]:
            arms[arm] = {metric: mean(r["arms"][arm][metric] / r["count"] for _, r in paired_runs)
                         for metric in spec["arms"][arm]}
        deltas = [(r["arms"]["one_pass_pair"]["joint"] - r["arms"]["sequential"]["joint"]) / r["count"]
                  for _, r in paired_runs]
        results[f"{spec['profile']}:{spec['depth_per_bank']}"] = {"arms": arms, "paired": paired(deltas),
                                                               "seeds": sorted(seeds)}
    return results


def decoder(directory, mode):
    data = runs(directory, f"{mode}-*.json", 10 if mode == "pilot" else 66)
    by_seed = defaultdict(dict)
    for r in data:
        assert r["arm"] not in by_seed[r["seed"]]
        by_seed[r["seed"]][r["arm"]] = r
    assert all(len(v) == (5 if mode == "pilot" else 11) for v in by_seed.values())
    arm_keys = list(next(iter(by_seed.values())))
    count = len(next(iter(by_seed.values()))[arm_keys[0]]["test"])
    for panel in by_seed.values():
        assert len({tuple(r["train_input_sha256"]) for r in panel.values()}) == 1
        assert len({r["train_tokens"] for r in panel.values()}) == 1
        assert len({tuple(x["inputs_sha256"] for x in r["test"]) for r in panel.values()}) == 1
    result = {"seeds": sorted(by_seed), "slices": {}}
    for i in range(count):
        first = next(iter(by_seed.values()))[arm_keys[0]]["test"][i]
        key = f"{first['profile']}:{first['depth_per_bank']}"
        cells = {}
        for arm in arm_keys:
            values = [panel[arm]["test"][i] for panel in by_seed.values()]
            ref = [panel["dot_context"]["test"][i] for panel in by_seed.values()]
            delta = [(r["joint"] - d["joint"]) / r["count"] for r, d in zip(values, ref, strict=True)]
            cells[arm] = {
                "joint": mean(r["joint"] / r["count"] for r in values),
                "value": mean(r["correct_value"] / r["count"] for r in values),
                "follows_intervention": mean(r["value_follows_intervention"] / r["count"] for r in values),
                "seed_joint": {k: r["joint"] / r["count"] for k, r in zip(by_seed, values, strict=True)},
                "vs_dot": paired(delta),
            }
            if values[0].get("query_position_control_joint") is not None:
                cells[arm]["pin_query"] = mean(r["query_position_control_joint"] / r["count"] for r in values)
        result["slices"][key] = cells
    result["models"] = {
        arm: {"median_training_seconds": median(p[arm]["train_seconds"] for p in by_seed.values()),
              "train_tokens": next(iter(by_seed.values()))[arm]["train_tokens"],
              "parameters": next(iter(by_seed.values()))[arm]["parameters"],
              "device": sorted({p[arm]["device"] for p in by_seed.values()})}
        for arm in arm_keys
    }
    return result


def musique_multihop(directory):
    rows = flattened(runs(directory, "eval-?.json", 6))
    result = {}
    for k in (2, 3, 4):
        all_rows = [r for r in rows if r["hops"] == k]
        assert len(all_rows) == 192
        groups = {"all": all_rows, "gold_connected": [r for r in all_rows if r["gold_connected"]],
                  "other": [r for r in all_rows if not r["gold_connected"]]}
        result[k] = {}
        for name, members in groups.items():
            if not members:
                continue
            output = {"n": len(members), "arms": {}}
            for arm in members[0]["arms"]:
                values = [r["arms"][arm] for r in members]
                timings = [x["route_seconds"] for x in values if x["route_seconds"] is not None]
                output["arms"][arm] = totals(values, ("all", "em", "support_count")) | {
                    "joint": sum(r["all"] and r["em"] for r in values),
                    "route_median_ms": median(timings) * 1000 if timings else None,
                    "all_vs_sequential": paired([r["arms"][arm]["all"] - r["arms"]["sequential"]["all"] for r in members]),
                }
            result[k][name] = output
    return result


def cluster_interval(rows, group, difference):
    clustered = defaultdict(lambda: [0, 0])
    for r in rows:
        value = clustered[group(r)]
        value[0] += difference(r)
        value[1] += 1
    members = list(clustered.values())
    rng = Random(143505)
    replicates = []
    for _ in range(5000):
        draw = rng.choices(members, k=len(members))
        replicates.append(sum(a for a, _ in draw) / sum(n for _, n in draw))
    replicates.sort()
    return {"delta": sum(a for a, _ in members) / sum(n for _, n in members),
            "bootstrap95": [replicates[125], replicates[4874]], "clusters": len(members)}


def timeqa_human(directory):
    data = runs(directory, "eval-?.json", 4)
    rows = flattened(data)
    assert len(rows) == data[0]["recruitment"]["selected"] == 827
    result = {"recruitment": data[0]["recruitment"], "subsets": {}}
    for title, subset in (("all", rows), ("edited_question", [r for r in rows if r["edited"]])):
        panel = {"n": len(subset), "entities": len({r["entity"] for r in subset}), "variants": {}}
        for variant in ("easy", "hard"):
            summaries = {}
            for arm in subset[0]["outcomes"][variant]["arms"]:
                metric = [r["outcomes"][variant]["arms"][arm] for r in subset]
                assert all(m["support"] == (m["slot"] in r["gold"]) for m, r in zip(metric, subset, strict=True))
                summaries[arm] = totals(metric, ("em", "support")) | {
                    "joint": sum(m["support"] and m["em"] for m in metric),
                    "support_vs_dense": cluster_interval(subset, lambda r: r["entity"],
                                                         lambda r: int(r["outcomes"][variant]["arms"][arm]["support"]) -
                                                                   int(r["outcomes"][variant]["arms"]["dense"]["support"])),
                }
            panel["variants"][variant] = {"scores": summaries,
                                          "date_tags": dict(Counter(r["outcomes"][variant]["date_rule"] for r in subset))}
        result["subsets"][title] = panel
    return result


def vintage_global(directory):
    paths = sorted(directory.glob("eval-*-0.json"))
    assert paths
    devices = [read(path) for path in paths]
    assert len({d["catalog_sha256"] for d in devices}) == len({d["fixture_sha256"] for d in devices}) == 1
    result = {}
    for run in devices:
        data = run["rows"]
        assert len(data) == 960 and run["corpus_records"] == 190764
        groups = defaultdict(list)
        for r in data:
            groups[r["style"], r["ground_truth"] is not None].append(r)
        reports = {}
        for (style, known), rows in groups.items():
            record = {"n": len(rows), "semantic_index_exact": sum(r["semantic_index"] == r["ground_truth"] for r in rows),
                      "scoring": {}}
            for arm in rows[0]["arms"]:
                together = [(r["ground_truth"], r["arms"][arm]) for r in rows]
                record["scoring"][arm] = {
                    "top64_has_target": sum(m["ranked_oracle"] is not None for _, m in together),
                    "top1_exact": sum(m["first"] == gold for gold, m in together),
                    "pool_guard_exact": sum(m["pool_guard"] == gold for gold, m in together),
                    "top1_future": sum(m["illegal_first"] for _, m in together),
                    "top1_value_exact": sum(m["value_exact"] for _, m in together),
                }
            reports[f"{style}:{'known' if known else 'unavailable'}"] = record
        result[run["device"]] = {"corpus_size": run["corpus_records"], "vectors_bytes": run["stage_vector_bytes"],
                                 "queries": reports, "timing": run["timing"]}
    if len(devices) == 2:
        rows = list(zip(*(d["rows"] for d in devices), strict=True))
        assert all((a["id"], a["period"], a["style"]) == (b["id"], b["period"], b["style"]) for a, b in rows)
        result["hardware_agreement"] = {m: {
            "identical_top64_order": sum(a["arms"][m]["records"] == b["arms"][m]["records"] for a, b in rows),
            "identical_top1": sum(a["arms"][m]["first"] == b["arms"][m]["first"] for a, b in rows),
        } for m in rows[0][0]["arms"]}
    return result


def vintage_stream(directory):
    data = runs(directory, "seed-*.json", 4)
    assert len({r["catalog_sha256"] for r in data}) == 1
    panels = {}
    for name in ("point", "independent"):
        rows = [r["profiles"][name] for r in data]
        methods = rows[0]["methods"]
        panels[name] = {"queries": sum(r["count"] for r in rows),
                        "static_vintage_only_mismatches": sum(r["count"] - r["methods"]["bisect_static_vintage_only"]["correct"]
                                                              for r in rows),
                        "exact_all": all(r["methods"][k]["correct"] == r["count"] for r in rows
                                         for k in ("descending_scan", "min_arrival_tree", "sqlite_index")),
                        "median_ns_lookup": {k: median(r["methods"][k]["ns_per_query"] for r in rows) for k in methods},
                        "median_insert_ns": median(r["prequential"]["incremental_ingest_ns_per_arrival"] for r in rows),
                        "median_live_lookup_ns": median(r["prequential"]["live_read_ns_per_query"] for r in rows),
                        "median_out_of_order_arrivals": median(r["prequential"]["out_of_order_insertions"] for r in rows)}
    return {"seeds": [r["seed"] for r in data], "records": data[0]["records"],
            "addresses": data[0]["addresses"], "median_late": median(r["late_arrivals"] for r in data),
            "panels": panels}


def vintage_aliases(directory):
    data = read(directory / "results.json")
    counts = {}
    for part, rows in data["rows"].items():
        groups = defaultdict(list)
        for r in rows:
            groups[r["status"]].append(r)
            if r["status"] == "known":
                groups[r["code"]].append(r)
        counts[part] = {label: {"n": len(subset),
                                "predictions": {arm: {
                                    "correct": sum(r["arms"][arm]["correct"] for r in subset),
                                    "abstained": sum(r["arms"][arm]["resolved"] is None for r in subset),
                                } for arm in subset[0]["arms"]}} for label, subset in groups.items()}
    return {"fit": data["fit_counts"], "unknown_threshold": data["admission_threshold"],
            "ambiguous_threshold": data["ambiguity_threshold"], "counts": counts}


def qasper_controls(directory):
    rows = flattened(runs(directory, "reader-controls-?.json", 4))
    assert len(rows) == 384
    answers = {}
    for name, group in (("all", rows), *((kind, [r for r in rows if r["kind"] == kind])
                                         for kind in ("one", "two", "three_plus"))):
        methods = group[0]["routing"]["clean"]
        outputs = {p: {arm: {"em": sum(r[p]["arms"][arm]["em"] for r in group),
                              "f1": mean(r[p]["arms"][arm]["f1"] for r in group)}
                        for arm in group[0][p]["arms"]} for p in ("small", "base")}
        visibility = {p: totals([r[p] for r in group], ("visibility_full_gold", "visibility_clipped_gold"))
                      for p in ("small", "base")}
        routes = {scene: {arm: totals([r["routing"][scene][arm] for r in group], ("complete", "outsiders"))
                          for arm in methods} for scene in ("clean", "noise")}
        answers[name] = {"n": len(group), "reader": outputs, "visibility": visibility, "routing": routes}
    return answers


def nolima(directory):
    data = runs(directory, "eval-?.json", 4)
    rows = flattened(data)
    assert len(rows) == data[0]["n_total"] == 156
    assert all(x["inputs"] == data[0]["inputs"] for x in data)
    output = {}
    for name, subset in (("direct", [r for r in rows if r["kind"] == "direct"]),
                         ("onehop", [r for r in rows if r["kind"] == "onehop"]),
                         ("twohop", [r for r in rows if r["kind"].startswith("twohop")])):
        output[name] = {"n": len(subset), "gold_reader_em": sum(r["oracle_reader"]["em"] for r in subset),
                        "closed_reader_em": sum(r["closed_reader"]["em"] for r in subset), "scenarios": {}}
        for scene in subset[0]["scenes"]:
            output[name]["scenarios"][scene] = {depth: {
                arm: totals([r["scenes"][scene][depth][arm] for r in subset], ("needle", "distractor", "em"))
                for arm in subset[0]["scenes"][scene][depth]} for depth in subset[0]["scenes"][scene]}
    return output


def matched_pools(directory):
    answer = {}
    for dataset, shards in (("twowiki", 4), ("musique", 2)):
        rows = flattened(runs(directory, f"{dataset}-?.json", shards))
        for row in rows:
            base = set(row["conditions"]["supplied:10"]["pool"])
            for kind in ("unrelated", "hard"):
                assert base <= set(row["conditions"][f"{kind}:20"]["pool"]) <= set(row["conditions"][f"{kind}:64"]["pool"])
            for scene in row["conditions"].values():
                gold = {i for i, x in enumerate(scene["pool"]) if x < 0 and -x - 1 in row["gold_base"]}
                assert len(gold) == 2
                assert all(p["complete"] == (set(p["selected"]) == gold) for p in scene["arms"].values())
        answer[dataset] = {"n": len(rows), "conditions": {}}
        for name, example in rows[0]["conditions"].items():
            panel = [r["conditions"][name] for r in rows]
            answer[dataset]["conditions"][name] = {
                "complete_support": {arm: sum(p["arms"][arm]["complete"] for p in panel) for arm in example["arms"]},
                "linked_gold": sum(p["gold_pairs_linked"] for p in panel),
                "mean_linked_pairs": mean(p["n_linked_pairs"] for p in panel),
                "mean_donor_relevance": mean(p["donor_relevance"] for p in panel)
                if example["donor_relevance"] is not None else None,
                "context_minus_no_coverage": paired([p["arms"]["context_only"]["complete"] -
                                                       p["arms"]["no_document_coverage"]["complete"] for p in panel]),
            }
    return answer


def corrections(directory):
    baseline = {r["seed"]: r for r in runs(directory, "baseline-*.json", 6)}
    current = {r["seed"]: r for r in runs(directory, "corrections-*.json", 6)}
    assert baseline.keys() == current.keys()
    result = {}
    for i, fixture in enumerate(next(iter(baseline.values()))["rows"]):
        heads = {}
        aligned = [(baseline[seed]["rows"][i], current[seed]["rows"][i]) for seed in baseline]
        assert all(a["matched_sha256"] == b["matched_sha256"] for a, b in aligned)
        for arm in ("sequential", "one_pass_pair"):
            original = [r["arms"][arm]["joint"] / r["count"] for r, _ in aligned]
            treatment = [r["arms"][arm]["joint"] / r["count"] for _, r in aligned]
            heads[arm] = {"baseline": mean(original), "correction_training": mean(treatment),
                          "paired": paired([b - a for a, b in zip(original, treatment, strict=True)])}
        result[f"{fixture['profile']}:{fixture['depth_per_bank']}"] = heads
    return {"seeds": sorted(baseline), "tests": result}


def curriculum(directory):
    data = runs(directory, "confirm-*.json", 40)
    by_seed = defaultdict(dict)
    for row in data:
        key = row["arm"] + ("/no-clock" if row["hide_query_clock"] else "") + "/" + row["schedule"]
        assert key not in by_seed[row["seed"]]
        by_seed[row["seed"]][key] = row
    assert all(len(r) == 10 for r in by_seed.values())
    result = {}
    for panel in by_seed.values():
        assert len({tuple(x["inputs_sha256"] for x in r["test"]) for r in panel.values()}) == 1
        assert len({r["train_tokens"] for r in panel.values()}) == 1
    for name, first in next(iter(by_seed.values())).items():
        cases = [panel[name] for panel in by_seed.values()]
        runs_ = {}
        for i, reference in enumerate(first["test"]):
            rows = [run["test"][i] for run in cases]
            epoch = f"{reference['profile']}:{reference['depth_per_bank']}"
            item = {"joint": mean(r["joint"] / r["count"] for r in rows),
                    "payload_intervention": mean(r["value_follows_intervention"] / r["count"] for r in rows)}
            if reference["robustness"]:
                item["robustness"] = {kind: mean(r["robustness"][kind] / r["count"] for r in rows)
                                        for kind in reference["robustness"]}
            runs_[epoch] = item
        result[name] = {"model": first["arm"], "schedule": first["schedule"],
                        "median_train_seconds": median(r["train_seconds"] for r in cases),
                        "tokens": first["train_tokens"], "slices": runs_}
    return {"seeds": sorted(by_seed), "models": result}


def longmemeval(directory):
    data = runs(directory, "eval-?.json", 8)
    rows = flattened(data)
    assert len(rows) == data[0]["n_source"] == 500
    outputs = {"after_question_dated": sum(bool(r["dated_post_cutoff_sessions"]) for r in rows),
               "gold_excluded_by_cutoff": sum(bool(r["unavailable_gold_under_cutoff"]) for r in rows),
               "abstention_questions": sum(r["abstention_item"] for r in rows),
               "questions": len(rows), "mean_sessions": mean(r["n_sessions"] for r in rows), "categories": {}}
    for kind in ("all", *sorted({r["kind"] for r in rows})):
        subset = [r for r in rows if (kind == "all" or kind == r["kind"]) and not r["abstention_item"]]
        visible = [r for r in subset if not r["unavailable_gold_under_cutoff"]]
        detail = {"n": len(subset), "eligible_with_complete_asof_evidence": len(visible),
                  "unavailable_under_cutoff": len(subset) - len(visible), "arms": {}}
        for a in subset[0]["arms"]:
            assert all(r["arms"]["dense_asof_original"] == r["arms"]["future_echo_asof"] for r in subset)
            groups = {"published": subset, "asof_eligible": visible}
            detail["arms"][a] = {name: {"n": len(group),
                                        "echo_top5": sum(any(c.startswith("__") for c in r["arms"][a]["ranks"][:5])
                                                        for r in group),
                                        "latest_gold_top5": sum(r["latest_gold"] in r["arms"][a]["ranks"][:5] for r in group),
                                        "complete": {k: sum(r["arms"][a]["scores"][k]["all"] for r in group)
                                                     for k in group[0]["arms"][a]["scores"]},
                                        "mean_recall": {k: mean(r["arms"][a]["scores"][k]["recall"] for r in group)
                                                        for k in group[0]["arms"][a]["scores"]}}
                                   for name, group in groups.items()}
        outputs["categories"][kind] = detail
    return outputs


def timeqa_date_roles(directory):
    fit = read(directory / "fit-donor-0.json")
    assert fit["train_entities"] == 296
    panels = {}
    for label, n in (("donor", 320), ("sham", 320), ("date_collision", 320), ("human", 827)):
        pattern = "human-donor-?.json" if label == "human" else f"test-{label}-?.json"
        data = runs(directory, pattern, 4)
        assert data[0]["weights_sha256"] == fit["weights_sha256"]
        assert data[0]["training_sha256"] == fit["training_sha256"]
        joined = flattened(data)
        assert len(joined) == n
        panels[label] = {r["id"]: r for r in joined if r["eligible"]}
    assert panels["sham"].keys() == panels["date_collision"].keys()
    assert len(panels["sham"]) == 292
    out = {"fit": {k: fit[k] for k in ("train_entities", "raw_refit_max_param_delta", "loss", "weights_sha256")}}
    for pre, post in (("sham", "date_collision"), ("donor", "donor"), ("human", "human")):
        rows = panels[post].values()
        variants = ("easy", "hard") if post == "human" else ("clean", "with_noise")
        names = next(iter(rows))["retrieval"][0]
        out[post] = {"groups": len(rows), "variants": {}}
        for i, variant in enumerate(variants):
            selected = tuple((r, q) for r in rows for q in ((i,) if post == "human" else (0, 1)))
            def hit(pair, arm, panel=post):
                row, q = pair
                field = "clean" if post == "human" else variant
                return int(panels[panel][row["id"]]["retrieval"][q][arm][field])

            out[post]["variants"][variant] = {
                arm: {"n": len(selected), "support": sum(hit(pair, arm) for pair in selected),
                      "sham_support": sum(hit(pair, arm, pre) for pair in selected) if post == "date_collision" else None,
                      "noise_selected": sum(r["retrieval"][j][arm]["noise_pick"] for r, j in selected)
                                        if variant == "with_noise" else None,
                      "vs_all_years": cluster_interval(selected, lambda pair: pair[0]["entity"],
                                                        lambda pair: hit(pair, arm) - hit(pair, "all_years")),
                      "collision_vs_sham": cluster_interval(selected, lambda pair: pair[0]["entity"],
                                                             lambda pair: hit(pair, arm) - hit(pair, arm, pre))
                                           if post == "date_collision" else None}
                for arm in names
            }
    for case_id in panels["sham"]:
        for sham, coll in zip(panels["sham"][case_id]["retrieval"], panels["date_collision"][case_id]["retrieval"], strict=True):
            assert all(sham[a]["clean"] == coll[a]["clean"] for a in sham)
    return out


def archive_projection(directory, study="baseline"):
    conditions = {
        "baseline": (4, 6, 1280, 2560, 640),
        "boundary": (4, 8, 5120, 2560, 640),
        "topology": (3, 6, 256, 384, 96),
    }
    n_seeds, n_arms, n_records, n_all, n_held_family = conditions[study]
    data = runs(directory, "eval-*.json", n_seeds * n_arms)
    seeds = sorted({r["seed"] for r in data})
    assert len(seeds) == n_seeds and len({r["archive_fixture_sha256"] for r in data}) == 1
    assert {r["train_tokens"] for r in data} == {13_516_800}
    assert all(len(r["archive"]) == n_records and r["hide_query_clock"] for r in data)
    if study == "topology":
        assert all(r["archive_test_years"] == [2023, 2026] for r in data)
    runs_by_condition = defaultdict(dict)
    for run in data:
        cases = runs_by_condition[run["arm"], run["training_data"]]
        assert run["seed"] not in cases
        cases[run["seed"]] = run["archive"]
    assert len(runs_by_condition) == n_arms
    boundary = study == "boundary"
    gaps = ((0, 0), (8, 0), (0, 8), (8, 8)) if boundary else ((0, 0),)
    panels = {}
    for (arm, source), seed_cases in sorted(runs_by_condition.items()):
        assert sorted(seed_cases) == seeds
        detail = panels[f"{arm}:{source}"] = {}
        for effective, publication in gaps:
            for series in ("all", "NCON"):
                for depth in (16, 48):
                    by_seed = [[r for r in seed_cases[seed] if r["depth"] == depth and
                                (series == "all" or r["series"] == series) and
                                r.get("effective_gap_intervention", 0) == effective and
                                r.get("publication_gap_intervention", 0) == publication] for seed in seeds]
                    panel = [r for seed_panel in by_seed for r in seed_panel]
                    assert len(panel) == (n_all if series == "all" else n_held_family)
                    joint = [mean(r["joint"] for r in seed_panel) for seed_panel in by_seed]
                    key = f"{series}:{depth}:{effective}:{publication}" if boundary else f"{series}:{depth}"
                    detail[key] = {"n": len(panel), "seed_range_joint": [min(joint), max(joint)],
                                   **{metric: mean(r[metric] for r in panel) for metric in
                                      ("joint", "numeric_exact", "record", "entity", "unbound_payload")}}
    result = {"seeds": seeds, "tokens": 13_516_800, "panels": panels}
    if study == "topology":
        result["paired_release_cluster_effects"] = archive_topology_effects(runs_by_condition, seeds)
    return result


def archive_topology_effects(runs_by_condition, seeds):
    output = {}
    for arm in ("dot_without_absolute", "radial_without_absolute"):
        for depth in (16, 48):
            for reference in ("procedural_dual_boundary", "archive"):
                matched = defaultdict(list)
                for seed in seeds:
                    panels = []
                    for method in ("procedural_archive_topology", reference):
                        panels.append({(r["id"], r["period"], r["recode"]): int(r["joint"])
                                       for r in runs_by_condition[arm, method][seed] if r["depth"] == depth})
                    proposed, baseline = panels
                    assert proposed.keys() == baseline.keys() and len(proposed) == 128
                    for key, value in proposed.items():
                        quarter = int(key[0].rsplit("-", 1)[1])
                        matched[seed, quarter].append(value - baseline[key])
                quarters = sorted({quarter for _, quarter in matched})
                assert len(quarters) == 8 and len(matched) == len(seeds) * len(quarters)
                assert {len(v) for v in matched.values()} == {16}
                averages = {key: mean(v) for key, v in matched.items()}
                rng = Random(2744601)
                bootstrap = []
                for _ in range(5000):
                    draw_q = rng.choices(quarters, k=len(quarters))
                    draw_s = rng.choices(seeds, k=len(seeds))
                    bootstrap.append(mean(averages[s, q] for s in draw_s for q in draw_q))
                bootstrap.sort()
                output[f"{arm}:{depth}:vs_{reference}"] = {
                    "delta": mean(averages.values()), "bootstrap95": [bootstrap[125], bootstrap[4874]],
                    "quarter_clusters": len(quarters), "paired_seeds": len(seeds),
                }
    return output


def archive_factors(directory):
    data = runs(directory, "eval-*.json", 24)
    fields = ("shared_effective", "many_future", "unique_payloads")
    seeds = sorted({r["seed"] for r in data})
    assert len(seeds) == 3 and {r["train_tokens"] for r in data} == {13_516_800}
    assert {r["arm"] for r in data} == {"radial_without_absolute"}
    assert len({r["archive_fixture_sha256"] for r in data}) == 1
    out = {"seeds": seeds, "archive_fixture_sha256": data[0]["archive_fixture_sha256"], "depths": {}}
    for depth in (16, 48):
        cells = defaultdict(dict)
        case_keys = {}
        for run in data:
            options = tuple(int(run["archive_generation"][field]) for field in fields)
            seed = run["seed"]
            cases = [r for r in run["archive"] if r["depth"] == depth]
            selected_keys = [(r["id"], r["period"], r["recode"]) for r in cases]
            assert len(cases) == len(set(selected_keys)) == 640
            assert case_keys.setdefault(seed, selected_keys) == selected_keys
            assert seed not in cells[options]
            cells[options][seed] = mean(r["joint"] for r in cases)
        assert len(cells) == 8 and all(sorted(c) == seeds for c in cells.values())
        marginals = {}
        for i, factor in enumerate(fields):
            deltas = [mean(value[seed] - cells[key[:i] + (0,) + key[i+1:]][seed]
                           for key, value in cells.items() if key[i]) for seed in seeds]
            marginals[factor] = {"delta": mean(deltas), "seed_range": [min(deltas), max(deltas)]}
        out["depths"][depth] = {"cases_per_seed": 640, "factor_order": fields, "marginal_joints": marginals,
                                "condition_joints": {"".join(map(str, k)): mean(cells[k].values()) for k in sorted(cells)}}
    return out


def timelineqa(directory):
    data = runs(directory, "eval-?.json", 8)
    rows = flattened(data)
    sources = sorted((s for d in data for s in d["sources"]), key=lambda s: s["seed"])
    assert len(sources) == 24 and len(rows) == 576
    assert set(Counter(r["persona"] for r in rows).values()) == {24}
    assert len({r["author_generator_revision"] for r in data}) == 1
    panels = {}
    for density in ("all", "sparse", "medium", "dense"):
        selected = [r for r in rows if density == "all" or r["density"] == density]
        panels[density] = {}
        for depth in ("16", "64", "256"):
            panels[density][depth] = {}
            for arm in ("dense", "reported_date", "authoritative_date"):
                d = panels[density][depth][arm] = {}
                for panel in ("true_date", "misdated_noise"):
                    d[panel] = totals([r["scores"][depth][panel][arm] for r in selected],
                                      ("support", "support_top2", "wrong_day_top1", "donor_top1"))
                d["effect_support"] = cluster_interval(selected, lambda r: r["persona"],
                                                        lambda r: (r["scores"][depth]["misdated_noise"][arm]["support"] -
                                                                   r["scores"][depth]["true_date"][arm]["support"]))
    return {"generator": data[0]["author_generator_revision"], "sources": sources, "panels": panels}


def unseentimeqa_probe(directory):
    data = runs(directory, "eval-?.json", 4)
    rows = flattened(data)
    assert len(rows) == 600
    for r in rows:
        panels = r["results"]
        assert all(p["compatible_time_and_scope"]["support"] for p in panels.values())
        assert panels["clean"]["compatible_time"]["support"] and panels["sham"]["compatible_time"]["support"]
    output = {"fixture_sha256": data[0]["fixture_sha256"], "all": {}, "by_question_type": {}}
    for key, group in (("all", rows), ("absolute", [r for r in rows if r["kind"] == 1]),
                       ("relative", [r for r in rows if r["kind"] == 2])):
        assert len(group) == (600 if key == "all" else 300)
        dest = output["all"] if key == "all" else output["by_question_type"].setdefault(key, {})
        dest["n"] = len(group)
        dest["initial_worlds"] = len({r["scope"] for r in group})
        for policy in group[0]["results"]["clean"]:
            panels = {name: totals([r["results"][name][policy] for r in group], ("support", "support_top2", "decoy"))
                      for name in ("clean", "sham", "collision")}
            dest[policy] = panels | {
                "collision_vs_sham": cluster_interval(group, lambda r: r["scope"],
                                                      lambda r: r["results"]["collision"][policy]["support"] -
                                                                r["results"]["sham"][policy]["support"])}
    return output


ROUTES = {
    "ablation": ablation,
    "link_noise": lambda d: link_noise(d, False),
    "link_noise_training": lambda d: link_noise(d, True),
    "news": news,
    "tempreason": tempreason,
    "qasper": qasper,
    "timeqa_collision": timeqa_collision,
    "vintage": vintage,
    "procedural": procedural,
    "decoder_pilot": lambda d: decoder(d, "pilot"),
    "decoder_confirm": lambda d: decoder(d, "confirm"),
    "musique_multihop": musique_multihop,
    "timeqa_human": timeqa_human,
    "vintage_global": vintage_global,
    "vintage_stream": vintage_stream,
    "vintage_aliases": vintage_aliases,
    "qasper_controls": qasper_controls,
    "nolima": nolima,
    "matched_pools": matched_pools,
    "corrections": corrections,
    "curriculum": curriculum,
    "longmemeval": longmemeval,
    "timeqa_date_roles": timeqa_date_roles,
    "archive_projection": archive_projection,
    "archive_boundary": lambda directory: archive_projection(directory, "boundary"),
    "archive_topology": lambda directory: archive_projection(directory, "topology"),
    "archive_factors": archive_factors,
    "timelineqa": timelineqa,
    "unseentimeqa_probe": unseentimeqa_probe,
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", choices=ROUTES)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    output = ROUTES[a.study](a.directory)
    a.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
