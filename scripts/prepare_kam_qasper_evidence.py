#!/usr/bin/env python3
"""Prepare exact paragraph evidence and section-matched distractors from QASPER."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import pyarrow.parquet as pq

from audit_kam_natural_data import digest, stable, words


def norm(text):
    return " ".join(words(text))


def answer_text(annotation):
    if annotation["yes_no"] is not None:
        return "Yes" if annotation["yes_no"] else "No"
    if annotation["extractive_spans"]:
        return ", ".join(t.strip() for t in annotation["extractive_spans"])
    return annotation["free_form_answer"]


def parse(papers):
    counts, cases = Counter(), {}
    donor_by_section = defaultdict(list)
    for p in papers:
        name = p["title"]
        sections = ["Abstract", *p["full_text"]["section_name"]]
        bodies = [[p["abstract"]], *p["full_text"]["paragraphs"]]
        context, index, headings = [], defaultdict(list), []
        package = []
        for heading, sections_text in zip(sections, bodies, strict=True):
            heading = heading or "Body"
            for text in sections_text:
                if not norm(text):
                    continue
                context.append((f"{name} — {heading}", [text]))
                key = norm(text)
                index[key].append(len(context) - 1)
                surface = norm(heading.split(" ::: ")[-1])
                headings.append(surface)
                donor_by_section[surface].append((p["id"], context[-1]))
        qs = p["qas"]
        for question, q_id, answers in zip(qs["question"], qs["question_id"], qs["answers"], strict=True):
            counts["raw_questions"] += 1
            all_annotations = answers["answer"]
            gold, answers = set(), set()
            for a in all_annotations:
                if a["unanswerable"]:
                    continue
                response = answer_text(a)
                if response.strip():
                    answers.add(response.strip())
                slots = [index[norm(text)] for text in a["evidence"]]
                if slots and all(len(x) == 1 for x in slots):
                    gold.add(tuple(sorted({x[0] for x in slots})))
            if not gold or not answers:
                counts["no_mapped_answerable_evidence"] += 1
                continue
            cardinality = min(map(len, gold))
            counts[f"mapped_minimum_{min(cardinality, 3)}"] += 1
            package.append({"id": q_id, "paper": p["id"], "question": f"In the paper {name}, {question.strip()}",
                            "answers": sorted(answers), "gold_sets": sorted(gold),
                            "kind": ("one" if cardinality == 1 else "two" if cardinality == 2 else "three_plus"),
                            "annotators": len(all_annotations)})
        if package:
            cases[p["id"]] = (context, headings, package)
    return cases, donor_by_section, counts


def build(source, cap):
    papers = pq.read_table(source).to_pylist()
    cases, donor_by_section, counts = parse(papers)
    output = []
    priority = {"two": 0, "three_plus": 1, "one": 2}
    for paper, (context, headings, questions) in sorted(cases.items(), key=lambda item: stable("kam-qasper-v1:" + item[0])):
        if len(context) < 3:
            counts["small_paper"] += 1
            continue
        # Prefer the rarer multi-paragraph slots, with no model-based sampling.
        q = min(questions, key=lambda row: (priority[row["kind"]], stable(row["id"])))
        gold_slots = {i for group in q["gold_sets"] for i in group}
        focus = sorted({headings[s] for s in gold_slots}, key=lambda h: stable(q["id"] + h))
        search = [*focus, *(x for x in ("introduction", "experiments", "results", "abstract") if x not in focus)]
        used, distractors = {paper}, []
        source_text = {norm(row[1][0]) for row in context}
        for section in search:
            group = sorted(donor_by_section[section], key=lambda item: stable(q["id"] + item[0] + item[1][1][0][:140]))
            tranche = 0
            for donor, para in group:
                text = norm(para[1][0])
                if donor in used or text in source_text:
                    continue
                used.add(donor)
                source_text.add(text)
                distractors.append(para)
                tranche += 1
                if tranche == 4 or len(distractors) == 16:
                    break
            if len(distractors) == 16:
                break
        if len(distractors) != 16:
            counts["insufficient_donors"] += 1
            continue
        output.append({**q, "context": context, "noise": distractors})
        if len(output) == cap:
            break
    counts["sampled_papers"] = len(output)
    counts.update({f"sampled_{kind}": count for kind, count in Counter(r["kind"] for r in output).items()})
    assert len({r["paper"] for r in output}) == len(output)
    return {"source_sha256": digest(source), "source_papers": len(papers),
            "recruitment": dict(counts), "rows": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cap", type=int, default=384)
    a = p.parse_args()
    data = build(a.source, a.cap)
    a.output.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(data["recruitment"])


if __name__ == "__main__":
    main()
