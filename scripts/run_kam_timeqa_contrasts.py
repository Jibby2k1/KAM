#!/usr/bin/env python3
"""Train and evaluate date-sensitive evidence routing on paired TimeQA periods."""

import argparse
from itertools import chain
from pathlib import Path
import re
from time import perf_counter
from urllib.parse import unquote

import torch
from torch import nn
from torch.nn.functional import log_softmax

from audit_kam_natural_data import digest, words
from run_kam_hotpot_independent import read, save
from run_kam_hotpot_pool_scaling import Reader
from run_kam_natural_feasibility import Encoder, measure_answer

YEAR = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
ARMS = ("dense_no_date", "dense", "learned_no_dates", "learned_dates")
SCENARIOS = ("clean", "matched_noise")
SIMPLE_FEATURES = ("affinity", "no_date_affinity", "entity_in_text", "lexical_overlap")
DATE_FEATURES = ("year_present", "within_interval", "nearest_year", "year_fraction")


def with_date_collisions(rows, variant):
    out = []
    for row in rows:
        revised = []
        for i, q in enumerate(row["questions"]):
            opposite = row["questions"][1 - i]
            answer = " " + " ".join(words(q["answer"])) + " "
            texts = [row["context"][idx][2] for idx in opposite["gold"]]
            text = next((t for t in texts if answer not in " " + " ".join(words(t)) + " "), None)
            if text is None:
                break
            dates = q["period"]
            if variant == "sham":
                dates = [re.sub(r"\d{4}$", lambda m: str(int(m[0]) - 37), bound) for bound in dates]
            revised.append({"entity": row["entity"], "period": dates,
                            "text": f"Archive review period from {dates[0]} to {dates[1]}. Historical extract: {text}"})
        eligible = len(revised) == 2
        noise = revised + row["noise"][2:] if eligible else row["noise"]
        out.append({**row, "noise": noise, "date_collision_eligible": eligible})
    return out


def data_matrix(encoder, rows):
    names, lengths, flattened = [], [], []
    for row in rows:
        title = row["title"]
        text = [chunk[2] for chunk in row["context"]] + [noise["text"] for noise in row["noise"]]
        flattened.extend(text)
        lengths.append(len(text))
        donor_titles = [unquote(n["entity"].removeprefix("/wiki/")).replace("_", " ") for n in row["noise"]]
        names.extend([title] * len(row["context"]) + donor_titles)
    passage = encoder([(f"{name}. {text}" if name else text) for name, text in zip(names, flattened, strict=True)],
                      "passage")
    dated = encoder([question["query"] for row in rows for question in row["questions"]], "query")
    undated = encoder([question["query_without_date"] for row in rows for question in row["questions"]], "query")
    matrices, contexts = [], []
    offset = 0
    for i, (row, bank) in enumerate(zip(rows, passage.split(lengths), strict=True)):
        docs = flattened[offset:offset + len(bank)]
        titles = names[offset:offset + len(bank)]
        offset += len(bank)
        contexts.append({"context": [(title, [doc]) for title, doc in zip(titles, docs, strict=True)]})
        entity = " ".join(words(row["title"].split(" (", 1)[0]))
        wordsets = [set(words(doc)) for doc in docs]
        years = [set(map(int, YEAR.findall(doc))) for doc in docs]
        entity_match = [float(f" {entity} " in f" {' '.join(words(doc))} ") for doc in docs]
        for j, question in enumerate(row["questions"]):
            q = dated[2 * i + j] @ bank.T
            q0 = undated[2 * i + j] @ bank.T
            terms = set(words(question["query_without_date"])) - {"which", "a", "who", "the", "a", "of", "for", "was", "did"}
            lo, hi = question["lower"] // 12, question["upper"] // 12
            supplemental = []
            for tokens, doc_years, has_entity in zip(wordsets, years, entity_match, strict=True):
                overlap = len(terms & tokens) / max(len(terms), 1)
                within = sum(lo <= y <= hi for y in doc_years)
                distance = min((max(0, lo - y, y - hi) for y in doc_years), default=100)
                supplemental.append((has_entity, overlap, bool(doc_years), bool(within),
                                     max(0.0, 1 - distance / 30), within / max(len(doc_years), 1)))
            matrices.append(torch.cat((q[:, None], q0[:, None], torch.tensor(supplemental, device="cuda")), dim=1))
    return matrices, contexts


def transform(matrix, width):
    if width == len(SIMPLE_FEATURES):
        # Replace the dated embedding score with the undated score as well.
        matrix = matrix[..., :width].clone()
        matrix[..., 0] = matrix[..., 1]
    return matrix


def padded(rows, matrices):
    size = max(len(matrix) for matrix in matrices)
    x = torch.zeros((len(matrices) * 2, size, len(SIMPLE_FEATURES) + len(DATE_FEATURES)), device="cuda")
    active = torch.zeros((len(x), size), dtype=torch.bool, device="cuda")
    gold = active.clone()
    questions = [(len(row["context"]), q["gold"]) for row in rows for q in row["questions"]]
    for i, (matrix, (clean, winners)) in enumerate(zip(matrices, questions, strict=True)):
        for j, count in enumerate((clean, len(matrix))):
            k = 2 * i + j
            x[k, :count] = matrix[:count]
            active[k, :count] = True
            gold[k, winners] = True
    return x, active, gold


def fit(rows, matrices):
    x, active, gold = padded(rows, matrices)
    heads = {}
    for width, arm in ((len(SIMPLE_FEATURES), "learned_no_dates"),
                       (len(SIMPLE_FEATURES) + len(DATE_FEATURES), "learned_dates")):
        torch.manual_seed(852761)
        model = nn.Sequential(nn.Linear(width, 16), nn.SiLU(), nn.Linear(16, 1)).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, fused=True)
        rng = torch.Generator(device="cuda").manual_seed(61529)
        for _ in range(30):
            for batch in torch.randperm(len(x), generator=rng, device="cuda").split(64):
                logits = model(transform(x[batch], width)).squeeze(-1)
                scores = log_softmax(logits.masked_fill(~active[batch], -torch.inf), -1)
                loss = -scores.masked_fill(~gold[batch], -torch.inf).logsumexp(-1).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            assert torch.isfinite(loss)
        heads[arm] = model.cpu().state_dict()
    return heads


@torch.inference_mode()
def evaluate(rows, matrices, contexts, state, reader):
    models = {}
    for name, weights in state.items():
        width = weights["0.weight"].shape[1]
        model = nn.Sequential(nn.Linear(width, 16), nn.SiLU(), nn.Linear(16, 1)).cuda().eval()
        model.load_state_dict(weights)
        models[name] = width, model
    records, requests = [], []
    for i, (row, context) in enumerate(zip(rows, contexts, strict=True)):
        questions = []
        for j, question in enumerate(row["questions"]):
            matrix = matrices[2 * i + j]
            all_logits = {"dense_no_date": matrix[:, 1], "dense": matrix[:, 0]}
            all_logits.update({name: model(transform(matrix, width)).flatten()
                               for name, (width, model) in models.items()})
            gold = set(question["gold"])
            other_gold = set(row["questions"][1 - j]["gold"])
            other_answer = row["questions"][1 - j]["answer"]
            scenarios = {}
            for scenario, cap in zip(SCENARIOS, (len(row["context"]), len(matrix)), strict=True):
                arms = {}
                for arm, logits in all_logits.items():
                    selection = int(logits[:cap].argmax())
                    record = {"selected": selection, "evidence_hit": selection in gold,
                              "opposite_period": selection in other_gold, "distractor": selection >= len(row["context"])}
                    arms[arm] = record
                    prompt = reader.prompt({"question": question["query"], **context}, [selection])
                    requests.append((prompt, record, question["answer"], other_answer))
                scenarios[scenario] = arms
            oracle = {}
            prompt = reader.prompt({"question": question["query"], **context}, [min(gold)])
            requests.append((prompt, oracle, question["answer"], other_answer))
            questions.append({"period": question["period"], "gold": sorted(gold), "oracle": oracle, "scenarios": scenarios})
        records.append({"id": row["id"], "kind": row["type"], "clean_depth": len(row["context"]),
                        "date_collision_eligible": row.get("date_collision_eligible", False), "questions": questions})
    prompts = list(dict.fromkeys(p for p, _, _, _ in requests))
    decoded = dict(zip(prompts, reader.answers(prompts), strict=True))
    for prompt, record, answer, opposite in requests:
        record.update(measure_answer(decoded[prompt], answer))
        record["opposite_answer_em"] = measure_answer(decoded[prompt], opposite)["em"]
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fit", "eval"))
    for name in ("fixture", "assets", "weights", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--noise-variant", choices=("donor", "sham", "date_collision"), default="donor")
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    start = perf_counter()
    torch.set_float32_matmul_precision("highest")
    fixture = read(a.fixture)
    meta = {"fixture_sha256": digest(a.fixture), "models": read(a.assets / "revisions.json"),
            "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__)}
    rows = fixture["rows"]["dev" if a.mode == "fit" else "test"][a.shard::a.shards]
    if a.noise_variant != "donor":
        if a.mode != "eval":
            p.error("--noise-variant applies only to held-out evaluation")
        rows = with_date_collisions(rows, a.noise_variant)
    encoder = Encoder(a.assets / "encoder", 64)
    matrices, contexts = data_matrix(encoder, rows)
    if a.mode == "fit":
        heads = fit(rows, matrices)
        torch.save({"heads": heads, "fixture_sha256": meta["fixture_sha256"],
                    "models": meta["models"]}, a.weights)
        records = [{"id": row["id"], "n_clean": len(row["context"])} for row in rows]
    else:
        fitted = torch.load(a.weights, map_location="cpu", weights_only=True)
        assert fitted["fixture_sha256"] == meta["fixture_sha256"] and fitted["models"] == meta["models"]
        records = evaluate(rows, matrices, contexts, fitted["heads"], Reader(a.assets / "reader"))
    save(a.output, {**meta, "weights_sha256": digest(a.weights), "noise_variant": a.noise_variant,
                    "elapsed_seconds": perf_counter() - start, "rows": records})


if __name__ == "__main__":
    main()
