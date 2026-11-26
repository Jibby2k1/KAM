#!/usr/bin/env python3
"""Evaluate archived-record routing under revision noise and alias transfer."""

import argparse
from pathlib import Path
import re
from time import perf_counter

import torch
from torch import nn
from torch.nn.functional import cross_entropy

from audit_kam_natural_data import digest
from prepare_kam_vintage_revisions import ORDINAL, SERIES, format_quarter, period
from run_kam_hotpot_independent import read, save
from run_kam_natural_feasibility import Encoder

CANONICAL = re.compile(r"^What was archived for (.+) observed in (\d{4}) Q([1-4]) as of (\d{4}) Q([1-4])\?$")
PARAPHRASE = re.compile(
    rf"^Using what was known by the ({'|'.join(ORDINAL)}) quarter of (\d{{4}}), "
    rf"what figure was listed for (.+) during the ({'|'.join(ORDINAL)}) quarter of (\d{{4}})\?$"
)
NAMES = tuple(SERIES)
ARMS = ("dense", "dense_calibrated", "learned", "semantic_guard", "lexical_guard", "exact_oracle")


def read_query(text):
    if match := CANONICAL.fullmatch(text):
        name, year, quarter, cutoff_year, cutoff_quarter = match.groups()
        return name, period(year, quarter), period(cutoff_year, cutoff_quarter)
    if match := PARAPHRASE.fullmatch(text):
        cutoff_quarter, cutoff_year, name, q, year = match.groups()
        return name, period(year, 1 + ORDINAL.index(q)), period(cutoff_year, 1 + ORDINAL.index(cutoff_quarter))
    return None


def document(record):
    canonical = SERIES[record["series"]][0]
    observation = format_quarter(record["observation"], "canonical")
    vintage = format_quarter(record["vintage"], "canonical")
    return f"{canonical}. Observation {observation}. Archived vintage {vintage}. Value {record['value']}."


@torch.no_grad()
def embed(encoder, groups):
    cases = [(group, q) for group in groups for q in group["questions"]]
    rows = [tuple(doc["series"] for doc in group["candidates"]) for group, _ in cases]
    strings = [[document(row) for row in group["candidates"]] for group, _ in cases]
    unique = list(dict.fromkeys(value for docs in strings for value in docs))
    ids = {text: i for i, text in enumerate(unique)}
    bank = encoder(unique, "passage")
    doc_rows = bank[torch.tensor([[ids[text] for text in docs] for docs in strings], device="cuda")]
    qemb = encoder([q["text"] for _, q in cases], "query")
    details = [read_query(q["text"]) for _, q in cases]
    assert all(parsed == (q["name"], g["observation"], q["cutoff"])
               for parsed, (g, q) in zip(details, cases, strict=True))
    names = encoder([item[0] for item in details], "query")
    labels = encoder([name for variants in SERIES.values() for name in variants[:2]], "passage")
    name_scores = (names @ labels.T).reshape(len(cases), len(NAMES), 2).amax(-1)
    series_ids = torch.tensor([[NAMES.index(series) for series in row] for row in rows], device="cuda")
    semantic = name_scores.gather(1, series_ids)
    affinity = torch.einsum("bd,bnd->bn", qemb, doc_rows)
    offsets = [
        [[r["observation"] - obs, r["vintage"] - cutoff] for r in g["candidates"]]
        for (g, _), (_, obs, cutoff) in zip(cases, details, strict=True)
    ]
    deltas = torch.tensor(offsets, device="cuda")
    obs, vintage = deltas.unbind(-1)
    x = torch.stack((affinity, semantic, obs.eq(0), -obs.abs().float() / 8,
                     vintage.le(0), vintage.float() / 20, -vintage.abs().float() / 20,
                     name_scores.argmax(-1)[:, None].eq(series_ids)), -1)
    masks = torch.zeros((len(cases), 48), device="cuda", dtype=torch.bool)
    for i, (g, _) in enumerate(cases):
        masks[i, g["scenarios"]["clean"]] = True
    exact_labels = torch.tensor([q["gold"] if q["gold"] is not None else 48 for _, q in cases], device="cuda")
    return cases, x, masks, exact_labels, name_scores


class Ranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 1))
        self.null = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, mask):
        scores = self.head(x).squeeze(-1).masked_fill(~mask, -torch.inf)
        return torch.cat((scores, self.null.expand(len(x), 1)), 1)


def fit(data, cfg):
    cases, x, clean_mask, y, _ = data
    # Reserve entire observation IDs for calibration before fitting any weights.
    calibration = torch.tensor([int(digest_text(g["id"])[:8], 16) % 4 == 0 for g, _ in cases], device="cuda")
    indices = (~calibration).nonzero().flatten()
    assert bool(calibration.any() and (~calibration).any())
    model = Ranker().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, fused=True)
    rng = torch.Generator(device="cuda").manual_seed(cfg.seed + 62378)
    for _ in range(44):
        for block in indices[torch.randperm(len(indices), generator=rng, device="cuda")].split(64):
            logits = torch.cat((model(x[block], clean_mask[block]),
                                model(x[block], torch.ones_like(clean_mask[block]))))
            error = cross_entropy(logits, y[block].repeat(2))
            optimizer.zero_grad(set_to_none=True)
            error.backward()
            optimizer.step()
    assert torch.isfinite(error)
    max_score, choice = x[calibration, :, 0].max(-1)
    null = y[calibration].eq(48)
    possibilities = [value / 1000 for value in range(500, 981, 2)]
    threshold = max(possibilities, key=lambda t: (
        float((max_score.lt(t) & null).sum()) / max(int(null.sum()), 1)
        + float((max_score.ge(t) & ~null).sum()) / max(int((~null).sum()), 1), t))
    return model.eval(), threshold, {"training_questions": len(indices), "calibration_questions": int(calibration.sum()),
                                     "training_groups": len({cases[i][0]["id"] for i in indices.tolist()}),
                                     "calibration_groups": len({cases[i][0]["id"] for i in calibration.nonzero().flatten().tolist()}),
                                     "loss": float(error.detach())}


def digest_text(text):
    from hashlib import sha256
    return sha256(text.encode()).hexdigest()


def guard(name, obs, cutoff, docs, pool):
    eligible = [idx for idx in pool if docs[idx]["series"] == name
                and docs[idx]["observation"] == obs and docs[idx]["vintage"] <= cutoff]
    return max(eligible, key=lambda idx: docs[idx]["vintage"], default=48)


def outcome(g, q, idx):
    doc = g["candidates"][idx] if idx != 48 else None
    return {"picked": idx if idx != 48 else None, "record_exact": idx == (q["gold"] if q["gold"] is not None else 48),
            "value_exact": (doc["value"] if doc is not None else None) == q["value"],
            "abstained": doc is None, "future": doc is not None and doc["vintage"] > q["cutoff"],
            "wrong_series": doc is not None and doc["series"] != g["series"],
            "wrong_period": doc is not None and doc["observation"] != g["observation"]}


@torch.inference_mode()
def evaluate(data, model, threshold):
    cases, x, masks, _, name_scores = data
    dictionary = {variant.casefold(): code for code, variants in SERIES.items() for variant in variants[:2]}
    routes = {}
    for scenario, mask in (("clean", masks), ("noise", torch.ones_like(masks))):
        score, dense = x[:, :, 0].masked_fill(~mask, -torch.inf).max(-1)
        learned = model(x, mask).argmax(-1)
        calibrated = dense.masked_fill(score < threshold, 48)
        routes[scenario] = (dense.tolist(), calibrated.tolist(), learned.tolist())
    sem = name_scores.argmax(-1).tolist()
    out = []
    for i, (g, q) in enumerate(cases):
        name, obs, cutoff = read_query(q["text"])
        referenced = {}
        for scenario, methods in routes.items():
            picks = {arm: slots[i] for arm, slots in zip(ARMS[:3], methods, strict=True)}
            for arm, code in (("semantic_guard", NAMES[sem[i]]),
                              ("lexical_guard", dictionary.get(name.casefold())), ("exact_oracle", g["series"])):
                picks[arm] = guard(code, obs, cutoff, g["candidates"], g["scenarios"][scenario])
            reference = outcome(g, q, picks["exact_oracle"])
            assert reference["record_exact"] and reference["value_exact"]
            referenced[scenario] = {arm: outcome(g, q, slot) for arm, slot in picks.items()}
        out.append({"id": g["id"], "period": q["period"], "series": g["series"], "style": q["style"],
                    "known": q["gold"] is not None, "semantic_series": NAMES[sem[i]], "scenarios": referenced})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fit", "eval"))
    for name in ("assets", "fixture", "weights", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--seed", type=int, default=201849)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    cfg = p.parse_args()
    assert 0 <= cfg.shard < cfg.shards
    start = perf_counter()
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("highest")
    data = read(cfg.fixture)
    bundle = {"fixture_sha256": digest(cfg.fixture), "models": read(cfg.assets / "revisions.json"),
              "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name()}
    enc = Encoder(cfg.assets / "encoder", 64)
    split = "train" if cfg.mode == "fit" else "test"
    rows = data["rows"][split][cfg.shard::cfg.shards]
    values = embed(enc, rows)
    if cfg.mode == "fit":
        model, threshold, training = fit(values, cfg)
        torch.save({"weights": model.cpu().state_dict(), "fixture_sha256": bundle["fixture_sha256"],
                    "models": bundle["models"], "threshold": threshold}, cfg.weights)
        output = {"training": training, "threshold": threshold}
    else:
        stored = torch.load(cfg.weights, map_location="cpu", weights_only=True)
        assert stored["fixture_sha256"] == bundle["fixture_sha256"] and stored["models"] == bundle["models"]
        model = Ranker().cuda().eval()
        model.load_state_dict(stored["weights"])
        output = {"rows": evaluate(values, model, stored["threshold"]), "threshold": stored["threshold"]}
    save(cfg.output, {**bundle, **output, "weight_sha256": digest(cfg.weights),
                      "elapsed_seconds": perf_counter() - start})


if __name__ == "__main__":
    main()
