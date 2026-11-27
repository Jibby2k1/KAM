#!/usr/bin/env python3
"""Measure yes/no evidence use with frozen QASPER paragraphs and reader weights."""

import argparse
import gzip
import json
from pathlib import Path

from audit_kam_natural_data import digest

ARMS = ("no_evidence", "paper_dense", "same_paper_sham", "gold")


def read(a, fixture):
    import torch

    from run_kam_hotpot_pool_scaling import Reader
    from run_kam_musique_bindings import BASE_REVISION, BASE_WEIGHT
    from run_kam_natural_feasibility import Encoder, batches, normalized_answer

    torch.set_float32_matmul_precision("highest")
    model_dir = a.assets / f"reader_base_{BASE_REVISION}"
    assert digest(model_dir / "model.safetensors") == BASE_WEIGHT
    reader = Reader(model_dir)
    tokenizer = reader.tokenizer
    encoder = Encoder(a.assets / "encoder", 128)
    panel = fixture["rows"][a.shard::a.shards]
    articles = fixture["articles"]

    def ids(terms):
        forms = [tokenizer(s, add_special_tokens=False).input_ids for s in terms]
        assert all(len(p) == 1 for p in forms)
        return sorted({p[0] for p in forms})

    yes, no = ids(("Yes", "yes")), ids(("No", "no"))
    assert set(yes).isdisjoint(no)

    def prepare_prompt(question, texts):
        budget = 384 // max(len(texts), 1)
        capped = [tokenizer.decode(tokenizer(text, add_special_tokens=False, max_length=budget,
                                              truncation=True).input_ids, skip_special_tokens=True) for text in texts]
        prompt = f"Answer with exactly Yes or No. Question: {question} Evidence: {' '.join(capped)}"
        assert len(tokenizer(prompt).input_ids) <= 512
        return prompt

    with torch.inference_mode():
        texts = {p: [f"{h}. {body}" for h, body in articles[p]] for p in {q["paper"] for q in panel}}
        keys = {p: encoder(rows, "passage") for p, rows in texts.items()}
        questions = encoder([q["question"] for q in panel], "query")
        prompts, output = {}, []
        for q, vector in zip(panel, questions, strict=True):
            p = q["paper"]
            order = (keys[p] @ vector).argsort(descending=True, stable=True)[:2].tolist()
            modes = dict(no_evidence=[], paper_dense=order, same_paper_sham=q["sham"], gold=q["gold"])
            arms = {}
            for mode, selected in modes.items():
                prompt = prepare_prompt(q["question"], [texts[p][i] for i in selected])
                r = {"slots": selected, "tokens": len(tokenizer(prompt).input_ids)}
                assert len(selected) == len(set(selected))
                prompts.setdefault(prompt, []).append(r)
                arms[mode] = r
            output.append({"id": q["id"], "arms": arms})

        decoded = reader.answers(list(prompts))
        for block in batches(list(prompts), 32):
            inputs = tokenizer(block, padding=True, return_tensors="pt").to("cuda")
            labels = torch.full((len(block), 1), reader.model.config.decoder_start_token_id, dtype=torch.long, device="cuda")
            logits = reader.model(**inputs, decoder_input_ids=labels, use_cache=False).logits[:, 0]
            relative = logits[:, yes].logsumexp(-1) - logits[:, no].logsumexp(-1)
            for prompt, margin in zip(block, relative.tolist(), strict=True):
                for record in prompts[prompt]:
                    record.update(yes=margin > 0, yes_minus_no_logit=margin)
        for name, response in zip(prompts, decoded, strict=True):
            words = normalized_answer(response)
            recognized = {"yes": True, "no": False}.get(" ".join(words))
            for record in prompts[name]:
                record["free_generation"] = recognized

    result = {"fixture_sha256": digest(a.fixture), "reader_weights_sha256": BASE_WEIGHT, "reader_revision": BASE_REVISION,
              "device": torch.cuda.get_device_name(), "codes": {"yes": yes, "no": no},
              "shard": a.shard, "shards": a.shards, "rows": output}
    a.output.write_bytes(gzip.compress(json.dumps(result, separators=(",", ":")).encode(), mtime=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "fixture", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    read(a, json.loads(gzip.decompress(a.fixture.read_bytes())))


if __name__ == "__main__":
    main()
