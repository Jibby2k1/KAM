#!/usr/bin/env python3
"""Build exact-length, different-topic controls for the archived TabMWP tables."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from kam_adapter_preservation import LLAMA_ID, digest, model_identity
from kam_math_kl_protocol import normalized, write_json
from kam_tabmwp_archive_noise import (
    SOURCE_HASHES, UNRELATED_PROTOCOL, eligible_table, encode_case, hash_text, load_panel, relation,
)


def numeric_shape(table):
    lines = table.splitlines()
    return len(lines), len(lines[0].split("|"))


def table_key(title, text):
    return normalized(title), normalized(text)


def archive_width(tokenizer, title, table):
    return len(tokenizer.encode(f"Archived table: {title}\n{table}\n\n", add_special_tokens=False))


def prepare(root, original, dev, output, reserve=None):
    from transformers import AutoTokenizer

    assert model_identity(root)[0] == LLAMA_ID
    token_file = root / "model" / "tokenizer.json"
    tokenizer_sha = digest(token_file)
    fingerprints = {name: fingerprint for fingerprint, name in (
        line.split() for line in (root / "model-sha256.txt").read_text().splitlines()
    )}
    assert tokenizer_sha == fingerprints["tokenizer.json"]
    tokenizer = AutoTokenizer.from_pretrained(root / "model", local_files_only=True)
    rows, manifest_sha = load_panel(original)
    assert digest(dev) == SOURCE_HASHES["dev"]
    used = {table_key(row["title"], row[field]) for row in rows for field in ("current", "archived")}
    reserved = json.loads(reserve.read_text())["cases"] if reserve else []
    heldout_donors = {r[field] for r in reserved for field in ("donor", "control_donor")}
    donors = defaultdict(list)
    for ident, donor in json.loads(dev.read_text()).items():
        if ident in heldout_donors or not eligible_table(donor):
            continue
        title, text = donor["table_title"], donor["table"]
        identity = table_key(title, text)
        if identity in used:
            continue
        donor_key = (donor["grade"], numeric_shape(text), archive_width(tokenizer, title, text))
        donors[donor_key].append((ident, donor, identity, relation(donor)))
    controls, grade_counts = [], Counter()
    output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        width = archive_width(tokenizer, row["title"], row["archived"])
        key = (row["grade"], numeric_shape(row["archived"]), width)
        title, header = relation({"table_title": row["title"], "table": row["archived"]})
        options = [choice for choice in donors[key]
                   if choice[2] not in used and choice[3][1] != header and set(title).isdisjoint(choice[3][0])]
        if not options:
            continue
        ident, donor, identity, _ = min(options, key=lambda pair: hash_text(f"{UNRELATED_PROTOCOL}:{row['id']}:{pair[0]}"))
        used.add(identity)
        candidate = {**row, "control_donor": ident, "archived_title": donor["table_title"], "archived": donor["table"]}
        previous = encode_case(tokenizer, row)
        current = encode_case(tokenizer, candidate)
        assert previous["current_only"] == current["current_only"]
        assert len(previous["archive_last"]) == len(current["archive_last"])
        controls.append(candidate)
        grade_counts[row["grade"]] += 1
    fixture = output / "unrelated-panel.json"
    write_json(fixture, controls)
    write_json(output / "unrelated-manifest.json", {
        "protocol": UNRELATED_PROTOCOL, "original_manifest_sha256": manifest_sha, "dev_sha256": digest(dev),
        "tokenizer_sha256": tokenizer_sha, "panel_sha256": digest(fixture), "n": len(controls),
        "n_by_grade": grade_counts, "eligible_original_n": len(rows),
        "cases": [{k: r[k] for k in ("id", "donor", "control_donor", "grade", "family")} for r in controls],
        **({"reserved_manifest_sha256": digest(reserve)} if reserve else {}),
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "original", "dev", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--reserve", type=Path)
    args = cli.parse_args()
    prepare(args.root, args.original, args.dev, args.output, args.reserve)
