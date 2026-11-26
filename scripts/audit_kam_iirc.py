#!/usr/bin/env python3
"""Audit evidence and visible hyperlink pools in the authors' IIRC release."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
import re
import tarfile


def title_key(title):
    return title.strip().replace("_", " ").lower()


def article_text(article):
    return re.sub(r"<[^>]*>", "", article)


def contexts(path, requested, text_keys):
    import ijson

    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.isfile() and Path(member.name).name == "context_articles.json":
                with archive.extractfile(member) as stream:
                    return {k: article_text(v) if k in text_keys else None
                            for k, v in ijson.kvitems(stream, "") if k in requested}
    raise ValueError("context_articles.json is missing from the source archive")


def audit(passages, articles):
    counts, answer_types, gold_sizes, link_counts = (Counter() for _ in range(4))
    ids, sources, labels = set(), set(), set()
    for p in passages:
        sources.add(title_key(p["title"]))
        links = {title_key(link["target"]) for link in p["links"]}
        assert all(0 <= start < end <= len(p["text"]) for start, end in (l["indices"] for l in p["links"]))
        available = {title_key(link["target"]) for link in p["links"]
                     if link["target"] in articles or link["target"].lower() in articles}
        link_counts[len(links)] += 1
        for q in p["questions"]:
            assert q["qid"] not in ids
            ids.add(q["qid"])
            kind = q["answer"]["type"]
            answer_types[kind] += 1
            if kind == "none":
                counts["unanswerable_with_context_annotation"] += bool(q["context"])
                continue
            supporting = {title_key(span["passage"]) for span in q["context"] if span["passage"] != "main"}
            writer_links = {title_key(t) for t in q["question_links"]}
            labels.update(supporting)
            gold_sizes[len(supporting)] += 1
            counts["answerable"] += 1
            counts["has_evidence_annotation"] += bool(q["context"])
            counts["uses_main_passage"] += any(span["passage"] == "main" for span in q["context"])
            counts["all_gold_visible_in_links"] += supporting <= links
            counts["all_gold_available_in_corpus_links"] += supporting <= available
            counts["writer_links_equal_gold_context_titles"] += writer_links == supporting
            counts["all_gold_in_writer_links"] += supporting <= writer_links
            if len(supporting) == 2:
                counts["two_link_gold_visible_with_distractors"] += supporting <= available and len(available) > 2
            for span in q["context"]:
                passage = span["passage"]
                source = p["text"] if passage == "main" else articles.get(passage, articles.get(passage.lower()))
                counts["main_spans" if passage == "main" else "linked_spans"] += 1
                if source is None:
                    counts["unavailable_span_sources"] += 1
                    continue
                start, end = span["indices"]
                counts["spans_with_valid_offsets"] += 0 <= start < end <= len(source)
                counts["exact_span_offsets"] += source[start:end] == span["text"]
    return ({"source_passages": len(passages), "questions": len(ids), "answer_types": answer_types,
             "gold_link_count_answerable": gold_sizes, "source_passage_link_count": link_counts,
             "checks": counts}, ids, sources, labels)


def digest(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "dev", "articles", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    paths = {s: getattr(args, s) for s in ("train", "dev", "articles")}
    passages = {s: json.loads(paths[s].read_text()) for s in ("train", "dev")}
    linked = {link["target"] for rows in passages.values() for item in rows for link in item["links"]}
    annotated = {c["passage"] for rows in passages.values() for item in rows for q in item["questions"]
                 for c in q["context"] if c["passage"] != "main"}
    text_keys = annotated | {t.lower() for t in annotated}
    requested = linked | {t.lower() for t in linked} | text_keys
    articles = contexts(args.articles, requested, text_keys)
    by_split = {s: audit(rows, articles) for s, rows in passages.items()}
    train, dev = by_split["train"], by_split["dev"]
    summary = {"sha256": {s: digest(path) for s, path in paths.items()},
               "requested_article_keys": len(requested), "available_article_keys": len(articles),
               "train_dev_overlap": {"question_ids": len(train[1] & dev[1]), "source_titles": len(train[2] & dev[2]),
                                     "annotated_linked_titles": len(train[3] & dev[3])},
               "splits": {s: data[0] for s, data in by_split.items()}}
    args.output.write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
