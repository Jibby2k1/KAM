#!/usr/bin/env python3
"""Read the publisher's untimed context-need ratings for the frozen panel."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from kam_adapter_preservation import digest
from kam_math_kl_protocol import write_json
from prepare_kam_quality_native import ENTRY, PROTOCOL


def audit(source,manifest,output):
    spec=json.loads(manifest.read_text())
    assert spec["protocol"]==PROTOCOL and spec["publisher_sha256"]==digest(source)
    with ZipFile(source) as release:
        member,=(p for p in release.namelist() if Path(p).name==ENTRY and not p.startswith("__MACOSX/"))
        records=[json.loads(line) for line in release.read(member).splitlines() if line.strip()]
    records={r["question_unique_id"]:(str(article["article_id"]),r) for article in records for r in article["questions"]}
    ratings=[]
    for row in spec["selected"]:
        article_id, question=records[row["id"]]
        assert article_id==row["article_id"] and int(question["gold_label"])-1==row["answer"]
        assert int(question["difficult"])==row["hard"]
        available=[x["untimed_eval2_context"] for x in question["validation"]]
        assert all(x in (None,1,2,3,4) for x in available)
        valid=[x for x in available if x is not None]
        assert valid
        ratings.append({"id":row["id"],"values":valid})
    write_json(output,{"quality_manifest_sha256":digest(manifest),"publisher_archive_sha256":digest(source),
                       "context_ratings":ratings})


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("source","manifest","output"):
        cli.add_argument(name,type=Path)
    a=cli.parse_args()
    audit(a.source,a.manifest,a.output)
