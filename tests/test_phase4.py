from kam.phase4.manifest import build_rows


def test_phase4_manifest_is_paired_and_bounded():
    rows = build_rows({
        "tasks": ["prototype_switch"], "conditions": ["recurring", "separated"],
        "variants": ["D0", "DD-b", "DD-b-staged", "RF-b"], "scales": ["S"],
        "seeds_per_cell": 2,
    })
    assert len(rows) == 16
    assert {row["condition"] for row in rows} == {"recurring", "separated"}
    assert {row["variant"] for row in rows} == {"D0", "DD-b", "DD-b-staged", "RF-b"}
    assert len({row["run_id"] for row in rows}) == len(rows)
