from kam.factory import make_model
from kam.phase5.manifest import build_rows as build_validity_rows


def test_phase5_validity_manifest_is_explicit_and_bounded():
    rows = build_validity_rows({})
    assert len(rows) == 24
    assert {row["variant"] for row in rows} == {"D0", "DD-L", "RF-FULL"}
    assert {row["training_protocol"] for row in rows} == {"iid_window_training", "ordered_stream_training"}
    assert all(row["route_projection_dim"] == 64 for row in rows)


def test_phase5_labels_build():
    for label in ["DD-L", "RF-KV", "RF-FULL", "KC-LV", "RFF", "DD-PF", "DD-DRIFT"]:
        model = make_model({
            "model_name": label, "task_type": "regression", "d_model": 16,
            "num_heads": 4, "num_layers": 1, "num_supports": 16,
            "max_seq_len": 16, "input_dim": 2, "output_dim": 1,
            "memory_output": "both", "route_features": "projected",
            "route_projection_dim": 64,
        })
        if label == "RFF":
            assert model.route_feature_dim == 0
        else:
            assert model.route_feature_dim == 64

from kam.phase5.pilot_manifest import build_rows as build_pilot_rows


def test_phase5_pilot_manifest_has_paired_stage1_matrix():
    rows = build_pilot_rows({})
    assert len(rows) == 144
    assert {row["variant"] for row in rows} == {"D0", "DD-L", "RF-KV", "RF-FULL", "KC-LV", "RFF"}
    assert {row["scale"] for row in rows} == {"P250", "P1M"}
    assert len({(row["task"], row["scale"], row["seed_index"]) for row in rows}) == 24
    assert all(row["active_match_tolerance"] == 0.01 for row in rows)
