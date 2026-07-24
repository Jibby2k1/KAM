from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_phase2_descriptive_plots import write_descriptive_metrics


def test_descriptive_metrics_capture_tail_and_calibration(tmp_path: Path) -> None:
    rows = [
        {"task": "toy", "variant": "D0", "adapter": "nlms", "seed": "7", "target": "0.0", "prediction": "0.1"},
        {"task": "toy", "variant": "D0", "adapter": "nlms", "seed": "7", "target": "1.0", "prediction": "0.8"},
        {"task": "toy", "variant": "D0", "adapter": "nlms", "seed": "7", "target": "2.0", "prediction": "2.0"},
    ]
    output = tmp_path / "descriptive.csv"
    records = write_descriptive_metrics(rows, output)
    assert output.exists()
    assert len(records) == 1
    record = records[0]
    assert record["n"] == 3
    assert record["rmse"] > 0
    assert record["p95_abs_error"] >= record["median_abs_error"]
    assert "r2" in record
    assert "log10_median_abs_error" in record
